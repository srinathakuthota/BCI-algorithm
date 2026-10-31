"""
pipeline.py — leakage-controlled preprocessing, FBCSP/MIBIF, and all models.

Every data-dependent step is a fit(train) -> transform(any) object, so that within
each partition (CV fold, pooled split, LOSO fold, inter-session split) the
coefficients are estimated on the training partition only and applied unchanged
to validation / test data.
"""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import numpy as np
from scipy.linalg import eigh
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers, constraints

tf.config.threading.set_inter_op_parallelism_threads(2)
tf.config.threading.set_intra_op_parallelism_threads(2)

PAD_CH = 64


# ----------------------------------------------------------------------------- preprocessing
class EOGRegression:
    """EEG_clean = EEG - EOG @ b, with b (3 x C_eeg) estimated by least squares on the
    training-partition epochs only (broadband 4-40 Hz). Because filtering and this spatial
    regression are both linear, the same b is applied to every sub-band signal."""

    def __init__(self, n_eeg):
        self.n_eeg = n_eeg

    def fit(self, W):
        eeg = W[:, :self.n_eeg].transpose(1, 0, 2).reshape(self.n_eeg, -1)
        eog = W[:, self.n_eeg:].transpose(1, 0, 2).reshape(W.shape[1] - self.n_eeg, -1)
        self.b = np.linalg.lstsq(eog.T.astype(np.float64), eeg.T.astype(np.float64), rcond=None)[0]  # (3, C)
        return self

    def transform(self, X):
        """X: (..., C_eeg + 3, T) -> (..., C_eeg, T)"""
        eeg, eog = X[..., :self.n_eeg, :], X[..., self.n_eeg:, :]
        return (eeg - np.einsum("ec,...et->...ct", self.b, eog)).astype(np.float32)


class ChannelZ:
    """Per-channel z-score with mean/std from the training partition."""

    def fit(self, X):
        self.mu = X.mean(axis=(0, 2), keepdims=True)
        self.sd = X.std(axis=(0, 2), keepdims=True) + 1e-8
        return self

    def transform(self, X):
        return ((X - self.mu) / self.sd).astype(np.float32)


# ----------------------------------------------------------------------------- FBCSP + MIBIF
def _covs(X):
    """trace-normalised sample covariance per trial. X: (n, C, T)"""
    Xc = X - X.mean(axis=2, keepdims=True)
    C = np.einsum("nct,ndt->ncd", Xc, Xc) / X.shape[2]
    return C / np.trace(C, axis1=1, axis2=2)[:, None, None]


class FBCSP:
    """Filter-bank CSP.
    Class covariance: arithmetic mean of the trace-normalised single-trial covariances of
    that class, with Tikhonov regularisation R <- (1-r) R + r * tr(R)/C * I, r = 0.01.
    Multi-class (2a): one-versus-rest, one CSP per (band, class). Binary (2b): one CSP per band.
    Filters: generalised eigenproblem R_a w = lambda (R_a + R_b) w; the m eigenvectors with the
    largest and the m with the smallest eigenvalues are kept (2m filters). If 2m > C (2b, C=3)
    all C filters are kept.
    Feature: f_j = log( var(z_j) / sum_i var(z_i) ), sum over the filters of that CSP problem."""

    def __init__(self, m=2, reg=0.01):
        self.m, self.reg = m, reg

    def _reg(self, R):
        C = R.shape[0]
        return (1 - self.reg) * R + self.reg * np.trace(R) / C * np.eye(C)

    def fit(self, B, y):
        # B: (n, n_bands, C, T)
        self.classes = np.unique(y)
        pairs = [(self.classes[0], None)] if len(self.classes) == 2 else [(c, "rest") for c in self.classes]
        self.filters = []  # list over bands of list over pairs of (C, nf)
        for b in range(B.shape[1]):
            covs = _covs(B[:, b].astype(np.float64))
            fb = []
            for c, _ in pairs:
                Ra = self._reg(covs[y == c].mean(0))
                Rb = self._reg(covs[y != c].mean(0))
                w, V = eigh(Ra, Ra + Rb)            # ascending eigenvalues
                C = V.shape[0]
                if 2 * self.m >= C:
                    idx = np.arange(C)
                else:
                    idx = np.r_[np.arange(self.m), np.arange(C - self.m, C)]
                fb.append(V[:, idx])
            self.filters.append(fb)
        return self

    def transform(self, B):
        feats = []
        for b, fb in enumerate(self.filters):
            Xb = B[:, b].astype(np.float64)
            for Wf in fb:
                Z = np.einsum("cf,nct->nft", Wf, Xb)
                v = Z.var(axis=2)
                feats.append(np.log(v / v.sum(axis=1, keepdims=True)))
        return np.concatenate(feats, axis=1).astype(np.float32)


class MIBIF:
    """Mutual-information-based best individual feature selection.
    MI between each feature and the (multi-class) label is estimated on the training
    partition with the k-nearest-neighbour estimator of Kraskov et al. as implemented in
    sklearn.feature_selection.mutual_info_classif (n_neighbors=3, fixed random_state);
    the k features with the largest MI are kept. For 2a the multi-class label is used
    directly (not per OvR problem); every OvR CSP feature competes in a single ranking."""

    def __init__(self, k, seed=0):
        self.k, self.seed = k, seed

    def fit(self, F, y):
        mi = mutual_info_classif(F, y, n_neighbors=3, random_state=self.seed)
        self.idx = np.argsort(mi)[::-1][: self.k]
        self.mi = mi
        return self

    def transform(self, F):
        return F[:, self.idx]


class Features:
    """Full leakage-controlled feature pipeline. fit() sees ONLY training trials."""

    def __init__(self, n_eeg, m=2, k=20, seed=0):
        self.n_eeg, self.m, self.k, self.seed = n_eeg, m, k, seed

    def fit(self, W, B, y):
        self.eog = EOGRegression(self.n_eeg).fit(W)
        Wc = self.eog.transform(W)
        self.z = ChannelZ().fit(Wc)
        Bc = self.eog.transform(B)
        self.csp = FBCSP(self.m).fit(Bc, y)
        F = self.csp.transform(Bc)
        self.mib = MIBIF(self.k, self.seed).fit(F, y)
        self.fs = StandardScaler().fit(self.mib.transform(F))
        return self

    def transform(self, W, B):
        Xt = self.z.transform(self.eog.transform(W))           # (n, C, T)  temporal input
        F = self.fs.transform(self.mib.transform(self.csp.transform(self.eog.transform(B)))).astype(np.float32)
        return Xt, F


def to_conv_input(Xt):
    """(n, C, T) -> (n, T, C).
    The manuscript zero-pads C to 64 channels. Zero input channels receive exactly zero
    gradient, so with Adam (no weight decay) their kernel weights never change and never
    contribute to any activation: the padded network is functionally identical to the
    unpadded one PROVIDED the first-layer initialisation uses the padded fan-in (3 x 64).
    We therefore feed the C real channels and initialise the first Conv1D with
    He-uniform limits computed for fan_in = 3 x 64, which reproduces the padded model while
    avoiding ~(64/C)x wasted computation. Parameter counts in Table 1 refer to the padded form."""
    return np.ascontiguousarray(Xt.transpose(0, 2, 1))


# ----------------------------------------------------------------------------- models
def _conv_trunk(inp):
    lim = float(np.sqrt(6.0 / (3 * PAD_CH)))                    # He-uniform for padded fan-in
    x = layers.Conv1D(64, 3, activation="relu",
                      kernel_initializer=tf.keras.initializers.RandomUniform(-lim, lim))(inp)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(64, 3, activation="relu", kernel_initializer="he_uniform")(x)
    x = layers.MaxPooling1D(2)(x)
    return layers.Flatten()(x)


def _head(x, n_cls, dropout=0.5):
    x = layers.Dense(128, activation="relu", kernel_initializer="he_uniform")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(64, activation="relu", kernel_initializer="he_uniform")(x)
    x = layers.Dropout(dropout)(x)
    return layers.Dense(n_cls, activation="softmax")(x)


def build_dual(T, k, n_cls, dropout=0.5, C=PAD_CH):
    a = layers.Input((T, C), name="eeg")
    f = layers.Input((k,), name="fbcsp")
    x = layers.Concatenate()([_conv_trunk(a), f])
    return models.Model([a, f], _head(x, n_cls, dropout))


def build_conv_only(T, n_cls, dropout=0.5, C=PAD_CH):
    a = layers.Input((T, C))
    return models.Model(a, _head(_conv_trunk(a), n_cls, dropout))


def build_fbcsp_mlp(k, n_cls, dropout=0.5):
    f = layers.Input((k,))
    return models.Model(f, _head(f, n_cls, dropout))


def build_eegnet(C, T, n_cls, F1=8, D=2, F2=16, kern=64, dropout=0.5):
    """EEGNet-8,2 (Lawhern et al., 2018), re-implemented; input (C, T, 1)."""
    i = layers.Input((C, T, 1))
    x = layers.Conv2D(F1, (1, kern), padding="same", use_bias=False)(i)
    x = layers.BatchNormalization()(x)
    x = layers.DepthwiseConv2D((C, 1), use_bias=False, depth_multiplier=D,
                               depthwise_constraint=constraints.max_norm(1.0))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("elu")(x)
    x = layers.AveragePooling2D((1, 4))(x)
    x = layers.Dropout(dropout)(x)
    x = layers.SeparableConv2D(F2, (1, 16), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("elu")(x)
    x = layers.AveragePooling2D((1, 8))(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Flatten()(x)
    o = layers.Dense(n_cls, activation="softmax", kernel_constraint=constraints.max_norm(0.25))(x)
    return models.Model(i, o)


def build_shallow_cnn(C, T, n_cls, dropout=0.5):
    """CNN baseline = ShallowConvNet (Schirrmeister et al., 2017): temporal conv (40, 1x25),
    spatial conv (40, Cx1), BN, square, avg-pool (1x75, stride 15), log, dropout, dense."""
    i = layers.Input((C, T, 1))
    x = layers.Conv2D(40, (1, 25))(i)
    x = layers.Conv2D(40, (C, 1), use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Lambda(lambda t: tf.square(t))(x)
    x = layers.AveragePooling2D((1, 75), strides=(1, 15))(x)
    x = layers.Lambda(lambda t: tf.math.log(tf.clip_by_value(t, 1e-6, 1e6)))(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Flatten()(x)
    return models.Model(i, layers.Dense(n_cls, activation="softmax")(x))


def build_lstm(C, T, n_cls, dropout=0.5):
    """LSTM baseline: input averaged over non-overlapping 4-sample windows (T/4 = 125 steps,
    62.5 Hz), two stacked LSTM layers (64, 32 units), dropout, softmax."""
    i = layers.Input((T // 4, C))
    x = layers.LSTM(64, return_sequences=True)(i)
    x = layers.LSTM(32)(x)
    x = layers.Dropout(dropout)(x)
    return models.Model(i, layers.Dense(n_cls, activation="softmax")(x))


DEEP = ["dual", "conv1d_only", "fbcsp_mlp", "eegnet", "cnn_shallow", "lstm"]
SHALLOW = ["fbcsp_svm", "fbcsp_knn", "fbcsp_lda"]


def model_inputs(name, Xt, F):
    if name == "dual":
        return [to_conv_input(Xt), F]
    if name == "conv1d_only":
        return to_conv_input(Xt)
    if name == "fbcsp_mlp":
        return F
    if name in ("eegnet", "cnn_shallow"):
        return Xt[..., None]
    if name == "lstm":
        n, C, T = Xt.shape
        return Xt[:, :, : T // 4 * 4].reshape(n, C, T // 4, 4).mean(-1).transpose(0, 2, 1)
    raise ValueError(name)


def build(name, C, T, k, n_cls, dropout=0.5, eegnet_dropout=0.5):
    return {"dual": lambda: build_dual(T, k, n_cls, dropout, C),
            "conv1d_only": lambda: build_conv_only(T, n_cls, dropout, C),
            "fbcsp_mlp": lambda: build_fbcsp_mlp(k, n_cls, dropout),
            "eegnet": lambda: build_eegnet(C, T, n_cls, dropout=eegnet_dropout),
            "cnn_shallow": lambda: build_shallow_cnn(C, T, n_cls),
            "lstm": lambda: build_lstm(C, T, n_cls)}[name]()


def _index(inp, idx):
    return [a[idx] for a in inp] if isinstance(inp, list) else inp[idx]


def train_deep(name, Xt_tr, F_tr, y_tr, n_cls, seed, lr=1e-3, epochs=100, patience=15,
               batch=32, val_frac=0.1, dropout=0.5, eegnet_dropout=0.5, verbose=0):
    """Adam(lr), sparse CE, batch 32, <=100 epochs, early stopping on the loss of a stratified
    10 % validation subset of the TRAINING partition (patience 15, best weights restored)."""
    tf.keras.utils.set_random_seed(seed)
    C, T = Xt_tr.shape[1], Xt_tr.shape[2]
    k = F_tr.shape[1]
    inp = model_inputs(name, Xt_tr, F_tr)
    idx = np.arange(len(y_tr))
    tr, va = train_test_split(idx, test_size=val_frac, stratify=y_tr, random_state=seed)
    m = build(name, C, T, k, n_cls, dropout, eegnet_dropout)
    m.compile(optimizers.Adam(lr), "sparse_categorical_crossentropy", metrics=["accuracy"])
    es = callbacks.EarlyStopping("val_loss", patience=patience, restore_best_weights=True)
    h = m.fit(_index(inp, tr), y_tr[tr], validation_data=(_index(inp, va), y_tr[va]),
              epochs=epochs, batch_size=batch, callbacks=[es], verbose=verbose)
    return m, h.history


def predict_deep(m, name, Xt, F):
    return m.predict(model_inputs(name, Xt, F), batch_size=256, verbose=0)


def train_shallow(name, F_tr, y_tr, seed):
    """FBCSP features (the same k selected, standardised features given to the dual model).
    Hyper-parameters selected by 5-fold stratified CV inside the training partition only."""
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    if name == "fbcsp_svm":
        g = GridSearchCV(SVC(kernel="rbf", probability=True, random_state=seed),
                         {"C": [0.1, 1, 10, 100], "gamma": ["scale", 0.01, 0.1]}, cv=cv)
    elif name == "fbcsp_knn":
        g = GridSearchCV(KNeighborsClassifier(), {"n_neighbors": [3, 5, 7, 9, 15, 25],
                                                  "weights": ["uniform", "distance"]}, cv=cv)
    elif name == "fbcsp_lda":
        return LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(F_tr, y_tr), {}
    g.fit(F_tr, y_tr)
    return g.best_estimator_, g.best_params_
