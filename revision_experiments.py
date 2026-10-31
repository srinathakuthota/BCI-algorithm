#!/usr/bin/env python3
"""
revision_experiments.py
=======================
Experiments needed to answer Reviewer 1 and Reviewer 2 (MTI, dual-branch FBCSP-deep learning
framework for MI classification on BCI Competition IV 2a / 2b).

Every number this script prints is written to ./results/*.csv so that each value placed in the
revised manuscript can be traced to the exact run, seed, split and subject that produced it.

What it produces (reviewer comment it answers in brackets)
---------------------------------------------------------
  pooled   : pooled 80/20 split, stratified by subject x class (x session), repeated over
             N_SPLITS splits x N_SEEDS seeds; per-subject accuracy is computed on each subject's
             own test trials                                   [R1.1, R1.2, R2.1, R2.3, R2.7]
  session  : true per-subject, session-to-session evaluation (2a: session T -> E for all 9
             subjects; 2b: sessions 1-3 -> 4-5), the protocol used by most papers in Table 13
                                                                [R1.3, R2.2, R2.3]
  loso     : leave-one-subject-out with NESTED hyperparameter selection: the grid search is
             repeated inside each outer fold using only the 8 training subjects
             (inner leave-one-subject-out), the held-out subject never touches model selection
                                                                [R2.4, R2.7]
  Models   : dual (FBCSP+Conv1D), conv_only, fbcsp_only (ablation, same dense head),
             svm, knn (on FBCSP+MIBIF features), cnn, lstm, eegnet  [R1.4, R2.5, R2.6]
  Noise    : clean and 5 dB AWGN for dual, conv_only, fbcsp_only    [R2.5]
  Confusion matrices are built from the SAME test predictions that give the accuracies [R2.1]

Leakage control: EOG regression coefficients, z-score statistics, CSP filters and MIBIF
selection are fitted on the training partition only and applied unchanged to test data [R2.9].

Usage
-----
  pip install moabb mne tensorflow scikit-learn scipy pandas
  python revision_experiments.py --dataset 2a --protocol pooled  --models dual conv_only fbcsp_only svm knn cnn lstm eegnet
  python revision_experiments.py --dataset 2a --protocol session --models dual conv_only fbcsp_only eegnet
  python revision_experiments.py --dataset 2b --protocol loso    --models dual
  python revision_experiments.py --synthetic --protocol pooled --models dual --seeds 1 --splits 1   # smoke test
"""
import argparse
import itertools
import os
import time

import numpy as np
import pandas as pd
from scipy.linalg import eigh
from scipy.signal import butter, sosfiltfilt
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

FS = 250
T_MIN, T_MAX = 0.5, 2.5                     # s relative to cue -> 500 samples
N_T = int(round((T_MAX - T_MIN) * FS))
BANDS = [(f, f + 4) for f in range(4, 40, 4)]  # 9 sub-bands, 4-40 Hz
PAD_CH = 64

# ----------------------------------------------------------------------------- data
def load_moabb(dataset):
    """Return dict with eeg (n, C, T), eog (n, 3, T), y, subject, session (all trials)."""
    import mne
    from moabb.datasets import BNCI2014_001, BNCI2014_004
    mne.set_log_level("ERROR")
    ds = BNCI2014_001() if dataset == "2a" else BNCI2014_004()
    classes = ["left_hand", "right_hand", "feet", "tongue"] if dataset == "2a" else ["left_hand", "right_hand"]
    out = {k: [] for k in ["eeg", "eog", "y", "subject", "session"]}
    for subj in ds.subject_list:
        data = ds.get_data([subj])[subj]
        for s_idx, sess in enumerate(sorted(data.keys())):
            for run in sorted(data[sess].keys()):
                raw = data[sess][run].copy()
                ev, ev_id = mne.events_from_annotations(raw, event_id={c: ds.event_id[c] for c in classes if c in ds.event_id})
                if len(ev) == 0:
                    continue
                eeg_picks = mne.pick_types(raw.info, eeg=True, eog=False, stim=False)
                eog_picks = mne.pick_types(raw.info, eeg=False, eog=True)
                sig = raw.get_data()
                sos = butter(5, [4, 40], btype="bandpass", fs=FS, output="sos")
                sig = sosfiltfilt(sos, sig, axis=-1)          # zero-phase, causal-free; per run
                inv = {v: i for i, v in enumerate([ds.event_id[c] for c in classes])}
                for onset, _, code in ev:
                    a = onset + int(T_MIN * FS)
                    b = a + N_T
                    if b > sig.shape[1] or code not in inv:
                        continue
                    out["eeg"].append(sig[eeg_picks, a:b])
                    out["eog"].append(sig[eog_picks, a:b] if len(eog_picks) else np.zeros((1, N_T)))
                    out["y"].append(inv[code]); out["subject"].append(subj); out["session"].append(s_idx)
    return {k: np.asarray(v) for k, v in out.items()}


def load_synthetic(dataset, n_per=40, seed=0):
    """Tiny synthetic stand-in, only to check that the pipeline runs end to end."""
    rng = np.random.default_rng(seed)
    C, K = (22, 4) if dataset == "2a" else (3, 2)
    n_sess = 2 if dataset == "2a" else 5
    out = {k: [] for k in ["eeg", "eog", "y", "subject", "session"]}
    t = np.arange(N_T) / FS
    for subj in range(1, 10):
        for sess in range(n_sess):
            for k in range(K):
                for _ in range(n_per // n_sess):
                    x = rng.standard_normal((C, N_T))
                    x[k % C] += 1.5 * np.sin(2 * np.pi * (10 + 2 * k) * t)
                    out["eeg"].append(x); out["eog"].append(rng.standard_normal((3, N_T)))
                    out["y"].append(k); out["subject"].append(subj); out["session"].append(sess)
    return {k: np.asarray(v) for k, v in out.items()}

# ----------------------------------------------------------------------------- preprocessing
class EOGRegression:
    """Least-squares ocular regression, EEG = EOG @ B + residual. Label-free; fitted on the
    training partition only and applied unchanged to validation/test trials (R2.9)."""
    def fit(self, eeg, eog):
        E = np.concatenate(list(eog), axis=1).T            # (n*T, n_eog)
        X = np.concatenate(list(eeg), axis=1).T            # (n*T, C)
        self.B = np.linalg.lstsq(E, X, rcond=None)[0]      # (n_eog, C)
        return self
    def transform(self, eeg, eog):
        return eeg - np.einsum("ec,net->nct", self.B, eog)


class ChannelZScore:
    def fit(self, X):
        self.mu = X.mean(axis=(0, 2), keepdims=True); self.sd = X.std(axis=(0, 2), keepdims=True) + 1e-8
        return self
    def transform(self, X):
        return (X - self.mu) / self.sd

# ----------------------------------------------------------------------------- FBCSP + MIBIF
def _cov(X, shrink):
    """Trace-normalised trial covariance averaged over trials, optional shrinkage (R2.8)."""
    C = np.mean([x @ x.T / np.trace(x @ x.T) for x in X], axis=0)
    if shrink > 0:
        C = (1 - shrink) * C + shrink * np.trace(C) / C.shape[0] * np.eye(C.shape[0])
    return C


def _parzen_mi(f, y):
    """Mutual information I(f; y) with a Gaussian Parzen-window density (Ang et al., 2008)."""
    n = len(f); classes = np.unique(y)
    h = 1.06 * np.std(f) * n ** (-1 / 5) + 1e-12
    K = np.exp(-0.5 * ((f[:, None] - f[None, :]) / h) ** 2)
    p_w = np.array([np.mean(y == c) for c in classes])
    H_w = -np.sum(p_w * np.log2(p_w))
    lik = np.stack([K[:, y == c].mean(axis=1) for c in classes], axis=1)   # p(f_i | w)
    post = lik * p_w; post /= post.sum(axis=1, keepdims=True) + 1e-300
    H_w_f = -np.mean(np.sum(post * np.log2(post + 1e-300), axis=1))
    return H_w - H_w_f


class FBCSP:
    """One-versus-rest FBCSP for K>2, standard two-class CSP for K=2.
    For each band b and each OvR problem c: solve R_c w = lambda (R_c + R_rest) w,
    R_rest = mean of the other class covariances. Keep m filters from each end of the
    spectrum (m is capped at floor(C/2), so 2b with 3 channels uses m=1 -> 2 filters/band).
    Feature: f = log( var(w^T x) / sum_j var(w_j^T x) ). MIBIF: Parzen MI against the full
    label vector, top-k features plus their paired filter (Ang et al., 2008)."""
    def __init__(self, m=2, k=20, shrink=0.0):
        self.m, self.k, self.shrink = m, k, shrink
        self.sos = [butter(5, b, btype="bandpass", fs=FS, output="sos") for b in BANDS]

    def _band(self, X, i):
        return sosfiltfilt(self.sos[i], X, axis=-1)

    def fit(self, X, y):
        self.classes = np.unique(y); C = X.shape[1]
        m = min(self.m, C // 2)
        problems = [None] if len(self.classes) == 2 else list(self.classes)
        self.W = []
        for i in range(len(BANDS)):
            Xb = self._band(X, i)
            covs = {c: _cov(Xb[y == c], self.shrink) for c in self.classes}
            Wb = []
            for c in problems:
                if c is None:
                    Ra, Rb = covs[self.classes[0]], covs[self.classes[1]]
                else:
                    Ra = covs[c]; Rb = np.mean([covs[o] for o in self.classes if o != c], axis=0)
                vals, vecs = eigh(Ra, Ra + Rb)
                idx = np.argsort(vals)[::-1]
                Wb.append(vecs[:, np.r_[idx[:m], idx[-m:]]])
            self.W.append(np.concatenate(Wb, axis=1))
        F = self._features(X)
        self.n_per_problem = 2 * m
        mi = np.array([_parzen_mi(F[:, j], y) for j in range(F.shape[1])])
        order = np.argsort(mi)[::-1]
        sel = []
        for j in order:                          # add feature and its CSP partner
            g, r = divmod(j, self.n_per_problem); half = self.n_per_problem // 2
            partner = g * self.n_per_problem + (self.n_per_problem - 1 - r)   # largest <-> smallest
            for q in (j, partner):
                if q not in sel and len(sel) < self.k:
                    sel.append(q)
            if len(sel) >= self.k:
                break
        self.sel = np.array(sel)
        return self

    def _features(self, X):
        feats = []
        for i, Wb in enumerate(self.W):
            Z = np.einsum("cf,nct->nft", Wb, self._band(X, i))
            v = Z.var(axis=-1)
            # normalise within each OvR group of 2m filters
            g = v.reshape(len(X), -1, 2 * min(self.m, X.shape[1] // 2))
            feats.append(np.log(g / g.sum(axis=-1, keepdims=True)).reshape(len(X), -1))
        return np.concatenate(feats, axis=1)

    def transform(self, X):
        return self._features(X)[:, self.sel]

# ----------------------------------------------------------------------------- networks
def _tf():
    import tensorflow as tf
    return tf


def build_net(kind, n_ch, n_fb, n_cls, lr=1e-3, drop=0.5, d1=128, d2=64):
    tf = _tf(); L = tf.keras.layers
    init = "he_uniform"
    inputs, branches = [], []
    if kind in ("dual", "conv_only", "cnn"):
        xin = L.Input((N_T, PAD_CH), name="eeg"); inputs.append(xin)
        h = L.Conv1D(64, 3, activation="relu", kernel_initializer=init)(xin)
        h = L.MaxPooling1D(2)(h)
        h = L.Conv1D(64, 3, activation="relu", kernel_initializer=init)(h)
        h = L.MaxPooling1D(2)(h)
        branches.append(L.Flatten()(h))
    if kind in ("dual", "fbcsp_only"):
        fin = L.Input((n_fb,), name="fbcsp"); inputs.append(fin); branches.append(fin)
    h = branches[0] if len(branches) == 1 else L.Concatenate()(branches)
    h = L.Dropout(drop)(L.Dense(d1, activation="relu", kernel_initializer=init)(h))
    h = L.Dropout(drop)(L.Dense(d2, activation="relu", kernel_initializer=init)(h))
    out = L.Dense(n_cls, activation="softmax")(h)
    model = tf.keras.Model(inputs, out)
    model.compile(tf.keras.optimizers.Adam(lr), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def build_lstm(n_ch, n_cls, lr=1e-3):
    tf = _tf(); L = tf.keras.layers
    xin = L.Input((N_T, n_ch))
    h = L.AveragePooling1D(4)(xin)                 # 500 -> 125 steps
    h = L.LSTM(64, return_sequences=True)(h)
    h = L.LSTM(64)(h)
    h = L.Dropout(0.5)(L.Dense(64, activation="relu")(h))
    m = tf.keras.Model(xin, L.Dense(n_cls, activation="softmax")(h))
    m.compile(tf.keras.optimizers.Adam(lr), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return m


def build_eegnet(n_ch, n_cls, F1=8, D=2, F2=16, kern=125, drop=0.5, lr=1e-3):
    """EEGNet-8,2 (Lawhern et al., 2018) re-implemented; kernLength = fs/2 = 125 at 250 Hz."""
    tf = _tf(); L = tf.keras.layers
    xin = L.Input((n_ch, N_T, 1))
    h = L.Conv2D(F1, (1, kern), padding="same", use_bias=False)(xin)
    h = L.BatchNormalization()(h)
    h = L.DepthwiseConv2D((n_ch, 1), depth_multiplier=D, use_bias=False,
                          depthwise_constraint=tf.keras.constraints.max_norm(1.0))(h)
    h = L.Activation("elu")(L.BatchNormalization()(h))
    h = L.Dropout(drop)(L.AveragePooling2D((1, 4))(h))
    h = L.SeparableConv2D(F2, (1, 16), padding="same", use_bias=False)(h)
    h = L.Activation("elu")(L.BatchNormalization()(h))
    h = L.Dropout(drop)(L.AveragePooling2D((1, 8))(h))
    out = L.Dense(n_cls, activation="softmax", kernel_constraint=tf.keras.constraints.max_norm(0.25))(L.Flatten()(h))
    m = tf.keras.Model(xin, out)
    m.compile(tf.keras.optimizers.Adam(lr), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return m

# ----------------------------------------------------------------------------- one train/test run
def add_awgn(X, snr_db, rng):
    p = X.var(axis=-1, keepdims=True)
    return X + rng.standard_normal(X.shape) * np.sqrt(p / 10 ** (snr_db / 10))


def prepare(Xtr_eeg, Xtr_eog, Xte_eeg, Xte_eog, ytr, fb_cfg):
    """All data-dependent steps fitted on the training partition only."""
    eogr = EOGRegression().fit(Xtr_eeg, Xtr_eog)
    Xtr = eogr.transform(Xtr_eeg, Xtr_eog); Xte = eogr.transform(Xte_eeg, Xte_eog)
    fb = FBCSP(**fb_cfg).fit(Xtr, ytr)
    Ftr, Fte = fb.transform(Xtr), fb.transform(Xte)
    fsc = StandardScaler().fit(Ftr); Ftr, Fte = fsc.transform(Ftr), fsc.transform(Fte)
    zs = ChannelZScore().fit(Xtr); Ztr, Zte = zs.transform(Xtr), zs.transform(Xte)
    return Ztr, Zte, Ftr, Fte, fb, zs, fsc


def to_conv(Z):
    """(n, C, T) -> (n, T, 64) with zero-padding of the channel axis (as in the manuscript)."""
    out = np.zeros((len(Z), N_T, PAD_CH), dtype=np.float32)
    out[:, :, :Z.shape[1]] = np.transpose(Z, (0, 2, 1))
    return out


def fit_predict(model_name, Ztr, Ftr, ytr, Zte, Fte, n_cls, seed, hp, epochs, noise_rng=None, Zte_noisy=None, Fte_noisy=None):
    tf = _tf()
    tf.keras.utils.set_random_seed(seed)
    if model_name == "svm":
        clf = SVC(C=hp.get("C", 1.0), gamma=hp.get("gamma", "scale"), kernel="rbf").fit(Ftr, ytr)
        return clf.predict(Fte), None
    if model_name == "knn":
        clf = KNeighborsClassifier(n_neighbors=hp.get("n_neighbors", 7)).fit(Ftr, ytr)
        return clf.predict(Fte), None
    es = tf.keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True)
    # validation subset is carved from the TRAINING partition only
    sss = StratifiedShuffleSplit(1, test_size=0.1, random_state=seed)
    tr_i, va_i = next(sss.split(Ftr, ytr))
    kw = dict(epochs=epochs, batch_size=32, verbose=0, callbacks=[es])
    if model_name in ("dual", "conv_only", "fbcsp_only", "cnn"):
        m = build_net("conv_only" if model_name == "cnn" else model_name, Ztr.shape[1], Ftr.shape[1], n_cls,
                      lr=hp.get("lr", 1e-3), drop=hp.get("drop", 0.5), d1=hp.get("d1", 128), d2=hp.get("d2", 64))
        def inp(Z, F):
            d = []                                   # same order as model.inputs
            if model_name in ("dual", "conv_only", "cnn"): d.append(to_conv(Z))
            if model_name in ("dual", "fbcsp_only"): d.append(F.astype(np.float32))
            return d[0] if len(d) == 1 else d
        m.fit(inp(Ztr[tr_i], Ftr[tr_i]), ytr[tr_i], validation_data=(inp(Ztr[va_i], Ftr[va_i]), ytr[va_i]), **kw)
        pred = m.predict(inp(Zte, Fte), verbose=0).argmax(1)
        pred_noisy = None
        if Zte_noisy is not None:
            pred_noisy = m.predict(inp(Zte_noisy, Fte_noisy), verbose=0).argmax(1)
        return pred, pred_noisy
    if model_name == "lstm":
        m = build_lstm(Ztr.shape[1], n_cls, lr=hp.get("lr", 1e-3))
        A = lambda Z: np.transpose(Z, (0, 2, 1)).astype(np.float32)
        m.fit(A(Ztr[tr_i]), ytr[tr_i], validation_data=(A(Ztr[va_i]), ytr[va_i]), **kw)
        return m.predict(A(Zte), verbose=0).argmax(1), None
    if model_name == "eegnet":
        m = build_eegnet(Ztr.shape[1], n_cls, drop=hp.get("drop", 0.5), lr=hp.get("lr", 1e-3))
        A = lambda Z: Z[..., None].astype(np.float32)
        m.fit(A(Ztr[tr_i]), ytr[tr_i], validation_data=(A(Ztr[va_i]), ytr[va_i]), **kw)
        return m.predict(A(Zte), verbose=0).argmax(1), None
    raise ValueError(model_name)

# ----------------------------------------------------------------------------- protocols
HP_GRID = {
    "dual":       [dict(lr=lr, drop=d) for lr in (1e-3, 5e-4) for d in (0.3, 0.5)],
    "conv_only":  [dict(lr=lr, drop=d) for lr in (1e-3, 5e-4) for d in (0.3, 0.5)],
    "fbcsp_only": [dict(lr=lr, drop=d) for lr in (1e-3, 5e-4) for d in (0.3, 0.5)],
    "cnn":        [dict(lr=lr, drop=d) for lr in (1e-3, 5e-4) for d in (0.3, 0.5)],
    "svm":        [dict(C=c, gamma=g) for c in (0.1, 1, 10) for g in ("scale", 0.01, 0.1)],
    "knn":        [dict(n_neighbors=k) for k in (3, 5, 7, 11)],
    "lstm":       [dict(lr=lr) for lr in (1e-3, 5e-4)],
    "eegnet":     [dict(drop=d) for d in (0.25, 0.5)],
}


def record(rows, cms, tag, model, seed, split, subj_te, y_te, pred, extra=None):
    for s in np.unique(subj_te):
        msk = subj_te == s
        p, r, f, _ = precision_recall_fscore_support(y_te[msk], pred[msk], average="macro", zero_division=0)
        rows.append(dict(protocol=tag, model=model, seed=seed, split=split, subject=int(s), n_test=int(msk.sum()),
                         acc=100 * accuracy_score(y_te[msk], pred[msk]), prec=p, rec=r, f1=f, **(extra or {})))
    p, r, f, _ = precision_recall_fscore_support(y_te, pred, average="macro", zero_division=0)
    rows.append(dict(protocol=tag, model=model, seed=seed, split=split, subject="ALL", n_test=len(y_te),
                     acc=100 * accuracy_score(y_te, pred), prec=p, rec=r, f1=f, **(extra or {})))
    cms.append(dict(protocol=tag, model=model, seed=seed, split=split, **(extra or {}),
                    cm=confusion_matrix(y_te, pred).tolist()))


def inner_select(model, D, tr_idx, groups, n_cls, fb_cfg, epochs, seed, inner="loso"):
    """Nested model selection on the training partition only. Returns best hp dict."""
    grid = HP_GRID[model]
    if len(grid) == 1:
        return grid[0]
    if inner == "loso":
        folds = [(tr_idx[groups[tr_idx] != g], tr_idx[groups[tr_idx] == g]) for g in np.unique(groups[tr_idx])]
    else:
        sss = StratifiedShuffleSplit(3, test_size=0.2, random_state=seed)
        folds = [(tr_idx[a], tr_idx[b]) for a, b in sss.split(tr_idx, D["y"][tr_idx])]
    prepared = [(a, b, prepare(D["eeg"][a], D["eog"][a], D["eeg"][b], D["eog"][b], D["y"][a], fb_cfg)[:4]) for a, b in folds]
    scores = []
    for hp in grid:
        acc = []
        for a, b, (Ztr, Zva, Ftr, Fva) in prepared:
            pred, _ = fit_predict(model, Ztr, Ftr, D["y"][a], Zva, Fva, n_cls, seed, hp, epochs)
            acc.append(accuracy_score(D["y"][b], pred))
        scores.append(np.mean(acc))
    return grid[int(np.argmax(scores))]


def run(D, dataset, protocol, models, n_seeds, n_splits, epochs, nested, snr_db, outdir):
    n_cls = len(np.unique(D["y"]))
    fb_cfg = dict(m=2, k=20, shrink=0.0) if dataset == "2a" else dict(m=1, k=10, shrink=0.0)
    rows, cms = [], []
    n = len(D["y"])
    splits = []
    if protocol == "pooled":
        strat = np.array([f"{s}_{c}_{e}" for s, c, e in zip(D["subject"], D["y"], D["session"])])
        sss = StratifiedShuffleSplit(n_splits, test_size=0.2, random_state=0)
        splits = [(f"split{i}", a, b) for i, (a, b) in enumerate(sss.split(np.zeros(n), strat))]
    elif protocol == "session":
        last_train = 0 if dataset == "2a" else 2
        for s in np.unique(D["subject"]):
            a = np.where((D["subject"] == s) & (D["session"] <= last_train))[0]
            b = np.where((D["subject"] == s) & (D["session"] > last_train))[0]
            splits.append((f"S{s}", a, b))
    elif protocol == "loso":
        for s in np.unique(D["subject"]):
            splits.append((f"LOSO_S{s}", np.where(D["subject"] != s)[0], np.where(D["subject"] == s)[0]))
    for (name, tr, te), seed in itertools.product(splits, range(n_seeds)):
        t0 = time.time()
        Ztr, Zte, Ftr, Fte, fb, zs, fsc = prepare(D["eeg"][tr], D["eog"][tr], D["eeg"][te], D["eog"][te], D["y"][tr], fb_cfg)
        rng = np.random.default_rng(1000 + seed)
        # noisy copy of the held-out trials: noise added BEFORE EOG regression / feature extraction
        noisy_eeg = add_awgn(D["eeg"][te], snr_db, rng)
        _, Zte_n, _, Fte_n, *_ = prepare(D["eeg"][tr], D["eog"][tr], noisy_eeg, D["eog"][te], D["y"][tr], fb_cfg)
        for model in models:
            hp = HP_GRID[model][0]
            if nested:
                inner = "loso" if protocol == "loso" else "shuffle"
                hp = inner_select(model, D, tr, D["subject"], n_cls, fb_cfg, epochs, seed, inner)
            with_noise = model in ("dual", "conv_only", "fbcsp_only")
            pred, pred_n = fit_predict(model, Ztr, Ftr, D["y"][tr], Zte, Fte, n_cls, seed, hp, epochs,
                                       Zte_noisy=Zte_n if with_noise else None, Fte_noisy=Fte_n if with_noise else None)
            record(rows, cms, protocol, model, seed, name, D["subject"][te], D["y"][te], pred, dict(condition="clean", hp=str(hp)))
            if pred_n is not None:
                record(rows, cms, protocol, model, seed, name, D["subject"][te], D["y"][te], pred_n, dict(condition=f"awgn{snr_db}dB", hp=str(hp)))
        print(f"[{protocol}] {name} seed={seed} done in {time.time()-t0:.1f}s", flush=True)
        pd.DataFrame(rows).to_csv(os.path.join(outdir, f"{dataset}_{protocol}_runs.csv"), index=False)
        pd.DataFrame(cms).to_json(os.path.join(outdir, f"{dataset}_{protocol}_confusion.json"), orient="records")
    return pd.DataFrame(rows)


def summarise(df, dataset, protocol, outdir):
    """Per-subject mean +/- SD across seeds/splits, and overall mean +/- SD + 95% CI (R2.7)."""
    from scipy import stats
    df = df.copy()
    subj = df[df.subject != "ALL"].groupby(["model", "condition", "subject"])["acc"].agg(["mean", "std", "count"]).reset_index()
    subj.to_csv(os.path.join(outdir, f"{dataset}_{protocol}_per_subject.csv"), index=False)
    if protocol == "pooled":
        overall = df[df.subject == "ALL"].groupby(["model", "condition"])["acc"]
    else:  # mean over subjects within a seed, then spread across seeds and subjects
        overall = df[df.subject != "ALL"].groupby(["model", "condition"])["acc"]
    o = overall.agg(["mean", "std", "count"]).reset_index()
    o["ci95"] = 1.96 * o["std"] / np.sqrt(o["count"])
    o.to_csv(os.path.join(outdir, f"{dataset}_{protocol}_overall.csv"), index=False)
    # Wilcoxon signed-rank: dual vs each other model on per-subject means (clean)
    clean = subj[subj.condition == "clean"].pivot(index="subject", columns="model", values="mean")
    tests = []
    if "dual" in clean:
        for m in clean.columns:
            if m != "dual":
                d = clean[["dual", m]].dropna()
                if len(d) >= 5 and not np.allclose(d["dual"], d[m]):
                    w = stats.wilcoxon(d["dual"], d[m])
                    tests.append(dict(comparison=f"dual vs {m}", n_subjects=len(d), W=w.statistic, p=w.pvalue,
                                      mean_diff=(d["dual"] - d[m]).mean()))
    pd.DataFrame(tests).to_csv(os.path.join(outdir, f"{dataset}_{protocol}_wilcoxon.csv"), index=False)
    print(o.to_string(index=False))
    print(pd.DataFrame(tests).to_string(index=False) if tests else "(no paired tests)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["2a", "2b"], default="2a")
    ap.add_argument("--protocol", choices=["pooled", "session", "loso"], default="pooled")
    ap.add_argument("--models", nargs="+", default=["dual", "conv_only", "fbcsp_only"])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--splits", type=int, default=5, help="repeated pooled 80/20 splits")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--no-nested", action="store_true", help="skip inner hyperparameter search (faster)")
    ap.add_argument("--snr", type=float, default=5.0)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    D = load_synthetic(a.dataset) if a.synthetic else load_moabb(a.dataset)
    print(f"Loaded {len(D['y'])} trials, {D['eeg'].shape[1]} EEG channels, sessions={np.unique(D['session'])}")
    df = run(D, a.dataset, a.protocol, a.models, a.seeds, a.splits, a.epochs, not a.no_nested, a.snr, a.out)
    summarise(df, a.dataset, a.protocol, a.out)
