"""
run_experiments.py — every evaluation protocol used in the revised manuscript.

Protocols
  pooled   : manuscript's original "subject-dependent" protocol, now re-run correctly.
             2a: all A0xT trials (9 x 288 = 2592); 2b: all sessions (6520 trials).
             80/20 split stratified jointly by subject x session x class; repeated over
             5 seeds (seed changes the split AND the network initialisation).
  subject  : subject-specific, cross-session (the official competition split).
             2a: train A0xT (288) -> test A0xE (288); 2b: train sessions 1-3 -> test 4-5.
             One model per subject; nothing from other subjects is used.
  loso     : leave-one-subject-out. Same trial pool as 'pooled'. 9 folds.
In every protocol: EOG regression, z-score, CSP, MIBIF, feature scaling, SVM/kNN grid search
and early-stopping validation use ONLY the training partition of that split. Hyper-parameters
(lr 1e-3, dropout 0.5, dense 128/64, m = 2, k = 20/10, batch 32, patience 15) are fixed a
priori to the values of the original submission; they are never tuned on any test partition.

Outputs (results/):
  <protocol>_<dataset>.csv   one row per (model, seed, fold)
  preds_<protocol>_<dataset>_<model>_s<seed>_f<fold>.npz  true / predicted labels, subjects
  hist_*.json                training curves (dual, pooled, seed 0)
  noise_<dataset>.csv        5 dB AWGN stress test (pooled protocol)

Usage:
  python run_experiments.py --protocol subject --dataset 2a --models all --seeds 0 1 2
"""
import argparse, json, os, time
import numpy as np
import pandas as pd
from scipy.signal import sosfiltfilt
from sklearn.metrics import accuracy_score, cohen_kappa_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

import pipeline as P
from prep import SOS_BROAD, SOS_BANDS

OUT = "results"
os.makedirs(OUT, exist_ok=True)


def load(ds):
    g = lambda k: np.load(f"{ds}_{k}.npy", mmap_mode="r")
    return dict(W=g("W"), B=g("B"), y=np.array(g("y")), subj=np.array(g("subj")),
                sess=np.array(g("sess")), n_eeg=int(np.load(f"{ds}_n_eeg.npy")))


def metrics(y, p):
    pr, rc, f1, _ = precision_recall_fscore_support(y, p, average="macro", zero_division=0)
    return dict(acc=100 * accuracy_score(y, p), kappa=cohen_kappa_score(y, p), precision=pr, recall=rc, f1=f1)


def splits(D, protocol, ds, seed):
    y, subj, sess = D["y"], D["subj"], D["sess"]
    if protocol == "pooled":
        pool = np.where(sess == 1)[0] if ds == "2a" else np.arange(len(y))
        strat = subj[pool] * 100 + sess[pool] * 10 + y[pool]
        tr, te = train_test_split(pool, test_size=0.2, stratify=strat, random_state=seed)
        yield 0, np.sort(tr), np.sort(te)
    elif protocol == "subject":
        for s in range(1, 10):
            if ds == "2a":
                tr, te = np.where((subj == s) & (sess == 1))[0], np.where((subj == s) & (sess == 2))[0]
            else:
                tr, te = np.where((subj == s) & (sess <= 3))[0], np.where((subj == s) & (sess >= 4))[0]
            yield s, tr, te
    elif protocol == "loso":
        pool = np.where(sess == 1)[0] if ds == "2a" else np.arange(len(y))
        for s in range(1, 10):
            yield s, pool[subj[pool] != s], pool[subj[pool] == s]


def add_noise(W, B, n_eeg, snr_db, rng):
    """Additive white Gaussian noise on the EEG channels of the raw trial, at a given SNR
    relative to the trial's 4-40 Hz EEG power (per channel). Because the preprocessing
    filters are linear, noise is generated as white noise, passed through the SAME
    broadband and sub-band filters (on a 3x longer segment, centre cropped), and added."""
    n, _, T = W.shape
    L = 3 * T
    white = rng.standard_normal((n, n_eeg, L))
    nb = sosfiltfilt(SOS_BROAD, white, axis=-1)[..., T:2 * T]
    nbands = np.stack([sosfiltfilt(s, white, axis=-1)[..., T:2 * T] for s in SOS_BANDS], 1)
    p_sig = W[:, :n_eeg].var(axis=-1)                        # (n, C)
    p_nb = nb.var(axis=-1)
    scale = np.sqrt(p_sig / (10 ** (snr_db / 10)) / p_nb)[..., None]
    Wn, Bn = np.array(W, np.float32), np.array(B, np.float32)
    Wn[:, :n_eeg] += (scale * nb).astype(np.float32)
    Bn[:, :, :n_eeg] += (scale[:, None] * nbands).astype(np.float32)
    return Wn, Bn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", choices=["pooled", "subject", "loso"], required=True)
    ap.add_argument("--dataset", choices=["2a", "2b"], required=True)
    ap.add_argument("--models", nargs="+", default=["all"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--folds", nargs="+", type=int, default=None)
    ap.add_argument("--noise", action="store_true", help="also run the 5 dB noise test (pooled)")
    ap.add_argument("--noise_reps", type=int, default=30)
    a = ap.parse_args()

    ds = a.dataset
    D = load(ds)
    n_cls = len(np.unique(D["y"]))
    k = 20 if ds == "2a" else 10
    names = P.DEEP + P.SHALLOW if a.models == ["all"] else a.models
    csv = f"{OUT}/{a.protocol}_{ds}.csv"
    done = set()
    if os.path.exists(csv):
        old = pd.read_csv(csv)
        done = set(zip(old.model, old.seed, old.fold))

    for seed in a.seeds:
        for fold, tr, te in splits(D, a.protocol, ds, seed):
            if a.folds and fold not in a.folds:
                continue
            todo = [m for m in names if (m, seed, fold) not in done]
            if not todo and not a.noise:
                continue
            t0 = time.time()
            ytr, yte = D["y"][tr], D["y"][te]
            Wtr, Btr = np.asarray(D["W"][tr]), np.asarray(D["B"][tr])
            fe = P.Features(D["n_eeg"], m=2, k=k, seed=seed).fit(Wtr, Btr, ytr)
            Xtr, Ftr = fe.transform(Wtr, Btr)
            del Btr
            Wte, Bte = np.asarray(D["W"][te]), np.asarray(D["B"][te])
            Xte, Fte = fe.transform(Wte, Bte)
            tfeat = time.time() - t0
            trained = {}
            for name in todo:
                t1 = time.time()
                info = {}
                if name in P.DEEP:
                    edrop = 0.25 if a.protocol == "loso" else 0.5
                    m, hist = P.train_deep(name, Xtr, Ftr, ytr, n_cls, seed, eegnet_dropout=edrop)
                    pred = P.predict_deep(m, name, Xte, Fte).argmax(1)
                    info = dict(epochs=len(hist["loss"]),
                                best_epoch=int(np.argmin(hist["val_loss"]) + 1))
                    trained[name] = m
                    if name == "dual" and seed == 0 and a.protocol == "pooled":
                        json.dump({kk: [float(v) for v in vv] for kk, vv in hist.items()},
                                  open(f"{OUT}/hist_{a.protocol}_{ds}_dual_s0.json", "w"))
                else:
                    clf, bp = P.train_shallow(name, Ftr, ytr, seed)
                    pred = clf.predict(Fte)
                    info = dict(params=json.dumps(bp))
                row = dict(protocol=a.protocol, dataset=ds, model=name, seed=seed, fold=fold,
                           n_train=len(tr), n_test=len(te), **metrics(yte, pred),
                           train_s=round(time.time() - t1, 1), feat_s=round(tfeat, 1), **info)
                np.savez(f"{OUT}/preds_{a.protocol}_{ds}_{name}_s{seed}_f{fold}.npz",
                         y=yte, p=pred, subj=D["subj"][te], sess=D["sess"][te], idx=te)
                pd.DataFrame([row]).to_csv(csv, mode="a", header=not os.path.exists(csv), index=False)
                print(f"[{a.protocol} {ds}] seed {seed} fold {fold} {name:12s} acc {row['acc']:.2f} "
                      f"kappa {row['kappa']:.3f} ({row['train_s']}s)", flush=True)

            if a.noise and a.protocol == "pooled":
                rng = np.random.default_rng(1000 + seed)
                rows = []
                for name in [n for n in ("dual", "conv1d_only", "fbcsp_mlp") if n in trained]:
                    clean = P.predict_deep(trained[name], name, Xte, Fte).argmax(1)
                    rows.append(dict(dataset=ds, model=name, seed=seed, condition="clean", rep=-1,
                                     acc=100 * accuracy_score(yte, clean)))
                rng = np.random.default_rng(1000 + seed)
                for r in range(a.noise_reps):
                    Wn, Bn = add_noise(Wte, Bte, D["n_eeg"], 5.0, rng)
                    Xn, Fn = fe.transform(Wn, Bn)       # same fitted (training) pipeline
                    for name in [n for n in ("dual", "conv1d_only", "fbcsp_mlp") if n in trained]:
                        pn = P.predict_deep(trained[name], name, Xn, Fn).argmax(1)
                        rows.append(dict(dataset=ds, model=name, seed=seed, condition="noisy_5dB", rep=r,
                                         acc=100 * accuracy_score(yte, pn)))
                nf = f"{OUT}/noise_{ds}.csv"
                pd.DataFrame(rows).to_csv(nf, mode="a", header=not os.path.exists(nf), index=False)
                print(pd.DataFrame(rows).groupby(["model", "condition"]).acc.mean(), flush=True)


if __name__ == "__main__":
    main()
