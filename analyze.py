"""
analyze.py — build every revised table / figure from results/*.csv and preds_*.npz.
Writes results/tables.md (all tables, markdown), results/summary.json and figures/*.png.
"""
import glob, json, os
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R, FIG = "results", "figures"
os.makedirs(FIG, exist_ok=True)
NICE = {"dual": "Dual-branch (FBCSP + Conv1D)", "conv1d_only": "Conv1D branch only (ablation)",
        "fbcsp_mlp": "FBCSP branch only (ablation)", "eegnet": "EEGNet-8,2 (re-implemented)",
        "cnn_shallow": "CNN (ShallowConvNet)", "lstm": "LSTM", "fbcsp_svm": "FBCSP + SVM (RBF)",
        "fbcsp_knn": "FBCSP + k-NN", "fbcsp_lda": "FBCSP + LDA"}
ORDER = list(NICE)
CLS = {"2a": ["Left hand", "Right hand", "Feet", "Tongue"], "2b": ["Left hand", "Right hand"]}
out, summary = [], {}


def ms(x, d=2):
    x = np.asarray(x, float)
    return f"{x.mean():.{d}f} ± {x.std(ddof=1):.{d}f}" if len(x) > 1 else f"{x.mean():.{d}f}"


def table(df, cols, title):
    out.append(f"\n### {title}\n")
    out.append("| " + " | ".join(cols) + " |")
    out.append("|" + "---|" * len(cols))
    for r in df:
        out.append("| " + " | ".join(str(v) for v in r) + " |")


def load(proto, ds):
    f = f"{R}/{proto}_{ds}.csv"
    return pd.read_csv(f) if os.path.exists(f) else None


def preds(proto, ds, model):
    return [np.load(f) for f in sorted(glob.glob(f"{R}/preds_{proto}_{ds}_{model}_s*_f*.npz"))]


# ---------------------------------------------------------------- pooled (manuscript protocol)
for ds in ("2a", "2b"):
    df = load("pooled", ds)
    if df is None:
        continue
    rows = []
    for m in ORDER:
        d = df[df.model == m]
        if len(d):
            rows.append([NICE[m], len(d), ms(d.acc), ms(d.kappa, 3), ms(d.precision, 3), ms(d.recall, 3), ms(d.f1, 3)])
            summary[f"pooled_{ds}_{m}_acc"] = [float(d.acc.mean()), float(d.acc.std(ddof=1)) if len(d) > 1 else 0.0]
    n_te = int(df.n_test.iloc[0]); n_tr = int(df.n_train.iloc[0])
    table(rows, ["Model", "runs (seeds)", "Accuracy (%)", "Kappa", "Precision", "Recall", "F1"],
          f"Pooled within-session protocol — Dataset {ds} (train {n_tr} / test {n_te} trials, mean ± SD over seeds)")

    # per-subject breakdown of the pooled test set (Reviewer 1: Tables 11/12 per subject)
    rows, per = [], {}
    for m in ORDER:
        P_ = preds("pooled", ds, m)
        if not P_:
            continue
        acc = np.array([[100 * np.mean(p["p"][p["subj"] == s] == p["y"][p["subj"] == s]) for s in range(1, 10)] for p in P_])
        per[m] = acc.mean(0)
    if per:
        for s in range(9):
            rows.append([f"S{s+1}"] + [f"{per[m][s]:.1f}" for m in per])
        rows.append(["Mean ± SD"] + [ms(per[m], 1) for m in per])
        table(rows, ["Subject"] + [NICE[m] for m in per],
              f"Pooled protocol — per-subject accuracy (%) on the held-out 20 % test trials, Dataset {ds} (mean over seeds)")
        summary[f"pooled_{ds}_persubject"] = {m: per[m].tolist() for m in per}

    # confusion matrix + class-wise (actual test partition, seed 0, and summed over seeds)
    P_ = preds("pooled", ds, "dual")
    if P_:
        for tag, sel in (("seed 0 test partition", P_[:1]), (f"summed over {len(P_)} seeds", P_)):
            y = np.concatenate([p["y"] for p in sel]); p = np.concatenate([p["p"] for p in sel])
            cm = confusion_matrix(y, p)
            rows = [[CLS[ds][i]] + list(cm[i]) for i in range(len(cm))]
            table(rows, ["Actual \\ Pred."] + CLS[ds], f"Confusion matrix, dual-branch, pooled protocol, Dataset {ds} ({tag}, n = {len(y)})")
            pr, rc, f1, sup = precision_recall_fscore_support(y, p, zero_division=0)
            rows = [[CLS[ds][i], f"{pr[i]:.3f}", f"{rc[i]:.3f}", f"{f1[i]:.3f}", sup[i]] for i in range(len(pr))]
            table(rows, ["Class", "Precision", "Recall", "F1", "Support"], f"Class-wise performance, dual-branch, pooled, Dataset {ds} ({tag})")

# ---------------------------------------------------------------- subject-specific / LOSO
for proto, label in (("subject", "Subject-specific cross-session (2a: T→E; 2b: sessions 1–3 → 4–5)"),
                     ("loso", "Leave-one-subject-out")):
    for ds in ("2a", "2b"):
        df = load(proto, ds)
        if df is None:
            continue
        g = df.groupby(["model", "fold"]).acc.mean().unstack().reindex(columns=range(1, 10))  # mean over seeds
        models_ = [m for m in ORDER if m in g.index]
        rows = []
        for s in range(1, 10):
            rows.append([f"S{s}"] + [f"{g.loc[m, s]:.1f}" if s in g.columns and not np.isnan(g.loc[m, s]) else "–" for m in models_])
        rows.append(["Mean ± SD"] + [ms(g.loc[m].dropna(), 1) for m in models_])
        kap = df.groupby(["model", "fold"]).kappa.mean().unstack()
        rows.append(["Mean kappa"] + [f"{kap.loc[m].mean():.3f}" for m in models_])
        seeds = df.groupby("model").seed.nunique()
        rows.append(["seeds"] + [str(seeds[m]) for m in models_])
        table(rows, ["Subject"] + [NICE[m] for m in models_], f"{label} — per-subject accuracy (%), Dataset {ds}")
        for m in models_:
            summary[f"{proto}_{ds}_{m}"] = dict(per_subject=g.loc[m].tolist(), mean=float(g.loc[m].mean()),
                                                 sd=float(g.loc[m].std(ddof=1)), kappa=float(kap.loc[m].mean()))
        # paired Wilcoxon (subjects) dual vs others
        if "dual" in g.index:
            rows = []
            for m in models_:
                if m == "dual" or g.loc[m].isna().any():
                    continue
                diff = g.loc["dual"] - g.loc[m]
                st = wilcoxon(g.loc["dual"], g.loc[m]) if np.any(diff != 0) else None
                rows.append([NICE[m], f"{diff.mean():+.1f}", f"{(diff > 0).sum()}/9", f"{st.pvalue:.3f}" if st else "–"])
            table(rows, ["Compared model", "Δ acc (dual − model)", "subjects dual better", "Wilcoxon p (n = 9)"],
                  f"{label}, Dataset {ds}: dual-branch vs other models")
        # seed variability of the dual model
        d = df[df.model == "dual"]
        if d.seed.nunique() > 1:
            per_seed = d.groupby("seed").acc.mean()
            out.append(f"\nDual-branch, {label}, {ds}: subject-mean accuracy per seed = "
                       + ", ".join(f"{v:.2f}" for v in per_seed) + f" (SD across seeds {per_seed.std(ddof=1):.2f})\n")
            summary[f"{proto}_{ds}_dual_seedmeans"] = per_seed.tolist()

# ---------------------------------------------------------------- noise
for ds in ("2a", "2b"):
    f = f"{R}/noise_{ds}.csv"
    if not os.path.exists(f):
        continue
    df = pd.read_csv(f)
    g = df.groupby(["model", "condition", "seed"]).acc.mean().reset_index()
    rows = []
    for m in ("conv1d_only", "fbcsp_mlp", "dual"):
        c = g[(g.model == m) & (g.condition == "clean")].acc
        n = g[(g.model == m) & (g.condition == "noisy_5dB")].acc
        if len(c):
            rows.append([NICE[m], ms(c), ms(n), f"{(c.values - n.values).mean():.2f}"])
            summary[f"noise_{ds}_{m}"] = [float(c.mean()), float(n.mean())]
    table(rows, ["Model", "Clean acc (%)", "Noisy 5 dB acc (%)", "Drop (pp)"],
          f"Noise stress test, pooled protocol, Dataset {ds} (30 noise realisations per seed; mean ± SD over seeds)")

# ---------------------------------------------------------------- figures
for ds in ("2a", "2b"):
    f = f"{R}/hist_pooled_{ds}_dual_s0.json"
    if os.path.exists(f):
        h = json.load(open(f))
        e = np.arange(1, len(h["loss"]) + 1)
        best = int(np.argmin(h["val_loss"]) + 1)
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
        ax[0].plot(e, h["loss"], label="Training"); ax[0].plot(e, h["val_loss"], label="Validation")
        ax[1].plot(e, h["accuracy"], label="Training"); ax[1].plot(e, h["val_accuracy"], label="Validation")
        for a_, t in zip(ax, ("Loss", "Accuracy")):
            a_.axvline(best, ls="--", c="gray", lw=1, label=f"Best val. loss (epoch {best})")
            a_.set_xlabel("Epoch"); a_.set_ylabel(t); a_.legend(frameon=False, fontsize=8)
        fig.suptitle(f"Dual-branch training curves, Dataset {ds}, pooled protocol (seed 0)")
        fig.tight_layout(); fig.savefig(f"{FIG}/training_curves_{ds}.png", dpi=200); plt.close(fig)
        summary[f"hist_{ds}"] = dict(best_epoch=best, stop_epoch=len(h["loss"]))

for proto in ("subject", "loso"):
    for ds in ("2a", "2b"):
        df = load(proto, ds)
        if df is None:
            continue
        g = df.groupby(["model", "fold"]).acc.mean().unstack().reindex(columns=range(1, 10))
        ms_ = [m for m in ("dual", "conv1d_only", "fbcsp_mlp", "fbcsp_lda", "eegnet") if m in g.index]
        fig, ax = plt.subplots(figsize=(10, 3.6))
        w = 0.8 / len(ms_)
        for i, m in enumerate(ms_):
            ax.bar(np.arange(9) + i * w, g.loc[m].values, w, label=NICE[m])
        ax.axhline(100 / (4 if ds == "2a" else 2), c="k", ls=":", lw=1, label="Chance")
        ax.set_xticks(np.arange(9) + 0.4 - w / 2); ax.set_xticklabels([f"S{s}" for s in range(1, 10)])
        ax.set_ylabel("Accuracy (%)"); ax.set_ylim(0, 100); ax.legend(fontsize=7, ncol=3, frameon=False)
        ax.set_title(f"{'Subject-specific cross-session' if proto == 'subject' else 'LOSO'} — Dataset {ds}")
        fig.tight_layout(); fig.savefig(f"{FIG}/{proto}_{ds}_per_subject.png", dpi=200); plt.close(fig)

open(f"{R}/tables.md", "w").write("\n".join(out))
json.dump(summary, open(f"{R}/summary.json", "w"), indent=1)
print("\n".join(out))
