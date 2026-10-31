"""
make_response.py — builds the point-by-point Response to Reviewers (.docx) from the
result files, so every number in the document comes directly from results/*.csv.
"""
import glob, json, os
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from docx import Document
from docx.shared import Pt, RGBColor, Inches, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

R = "results"
NICE = {"dual": "Dual-branch (FBCSP + Conv1D)", "conv1d_only": "Conv1D branch only",
        "fbcsp_mlp": "FBCSP branch only", "eegnet": "EEGNet-8,2 (re-impl.)",
        "cnn_shallow": "CNN (ShallowConvNet)", "lstm": "LSTM", "fbcsp_svm": "FBCSP + SVM",
        "fbcsp_knn": "FBCSP + k-NN", "fbcsp_lda": "FBCSP + LDA"}
ORDER = list(NICE)
CLS = {"2a": ["Left hand", "Right hand", "Feet", "Tongue"], "2b": ["Left hand", "Right hand"]}
BLUE = RGBColor(0x1F, 0x4E, 0x9A)


def csv(p, ds):
    f = f"{R}/{p}_{ds}.csv"
    return pd.read_csv(f) if os.path.exists(f) else pd.DataFrame(columns=["model", "fold", "seed", "acc"])


def complete(df, m):
    d = df[df.model == m]
    return len(d) and d.fold.nunique() == (9 if d.fold.max() > 0 else 1)


def persub(p, ds):
    df = csv(p, ds)
    g = df.groupby(["model", "fold"]).acc.mean().unstack()
    return df, g


def ms(x, d=1):
    x = np.asarray(x, float)
    return f"{x.mean():.{d}f} ± {x.std(ddof=1):.{d}f}" if len(x) > 1 else f"{x.mean():.{d}f}"


# ------------------------------------------------------------------ docx helpers
doc = Document()
st = doc.styles["Normal"]; st.font.name = "Calibri"; st.font.size = Pt(10.5)
for s in doc.sections:
    s.left_margin = s.right_margin = Cm(2.0); s.top_margin = s.bottom_margin = Cm(2.0)


def H(t, lvl=1):
    return doc.add_heading(t, lvl)


def Pp(t="", bold=False, italic=False, color=None, size=None):
    p = doc.add_paragraph()
    r = p.add_run(t); r.bold = bold; r.italic = italic
    if color: r.font.color.rgb = color
    if size: r.font.size = Pt(size)
    p.paragraph_format.space_after = Pt(4)
    return p


def runs(parts):
    """parts: list of (text, style) where style in {'', 'b', 'i'}"""
    p = doc.add_paragraph()
    for t, s in parts:
        r = p.add_run(t); r.bold = "b" in s; r.italic = "i" in s
    p.paragraph_format.space_after = Pt(4)
    return p


def bullet(t):
    p = doc.add_paragraph(t, style="List Bullet"); p.paragraph_format.space_after = Pt(2); return p


def shade(cell, hexcol):
    tcPr = cell._tc.get_or_add_tcPr()
    sh = OxmlElement("w:shd"); sh.set(qn("w:val"), "clear"); sh.set(qn("w:color"), "auto"); sh.set(qn("w:fill"), hexcol)
    tcPr.append(sh)


def T(caption, header, rows, widths=None, bold_first_col=False, small=True):
    c = doc.add_paragraph(); r = c.add_run(caption); r.bold = True; r.font.size = Pt(9.5)
    c.paragraph_format.space_before = Pt(6); c.paragraph_format.space_after = Pt(2); c.paragraph_format.keep_with_next = True
    t = doc.add_table(rows=1 + len(rows), cols=len(header)); t.style = "Table Grid"; t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for j, h in enumerate(header):
        cell = t.rows[0].cells[j]; cell.text = ""; rr = cell.paragraphs[0].add_run(str(h)); rr.bold = True
        rr.font.size = Pt(8.5 if small else 10); shade(cell, "D9E2F3")
    for i, row in enumerate(rows):
        for j, v in enumerate(row):
            cell = t.rows[i + 1].cells[j]; cell.text = ""
            rr = cell.paragraphs[0].add_run(str(v)); rr.font.size = Pt(8.5 if small else 10)
            if (j == 0 and bold_first_col) or str(row[0]).startswith("Mean"): rr.bold = True
    if widths:
        t.autofit = False
        grid = t._tbl.tblGrid
        for j, gc in enumerate(grid.findall(qn("w:gridCol"))):
            gc.set(qn("w:w"), str(int(widths[j] * 567)))
        for row in t.rows:
            for j, w in enumerate(widths):
                row.cells[j].width = Cm(w)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    return t


def comment(txt):
    p = doc.add_paragraph(); p.paragraph_format.left_indent = Cm(0.5)
    r = p.add_run("Comment: "); r.bold = True
    r2 = p.add_run(txt); r2.italic = True
    shade_par(p, "F2F2F2")


def shade_par(p, hexcol):
    pPr = p._p.get_or_add_pPr(); sh = OxmlElement("w:shd")
    sh.set(qn("w:val"), "clear"); sh.set(qn("w:color"), "auto"); sh.set(qn("w:fill"), hexcol); pPr.append(sh)


def resp(txt=None):
    p = doc.add_paragraph(); r = p.add_run("Response: "); r.bold = True; r.font.color.rgb = BLUE
    if txt: p.add_run(txt)
    return p


def persub_table(caption, p, ds, models):
    df, g = persub(p, ds)
    models = [m for m in models if m in g.index and g.loc[m].notna().sum() == 9]
    rows = [[f"S{s}"] + [f"{g.loc[m, s]:.1f}" for m in models] for s in range(1, 10)]
    rows.append(["Mean ± SD"] + [ms(g.loc[m]) for m in models])
    kap = df.groupby(["model", "fold"]).kappa.mean().unstack()
    rows.append(["κ (mean)"] + [f"{kap.loc[m].mean():.2f}" for m in models])
    T(caption, ["Subject"] + [NICE[m] for m in models], rows)
    return g


def wil_rows(g, ref="dual", models=None):
    rows = []
    for m in models or ORDER:
        if m == ref or m not in g.index or g.loc[m].notna().sum() < 9:
            continue
        d = g.loc[ref] - g.loc[m]
        p = wilcoxon(g.loc[ref], g.loc[m]).pvalue if np.any(d != 0) else np.nan
        rows.append([NICE[m], f"{g.loc[m].mean():.1f}", f"{d.mean():+.1f}", f"{int((d > 0).sum())}/9", f"{p:.3f}"])
    return rows


def pooled_summary(ds, models):
    df = csv("pooled", ds)
    rows = []
    for m in models:
        d = df[df.model == m]
        if len(d):
            rows.append([NICE[m], len(d), ms(d.acc, 2), ms(d.kappa, 3), ms(d.precision, 3), ms(d.recall, 3), ms(d.f1, 3)])
    return df, rows


def pooled_persub(ds, models):
    out = {}
    for m in models:
        fs = sorted(glob.glob(f"{R}/preds_pooled_{ds}_{m}_s*_f0.npz"))
        if not fs: continue
        acc = []
        for f in fs:
            z = np.load(f); acc.append([100 * np.mean(z["p"][z["subj"] == s] == z["y"][z["subj"] == s]) for s in range(1, 10)])
        out[m] = np.array(acc).mean(0)
    return out


# ================================================================== numbers
S = {}
for ds in ("2a", "2b"):
    for p in ("subject", "loso"):
        S[p, ds] = persub(p, ds)[1]
    S["pooled", ds] = csv("pooled", ds)
d2a, d2b = S["pooled", "2a"], S["pooled", "2b"]


def pm(ds, m, col="acc"):
    d = S["pooled", ds]; d = d[d.model == m][col]
    return d.mean(), (d.std(ddof=1) if len(d) > 1 else 0.0), len(d)


def sm(p, ds, m):
    g = S[p, ds]; return g.loc[m].mean(), g.loc[m].std(ddof=1)


n_te = {ds: int(S["pooled", ds].n_test.iloc[0]) for ds in ("2a", "2b")}
n_tr = {ds: int(S["pooled", ds].n_train.iloc[0]) for ds in ("2a", "2b")}
nseed_pooled = {ds: S["pooled", ds][S["pooled", ds].model == "dual"].seed.nunique() for ds in ("2a", "2b")}
lat = json.load(open(f"{R}/latency.json")) if os.path.exists(f"{R}/latency.json") else None
noise = {ds: pd.read_csv(f"{R}/noise_{ds}.csv") for ds in ("2a", "2b") if os.path.exists(f"{R}/noise_{ds}.csv")}

# ================================================================== document
t = doc.add_heading("Response to Reviewers", 0)
Pp("Manuscript: “A Dual-Branch FBCSP–Deep Learning Framework for Motor Imagery Classification in "
   "Brain–Computer Interface Systems” — Multimodal Technologies and Interaction (MDPI), second-round revision", italic=True)

H("Cover Letter", 1)
Pp("We greatly appreciate the reviewers' thoughtful comments and suggestions, which have proven instrumental in "
   "enhancing the quality of our work. We acknowledge the importance of their insights and have diligently addressed "
   "each of their points in the revised manuscript. All changes made in response to the editor’s and reviewers’ "
   "comments are highlighted in blue in the revised manuscript.")
Pp("In addressing the reviewers’ questions on data partitioning, leakage control, reproducibility, ablation and "
   "variability, we rebuilt the complete experimental pipeline from the original BCI Competition IV recordings and "
   "re-ran every experiment from scratch. This re-analysis led to substantive changes that we wish to state plainly "
   "to the Editor and the reviewers:", bold=False)
bullet("All results are now produced by a single, documented, leakage-controlled pipeline whose code and raw "
       "per-run outputs are released with the paper. Every number in this response and in the revised manuscript "
       "is generated by that code.")
bullet(f"The previously reported subject-dependent accuracies (98.58 % for Dataset 2a, 97.12 % for Dataset 2b), the "
       "LOSO accuracies (77.3 %, 74.4 %), and the associated per-subject, class-wise, inter-session and noise results "
       "could not be reproduced. Under the same pooled 80/20 protocol, the re-run gives "
       f"{pm('2a','dual')[0]:.1f} ± {pm('2a','dual')[1]:.1f} % (2a) and {pm('2b','dual')[0]:.1f} ± {pm('2b','dual')[1]:.1f} % (2b); "
       f"LOSO gives {sm('loso','2a','dual')[0]:.1f} % and {sm('loso','2b','dual')[0]:.1f} %. All affected tables and "
       "figures have been replaced, and the earlier values have been withdrawn.")
bullet("The branch ablation requested by both reviewers shows that the FBCSP branch carries most of the "
       "discriminative information, and that the Conv1D branch, as configured, does not improve on it. The "
       "Abstract, Discussion and Conclusions have been rewritten so that the claims match these results.")
Pp("We are grateful for the opportunity to correct and improve our work, and we sincerely thank the reviewers for "
   "their time and expertise. Their questions exposed problems that we would otherwise have published.")
Pp("Sincerely,"); Pp("On behalf of all authors,")

# ---------------------------------------------------------------- common: what was re-done
H("Summary of the re-analysis (applies to all responses)", 1)
Pp("Data. BCI Competition IV 2a and 2b, official GDF recordings and the official true-label files of all sessions. "
   "All trials are retained (trials flagged by event 1023 are kept, as in the original submission).")
T("Table R1. Trials used in each protocol.",
  ["Protocol", "Dataset 2a", "Dataset 2b"],
  [["Pooled within-session (previously “subject-dependent”)", f"A01T–A09T, 9 × 288 = 2592 trials; {n_tr['2a']} train / {n_te['2a']} test",
    f"all 5 sessions, 6520 trials; {n_tr['2b']} train / {n_te['2b']} test"],
   ["Subject-specific cross-session (new; = inter-session)", "per subject: train A0xT (288) → test A0xE (288)",
    "per subject: train sessions 1–3 (400–440) → test sessions 4–5 (280–320)"],
   ["Leave-one-subject-out", "A0xT trials; train 8 subjects (2304) → test 1 (288)", "all sessions; train 8 subjects (5760–5840) → test 1 (680–760)"]],
  widths=[5.5, 6, 6])
Pp("Pipeline. Band-pass 4–40 Hz and nine 4-Hz sub-bands (4–40 Hz), 5th-order zero-phase Butterworth applied to the "
   "continuous recording; epochs 0.5–2.5 s after the cue (500 samples). Within every training partition, and only "
   "there, we estimate: EOG regression coefficients, per-channel z-score statistics, CSP filters, MIBIF ranking, the "
   "feature standardiser, the SVM/k-NN hyper-parameter grid search (inner 5-fold CV), and the early-stopping "
   "validation subset (stratified 10 % of the training partition). The fitted objects are then applied unchanged "
   "to the test partition. Network: identical to Table 1 of the manuscript (2 × Conv1D(64, 3) + max-pool, flatten "
   "7872, concatenation with the k FBCSP features, Dense 128 – Dropout 0.5 – Dense 64 – Dropout 0.5 – softmax; Adam "
   "1e-3, batch 32, ≤ 100 epochs, early stopping patience 15 with best-weight restoration).")
Pp("Implementation note on zero-padding. Zero-padded input channels receive exactly zero gradient, so their weights "
   "never change and never contribute to any activation. The 64-channel padded network is therefore functionally "
   "identical to a network fed the real channels (22 or 3), provided the first-layer He-uniform initialisation uses "
   "the padded fan-in. We implement it this way, and we now explain in Section 3.1 that the padding has no effect "
   "on the model.", italic=True)
Pp("Validity check. FBCSP + LDA on the official 2a T→E split gives "
   f"{sm('subject','2a','fbcsp_lda')[0]:.1f} % mean accuracy (κ = "
   f"{csv('subject','2a').query('model==\"fbcsp_lda\"').kappa.mean():.2f}). This is in line with the published "
   "FBCSP result on this dataset (winning method of the competition, κ = 0.57), which indicates that the "
   "re-implemented pipeline behaves as expected.")

# ================================================================== Reviewer 1
H("Reviewer #1", 1)

comment("Table 11. “EEGNet [42]”. In the original paper [42], the result presented in Table 11 is not present. You have "
        "to explain how did you get this result. Moreover, the reference number present next to EEGNet leads to the "
        "conclusion that this result is taken from the original paper.")
resp("We thank the reviewer and agree. The EEGNet value in the previous Table 11 was not taken from [42], and we "
     "could not trace it to a documented run. It has been removed. We have now re-implemented EEGNet-8,2 exactly as "
     "specified in [42]: F1 = 8 temporal filters (kernel 64), depth multiplier D = 2, depthwise spatial convolution "
     "with max-norm 1, F2 = 16 separable filters (kernel 16), average pooling 4 and 8, ELU, dropout 0.5 "
     "(within-subject) or 0.25 (LOSO, as recommended in [42]), dense max-norm 0.25. It is trained on exactly the same "
     "partitions, the same EOG-corrected and z-scored 4–40 Hz epochs (22 or 3 channels × 500 samples), with the same "
     "optimiser and early-stopping rule as the dual-branch model. In all tables it is now labelled “EEGNet-8,2 "
     "(re-implemented; architecture of [42])”, and the text states explicitly that the numbers come from our runs "
     "and not from [42]. Its results are given in Tables R2–R7.")

comment("It is not enough to present only final averaged result in Table 11. It is very interesting to observe the "
        "results for all the subjects separately. The same rule must be applied to Table 12.")
resp("Agreed. Tables 11 and 12 now report per-subject accuracy for every model under each protocol. For the pooled "
     "protocol (the protocol of the previous Tables 11/12), the per-subject value is the accuracy on that subject’s "
     "trials within the held-out 20 % test partition, averaged over seeds (Tables R2a/R2b). We additionally report "
     "the subject-specific cross-session protocol, where each subject has a separately trained model (Tables R3a/R3b).")
for ds in ("2a", "2b"):
    models = [m for m in ORDER if m in set(S["pooled", ds].model)]
    per = pooled_persub(ds, models)
    rows = [[f"S{s+1}"] + [f"{per[m][s]:.1f}" for m in per] for s in range(9)]
    rows.append(["Mean ± SD (subjects)"] + [ms(per[m]) for m in per])
    rows.append(["Overall test acc."] + [ms(S["pooled", ds][S["pooled", ds].model == m].acc) for m in per])
    rows.append(["seeds"] + [str(S["pooled", ds][S["pooled", ds].model == m].seed.nunique()) for m in per])
    T(f"Table R2{'a' if ds == '2a' else 'b'} (revised Table {'11' if ds == '2a' else '12'}). Pooled within-session protocol, "
      f"Dataset {ds}: per-subject accuracy (%) on the held-out test trials.", ["Subject"] + [NICE[m] for m in per], rows)
for ds in ("2a", "2b"):
    persub_table(f"Table R3{'a' if ds == '2a' else 'b'}. Subject-specific cross-session protocol, Dataset {ds} "
                 f"({'T → E' if ds == '2a' else 'sessions 1–3 → 4–5'}): per-subject accuracy (%). Dual, ablations and FBCSP "
                 "classifiers: mean of 3 seeds; deep baselines: 1 seed.", "subject", ds, ORDER)

comment("It is not enough to present only final averaged result in Table 13. It is very interesting to observe the "
        "results for all the subjects separately. In addition, you have to comment why your method obtains almost the "
        "same high accuracy for all the subjects, meanwhile the accuracy of competitor methods for different subject "
        "differs quite substantially.")
gsub = S["subject", "2a"]
resp(f"We thank the reviewer for this observation, which led directly to the re-analysis. The near-uniform values "
     f"of 97.5–99.4 % in the previous Tables 3/4/10 are not reproduced. With leakage-controlled processing, per-subject "
     f"accuracy of the dual-branch model varies widely, as it does for the competitor methods. In the subject-specific "
     f"protocol on 2a it ranges from {gsub.loc['dual'].min():.1f} % to {gsub.loc['dual'].max():.1f} % "
     f"(SD {gsub.loc['dual'].std(ddof=1):.1f} pp). FBCSP + LDA ranges from {gsub.loc['fbcsp_lda'].min():.1f} % "
     f"to {gsub.loc['fbcsp_lda'].max():.1f} %. The ranking of subjects is also the one commonly reported for this "
     "dataset: S2, S5 and S6 are difficult, and S3, S7, S8 and S9 are good performers. The uniform high values "
     "reported earlier were therefore not a property of the method. We withdraw them. The pooled 80/20 protocol "
     "remains in the paper only as a secondary analysis, and it is explicitly marked as subject-overlapping.")
Pp("Table 13 has been restructured. (i) The rows are grouped by evaluation protocol, and only results obtained under "
   "the official 2a session-wise protocol (T → E) are placed in the same block as our subject-specific result. "
   "(ii) Our own per-subject results for all protocols are given in Tables R2, R3 and R9. (iii) Where the cited works report "
   "per-subject results, these are reproduced in the revised Table 13 with their source; methods for which "
   "per-subject or protocol-matched values are not available are listed separately as contextual references, "
   "without a ranking claim.")
Pp("[AUTHORS — ACTION REQUIRED BEFORE SUBMISSION: insert the per-subject values of the cited methods in revised "
   "Table 13 from their original papers, and keep in the T→E block only methods evaluated on the official 2a/2b "
   "session split. Delete this note.]", bold=True, color=RGBColor(0xC0, 0x00, 0x00))

comment("What about ablation experiment to see impact of every branch separately? You are advised to perform such an experiment.")
resp("We performed the ablation under all three protocols, for both datasets. The three configurations share the "
     "same partitions, pipeline, dense head and training rule: (a) the full dual-branch model, (b) the Conv1D branch "
     "only (the FBCSP input removed), and (c) the FBCSP branch only (Dense 128–64 head on the k MIBIF features). "
     "Table R4 summarises the results; per-subject values are in Tables R2, R3 and R9.")
rows = []
for p, lab in (("pooled", "Pooled"), ("subject", "Subject-specific"), ("loso", "LOSO")):
    for ds in ("2a", "2b"):
        r = [f"{lab}, {ds}"]
        for m in ("conv1d_only", "fbcsp_mlp", "dual"):
            if p == "pooled":
                a, s_, n = pm(ds, m); r.append(f"{a:.1f} ± {s_:.1f}")
            else:
                g = S[p, ds]; r.append(ms(g.loc[m]) if m in g.index else "–")
        rows.append(r)
T("Table R4 (new). Branch ablation: accuracy (%), mean ± SD (pooled: over seeds; other protocols: over subjects).",
  ["Protocol, dataset", "Conv1D branch only", "FBCSP branch only", "Dual-branch (fusion)"], rows, widths=[4.5, 4, 4, 4])
gs2a, gs2b = S["subject", "2a"], S["subject", "2b"]
def abl(p, ds):
    if p == "pooled":
        return {m: pm(ds, m)[0] for m in ("conv1d_only", "fbcsp_mlp", "dual")}
    return {m: S[p, ds].loc[m].mean() for m in ("conv1d_only", "fbcsp_mlp", "dual")}
A = {(p, ds): abl(p, ds) for p in ("pooled", "subject", "loso") for ds in ("2a", "2b")}
g2 = S["subject", "2a"]; g2b = S["subject", "2b"]
w2 = wilcoxon(g2.loc["dual"], g2.loc["fbcsp_mlp"]).pvalue
w2b = wilcoxon(g2b.loc["dual"], g2b.loc["fbcsp_mlp"]).pvalue
Pp(f"Dataset 2a. The Conv1D branch alone is close to chance (25 %) in every protocol "
   f"({A['pooled','2a']['conv1d_only']:.1f} / {A['subject','2a']['conv1d_only']:.1f} / {A['loso','2a']['conv1d_only']:.1f} % "
   f"for pooled / subject-specific / LOSO). Fusing it with FBCSP lowers accuracy relative to the FBCSP branch alone: "
   f"{A['pooled','2a']['dual']:.1f} vs {A['pooled','2a']['fbcsp_mlp']:.1f} % (pooled), "
   f"{A['subject','2a']['dual']:.1f} vs {A['subject','2a']['fbcsp_mlp']:.1f} % (subject-specific; fusion lower for "
   f"{int((g2.loc['dual'] < g2.loc['fbcsp_mlp']).sum())}/9 subjects, Wilcoxon p = {w2:.3f}) and "
   f"{A['loso','2a']['dual']:.1f} vs {A['loso','2a']['fbcsp_mlp']:.1f} % (LOSO).")
Pp(f"Dataset 2b. With only three channels, the Conv1D branch alone is well above chance "
   f"({A['pooled','2b']['conv1d_only']:.1f} / {A['subject','2b']['conv1d_only']:.1f} / {A['loso','2b']['conv1d_only']:.1f} %). "
   f"Fusion and the FBCSP branch alone perform equivalently: {A['pooled','2b']['dual']:.1f} vs {A['pooled','2b']['fbcsp_mlp']:.1f} % "
   f"(pooled), {A['subject','2b']['dual']:.1f} vs {A['subject','2b']['fbcsp_mlp']:.1f} % (subject-specific, Wilcoxon p = {w2b:.3f}) "
   f"and {A['loso','2b']['dual']:.1f} vs {A['loso','2b']['fbcsp_mlp']:.1f} % (LOSO).")
Pp("Interpretation (revised Discussion). The spatial–spectral FBCSP features carry essentially all of the "
   "discriminative information that the dual-branch model exploits. In the fused layer, the 7872-dimensional "
   "flattened Conv1D output outnumbers the 20 (2a) or 10 (2b) FBCSP inputs by more than two orders of magnitude. "
   "With 240–5800 training trials, the network fits the high-dimensional temporal features, which do not "
   "generalise across sessions or subjects, at the expense of the FBCSP features. On 2a this costs accuracy; on "
   "2b, where the temporal branch is itself informative, it is neutral. The revised manuscript therefore no longer "
   "claims that FBCSP–Conv1D fusion improves accuracy, and it reports the ablation as a central result.")

# ================================================================== Reviewer 2
H("Reviewer #2", 1)
comment("Tables 5–8 do not appear to correspond to the test set defined in Section 5.1 … From which partition, then, "
        "were these observations obtained, and how do they relate to the main results of 98.58% and 97.12%?")
resp(f"The reviewer is correct. The 780- and 770-observation tables did not originate from the test partition "
     f"defined in Section 5.1, and we cannot document their provenance. They have been withdrawn. The revised Tables "
     f"5–8 are computed from the predictions on the actual held-out test partition of the pooled protocol: "
     f"{n_te['2a']} trials for 2a (20 % of 2592) and {n_te['2b']} trials for 2b (20 % of 6520). They are the same "
     "predictions that produce the reported accuracy, so the class-wise metrics, the confusion matrices and the "
     "main accuracy now correspond exactly. We show seed 0 (one complete test partition) and, for stability, the "
     "matrix summed over all seeds.")
for ds in ("2a", "2b"):
    fs = sorted(glob.glob(f"{R}/preds_pooled_{ds}_dual_s*_f0.npz"))
    if not fs: continue
    z0 = np.load(fs[0]); cm = confusion_matrix(z0["y"], z0["p"])
    T(f"Table R5{'a' if ds == '2a' else 'b'} (revised Table {'7' if ds == '2a' else '8'}). Confusion matrix, dual-branch, "
      f"pooled protocol, Dataset {ds}, seed-0 held-out test partition (n = {len(z0['y'])}; accuracy "
      f"{100*np.mean(z0['y']==z0['p']):.2f} %).", ["Actual \\ Predicted"] + CLS[ds],
      [[CLS[ds][i]] + list(cm[i]) for i in range(len(cm))])
    y = np.concatenate([np.load(f)["y"] for f in fs]); p = np.concatenate([np.load(f)["p"] for f in fs])
    pr, rc, f1, sup = precision_recall_fscore_support(y, p, zero_division=0)
    pr0, rc0, f10, sup0 = precision_recall_fscore_support(z0["y"], z0["p"], zero_division=0)
    T(f"Table R6{'a' if ds == '2a' else 'b'} (revised Table {'5' if ds == '2a' else '6'}). Class-wise performance, "
      f"dual-branch, pooled protocol, Dataset {ds}: seed-0 test partition, and pooled over {len(fs)} seeds.",
      ["Class", "Precision (s0)", "Recall (s0)", "F1 (s0)", "Support (s0)", f"F1 ({len(fs)} seeds)"],
      [[CLS[ds][i], f"{pr0[i]:.3f}", f"{rc0[i]:.3f}", f"{f10[i]:.3f}", sup0[i], f"{f1[i]:.3f}"] for i in range(len(pr))])

comment("For Dataset 2a, the manuscript reports 288 trials per subject and 2592 trials in total, yet the later "
        "inter-session analysis is described as training and evaluation on different sessions … The restriction of "
        "this analysis to S1–S5 is also unexplained …")
resp("We have clarified the composition and replaced the inter-session analysis. Table R1 (now Table 3 in Section "
     "5.1) lists exactly which files and sessions enter each protocol. The pooled and LOSO analyses of Dataset 2a use "
     "only the training-session files A01T–A09T (9 × 288 = 2592 trials). The inter-session analysis now covers all "
     "nine subjects. It uses the official session split (train A0xT, test A0xE, recorded on different days) and is "
     "reported as the subject-specific cross-session protocol (Table R3a). For Dataset 2b, the same protocol trains "
     "on sessions 1–3 and tests on sessions 4–5 (Table R3b). The earlier S1–S5 figure has been withdrawn.")
d = S["subject", "2a"]
Pp(f"Subject-specific cross-session accuracy of the dual-branch model is {sm('subject','2a','dual')[0]:.1f} ± "
   f"{sm('subject','2a','dual')[1]:.1f} % (2a) and {sm('subject','2b','dual')[0]:.1f} ± {sm('subject','2b','dual')[1]:.1f} % (2b). "
   f"The best model in this protocol is FBCSP + LDA on 2a ({sm('subject','2a','fbcsp_lda')[0]:.1f} %) and "
   f"{NICE[S['subject','2b'].mean(axis=1).idxmax()]} on 2b ({S['subject','2b'].mean(axis=1).max():.1f} %).")
for ds in ("2a", "2b"):
    rows = wil_rows(S["subject", ds])
    T(f"Table R7{'a' if ds == '2a' else 'b'}. Subject-specific cross-session, Dataset {ds}: dual-branch vs each model "
      "(paired Wilcoxon signed-rank test over the 9 subjects).", ["Model", "Mean acc. (%)", "Δ (dual − model), pp",
                                                                   "Subjects where dual is better", "p"], rows)

comment("The protocol labeled subject-dependent is not an independent per-subject evaluation, because trials from all "
        "nine participants are pooled before applying the 80/20 split … It would also be relevant to state whether "
        "the split preserved the distributions of class, subject, and session.")
resp("We agree completely. The pooled protocol has been renamed “pooled within-session (subject-overlapping)”, and "
     "Section 5.1 states that training and test sets contain different trials from the same subjects and sessions. "
     "The 80/20 split is stratified jointly by subject × session × class, so each stratum keeps its proportion in "
     f"both partitions. The split is repeated with {nseed_pooled['2a']} random seeds. A genuinely subject-specific "
     "protocol has been added: one model per subject, trained on one session and tested on a different session "
     "(Table R3). It is now the primary within-subject result, and only results obtained under the same official "
     "session-wise protocol are compared directly in Table 13.")
rows = []
for ds in ("2a", "2b"):
    _, rr = pooled_summary(ds, ORDER)
    rows += [[ds] + r for r in rr]
T_w = [1.1, 3.9, 1.3, 2.3, 2.1, 2.1, 2.1, 2.1]
T(f"Table R8 (revised Tables 2, 11, 12). Pooled within-session protocol: overall results, mean ± SD over seeds "
  f"(2a: train {n_tr['2a']} / test {n_te['2a']}; 2b: train {n_tr['2b']} / test {n_te['2b']}).",
  ["Data", "Model", "Seeds", "Accuracy (%)", "κ", "Precision", "Recall", "F1"], rows, widths=T_w)

comment("The independence of the LOSO analysis depends on how hyperparameters were selected … Was hyperparameter "
        "optimization repeated within each LOSO iteration using an internal validation procedure, or were these "
        "values fixed beforehand?")
resp("The hyper-parameters were fixed beforehand, and none of them was tuned on any test subject in the re-analysis. "
     "The learning rate (1e-3), dropout (0.5), dense widths (128/64), batch size (32), CSP m = 2, MIBIF k = 20 "
     "(2a) / 10 (2b), patience (15) and the maximum number of epochs (100) are those of the original submission. "
     "They are used unchanged in every protocol and every LOSO fold. The only data-driven choices are made inside "
     "the training subjects of each fold: the early-stopping epoch (internal stratified 10 % validation subset of the "
     "eight training subjects) and the SVM/k-NN grids (inner 5-fold CV on the eight training subjects). The sentence "
     "in Section 3.4 stating that these values were selected by 10-fold cross-validation was inaccurate and has been "
     "replaced by this description. The LOSO estimates are therefore free of model-selection leakage from the "
     "held-out subject.")
for ds in ("2a", "2b"):
    persub_table(f"Table R9{'a' if ds == '2a' else 'b'} (revised Table 10). LOSO per-subject accuracy (%), Dataset {ds}.",
                 "loso", ds, ORDER)
Pp(f"LOSO accuracy of the dual-branch model is {sm('loso','2a','dual')[0]:.1f} ± {sm('loso','2a','dual')[1]:.1f} % on 2a "
   f"(chance 25 %) and {sm('loso','2b','dual')[0]:.1f} ± {sm('loso','2b','dual')[1]:.1f} % on 2b (chance 50 %), replacing the "
   "previously reported 77.3 % and 74.4 %. Dataset 2b is now reported per subject with SD, exactly as 2a, which "
   "addresses the reviewer’s point about uneven reporting. The dual-branch LOSO values are means over 3 seeds; the "
   "other models use 1 seed. Because of computational cost, the ShallowConvNet and LSTM baselines were evaluated "
   "in the pooled and subject-specific protocols only.")

comment("The discussion attributes the advantage of the proposed method to the complementarity between FBCSP and "
        "Conv1D, yet the experiments do not isolate that contribution … the noise experiment compares clean and noisy "
        "Conv1D with the dual-branch model only under noisy conditions.")
resp("The reviewer is right. The ablation (Table R4 and the Reviewer 1 response above) has now been performed, and "
     "it does not support the claim. The Discussion has been rewritten accordingly. The noise experiment has also "
     "been redone with all three configurations under both the clean and the noisy (5 dB) condition. Noise is "
     "white Gaussian, scaled per trial and channel to 5 dB SNR relative to the 4–40 Hz EEG power, added to the EEG "
     "channels of the test trials before any feature extraction, and passed through the same fixed filters. "
     "Thirty noise realisations are drawn per seed, and the fitted training pipeline is applied unchanged.")
rows = []
for ds in noise:
    g = noise[ds].groupby(["model", "condition", "seed"]).acc.mean().reset_index()
    for m in ("conv1d_only", "fbcsp_mlp", "dual"):
        c = g[(g.model == m) & (g.condition == "clean")].acc.values
        n = g[(g.model == m) & (g.condition == "noisy_5dB")].acc.values
        if len(c):
            rows.append([ds, NICE[m], ms(c), ms(n), f"{(c - n).mean():.1f}"])
NZ = {}
for ds in noise:
    g = noise[ds].groupby(["model", "condition"]).acc.mean()
    NZ[ds] = {m: (g[m, "clean"], g[m, "noisy_5dB"]) for m in ("conv1d_only", "fbcsp_mlp", "dual")}
T("Table R10 (revised Figure 5). Noise stress test, pooled protocol (mean ± SD over seeds; 30 realisations per seed).",
  ["Data", "Model", "Clean acc. (%)", "5 dB noise acc. (%)", "Drop (pp)"], rows)

if "2a" in NZ:
    z = NZ["2a"]
    Pp(f"With the clean condition added, the stress test no longer shows a fusion advantage. On 2a the dual-branch "
       f"model drops from {z['dual'][0]:.1f} % to {z['dual'][1]:.1f} % and the FBCSP branch alone from "
       f"{z['fbcsp_mlp'][0]:.1f} % to {z['fbcsp_mlp'][1]:.1f} %. The Conv1D branch is near chance in both conditions "
       f"({z['conv1d_only'][0]:.1f} → {z['conv1d_only'][1]:.1f} %). The previously reported 96.0 / 88.0 / 98.6 % "
       "values are withdrawn.")

comment("The baseline comparisons are difficult to assess without a reproducible description of their implementation …")
resp("All baselines are now fully specified (new Table 4 in Section 5.1, reproduced below) and released in code. "
     "Every model receives the same EOG-corrected, training-normalised epochs. The three conventional classifiers "
     "receive exactly the same k FBCSP–MIBIF features that enter the dual-branch model, so they differ from it only "
     "in the classifier.")
T("Table R11 (new Table 4). Baseline definitions.",
  ["Model", "Input representation", "Architecture / hyper-parameters", "Tuning"],
  [["FBCSP + SVM", "k = 20 / 10 standardised FBCSP-MIBIF features", "RBF kernel", "C ∈ {0.1, 1, 10, 100}, γ ∈ {scale, 0.01, 0.1}; inner 5-fold CV on training partition"],
   ["FBCSP + k-NN", "same", "Euclidean", "k ∈ {3, 5, 7, 9, 15, 25}, uniform/distance weights; inner 5-fold CV"],
   ["FBCSP + LDA", "same", "LDA, Ledoit–Wolf shrinkage", "analytic shrinkage (no tuning)"],
   ["FBCSP branch only", "same", "Dense 128–Dropout 0.5–Dense 64–Dropout 0.5–softmax", "same training rule as dual"],
   ["Conv1D branch only", "z-scored 4–40 Hz epoch (C × 500)", "dual-branch Conv1D trunk + same head", "same training rule"],
   ["EEGNet-8,2", "z-scored 4–40 Hz epoch (C × 500)", "as in [42]; dropout 0.5 (0.25 LOSO)", "same training rule"],
   ["CNN (ShallowConvNet)", "z-scored 4–40 Hz epoch (C × 500)", "Conv(40, 1×25) – Conv(40, C×1) – BN – square – AvgPool(1×75, s15) – log – Dropout 0.5 – softmax", "same training rule"],
   ["LSTM", "epoch averaged over 4-sample windows (125 × C)", "LSTM 64 (seq.) – LSTM 32 – Dropout 0.5 – softmax", "same training rule"]],
  widths=[3, 4, 6, 4.5])
Pp("Same training rule = Adam 1e-3, sparse categorical cross-entropy, batch 32, ≤ 100 epochs, early stopping "
   "(patience 15) on a stratified 10 % validation subset of the training partition, best weights restored. "
   "Hyper-parameters of the deep baselines follow their original publications and are not tuned. This is the same "
   "treatment as the dual-branch model.", italic=True)

comment("The final accuracies of 98.58% and 97.12% are reported to two decimal places from a single 80/20 split and "
        "one final training run … The LOSO reporting is also uneven across datasets …")
resp(f"The pooled protocol is now repeated over {nseed_pooled['2a']} seeds (2a) and {nseed_pooled['2b']} seeds (2b). Each seed "
     "draws a new stratified 80/20 split and a new network initialisation, minibatch order and dropout masks, and "
     "we report mean ± SD (Table R8). The subject-specific protocol is repeated over 3 seeds for the dual-branch "
     "model, the ablations and the FBCSP classifiers. LOSO is reported per subject with SD for both datasets "
     "(Tables R9a/b). Accuracies are reported to one decimal place, consistent with the observed variability. The "
     "earlier Figure 3 (10-fold CV within one training pool) has been replaced by this seed-level analysis.")
for p, lab in (("subject", "Subject-specific"), ("loso", "LOSO")):
    for ds in ("2a", "2b"):
        df = csv(p, ds); dd = df[df.model == "dual"]
        if dd.seed.nunique() > 1:
            v = dd.groupby("seed").acc.mean()
            Pp(f"{lab}, {ds}: subject-mean accuracy of the dual-branch model per seed = "
               + ", ".join(f"{x:.1f}" for x in v) + f" % (SD across seeds {v.std(ddof=1):.2f} pp).")

comment("The FBCSP formulation does not uniquely determine the features that are eventually passed to the network …")
resp("Section 3.2 now gives the complete formulation, reproduced here, and the released code implements exactly these equations.")
for line in [
    "Single-trial covariance (trial X ∈ ℝ^{C×T}, band-pass filtered, mean removed): C_i = X_i X_iᵀ / T, trace-normalised Ĉ_i = C_i / tr(C_i).",
    "Class covariance: R_c = (1/|I_c|) Σ_{i∈I_c} Ĉ_i (arithmetic mean over the training trials of class c); “rest” covariance R_¬c the mean over all other training trials. Tikhonov regularisation: R ← (1 − r) R + r · tr(R)/C · I, with r = 0.01.",
    "CSP: generalised eigenproblem R_c w = λ (R_c + R_¬c) w. The m = 2 eigenvectors with the largest and the m = 2 with the smallest λ form W ∈ ℝ^{C×2m}. Dataset 2a: one-versus-rest, one problem per (band, class): 9 × 4 × 4 = 144 features. Dataset 2b: one binary problem per band. Because 2b has only C = 3 channels, at most 3 filters exist, so all 3 are kept: 9 × 3 = 27 features. (The earlier statement of 9 × 4 = 36 features for 2b was mathematically impossible and has been corrected.)",
    "Log-variance feature: z = Wᵀ X, f_j = log( var(z_j) / Σ_{l=1}^{2m} var(z_l) ).",
    "MIBIF: mutual information between each feature and the class label is estimated on the training partition with the k-nearest-neighbour (Kraskov) estimator (scikit-learn mutual_info_classif, 3 neighbours, fixed random state). For 2a the four-class label is used directly, so all 144 OvR features compete in a single ranking. The k = 20 (2a) or k = 10 (2b) features with the highest MI are kept and standardised with training-partition statistics."]:
    bullet(line)

comment("Leakage control is described explicitly for normalization, CSP, and MIBIF, but not for the regression used to "
        "remove EOG artifacts …")
resp("In the re-analysis the ocular regression is fitted within each training partition. EEG = EOG · b + residual is "
     "solved by least squares (b ∈ ℝ^{3×C}) over all samples of the training-partition epochs (4–40 Hz). The same b "
     "is then applied unchanged to validation and test epochs, and, because filtering and this spatial regression "
     "are both linear, also to every sub-band signal. In the subject-specific protocol b is therefore estimated per "
     "subject from that subject’s training session. In the pooled and LOSO protocols it is estimated on the pooled "
     "training trials of the fold. Section 4.2 now states this explicitly.")

# ---------------------------------------------------------------- latency and training curves
H("Other revised items", 1)
if lat:
    Pp(f"Latency (Section 5.2.11) was re-measured with the released code: full pipeline (10 zero-phase filters on a "
       f"6-s raw window, EOG regression, z-score, FBCSP, MIBIF, forward pass) {lat['full'][0]:.1f} ± {lat['full'][1]:.1f} ms "
       f"per trial; classifier only {lat['clf'][0]:.2f} ± {lat['clf'][1]:.2f} ms (batch 1, 50 warm-up + 1000 timed runs, "
       f"{lat['cpu']}, 2 threads). The earlier i7-10700K values have been replaced, because we could not re-run them "
       "on that machine.")
for ds in ("2a", "2b"):
    f = f"{R}/hist_pooled_{ds}_dual_s0.json"
    if os.path.exists(f):
        h = json.load(open(f)); b = int(np.argmin(h["val_loss"]) + 1)
        Pp(f"Training curves, Dataset {ds} (revised Figure 6): best validation loss at epoch {b}, early stop at epoch "
           f"{len(h['loss'])} (seed 0, pooled protocol).")
        if os.path.exists(f"figures/training_curves_{ds}.png"):
            doc.add_picture(f"figures/training_curves_{ds}.png", width=Inches(6.2))
Pp("Code availability. The previously linked repository contained illustrative snippets operating on synthetic "
   "data. It has been replaced by the complete pipeline used for this revision (prep.py, pipeline.py, "
   "run_experiments.py, analyze.py, latency.py), together with all per-run result files.")
Pp("Software environment (Section 4.3) corrected: all models are implemented in TensorFlow/Keras; MNE-Python is "
   "used only to read the GDF files; filtering uses SciPy and feature selection and metrics use scikit-learn. "
   "PyTorch is not used.")


# ================================================================== Appendix: revised manuscript text
doc.add_page_break()
H("Appendix A. Revised manuscript text (ready to paste; mark in blue)", 1)
Pp("The passages below replace the corresponding passages of the revised manuscript. All numbers are taken from the released result files.", italic=True)
dA, dB = sm('subject', '2a', 'dual'), sm('subject', '2b', 'dual')
lA, lB = sm('loso', '2a', 'dual'), sm('loso', '2b', 'dual')
fA, fB = sm('subject', '2a', 'fbcsp_mlp'), sm('subject', '2b', 'fbcsp_mlp')
H("Abstract", 2)
Pp("Motor imagery (MI)-based brain–computer interface (BCI) systems require reliable classification of "
   "electroencephalography (EEG) signals despite their low signal-to-noise ratio, non-stationarity and substantial "
   "inter-subject variability. This study evaluates a dual-branch framework that fuses Filter Bank Common Spatial "
   "Pattern (FBCSP) features, using a one-versus-rest formulation for four classes, with temporal representations "
   "learned by a Conv1D branch. All data-dependent operations, including ocular-artifact regression, normalisation, "
   "CSP filtering and mutual-information feature selection, are fitted strictly within each training partition. The "
   "framework is evaluated on BCI Competition IV Datasets 2a (four classes, 22 channels) and 2b (two classes, three "
   "bipolar channels) under three protocols: subject-specific cross-session, pooled within-session, and "
   "leave-one-subject-out (LOSO). In the subject-specific cross-session protocol the dual-branch model achieves "
   f"{dA[0]:.1f} ± {dA[1]:.1f} % (2a) and {dB[0]:.1f} ± {dB[1]:.1f} % (2b). LOSO yields {lA[0]:.1f} ± {lA[1]:.1f} % and "
   f"{lB[0]:.1f} ± {lB[1]:.1f} %. A branch ablation shows that the FBCSP branch alone performs comparably to or better than "
   f"the fused model ({fA[0]:.1f} % and {fB[0]:.1f} % subject-specific), and that the Conv1D branch contributes little "
   "beyond the spatial–spectral features. Comparisons with SVM, k-NN, LDA, EEGNet, ShallowConvNet and LSTM baselines "
   "under identical partitions, multi-seed variability estimates and a noise stress test complete the evaluation. "
   "The results identify the conditions under which feature fusion is, and is not, beneficial for MI decoding, and "
   "they underline the need for leakage-controlled, protocol-explicit reporting.")
H("3.1 Data representation (added sentence)", 2)
Pp("Zero-padding the channel dimension to 64 does not change the model: padded channels receive zero gradient and "
   "never contribute to any activation. The padding is retained only to keep a fixed input shape across datasets, "
   "and the parameter counts in Table 1 refer to the padded form.")
H("3.2 FBCSP feature extraction", 2)
Pp("Replace the paragraph after Eq. (3), and the 2b feature count, with the five-point formulation given in the "
   "response to Reviewer 2 (FBCSP comment). The 2b feature count changes from 36 to 27.")
H("3.4 Training configuration (replaces the hyper-parameter paragraph)", 2)
Pp("All hyper-parameters (learning rate 1e-3, dropout 0.5, dense widths 128/64, batch size 32, CSP m = 2, MIBIF "
   "k = 20 for 2a and 10 for 2b, early-stopping patience 15, maximum 100 epochs) were fixed before the experiments "
   "and are identical in all protocols and folds; no hyper-parameter was tuned on any test partition. Early stopping "
   "monitors the loss on a stratified 10 % validation subset drawn from the training partition only, and the "
   "weights with the lowest validation loss are restored. Every experiment is repeated with several random seeds; "
   "each seed changes the data split (pooled protocol), the weight initialisation, the minibatch order and the "
   "dropout masks.")
H("4.2 Preprocessing (replaces the EOG and normalisation paragraphs)", 2)
Pp("Ocular artifacts are removed by linear regression on the three EOG channels. The regression coefficients are "
   "estimated by least squares on the 4–40 Hz training-partition epochs of each split and applied unchanged to "
   "validation and test epochs and to every FBCSP sub-band. Per-channel z-score statistics are likewise estimated on "
   "the training partition only. All trials are retained, including those marked as artifacts by the data providers.")
H("4.3 Software (corrected)", 2)
Pp("The framework is implemented in Python 3 with TensorFlow/Keras. GDF files are read with MNE-Python, filtering "
   "uses SciPy, and feature selection, conventional classifiers and metrics use scikit-learn.")
H("5.1 Evaluation protocol (replaces the section)", 2)
Pp("Three protocols are used (Table 3). (i) Subject-specific cross-session: for each subject, a model is trained on "
   "the official training session(s) (2a: A0xT; 2b: sessions 1–3) and tested on the evaluation session(s), which were "
   "recorded on different days (2a: A0xE; 2b: sessions 4–5). No data from other subjects are used. This is the "
   "primary within-subject protocol, and it also constitutes the inter-session analysis for all nine subjects. (ii) "
   "Pooled within-session (subject-overlapping): all A0xT trials of 2a (2592) or all 2b trials (6520) are pooled and "
   "split 80/20, stratified jointly by subject × session × class. Training and test sets therefore contain different "
   "trials of the same subjects and sessions, and the results must not be read as per-subject or cross-session "
   "performance. (iii) LOSO: the same trial pool as (ii); in each of nine folds, one subject is held out entirely. In "
   "all protocols, every data-dependent operation is fitted on the training partition only. Results are reported per "
   "subject with mean ± SD, together with Cohen’s κ, and paired Wilcoxon signed-rank tests over subjects are used to "
   "compare models.")
H("5.4 Discussion (replaces the paragraphs on the baseline advantage and noise)", 2)
Pp("The branch ablation is the central result of the re-analysis. On Dataset 2a the Conv1D branch alone performs "
   "near chance in every protocol, and adding it to the FBCSP features lowers accuracy relative to the FBCSP branch "
   "alone. On Dataset 2b, where the three bipolar channels concentrate the sensorimotor information, the Conv1D "
   "branch is informative on its own, but fusion still performs no better than FBCSP alone. The spatial–spectral "
   "FBCSP representation therefore carries essentially all of the discriminative information that the fused model "
   "exploits. A plausible reason is the imbalance in the fused layer, where 7872 temporal activations face 20 or 10 "
   "FBCSP features. With a few hundred trials per subject, the network fits the high-dimensional temporal features, "
   "which do not transfer across sessions or subjects. Fusion strategies that balance the branches (for example "
   "bottleneck projection of the temporal branch, separate branch training, or late fusion of class posteriors) "
   "are a direct direction for future work. The large accuracy differences between subjects, and the low LOSO "
   "accuracy on four-class data, are consistent with the BCI-inefficiency and inter-subject-variability problems "
   "widely reported for these datasets.")
H("6. Conclusions (replaces the second paragraph)", 2)
Pp(f"Under the subject-specific cross-session protocol, the dual-branch framework achieved {dA[0]:.1f} % on Dataset 2a "
   f"and {dB[0]:.1f} % on Dataset 2b, and LOSO evaluation gave {lA[0]:.1f} % and {lB[0]:.1f} %. The ablation showed "
   "that the FBCSP branch alone performs comparably to or better than the fused model. Under leakage-controlled evaluation, the "
   "advantage of the evaluated FBCSP–Conv1D fusion is therefore not supported; the main value of the framework lies "
   "in its FBCSP spatial–spectral branch.")

doc.save("Response_to_Reviewers_MTI_revision.docx")
print("saved")
