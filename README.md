# Dual-Branch FBCSP–Conv1D for MI-EEG — leakage-controlled re-analysis (MTI revision)

This folder contains everything needed to regenerate every number, table and figure in the revised manuscript and in the response to reviewers.

## 1. Data (not included; public)
- BCI Competition IV 2a GDF files `A01T.gdf … A09E.gdf`, plus the official true labels `A01T.mat … A09E.mat` (`true_labels.zip`, ds2a).
- BCI Competition IV 2b GDF files `B0101T.gdf … B0905E.gdf`, plus the official true labels `B0101T.mat … B0905E.mat` (`true_labels.zip`, ds2b).
- Source: https://www.bbci.de/competition/iv/

## 2. Environment
```
pip install tensorflow-cpu==2.21.0 mne scikit-learn scipy pandas matplotlib python-docx
```
(The results were produced with TF 2.21, MNE 1.13, scikit-learn 1.9, CPU only, 2 threads.)

## 3. Run (in this order)
```
# 1) epoch caches (fixed filters + epoching only; no data-dependent step)
python prep.py --dataset 2a --gdf_dir BCICIV_2a_gdf --label_dir true_labels_2a --out cache_2a.npz
python prep.py --dataset 2b --gdf_dir BCICIV_2b_gdf --label_dir true_labels_2b --out cache_2b.npz
python -c "import numpy as np
for ds in ['2a','2b']:
    d=np.load(f'cache_{ds}.npz')
    [np.save(f'{ds}_{k}.npy', d[k]) for k in d.files]"

# 2) all experiments (resumable; ~8 h on 2 CPU cores, much faster on a GPU / Colab / Kaggle)
bash queue2.sh

# 3) tables, figures, response document
python analyze.py          # -> results/tables.md, figures/*.png
python latency.py          # run on an idle machine
python make_response.py    # -> Response_to_Reviewers_MTI_revision.docx
```

## 4. Files
| File | Purpose |
|---|---|
| `prep.py` | GDF loading, labels, 4–40 Hz + nine 4-Hz sub-band zero-phase Butterworth (order 5), epochs 0.5–2.5 s after the cue |
| `pipeline.py` | Fit-on-train-only objects: EOG regression, channel z-score, FBCSP (OvR for 2a), MIBIF, scaler. All models (dual-branch, ablations, EEGNet-8,2, ShallowConvNet, LSTM, FBCSP+SVM/k-NN/LDA) |
| `run_experiments.py` | Protocols `subject` (cross-session, per subject), `pooled` (80/20 stratified by subject × session × class, multi-seed), `loso`; noise stress test |
| `analyze.py` | All tables (markdown) + figures |
| `latency.py` | Single-trial latency, full pipeline vs classifier only |
| `make_response.py` | Builds the response-to-reviewers .docx directly from the result files |
| `results/*.csv` | One row per (model, seed, fold): accuracy, κ, macro P/R/F1, epochs |
| `results/preds_*.npz` | Per-trial true/predicted labels with subject and session (confusion matrices, per-subject tables) |
| `results/noise_*.csv` | Noise stress test, 30 realisations per seed |

## 5. Leakage control
Every data-dependent step is fitted inside the training partition of each split and applied unchanged to the test partition: EOG coefficients, z-score statistics, CSP filters, MIBIF ranking, the feature scaler, SVM/k-NN grid search (inner 5-fold CV), and early stopping (stratified 10 % validation subset of the training partition). The hyper-parameters of the networks are fixed a priori and are never tuned on test data.
