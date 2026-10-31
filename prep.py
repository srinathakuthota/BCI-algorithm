"""
prep.py — Load BCI Competition IV 2a / 2b GDF files and build epoch caches.

Only operations that are NOT data-dependent (fixed zero-phase Butterworth filters,
epoch cutting) are done here, once, on the continuous recordings. Everything that
is estimated from data (EOG-regression coefficients, z-score statistics, CSP
filters, MIBIF selection) is fitted later, inside each training partition
(see pipeline.py).

Output per dataset (npz):
  B   : (n_trials, 9, C_eeg + C_eog, 500) float32 — sub-band (4-8 ... 36-40 Hz) epochs
  W   : (n_trials, C_eeg + C_eog, 500)    float32 — broadband 4-40 Hz epochs
  y   : class label 0..K-1
  subj: subject index 1..9
  sess: session index (2a: 1 = A0xT, 2 = A0xE; 2b: 1..5 = B0x01T ... B0x05E)
  art : 1 if the trial was marked as artifact (event 1023) by the data providers

Usage:
  python prep.py --dataset 2a --gdf_dir /path/to/BCICIV_2a_gdf --label_dir /path/to/true_labels_2a --out cache_2a.npz
  python prep.py --dataset 2b --gdf_dir /path/to/BCICIV_2b_gdf --label_dir /path/to/true_labels_2b --out cache_2b.npz
"""
import argparse, os, warnings
import numpy as np
import mne
import scipy.io as sio
from scipy.signal import butter, sosfiltfilt

warnings.filterwarnings("ignore")
mne.set_log_level("ERROR")

FS = 250
BANDS = [(4, 8), (8, 12), (12, 16), (16, 20), (20, 24), (24, 28), (28, 32), (32, 36), (36, 40)]
T0, T1 = 0.5, 2.5          # epoch window relative to cue onset (s) -> 500 samples
N_SAMP = int(round((T1 - T0) * FS))
CUE_CODES = {"769": 0, "770": 1, "771": 2, "772": 3, "783": -1}  # 783 = cue of unknown class (E sessions)


def bp(lo, hi, order=5):
    return butter(order, [lo, hi], btype="bandpass", fs=FS, output="sos")


SOS_BROAD = bp(4, 40)
SOS_BANDS = [bp(lo, hi) for lo, hi in BANDS]


def load_file(path, n_eeg, label_mat=None):
    raw = mne.io.read_raw_gdf(path, preload=True)
    data = raw.get_data().astype(np.float64) * 1e6        # microvolts, (C, N)
    data = data[: n_eeg + 3]                               # EEG + 3 EOG
    data[~np.isfinite(data)] = 0.0                         # GDF overflow samples -> 0
    ev, ed = mne.events_from_annotations(raw)
    inv = {v: k for k, v in ed.items()}
    codes = [inv[e] for e in ev[:, 2]]
    onsets = ev[:, 0]
    # artifact flags: event 1023 occurs at the trial start (768), before the cue
    trials, art = [], []
    last_start, last_rej = -1, False
    for o, c in zip(onsets, codes):
        if c == "768":
            last_start, last_rej = o, False
        elif c == "1023":
            last_rej = True
        elif c in CUE_CODES:
            trials.append((o, CUE_CODES[c]))
            art.append(1 if (last_rej and o - last_start < 10 * FS) else 0)
    if label_mat is not None:     # official true labels (classlabel 1..K, in trial order)
        lab = sio.loadmat(label_mat)["classlabel"].ravel().astype(int) - 1
        assert len(lab) == len(trials), (path, len(lab), len(trials))
        for (o, c), l in zip(trials, lab):
            assert c in (-1, l), f"label mismatch in {path}"
        trials = [(o, int(l)) for (o, _), l in zip(trials, lab)]
    assert all(c >= 0 for _, c in trials), f"{path}: unlabeled trials and no label file given"
    return data, trials, art


def epochs_from(data, trials):
    s0 = int(T0 * FS)
    broad = sosfiltfilt(SOS_BROAD, data, axis=1)
    bands = [sosfiltfilt(s, data, axis=1) for s in SOS_BANDS]
    W, B = [], []
    for o, _ in trials:
        a = o + s0
        W.append(broad[:, a:a + N_SAMP])
        B.append(np.stack([b[:, a:a + N_SAMP] for b in bands]))
    return np.asarray(W, np.float32), np.asarray(B, np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["2a", "2b"], required=True)
    ap.add_argument("--gdf_dir", required=True)
    ap.add_argument("--label_dir", required=True, help="folder with the official true_labels *.mat files")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    if a.dataset == "2a":
        n_eeg = 22
        files = [(s, k, f"A0{s}{'T' if k == 1 else 'E'}.gdf") for s in range(1, 10) for k in (1, 2)]
    else:
        n_eeg = 3
        files = [(s, k, f"B0{s}0{k}{'T' if k <= 3 else 'E'}.gdf") for s in range(1, 10) for k in (1, 2, 3, 4, 5)]

    Ws, Bs, ys, subj, sess, arts = [], [], [], [], [], []
    for s, k, fn in files:
        data, trials, art = load_file(os.path.join(a.gdf_dir, fn), n_eeg,
                                      os.path.join(a.label_dir, fn.replace(".gdf", ".mat")))
        W, B = epochs_from(data, trials)
        Ws.append(W); Bs.append(B)
        ys += [t[1] for t in trials]; subj += [s] * len(trials); sess += [k] * len(trials); arts += art
        print(f"{fn}: {len(trials)} trials, class counts {np.bincount([t[1] for t in trials])}, artifacts {sum(art)}")
    np.savez(a.out, W=np.concatenate(Ws), B=np.concatenate(Bs), y=np.array(ys), subj=np.array(subj),
             sess=np.array(sess), art=np.array(arts), n_eeg=n_eeg)
    print("saved", a.out, "total trials", len(ys))


if __name__ == "__main__":
    main()
