"""
latency.py — single-trial inference latency of the complete pipeline (batch size 1).
Full pipeline = broadband + 9 sub-band zero-phase filters on a padded raw window,
EOG regression, z-score, FBCSP log-variance, MIBIF selection, scaling, network forward pass.
Classifier-only = forward pass of the dual-branch network on pre-computed inputs.
50 warm-up runs, then 1000 timed runs. Run on an otherwise idle machine.
"""
import platform, subprocess, time
import numpy as np
from scipy.signal import sosfiltfilt
import tensorflow as tf
import pipeline as P
from prep import SOS_BROAD, SOS_BANDS

D = {k: np.load(f"2a_{k}.npy", mmap_mode="r") for k in ("W", "B", "y", "sess")}
tr = np.where(np.asarray(D["sess"]) == 1)[0][:2000]
fe = P.Features(22, 2, 20).fit(np.asarray(D["W"][tr]), np.asarray(D["B"][tr]), np.asarray(D["y"][tr]))
m = P.build_dual(500, 20, 4, C=22)
fwd = tf.function(lambda a, f: m([a, f], training=False))
rng = np.random.default_rng(0)
raw = rng.standard_normal((25, 1500)).astype(np.float64)   # 6 s raw window (25 ch), trial centred


def full(x):
    w = sosfiltfilt(SOS_BROAD, x, axis=1)[:, 500:1000][None]
    b = np.stack([sosfiltfilt(s, x, axis=1)[:, 500:1000] for s in SOS_BANDS])[None]
    Xt, F = fe.transform(w.astype(np.float32), b.astype(np.float32))
    return fwd(tf.constant(P.to_conv_input(Xt)), tf.constant(F)).numpy()


Xt, F = fe.transform(np.asarray(D["W"][:1]), np.asarray(D["B"][:1]))
a_, f_ = tf.constant(P.to_conv_input(Xt)), tf.constant(F)
res = {}
for name, fn in (("full_pipeline", lambda: full(raw)), ("classifier_only", lambda: fwd(a_, f_).numpy())):
    for _ in range(50):
        fn()
    t = []
    for _ in range(1000):
        t0 = time.perf_counter(); fn(); t.append(1e3 * (time.perf_counter() - t0))
    res[name] = (np.mean(t), np.std(t))
    print(f"{name}: {np.mean(t):.2f} ± {np.std(t):.2f} ms per trial")
cpu = subprocess.run("lscpu | grep 'Model name'", shell=True, capture_output=True, text=True).stdout.strip()
print("CPU:", cpu, "| threads:", 2, "| TF", tf.__version__, "|", platform.platform())
import json
json.dump({"full": res["full_pipeline"], "clf": res["classifier_only"],
           "cpu": cpu.replace("Model name:", "").strip()}, open("results/latency.json", "w"))
