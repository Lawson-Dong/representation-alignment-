"""Animate actual per-image attention-model geometry with time dilation at CKA dips.

The shared 2D PCA is fitted to every image's similarity fingerprint against
the same 200 images, pooled over all observed blocks. Motion between measured
blocks is an illustration; only the endpoints are actual model outputs.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation
import numpy as np
import sys
import pandas as pd
from sklearn.decomposition import PCA

ROOT = Path(__import__("os").environ.get("GEOMETRY_OUTPUT_DIR", "outputs")).resolve()
ROOT.mkdir(parents=True, exist_ok=True)
slug = sys.argv[1]
df = pd.read_csv(ROOT / "attention_geometry_metrics.csv")
df = df.loc[df.slug.eq(slug)].reset_index(drop=True)
stages = df.stage.to_list()
source = np.load(ROOT / f"{slug}_per_image_vectors.npz")
labels = source["labels"]
assert np.array_equal(source["stages"], stages)
L = len(stages)
model_title = str(source["model"]) + " / " + str(source["readout"])
assert np.array_equal(np.unique(labels, return_counts=True)[1], [100, 100])

# Channel counts vary between stages. A same-image 200-coordinate relational
# fingerprint gives a comparable feature space for one shared PCA fit.
profiles = []
for k in range(len(stages)):
    raw = source[f"features_{k}"].astype(np.float64)
    unit = raw / np.maximum(np.linalg.norm(raw, axis=1, keepdims=True), 1e-12)
    sim = unit @ unit.T
    profiles.append(sim - sim.mean(axis=0, keepdims=True))
profiles = np.asarray(profiles)
pca = PCA(n_components=2, svd_solver="full")
clouds = pca.fit_transform(profiles.reshape(-1, 200)).reshape(L, 200, 2)
print("Shared PCA explained variance:", pca.explained_variance_ratio_, flush=True)

x = np.arange(L)
within = df.d_within_cos.to_numpy()
between = df.d_between_cos.to_numpy()
separation = df.S.to_numpy()
cka = df.CKA_prev.to_numpy()

FPS, SECONDS = 20, 12
FRAMES = FPS * SECONDS
# Segment j is the passage from observation j-1 to j. Slower segments have
# more screen time, but the measurement positions on the plots remain discrete.
focus = np.argsort(cka[1:])[:2] + 1
weights = np.full(L-1, .45)
for j in focus: weights[j-1] = 1.8 if j != focus[0] else 2.7
stage_time = np.r_[0, np.cumsum(weights)]
stage_time *= SECONDS / stage_time[-1]
print("Slow intervals (s):", {stages[i]: round(stage_time[i] - stage_time[i-1], 2)
                              for i in focus}, flush=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "text.color": "#eef3fc", "axes.labelcolor": "#aab8ce",
    "xtick.color": "#9aabc0", "ytick.color": "#9aabc0",
})
fig = plt.figure(figsize=(14.4, 8.1), dpi=100, facecolor="#0b1424")
ax = fig.add_axes([.057, .20, .49, .60], facecolor="#111f33")
ax_s = fig.add_axes([.63, .62, .32, .18], facecolor="#111f33")
ax_c = fig.add_axes([.63, .39, .32, .17], facecolor="#111f33")
ax_d = fig.add_axes([.63, .20, .32, .14], facecolor="#111f33")
for a in (ax, ax_s, ax_c, ax_d):
    for spine in a.spines.values():
        spine.set_color("#34475e")
    a.grid(color="#52637c", alpha=.22, lw=.7)

limits = np.array([[clouds[:, :, j].min(), clouds[:, :, j].max()]
                   for j in range(2)])
padding = .08 * (limits[:, 1] - limits[:, 0])
ax.set(xlim=(limits[0, 0]-padding[0], limits[0, 1]+padding[0]),
       ylim=(limits[1, 0]-padding[1], limits[1, 1]+padding[1]),
       xlabel="Shared PCA 1  ·  similarity fingerprint", ylabel="Shared PCA 2")
ax.set_aspect("equal", adjustable="box")
ax.set_title("200 real image vectors  ·  shared projection", loc="left",
             pad=12, weight="bold", fontsize=16)
colors = ["#65d7da", "#ff9f79"]
scats = [ax.scatter([], [], s=25, c=color, alpha=.76, lw=0, label=name)
         for color, name in zip(colors, ["cat (100)", "dog (100)"])]
centers = [ax.scatter([], [], s=180, c=color, marker="x", lw=2.6)
           for color in colors]
ax.legend(loc="upper right", frameon=False, fontsize=10, ncol=2)

for a in (ax_s, ax_c, ax_d):
    a.set_xlim(-.5, L-.5)
    for boundary in focus:
        a.axvline(boundary, color="#f2c56b", alpha=.30, lw=1.2)
    a.set_xticks(sorted(set([0, L-1, *focus, *range(0,L,4)])))
ax_s.plot(x, separation, color="#ccb1ff", lw=2.5, marker="o", ms=3)
dot_s, = ax_s.plot([], [], "o", ms=10, color="#ccb1ff", mec="white", mew=1.2)
ax_s.set(ylim=(min(.98, separation.min()-.03), max(1.39, separation.max()+.03)), ylabel="S", title="Relative separation  ·  between / within")
ax_s.tick_params(axis="x", labelbottom=False)
ax_c.plot(x[1:], cka[1:], color="#f4c778", lw=2.2, marker="o", ms=3)
for j in focus:
    ax_c.scatter([j], [cka[j]], s=90, facecolor="none", edgecolor="#ff7c8a", lw=1.6, zorder=5)
dot_c, = ax_c.plot([], [], "o", ms=10, color="#f4c778", mec="white", mew=1.2)
ax_c.set(ylim=(0, 1.05), ylabel="CKA", title="Adjacent CKA  ·  measured at each endpoint")
ax_c.tick_params(axis="x", labelbottom=False)
ax_d.plot(x, within, color=colors[0], lw=2.1, label="within")
ax_d.plot(x, between, color=colors[1], lw=2.1, label="between")
dot_w, = ax_d.plot([], [], "o", ms=8, color=colors[0])
dot_b, = ax_d.plot([], [], "o", ms=8, color=colors[1])
ax_d.set(ylim=(0, max(between.max(), within.max())*1.1), ylabel="Cosine distance", xlabel="Observed block / transition")
ax_d.legend(frameon=False, loc="upper left", fontsize=9, ncol=2)

fig.text(.057, .94, model_title + "  /  cat versus dog", fontsize=23, weight="bold")
fig.text(.057, .89, f"ImageNet pretrained  ·  {L} observed outputs  ·  same 200 images and preprocessing",
         fontsize=12, color="#b6c5d9")
layer_text = fig.text(.057, .135, "", fontsize=19, weight="bold")
event_text = fig.text(.057, .091, "", fontsize=13, color="#f4c778")
metric_text = fig.text(.63, .865, "", fontsize=13, color="#e0e9f6")
fig.text(.057, .034,
         "Measured dots use one shared PCA of same-image cosine profiles; motion is interpolated. A CKA dip signals reorganization, not proof of a phase transition.",
         fontsize=9.6, color="#aabbd0")
progress = fig.add_axes([.057, .066, .893, .006], facecolor="#24364b")
progress.set(xlim=(0, 1), ylim=(0, 1))
progress.set_axis_off()
bar = progress.barh(.5, 0, height=1, color="#f4c778", left=0)[0]

def update(frame):
    t = frame / (FRAMES-1) * SECONDS
    q = float(np.interp(t, stage_time, x))
    i = min(int(q), len(stages)-2)
    blend = q - i
    eased = 3*blend**2 - 2*blend**3
    pts = (1-eased)*clouds[i] + eased*clouds[i+1]
    for j, label in enumerate((3, 5)):
        selected = pts[labels == label]
        scats[j].set_offsets(selected)
        centers[j].set_offsets(selected.mean(axis=0))
    w = np.interp(q, x, within)
    b = np.interp(q, x, between)
    s = b/w
    dot_s.set_data([q], [s])
    dot_w.set_data([q], [w])
    dot_b.set_data([q], [b])
    current = min(round(q), L-1)
    target = min(int(np.ceil(q)), L-1)
    if q == 0:
        dot_c.set_data([], [])
    else:
        dot_c.set_data([target], [cka[target]])
    layer_text.set_text(f"{current+1:02d} / {L:02d}    {stages[current]}")
    metric_text.set_text(f"S  {s:.3f}   ·   within  {w:.3f}   ·   between  {b:.3f}")
    if target in focus and q < target:
        event_text.set_text(f"SLOW MOTION   {stages[target-1]} → {stages[target]}   ·   adjacent CKA {cka[target]:.3f}")
        event_text.set_color("#ffb879")
    else:
        event_text.set_text("Lower adjacent CKA = stronger change in the 200-image relational geometry")
        event_text.set_color("#aabbd0")
    bar.set_width((frame+1)/FRAMES)
    return (*scats, *centers, dot_s, dot_c, dot_w, dot_b,
            layer_text, event_text, metric_text, bar)

animation = FuncAnimation(fig, update, frames=FRAMES, interval=1000/FPS, blit=False)
out = ROOT / f"{slug}_geometry_slow_12s.mp4"
animation.save(out, writer=FFMpegWriter(fps=FPS, codec="libx264", bitrate=3000,
                                      extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"]))
update(int(stage_time[focus[0]]/SECONDS*(FRAMES-1))-12)
fig.savefig(ROOT / f"{slug}_geometry_preview.png", facecolor=fig.get_facecolor(), dpi=100)
print("Video:", out)
