"""Example 7 -- adding a new scan path, and the programmable-beam scans of Palos et al.

    python examples/07_add_a_path.py        (about 5 seconds)

Part 1 adds a path from *outside* the package: one function and one decorator, in your
own script or notebook. Nothing in scanpath/ is edited, yet the path works everywhere.

Part 2 uses the pixel-list scans that now live in scanpath/paths.py (grid_spiral in and
out, n_sequential, random_order; Palos et al., Microsc. Microanal. 32, ozag080, 2026) and
compares them on a lagging scanner that, like their scan engine, steps from pixel to pixel.

See docs/adding_a_path.md for the step-by-step guide.
"""
import numpy as np

from _style import BLUE, INK2, ORANGE, plt, save

import scanpath as sp


# --- part 1: a new path in four lines ---------------------------------------------------
@sp.register_path("spiral_in_out", knob="turns")
def spiral_in_out(s, turns=20, radius=1.0):
    """Archimedean spiral from the edge to the centre, then back out to the edge."""
    r = radius * np.abs(1 - 2 * s)                 # 1 -> 0 -> 1 over the scan
    angle = 2 * np.pi * turns * 2 * s
    return np.column_stack((r * np.cos(angle), r * np.sin(angle)))


budget = sp.Budget()
sample = sp.Sample.synthetic(budget.fov_nm)
print("parameters:", sp.parameters("spiral_in_out"))
result = sp.run(("spiral_in_out", {"turns": 30}), "lowpass", ["image_nrmse", "coverage"],
                budget, sample)
print("spiral_in_out on the 1.5 kHz scanner:",
      {k: round(v, 3) for k, v in result.scores.items()})

# --- part 2: the paper's scan modes, through the same pipeline ---------------------------
MODES = [("raster", {"lines": 64}),
         ("grid_spiral", {"direction": "in"}), ("grid_spiral", {"direction": "out"}),
         ("n_sequential", {"n": 2}), ("n_sequential", {"n": 4}), ("random_order", {})]
LABELS = ["raster", "spiral in", "spiral out", "N = 2", "N = 4", "random"]
STEPPING = ("lowpass", {"bandwidth_hz": 1500, "hold": "step"})   # command steps, scanner lags
METRICS = ["image_nrmse", "rms_error_nm", "path_length_um"]

print(f"\n{'scan':12s}" + "".join(f"{m:>16s}" for m in METRICS))
scores = []
for label, mode in zip(LABELS, MODES):
    r = sp.run(mode, STEPPING, METRICS, budget, sample)
    scores.append(r.scores)
    print(f"{label:12s}" + "".join(f"{r.scores[m]:16.3f}" for m in METRICS))

# The direction of the spiral is just another parameter, so a search can choose it:
rows, best = sp.grid_search("grid_spiral", {"direction": ["in", "out"], "pixels": [32, 48, 64]},
                            "image_nrmse", STEPPING, budget=budget, sample=sample)
print(f"\nbest grid_spiral: direction {best['direction']}, pixels {best['pixels']}, "
      f"image_nrmse {best['image_nrmse']:.3f}")

# --- figure: acquisition order (as in the paper's Fig. 2) and the scores -----------------
small = sp.Budget(fov_nm=16, duration_s=1.0, rate_hz=256)          # 16 x 16, one sample each
fig = plt.figure(figsize=(15, 6.2))
for i, (label, (family, params)) in enumerate(zip(LABELS[1:], MODES[1:])):
    ax = fig.add_subplot(2, 6, i + 1 if i < 3 else i + 4)
    path = sp.make_path(family, small, **{**params, "pixels": 16})
    order = np.zeros((16, 16))
    col = np.round(path.xy[:, 0] + 7.5).astype(int)
    row = np.round(7.5 - path.xy[:, 1]).astype(int)
    order[row, col] = np.arange(len(path.xy))
    ax.imshow(order, cmap="Spectral")
    ax.set_title(label, fontsize=10); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
fig.text(0.01, 0.97, "Acquisition order, red first, blue last (16 x 16 pixels)",
         fontsize=10, fontweight="bold", color=INK2)

x = np.arange(len(LABELS))
for k, (metric, colour) in enumerate((("image_nrmse", BLUE), ("rms_error_nm", ORANGE))):
    ax = fig.add_axes([0.53 + k * 0.245, 0.12, 0.2, 0.74])
    ax.bar(x, [s[metric] for s in scores], 0.6, color=colour)
    ax.set_xticks(x, LABELS, rotation=35, fontsize=8.5); ax.grid(axis="x", visible=False)
    ax.set_title(metric.replace("_", " "), fontsize=10)
fig.text(0.53, 0.97, "On a 1.5 kHz scanner stepping pixel to pixel (64 x 64, 1 s)",
         fontsize=10, fontweight="bold", color=INK2)
save(fig, "07_add_a_path.png")
