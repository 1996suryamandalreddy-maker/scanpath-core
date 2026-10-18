"""Example 2 -- optimize within one path class by evaluating a parameter grid.

    python examples/02_lissajous_grid.py            # about 3 minutes
    python examples/02_lissajous_grid.py --quick    # smaller grid, about 1 minute

Left: image error over the full (nx, ny) grid on a 1.5 kHz scanner.
Right: the same score along ny = nx + 1 for scanners of different bandwidth --
more cycles fill the field better but drive the scanner harder, so the best
Lissajous depends on the instrument.
"""
import csv
import sys

import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from _style import AQUA, BLUE, FIGURES, INK, INK2, ORANGE, ROOT, SEQ, VIOLET, plt, save

import scanpath as sp

SCORE = "image_nrmse"                              # the selected quality function
quick = "--quick" in sys.argv
budget = sp.Budget(fov_nm=100, duration_s=1.0, rate_hz=20_000)
sample = sp.Sample.synthetic(budget.fov_nm, seed=0)
scanner = ("lowpass", {"bandwidth_hz": 1500})

# --- 2-D grid ----------------------------------------------------------------------
n = np.arange(16, 41) if quick else np.arange(6, 51)   # step 1: ratios matter
rows, best = sp.grid_search("lissajous", {"nx": list(n), "ny": list(n)}, SCORE,
                            instrument=scanner, budget=budget, sample=sample)
print(f"best on 1.5 kHz scanner: nx={best['nx']}, ny={best['ny']}, {SCORE}={best[SCORE]:.3f}")
with open(FIGURES / "02_lissajous_grid.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
grid = np.array([r[SCORE] for r in rows]).reshape(len(n), len(n))     # [nx, ny]

# --- 1-D cut along ny = nx + 1, for four instruments -----------------------------------
cut = np.arange(5, 122, 6 if quick else 2)
models = [("ideal scanner", "ideal", BLUE),
          ("3 kHz", ("lowpass", {"bandwidth_hz": 3000}), AQUA),
          ("1.5 kHz", ("lowpass", {"bandwidth_hz": 1500}), ORANGE),
          ("0.8 kHz", ("lowpass", {"bandwidth_hz": 800}), VIOLET)]
curves = {}
for label, model, _ in models:
    points = [{"nx": int(k), "ny": int(k) + 1} for k in cut]   # explicit, coupled points
    line, _ = sp.grid_search("lissajous", points, SCORE, instrument=model,
                             budget=budget, sample=sample)
    curves[label] = [r[SCORE] for r in line]
    i = int(np.argmin(curves[label]))
    print(f"  ny = nx + 1, {label:13s}: best nx = {cut[i]:3d}  {SCORE} = {curves[label][i]:.3f}")

# --- figure --------------------------------------------------------------------------
fig, (a, b) = plt.subplots(1, 2, figsize=(12.5, 4.8), gridspec_kw={"width_ratios": [1, 1.25]})
cmap = LinearSegmentedColormap.from_list("seq", SEQ)
step = n[1] - n[0]
im = a.imshow(grid.T, origin="lower", cmap=cmap, vmin=0, vmax=1,
              extent=[n[0] - step / 2, n[-1] + step / 2] * 2)
a.plot(best["nx"], best["ny"], marker="*", ms=14, color=ORANGE, mec="white")
a.annotate(f"best {best['nx']}/{best['ny']}\nNRMSE {best[SCORE]:.2f}", (best["nx"], best["ny"]),
           xytext=(10, 10), textcoords="offset points", color=INK, fontsize=9,
           bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.9))
a.set_xlabel("nx  (x cycles per frame)"); a.set_ylabel("ny  (y cycles per frame)")
a.set_title("Image NRMSE over the Lissajous grid (1.5 kHz scanner)")
a.grid(False)
fig.colorbar(im, ax=a, fraction=0.046, label="image NRMSE (lower is better)")

for label, _, color in models:
    y = curves[label]
    i = int(np.argmin(y))
    b.plot(cut, y, color=color, label=label)
    b.plot(cut[i], y[i], "o", ms=8, color=color, mec="white", mew=2)
    b.annotate(label, (cut[-1], y[-1]), xytext=(6, 0), textcoords="offset points",
               va="center", color=INK2, fontsize=9)
b.set_xlim(0, cut[-1] + 22)
b.set_ylim(0, 1.0)
b.set_xlabel("nx  (with ny = nx + 1)"); b.set_ylabel("image NRMSE (lower is better)")
b.set_title("The optimum moves with scanner bandwidth (dots = best)")
b.legend(loc="upper left", bbox_to_anchor=(0.06, 0.99), ncol=2, fontsize=9)
fig.tight_layout()
save(fig, "02_lissajous_grid.png")
