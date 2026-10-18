"""Example 1 -- generate one path, pass it through an instrument model, score it.

    python examples/01_run_one.py
"""
from _style import BLUE, INK2, ORANGE, plt, save

import numpy as np

import scanpath as sp

budget = sp.Budget(fov_nm=100, duration_s=1.0, rate_hz=20_000)   # shared by every path
sample = sp.Sample.synthetic(budget.fov_nm, seed=0)               # ideal image

# --- the whole pipeline, one stage at a time ------------------------------------
commanded = sp.make_path("lissajous", budget, nx=23, ny=24)          # 1. generator
realized = sp.simulate(commanded, "lowpass", bandwidth_hz=1500)      # 2. instrument
scores = sp.evaluate(commanded, realized, sample)                    # 3. evaluator

print("Lissajous 23/24 on a 1.5 kHz scanner")
for name, value in scores.items():
    print(f"  {name:20s} {value:9.3f}")

# --- the same thing in one call, for three families at the same budget ----------
print("\nSame budget, three families (path length is a soft constraint: reported, not forced)")
keys = ["path_length_um", "speed_p99_um_s", "rms_error_nm", "coverage", "image_nrmse"]
print(f"  {'family':10s}" + "".join(f"{k:>16s}" for k in keys))
for family in ("raster", "spiral", "lissajous"):
    r = sp.run(family, ("lowpass", {"bandwidth_hz": 1500}), keys, budget, sample)
    print(f"  {family:10s}" + "".join(f"{r.scores[k]:16.3f}" for k in keys))

# --- figure ---------------------------------------------------------------------
image = sp.image_as_acquired(commanded, realized, sample)   # read at realized, drawn at commanded
extent = [-budget.fov_nm / 2, budget.fov_nm / 2] * 2

fig, ax = plt.subplots(1, 4, figsize=(15, 3.8))
ax[0].plot(*commanded.xy.T, color=BLUE, lw=0.3)
ax[0].add_patch(plt.Rectangle((41, 41), 10, 10, fill=False, ec=ORANGE, lw=1.5))
ax[0].set_title("1  Commanded path")
ax[1].plot(*commanded.xy.T, color="#c3c2b7", lw=0.8, zorder=1)
tick = slice(None, None, 15)                       # same instants on both paths
c, r = commanded.xy[tick], realized.xy[tick]
ax[1].plot(np.stack((c[:, 0], r[:, 0])), np.stack((c[:, 1], r[:, 1])), color=INK2, lw=0.8)
ax[1].scatter(*c.T, s=10, color=BLUE, label="commanded at t", zorder=3)
ax[1].scatter(*r.T, s=10, color=ORANGE, label="realized at t (1.5 kHz)", zorder=3)
ax[1].set_xlim(41, 51); ax[1].set_ylim(41, 51)
ax[1].legend(loc="lower left", fontsize=8, framealpha=0.9, frameon=True)
ax[1].set_title("2  Instrument: lag (corner zoom)")
ax[2].imshow(sample.image, extent=extent, origin="lower", cmap="gray")
ax[2].set_title("Ideal image")
ax[3].imshow(image, extent=extent, origin="lower", cmap="gray")
ax[3].set_title(f"3  As acquired: NRMSE {scores['image_nrmse']:.2f}")
for a in ax:
    a.set_aspect("equal"); a.set_xlabel("x (nm)")
    a.grid(a in ax[:2])
ax[0].set_ylabel("y (nm)")
fig.suptitle("make_path  ->  simulate  ->  evaluate", x=0.01, ha="left",
             color=INK2, fontsize=10)
save(fig, "01_run_one.png")
