"""Example 3 -- substitute one module, add another, change nothing else.

    python examples/03_swap_modules.py

(a) Swap the instrument model: first-order 'lowpass' -> second-order 'resonant'.
    One string changes; the path, evaluator and optimizer calls are identical.
(b) Add a new quality function, defined in THIS file. The package is not edited.
(c) Rerun the same grid search with the new instrument: the optimum moves.
"""
import numpy as np

from _style import AQUA, BLUE, ORANGE, plt, save

import scanpath as sp


# (b) a new metric, registered from outside the package ------------------------------
@sp.register_metric("affine_residual_nm")
def affine_residual_nm(commanded, realized, sample):
    """Distortion left after the best affine correction (offset, scale, shear,
    rotation). Affine errors are benign -- a calibration removes them; this residual
    is the part that is not."""
    design = np.column_stack((commanded.xy, np.ones(len(commanded.xy))))
    fit, *_ = np.linalg.lstsq(design, realized.xy, rcond=None)
    residual = realized.xy - design @ fit
    return np.sqrt(np.mean(np.sum(residual**2, axis=1)))


budget = sp.Budget(fov_nm=100, duration_s=1.0, rate_hz=20_000)
sample = sp.Sample.synthetic(budget.fov_nm, seed=0)
instruments = {"lowpass": ("lowpass", {"bandwidth_hz": 1500}),
               "resonant": ("resonant", {"resonance_hz": 1500, "damping": 0.15})}
keys = ["rms_error_nm", "affine_residual_nm", "image_nrmse"]

# (a) same paths, two instrument models ---------------------------------------------
print(f"{'path':16s}{'instrument':11s}" + "".join(f"{k:>20s}" for k in keys))
runs = {}
for path in (("raster", {}), ("spiral", {}), ("lissajous", {"nx": 29, "ny": 30})):
    for name, model in instruments.items():
        r = sp.run(path, model, keys, budget, sample)          # <- only `model` differs
        runs[path[0], name] = r
        print(f"{path[0]:16s}{name:11s}" + "".join(f"{r.scores[k]:20.3f}" for k in keys))

# (c) the same grid search, the other instrument -------------------------------------
cut = np.arange(5, 122, 2)
best = {}
curves = {}
for name, model in instruments.items():
    points = [{"nx": int(k), "ny": int(k) + 1} for k in cut]      # coupled: ny = nx + 1
    rows, best[name] = sp.grid_search("lissajous", points, "image_nrmse",
                                      instrument=model, budget=budget, sample=sample)
    curves[name] = [r["image_nrmse"] for r in rows]
    print(f"best Lissajous (ny = nx + 1) on {name:8s}: nx = {best[name]['nx']}, "
          f"image_nrmse = {best[name]['image_nrmse']:.3f}")

# --- figure --------------------------------------------------------------------------
fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
r_lp, r_res = runs["raster", "lowpass"], runs["raster", "resonant"]
window = slice(5 * 417 - 40, 5 * 417 + 120)          # one raster line turnaround
t_ms = r_lp.commanded.t[window] * 1e3
ax[0].plot(t_ms, r_lp.commanded.xy[window, 0], color=BLUE, label="commanded")
ax[0].plot(t_ms, r_lp.realized.xy[window, 0], color=ORANGE, label="lowpass 1.5 kHz")
ax[0].plot(t_ms, r_res.realized.xy[window, 0], color=AQUA,
           label="resonant 1.5 kHz, damping 0.15")
ax[0].set_xlabel("time (ms)"); ax[0].set_ylabel("x (nm)")
ax[0].set_title("(a) Same raster command, two instrument models")
ax[0].legend(fontsize=8, loc="upper right")

order = [(p, m) for p in ("raster", "spiral", "lissajous") for m in ("lowpass", "resonant")]
labels = [f"{p}\n{m}" for p, m in order]
xpos = np.arange(len(order))
rms = [runs[o].scores["rms_error_nm"] for o in order]
aff = [runs[o].scores["affine_residual_nm"] for o in order]
ax[1].bar(xpos - 0.2, rms, 0.38, color=BLUE, edgecolor="#fcfcfb", linewidth=2, label="rms_error_nm (built in)")
ax[1].bar(xpos + 0.2, aff, 0.38, color=ORANGE, edgecolor="#fcfcfb", linewidth=2, label="affine_residual_nm (added here)")
ax[1].set_xticks(xpos, labels, fontsize=8)
ax[1].set_ylabel("nm")
ax[1].set_title("(b) New metric plugged in from outside")
ax[1].legend(fontsize=8, loc="upper right")
ax[1].grid(axis="x", visible=False)

for name, color in (("lowpass", ORANGE), ("resonant", AQUA)):
    ax[2].plot(cut, curves[name], color=color, label=name)
    ax[2].plot(best[name]["nx"], best[name]["image_nrmse"], "o", ms=8, color=color,
               mec="white", mew=2)
ax[2].set_xlabel("nx  (with ny = nx + 1)"); ax[2].set_ylabel("image NRMSE")
ax[2].set_ylim(0, 1)
ax[2].set_title("(c) Same search, other instrument: optimum moves")
ax[2].legend(fontsize=9, loc="upper right")
fig.tight_layout()
save(fig, "03_swap_modules.png")
