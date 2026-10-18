"""Example 4 -- the Scan_Path_Loop_Prototype notebook's experiment on the shared core.

    python examples/04_loop_prototype.py        (about 10 seconds)

Same budget as the notebook (20 µm field, 64 s, 12,800 samples, 1,280 µm of path for
every family with a knob), the same four-stage instrument (lag -> coarse-stage moves ->
drift -> noise), the same eight-feature sample, his quality functions and his paired-seed
decision rule. Coarse-stage move counts match the notebook exactly; continuous metrics
differ only through the random draws. See docs/integrating_loop_prototype.md.
"""
import math

import numpy as np

from _style import BLUE, GRID, INK2, ORANGE, plt, save

import scanpath as sp

budget = sp.Budget(fov_nm=20_000, duration_s=64.0, rate_hz=200.0, length_um=1280.0)
sample = sp.Sample.point_features()      # his eight-feature test image
instrument = ("chain", {"stages": [
    ("lowpass", {"bandwidth_hz": 1 / (2 * math.pi * 5e-3)}),   # his lag, tau = 5 ms
    ("stage_moves", {}),                                        # 5 µm piezo window, 50 nm landing
    ("drift", {}),                                              # 2, 1 nm/s
    ("noise", {}),                                              # 2 nm
]})

# --- stages 1-3: every family through the same pipeline -----------------------------
families = [("raster", {}), ("serpentine", {}), ("spiral", {}),
            ("lissajous", {"nx": 21, "ny": 22, "phase_deg": 90}), ("radial", {}),
            ("hilbert", {}), ("sparse", {}), ("hybrid", {})]
keys = ["path_length_um", "rms_error_nm", "local_misalignment_nm", "noise_sensitivity_nm",
        "placement_error_p99_nm", "feature_position_error_nm", "stage_moves",
        "switch_confidence"]
print(f"{'path':12s}" + "".join(f"{k[:15]:>16s}" for k in keys))
results = {}
for family in families:
    r = sp.run(family, instrument, keys, budget, sample, seed=0)
    results[family[0]] = r.scores
    print(f"{family[0]:12s}" + "".join(f"{r.scores[k]:16.4g}" for k in keys))

# his rule for which family goes forward: highest switching confidence, ties -> lower RMS
chosen = max(results, key=lambda n: (round(results[n]["switch_confidence"], 6),
                                     -results[n]["rms_error_nm"]))
print(f"\nchosen by switching confidence: {chosen}")

# --- stage 4: grid search with paired seeds; accept only beyond two standard errors -------
grid = list(np.round(np.linspace(0.2, 0.6, 9), 2))
rows, best = sp.grid_search("hybrid", {"roi_frac": grid}, "feature_position_error_nm",
                            instrument=instrument, budget=budget, sample=sample,
                            seeds=range(8), baseline={"roi_frac": 0.4})
print("\nhybrid, roi_frac grid, 8 paired seeds against roi_frac = 0.4:")
for r in rows:
    flag = "  significant" if r["significant"] else ""
    print(f"  roi_frac {r['roi_frac']:.2f}: {r['feature_position_error_nm']:6.1f} nm"
          f"   change {r['change']:+6.1f} ± {r['change_sem']:4.1f}{flag}")
print(f"best mean: roi_frac {best['roi_frac']}; accepted: {best['accepted']}")

if best["accepted"]:      # confirm on fresh seeds before believing a win, as he does
    fresh, _ = sp.grid_search("hybrid", [{"roi_frac": 0.4}, {"roi_frac": best["roi_frac"]}],
                              "feature_position_error_nm", instrument=instrument,
                              budget=budget, sample=sample, seeds=range(100, 116))
    print(f"on 16 fresh seeds: change {fresh[1]['change']:+.1f} ± {fresh[1]['change_sem']:.1f} nm")

# --- figure ---------------------------------------------------------------------------
fig, (a, b) = plt.subplots(1, 2, figsize=(13, 4.2), gridspec_kw={"width_ratios": [1.3, 1]})
names = list(results)
x = np.arange(len(names))
a.bar(x - 0.2, [results[n]["rms_error_nm"] for n in names], 0.38, color=BLUE,
      label="RMS trajectory error", edgecolor="#fcfcfb", linewidth=2)
a.bar(x + 0.2, [results[n]["feature_position_error_nm"] for n in names], 0.38, color=ORANGE,
      label="feature-position error", edgecolor="#fcfcfb", linewidth=2)
a.set_xticks(x, [f"{n}\n{results[n]['stage_moves']:.0f} moves" for n in names], fontsize=8.5)
a.set_ylabel("nm"); a.grid(axis="x", visible=False)
a.set_title("Same budget, same four-stage instrument")
a.legend(fontsize=8.5, loc="upper right")

b.axhline(0, color=GRID, lw=1)
b.errorbar([r["roi_frac"] for r in rows], [r["change"] for r in rows],
           yerr=[2 * r["change_sem"] for r in rows], color=INK2, marker="o", ms=5, capsize=3, lw=1.2)
b.plot(best["roi_frac"], best["change"], "o", ms=11, mfc="none", mec=ORANGE, mew=2)
b.set_xlabel("hybrid roi_frac"); b.set_ylabel("change in feature-position error (nm)")
b.set_title(f"Paired seeds vs roi_frac 0.4 (bars ±2 s.e.); "
            f"{'accepted' if best['accepted'] else 'not accepted'}")
fig.tight_layout()
save(fig, "04_loop_prototype.png")
