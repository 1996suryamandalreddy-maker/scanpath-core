"""Example 6 -- the workflow notebooks' experiment, on the shared package.

    python examples/06_constrained_benchmark.py        (about a minute)

Scan_Path_Workflow.ipynb and Optimization_Benchmark.ipynb (notebooks/) asked: with the
same scan time, the same 200 measurements and the same total dose for every path, a
lagging and drifting scanner, a combined loss and hard motion limits, which path family
and which parameters are best -- and does the choice of search method matter?

Here the same question runs through scanpath. His field-of-view unit is 1 µm, so speeds
and lengths read the same in both. See docs/integrating_workflow_notebooks.md.
"""
import math

import numpy as np
from scipy.spatial import cKDTree

from _style import AQUA, BLUE, GRID, INK2, ORANGE, VIOLET, plt, save

import scanpath as sp

# --- the experiment: one place to change it --------------------------------------------
BUDGET = sp.Budget(fov_nm=1000, duration_s=10.0, rate_hz=100.0,   # 1000 samples
                   acquisitions=200, dose=1.0)                      # equal events and dose
SAMPLE = sp.Sample.synthetic(BUDGET.fov_nm)
INSTRUMENT = ("chain", {"stages": [
    ("lowpass", {"bandwidth_hz": 1 / (2 * math.pi * 0.08)}),      # lag, tau = 80 ms
    ("drift", {"vx_nm_s": 1.0, "vy_nm_s": -1.0}),
]})
MOTION_LIMITS = sp.limits(at_most={
    "speed_peak_um_s": 4.0, "realized_speed_peak_um_s": 4.0,
    "accel_peak_um_s2": 200.0, "realized_accel_peak_um_s2": 100.0,
})
METRICS = ["rms_error_nm", "coverage_30nm", "path_length_um"]
SEARCH_BUDGET = 48                      # pipeline runs per search
SEEDS = [0, 1, 2]                       # for the searches that are random

# Lists are choices, (low, high) are ranges, as in his notebook. radius is a fraction
# of the half-field here (his 0.35-0.5 field units = our 0.7-1.0).
SPACES = {
    "strokes":       {"strokes": [4, 8, 12], "angle_deg": (0, 90), "radius": (0.7, 1.0)},
    "serpentine":    {"lines": [8, 12, 16], "angle_deg": (0, 90)},
    "pulse_lattice": {"rows": [4, 6, 8], "columns": [4, 6, 8],
                      "dwell_fraction": (0.35, 0.75), "radius": (0.7, 1.0)},
    "spiral":        {"turns": [5, 8, 11], "radius": (0.7, 1.0)},
    "closed_loops":  {"loops": [3, 6, 9], "axis_ratio": (0.5, 1.0), "radius": (0.7, 1.0)},
    "radial":        {"spokes": [8, 12, 16], "angle_deg": (0, 90), "radius": (0.7, 1.0)},
    "rose":          {"order": [2, 3, 4], "cycles": [1, 2, 3], "angle_deg": (0, 90),
                      "radius": (0.7, 1.0)},
    "lissajous":     {"nx": [3, 5, 7], "ny": (4, 11), "phase_deg": (0, 90)},
}


# --- his coverage definition, added from outside the package ---------------------------
@sp.register_metric("coverage_30nm", higher_is_better=True)
def coverage_30nm(commanded, realized, sample):
    """Fraction of a 48 x 48 grid of query points within 30 nm of a measurement."""
    axis = ((np.arange(48) + 0.5) / 48 - 0.5) * sample.fov_nm
    grid = np.stack(np.meshgrid(axis, axis), axis=-1).reshape(-1, 2)
    return float(np.mean(cKDTree(realized.events).query(grid)[0] <= 30.0))


def quality(scores):
    """His loss: distortion in units of 50 nm, plus the uncovered fraction, plus a soft
    penalty for straying from 20 µm of path."""
    return (scores["rms_error_nm"] / 50.0 + 1.0 - scores["coverage_30nm"]
            + 0.1 * (scores["path_length_um"] / 20.0 - 1.0) ** 2)


COMMON = dict(instrument=INSTRUMENT, metrics=METRICS, constraints=MOTION_LIMITS,
              budget=BUDGET, sample=SAMPLE)

# --- part 1: a three-level grid for every family (his main notebook) -------------------
print(f"{'family':14s} {'runs':>5s} {'feasible':>9s} {'best loss':>10s}   best parameters")
grid_best, grid_runs = {}, {}
for family, space in SPACES.items():
    rows, best = sp.grid_search(family, space, quality, levels=3, **COMMON)
    grid_best[family], grid_runs[family] = best, len(rows)
    params = {k: round(best[k], 2) if isinstance(best[k], float) else best[k] for k in space}
    print(f"{family:14s} {len(rows):5d} {sum(r['feasible'] for r in rows):9d} "
          f"{best['loss']:10.4f}   {params}")
winner = min(grid_best, key=lambda f: grid_best[f]["loss"])
print(f"lowest loss: {winner} (coverage {grid_best[winner]['coverage_30nm']:.2f})")

# The loss lets a path that covers little of the field win. A coverage floor, added as
# one more constraint, says what we actually require:
floor = sp.limits(at_most={"speed_peak_um_s": 4.0, "realized_speed_peak_um_s": 4.0,
                           "accel_peak_um_s2": 200.0, "realized_accel_peak_um_s2": 100.0},
                  at_least={"coverage_30nm": 0.4})
with_floor = {f: sp.grid_search(f, s, quality, levels=3, **{**COMMON, "constraints": floor})[1]
              for f, s in SPACES.items()}
feasible = {f: b for f, b in with_floor.items() if b["feasible"]}
best_floor = min(feasible, key=lambda f: feasible[f]["loss"])
print(f"with coverage >= 0.4 required: {best_floor} (loss {feasible[best_floor]['loss']:.3f}, "
      f"coverage {feasible[best_floor]['coverage_30nm']:.2f}); "
      f"{len(SPACES) - len(feasible)} families cannot meet it")

# --- part 2: every search method on the same budget (his benchmark notebook) -----------
METHODS = ["random", "pattern", "de", "gp"]
results = {m: {} for m in METHODS}       # method -> family -> list of (rows, best)
for family, space in SPACES.items():
    for method in METHODS:
        seeds = [0] if method == "pattern" else SEEDS      # pattern search is deterministic
        results[method][family] = [
            sp.SEARCHES[method](family, space, quality, n_evals=SEARCH_BUDGET, seed=s, **COMMON)
            for s in seeds]

print(f"\nbest feasible loss after {SEARCH_BUDGET} runs (median over seeds)")
print(f"{'family':14s}" + "".join(f"{m:>10s}" for m in METHODS) + f"{'grid':>10s}")
for family in SPACES:
    medians = [np.median([b["loss"] if b["feasible"] else np.nan for _, b in results[m][family]])
               for m in METHODS]
    print(f"{family:14s}" + "".join(f"{v:10.4f}" for v in medians)
          + f"{grid_best[family]['loss']:10.4f}")

# --- figure -------------------------------------------------------------------------------
colours = dict(zip(METHODS, (ORANGE, BLUE, AQUA, VIOLET)))
fig, (a, b, c) = plt.subplots(1, 3, figsize=(17, 4.6),
                              gridspec_kw={"width_ratios": [1, 1.4, 1]})

family = "rose"
for method in METHODS:
    for i, (rows, _) in enumerate(results[method][family]):
        a.plot([r["step"] + 1 for r in rows], [r["best_so_far"] for r in rows],
               color=colours[method], alpha=1 if i == 0 else 0.35, lw=1.6,
               drawstyle="steps-post", label=method if i == 0 else None)
a.axhline(grid_best[family]["loss"], color=INK2, ls="--", lw=1,
          label=f"3-level grid ({grid_runs[family]} runs)")
a.set_xlabel("pipeline runs"); a.set_ylabel("best feasible loss so far")
a.set_title(f"Convergence, {family} (3 seeds faint)"); a.legend(fontsize=8)

x = np.arange(len(SPACES))
for k, method in enumerate(METHODS):
    losses = [[bst["loss"] for _, bst in results[method][f] if bst["feasible"]] for f in SPACES]
    med = [np.median(v) if v else np.nan for v in losses]
    lo = [min(v) if v else np.nan for v in losses]
    hi = [max(v) if v else np.nan for v in losses]
    pos = x + (k - 1.5) * 0.2
    b.bar(pos, med, 0.19, color=colours[method], label=method, edgecolor="#fcfcfb", lw=0.5)
    b.vlines(pos, lo, hi, color=INK2, lw=1)
b.set_xticks(x, list(SPACES), rotation=20, fontsize=8.5)
b.set_ylim(1.2, None); b.set_ylabel("best feasible loss (median, seed range)")
b.set_title(f"Equal budget ({SEARCH_BUDGET} runs) per method and family")
b.legend(fontsize=8, ncol=4, loc="upper left"); b.grid(axis="x", visible=False)

all_rows = [r for m in METHODS for f in SPACES for rows, _ in results[m][f] for r in rows]
feasible_rows = [r for r in all_rows if r["feasible"]]
front = sp.pareto_front(feasible_rows, ["rms_error_nm", "coverage_30nm"])
c.scatter([r["rms_error_nm"] for r in feasible_rows], [r["coverage_30nm"] for r in feasible_rows],
          s=6, color=GRID, label="feasible runs")
front = sorted(front, key=lambda r: r["rms_error_nm"])
c.plot([r["rms_error_nm"] for r in front], [r["coverage_30nm"] for r in front], "o-",
       ms=4, color=ORANGE, lw=1.2, label="Pareto front")
c.set_xlabel("RMS distortion (nm)"); c.set_ylabel("coverage within 30 nm")
c.set_title("Distortion vs coverage: no weights needed"); c.legend(fontsize=8)
fig.tight_layout()
save(fig, "06_constrained_benchmark.png")
print(f"Pareto front: {len(front)} of {len(feasible_rows)} feasible runs; families on it: "
      f"{sorted({r['family'] for r in front})}")
