"""Example 5 -- the full pipeline end to end, with every choice in one CONFIG block.

    python examples/05_full_pipeline.py        (about 5 seconds)

Stage 1 path -> stage 2 instrument model -> stage 3 metrics -> stage 4 optimisation,
then the optimised path is run again and compared with the starting one.

To swap a component, change one line of CONFIG:
  * path:       any family in paths.py, with its parameters (see sp.parameters(name))
  * instrument: a path-to-path model (instruments.py, case A) or an image-to-image
                model (case B); the pipeline picks the right route
  * metrics:    any names in metrics.py; image metrics work with both kinds of model,
                path metrics only with path-to-path models
  * optimiser:  any of sp.SEARCHES -- "grid", "random", "pattern", "de", "gp" -- all
                take the same search space
  * constraints (optional): limits a candidate must respect, e.g. motion limits
"""
from _style import BLUE, ORANGE, plt, save

import scanpath as sp

CONFIG = dict(
    budget=dict(fov_nm=100, duration_s=1.0, rate_hz=20_000),
    sample="synthetic",                                    # or "point_features"

    # stage 1: path family and its explicit parameters
    path=("raster", {"lines": 48, "flyback": 0.1}),

    # stage 2: a path-to-path model ...
    instrument=("lowpass", {"bandwidth_hz": 1500}),
    # ... or an image-to-image model, e.g.
    # instrument=("drift_warp", {"vx_nm_s": 5.0, "vy_nm_s": 2.0}),

    # stage 3: what to report, and the one score the optimiser minimises / maximises
    metrics=["image_nrmse", "image_correlation", "rms_error_nm", "path_length_um"],
    score="image_nrmse",

    # stage 4: the parameters to search, the method and its budget ...
    space={"lines": (16, 128, "int"), "flyback": (0.05, 0.8)},
    optimiser=("gp", {"n_evals": 25}),
    # ... or any other method on the same space, e.g.
    # optimiser=("grid", {"levels": 8}),       # 8 x 8 points
    # optimiser=("de", {"n_evals": 25}),
    constraints=None,     # e.g. sp.limits(at_most={"realized_speed_peak_um_s": 15})
)


def tidy(params):
    """Parameters rounded for display."""
    return {k: round(v, 3) if isinstance(v, float) else v for k, v in params.items()}


def run_pipeline(config):
    budget = sp.Budget(**config["budget"])
    sample = getattr(sp.Sample, config["sample"])(budget.fov_nm)
    family, start = config["path"]
    model = config["instrument"][0]

    # image-to-image models produce no realized path, so only image metrics apply
    metrics = [m for m in config["metrics"]
               if model in sp.INSTRUMENTS or sp.METRICS[m].kind == "image"]
    skipped = sorted(set(config["metrics"]) - set(metrics))
    kind = "path-to-path" if model in sp.INSTRUMENTS else "image-to-image"
    print(f"stage 1  path        {family} {start}")
    print(f"stage 2  instrument  {model} ({kind}) {config['instrument'][1]}")
    print(f"stage 3  metrics     {metrics}" + (f"  (skipped, need a path-to-path model: {skipped})" if skipped else ""))

    before = sp.run(config["path"], config["instrument"], metrics, budget, sample)
    print("         scores      " + ", ".join(f"{k} {v:.3f}" for k, v in before.scores.items()))

    method, options = config["optimiser"]
    space = config["space"]
    rows, best = sp.SEARCHES[method](family, space, config["score"], config["instrument"],
                                     fixed={k: v for k, v in start.items() if k not in space},
                                     constraints=config["constraints"], budget=budget,
                                     sample=sample, **options)
    found = {k: best[k] for k in space}
    print(f"stage 4  {method:11s} {len(rows)} runs -> best {tidy(found)}, {config['score']} {best[config['score']]:.3f}")

    after = sp.run((family, {**start, **found}), config["instrument"], metrics, budget, sample)
    print("         scores      " + ", ".join(f"{k} {v:.3f}" for k, v in after.scores.items()))
    return budget, sample, before, after, rows, found


budget, sample, before, after, rows, found = run_pipeline(CONFIG)

# --- figure: start vs optimised -------------------------------------------------------------
extent = [-budget.fov_nm / 2, budget.fov_nm / 2] * 2
score = CONFIG["score"]
fig, ax = plt.subplots(1, 4, figsize=(15, 3.8))
for a, result, title in ((ax[0], before, "start"), (ax[2], after, "optimised")):
    a.plot(*result.commanded.xy.T, lw=0.4, color=BLUE if title == "start" else ORANGE)
    a.set_title(f"{title} path: {tidy(result.commanded.info['params'])}", fontsize=9)
    a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
for a, result, title in ((ax[1], before, "start"), (ax[3], after, "optimised")):
    a.imshow(result.image, extent=extent, origin="lower", cmap="gray")
    a.set_title(f"{title}: {score} {result.scores[score]:.3f}")
    a.set_xticks([]); a.set_yticks([]); a.grid(False)
fig.suptitle(f"Full pipeline: {CONFIG['path'][0]} on {CONFIG['instrument'][0]}, "
             f"optimised by {CONFIG['optimiser'][0]}", x=0.01, ha="left", fontsize=10)
fig.tight_layout()
save(fig, "05_full_pipeline.png")
