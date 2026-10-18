# scanpath — interface agreement

**Status: proposal for team sign-off.** Once agreed, changing anything in sections 1–3 needs
the whole team; everything else is owned by whoever writes the plug-in. Items marked
**(new)** came in with the workflow-notebook integration
(`docs/integrating_workflow_notebooks.md`): measurement events and dose, constraints,
three more searches.

## 0. The workflow

```
 family + params ─► [1 path generator] ─► Trajectory (commanded)
                                                 │
                    ┌────────────────────────────┴────────────────────────────┐
         case A: path → path                                       case B: image → image
   [2 instrument model] ─► Trajectory (realized)        Sample + commanded ─► [2 image model] ─► image
                    │                                                          │
                    └──────────────► [3 evaluator: metrics] ◄──────────────────┘
                                              │ scores {name: float}
                       best params ◄── [4 optimiser: grid_search | gp_search]
```

Stage 2 has **two kinds of model**, kept in separate registries so a model is always one or
the other:

- **Case A, path-to-path** (`register_instrument`): commanded path → realized path. Never sees
  the sample. The image a user would see is formed afterwards by `image_as_acquired` (signal
  read at the *realized* positions, drawn at the *commanded* ones). Every metric applies.
- **Case B, image-to-image** (`register_image_model`): sample + commanded path → acquired
  image. There is no realized path, so only **image metrics** apply. `from_path` wraps any
  case-A model as a case-B one (the two routes give identical images; tested).

`run` picks the route from the model's name; the caller never branches.

## 1. Objects passed between modules

| Object | Fields | Units | Produced by → consumed by |
|---|---|---|---|
| `Budget` | `fov_nm`, `duration_s`, `rate_hz` (→ `n`, `t`); optional `length_um`; optional `acquisitions`, `dose` **(new)** | nm, s, Hz, µm, dose units | user → generator |
| `Trajectory` | `t (N,)`, `xy (N,2)`, `info: dict`; `acquire (N,)` bool and `dose (N,)` **(new)**; `.events` = measured positions | s, nm | generator → instrument → evaluator |
| `Sample` | `image (H,W)`, `fov_nm`; optional `features (K,2)`, `feature_size_nm` | —, nm | user → evaluator |
| image | `(H,W)` array, same shape as `Sample.image` | — | case-B model (or `image_as_acquired`) → image metrics |
| scores | `dict[str, float]` | per metric | evaluator → optimiser |
| `Result` | `commanded`, `realized` (None in case B), `scores`, `image` | — | `run` → user |

Conventions: origin at the centre of the field of view; `xy[:,0]` is x; images are
`[row, col] = [y, x]`; the time axis is uniform. The same `Trajectory` type is used for
commanded and realized paths, so any metric can compare them point by point. Whatever an
instrument reports (a count of stage moves, say) goes in `realized.info` and is kept.

**(new)** Records check their inputs when created (shapes, finite values, increasing
time, dose positive exactly at measurement events) and are frozen: arrays are read-only,
so no stage can change data another stage holds; use `dataclasses.replace` to derive a new
record. The probe moves at every sample but measures only where `acquire` is True (default:
everywhere). Image formation, coverage and feature metrics use the measurement events;
trajectory and motion metrics use every sample.

## 2. Public functions (the whole API)

| Function | Signature | Stage |
|---|---|---|
| `make_path` | `(family, budget, **params) -> Trajectory` | 1 |
| `simulate` | `(commanded, model, seed=None, **params) -> Trajectory` | 2, case A |
| `make_image` | `(sample, commanded, model, seed=None, **params) -> image` | 2, case B |
| `image_as_acquired` | `(commanded, realized, sample, positions="commanded") -> image` | 2A → 3 |
| `evaluate` | `(commanded, realized, sample, metrics=None, rerun=None, image=None) -> dict` | 3 |
| `run` | `(path, instrument, metrics, budget, sample, seed=None) -> Result` | 1→3, either case |
| `grid_search` | `(family, grid, score, instrument, *, levels=5, ..., seeds=None, baseline=None) -> (rows, best)` | 4 |
| `random_search`, `pattern_search`, `de_search` **(new)** | `(family, space, score, instrument, *, n_evals=50, seed=0, ...) -> (rows, best)` | 4 |
| `gp_search` | `(family, space, score, instrument, *, n_evals=28, n_init=8, ...) -> (rows, best)` | 4 |
| `SEARCHES` **(new)** | `{"grid", "random", "pattern", "de", "gp"} -> search` | 4 |
| `limits` **(new)** | `(at_most={metric: value}, at_least={...}) -> constraints` | 4 |
| `pareto_front` **(new)** | `(rows, objectives) -> rows` no other feasible row beats on every objective | 4 |
| `parameters` | `(name) -> {param: default}` for any path, model or metric | — |
| `describe` | `() -> str`: everything registered, with defaults and knobs | — |

`path` / `instrument` arguments are `"name"` or `("name", {params})`. `grid` is a dict of
lists (Cartesian product) or a list of parameter dicts (explicit or coupled points). `space`
maps each parameter to `(low, high)`, `(low, high, "int")`, `(low, high, "log")` or a list
of choices; `grid_search` accepts the same `space` and gives each range `levels` points.
`metrics=None` means every metric that applies.

Every search takes the same keywords (`fixed`, `metrics`, `constraints`, `budget`,
`sample`, `run_seed`) and returns the same `(rows, best)`, so one can replace another.
**(new)** `score` is a metric name or a function `scores -> loss` (lower is better; list the
metrics it needs in `metrics`). `constraints` is a function `scores -> {rule: violation ≥
0}`, usually built by `limits`. Rows carry `family`, `step`, `phase`, `loss`, `feasible`,
`violation`, `violations` and `best_so_far`. The best row is the feasible one with the
lowest loss. If nothing is feasible, it is the row with the smallest violation, and
`best["feasible"]` is False. `n_evals` counts pipeline runs, each at a different point.

## 3. Plug-in contracts

Each kind is one function plus one decorator. **A plug-in's keyword defaults are its
parameters**; nothing else declares them.

| Add a… | Write | Contract |
|---|---|---|
| path family | `@register_path("name", knob=None)`<br>`def f(s, **params)` | `s = t/duration ∈ [0,1)`; return `(N,2)` in `[-1,1]`, or **(new)** `(xy, acquire)` if the probe moves without measuring. Scaling to nm, the time axis and the `Budget`'s events and dose are done for you. `knob` names the density parameter (lines, turns, spokes…) that `Budget(length_um=…)` may tune. `radius` is a fraction of the half-field; angles are `angle_deg`. |
| instrument model, case A | `@register_instrument("name")`<br>`def f(commanded, **params)` | Return a `Trajectory` on the **same time axis**; start settled; never read the sample; move the probe but never decide when it measures (the commanded `acquire` and `dose` are carried over). A random model takes a `seed` parameter and must repeat exactly for the same seed. Any model is also a stage of `"chain"`. |
| image model, case B | `@register_image_model("name")`<br>`def f(sample, commanded, **params)` | Return a finite image of `sample.image.shape`. A random model takes a `seed`. Its name must not clash with a case-A model (enforced). |
| metric | `@register_metric("name", higher_is_better=False, kind="path")`<br>`def f(commanded, realized, sample)` | Return one finite float; units in the name (`_nm`, `_um_s`); one physical idea per metric; anything about what was *measured* uses `.events`. A metric that needs repeat runs adds a 4th argument `rerun`: `rerun(seed)` returns a fresh realized path. |
| image metric | `@register_metric("name", kind="image")`<br>`def f(image, sample)` | Compares an acquired image with `sample.image`. Works with **both** cases of stage 2, so it is the score to use when comparing a path model with an image model. |

A plug-in can live in its own file or notebook; just import it before calling `run`. The
package itself is not edited (`examples/03_swap_modules.py` adds a metric this way).
`tests/test_contract.py` checks **every** registered plug-in automatically, including new ones
(30 tests: records, paths, both kinds of model, metrics, budgets, every search).

Combining models: `simulate(path, "chain", seed=0, stages=[("lowpass", {...}), ("stage_moves", {}), ("drift", {})])`
applies them in order and collects what each reports.

## 4. Comparing like with like

- **Hard (by construction):** same field of view, same scan time, same number of samples.
- **Optional hard constraint:** `Budget(length_um=…)` also matches path length, and so
  mean speed, for every family that declares a knob.
- **Optional hard constraint (new):** `Budget(acquisitions=…, dose=…)` gives every path the
  same number of measurements and the same total dose (local dose can still differ).
- **Reported, or enforced as constraints (new):** peak speed and acceleration, commanded
  and realized (`speed_peak_um_s`, `accel_peak_um_s2`, `realized_speed_peak_um_s`,
  `realized_accel_peak_um_s2`); pass them to a search with `constraints=limits(at_most=…)`.
  Our own families keep their natural velocity profiles (a Lissajous stays sinusoidal); the
  loop-prototype shapes are walked at constant speed, as designed.
- **Random instruments:** compare paths on the **same seeds**, seed by seed
  (`grid_search(..., seeds=range(16))`), accept a change only beyond two standard errors,
  and confirm a winner on fresh seeds.
- Metrics stay **separate**. Any combined objective is written down explicitly with its weights.

## 5. Repository layout

```
scanpath/
  core.py           shared by all stages: records (Budget, Trajectory, Sample, Result),
                    registries, image formation (reconstruct, image_as_acquired),
                    make_path / simulate / make_image / evaluate / run, parameters, describe
  paths.py          1  driven curves: raster, spiral, lissajous, closed_loops, rose
                       walked at constant speed: serpentine, radial, hilbert, hybrid, strokes
                       dwell paths: sparse, pulse_lattice (transfers blanked)
                       pixel-list scans: grid_spiral (in/out), n_sequential, random_order
  instruments.py    2  case A (path → path): ideal, lowpass (exact first-order), resonant,
                         drift, noise, stage_moves, creep, chain
                       case B (image → image): sampled, from_path, drift_warp, line_jitter,
                         probe_blur, detector_noise, affine
  metrics.py        3  path metrics: trajectory distortion · coverage · feature positions ·
                         repeatability · coarse-stage switching · path length · motion
                         (speed, acceleration) · measurement budget (acquisitions, dose)
                       image metrics: image_nrmse, image_correlation
  optimisation.py   4  grid_search (optionally paired seeds, 2 s.e. rule) · random_search ·
                       pattern_search · de_search · gp_search (GP + EI); limits, pareto_front
examples/           01 run one path · 02 Lissajous grid · 03 swap & add modules ·
                    04 loop prototype · 05 full pipeline (one CONFIG block, any search) ·
                    06 constrained, equal-dose comparison and search benchmark ·
                    07 adding a path; pixel-list scans of Palos et al. 2026
notebooks/          full_pipeline.ipynb (end to end, start here) ·
                    originals from teammates: Scan_Path_Loop_Prototype.ipynb,
                    Scan_Path_Workflow.ipynb, Optimization_Benchmark.ipynb
docs/               integrating_loop_prototype.md · integrating_workflow_notebooks.md ·
                    adding_a_path.md
tests/              contract tests
```

## 6. Decisions for the team

| # | Question | Proposal |
|---|---|---|
| 1 | Is `Trajectory(t, xy, info)` enough? | **(new)** `acquire` and `dose` were added once a plug-in needed them (pulse lattice, equal-dose budgets). Dwell is still expressed by repeating positions. Instrument state and reports go in `info`. |
| 2 | Uniform time axis required? | Yes for now (filters assume it). Dwell is expressed by repeating positions (see `sparse`). |
| 3 | Units | nm and s everywhere; unit suffix in every parameter and metric name. |
| 4 | Metric signature | Path metrics `(commanded, realized, sample[, rerun])`; image metrics `(image, sample)`. |
| 5 | Selected score for optimisation | `image_nrmse` for imaging studies (works with both kinds of model); `feature_position_error_nm` for the feature-placement study; report all other metrics alongside. |
| 6 | Optimiser | **(new)** Five searches with one signature, in `SEARCHES`. Grid for maps and small spaces (paired seeds when the instrument is random); pattern or GP when runs are expensive (the most consistent in example 06); random as the baseline any other method must beat. |
| 7 | Ownership | `instruments.py`: Richard (SPM models), Surya (EM models). `metrics.py`: Greg, Aidan. `paths.py`: Richard (path algebra). `optimisation.py` and `core.py`: changes reviewed by all. |
| 8 | Relation to `stem_scan` | `stem_scan` remains the research sandbox. Models move into `scanpath` as plug-ins once they fit the contracts above. |
| 9 | What does rotating a path mean? | Today a rotated `serpentine` shrinks to stay inside the field (corners unscanned). The alternative is rotating the field and the sample together, like a real scan rotation. |
| 10 | Match speed exactly, or only length and time? | Proposal: length and time (mean speed), with peak speed and acceleration reported. Constant-speed walking costs a Lissajous 27× in peak acceleration. |
| 11 | Run log format | Port the loop prototype's run log (one section per stage, arrays in `.npz`) as a small `scanpath.log` module. YAML needs PyYAML; JSON needs nothing new. |
| 12 | Weighted objective | Now possible: `score` can be any function of the scores. The weights still need agreeing (see 14). |
| 13 | Which image-to-image models are worth keeping? | Today they are simple stand-ins (drift warp, line jitter, probe blur, detector noise, affine). Replace with calibrated ones as data arrive; the contract stays. |
| 14 **(new)** | Loss weights vs coverage | With the workflow notebook's weights, a path covering 17 % of the field has the lowest loss. Proposal: keep the loss but add a coverage floor as a constraint (`at_least={"coverage": …}`), or report the Pareto front instead of one winner. |
| 15 **(new)** | Scanner command between samples | `lowpass(hold="linear")` (path drawn as straight segments; default) or `hold="step"` (DAC staircase, half a sample more lag). Which matches our hardware? |
| 16 **(new)** | Pulse-lattice timing | Equal time per site (workflow notebook) or travel time ∝ hop length (ours)? |

Deliberately **out of scope** for now: calibration against the real microscope, scan-time
optimisation, closed-loop control, detailed detector physics.
