# Integrating `Scan_Path_Loop_Prototype.ipynb` into the shared package

The loop prototype is a complete, well-structured run of the four-stage loop. Its strongest
contributions are exactly the parts the shared core was missing: a stricter budget, a
composable instrument, task-level quality functions and a statistically honest optimizer.
This note records what was ported, how it maps onto the shared interfaces, and what is
still open. His notebook itself is unchanged (`notebooks/Scan_Path_Loop_Prototype.ipynb`).

> **Update (workflow-notebook integration).** `lowpass` is now the exact first-order
> response. His lag filter, like our old `lowpass`, lagged half a sample less than the true
> scanner. Example 04's RMS and feature errors are therefore 25–40 % higher than the
> figures below. On the ROI grid, 0.20 and 0.30 are still about equal; 0.30 now has the
> best mean, passes the 2 s.e. rule and holds on fresh seeds (−10.8 ± 2.3 nm). See
> `docs/integrating_workflow_notebooks.md`.

## What maps to what

| In the notebook | In `scanpath` | Status |
|---|---|---|
| path dict `{name, family, params, x, y, t}` in metres | `Trajectory(t, xy, info)` in nm | same content, unit change only |
| `@family(free=...)` + `make_path` tuning a density knob to the budget | `@register_path(name, knob=...)` + `Budget(length_um=...)` | **core change**: `make_path` tunes the knob |
| families raster (serpentine), radial, Hilbert, sparse, hybrid | `serpentine`, `radial`, `hilbert`, `sparse`, `hybrid` in `scanpath/paths.py` | merged; spiral and Lissajous already existed |
| instrument = ordered list of `@stage` functions, `info` report | instrument `"chain"` with `stages=[(name, params), ...]`; each stage is a normal instrument | **core change**: `chain`, seeds, `info` kept |
| stages lag, stage_moves, drift, noise, creep | `lowpass` (= his lag with bandwidth 1/2πτ), `stage_moves`, `drift`, `noise`, `creep` in `scanpath/instruments.py` | merged |
| `@quality(name, unit, weight, about)` + `ctx` | `@register_metric` in `scanpath/metrics.py`; docstring = about, unit in the name; weights stay in the objective | merged |
| `ctx["rerun"](seed)` for noise sensitivity | metrics may take a 4th argument `rerun`; `run()` supplies it | **core change** |
| `ctx["sample"]` with feature positions | `Sample(features=..., feature_size_nm=...)`; `Sample.point_features()` builds his image | **core change** (optional fields) |
| paired-seed grid, accept only beyond 2 s.e. | `grid_search(..., seeds=..., baseline=...)` → `change`, `change_sem`, `significant`, `accepted` | **core change** |
| switching confidence, path choice | metrics `stage_moves`, `switch_confidence` in `scanpath/metrics.py`; the choice rule is in example 04 | merged |
| `run_log.yaml` + `paths.npz` | not ported yet | open (see below) |

Every core change is backward compatible: the earlier examples and the walkthrough
notebook give identical numbers.

## Check: his experiment on the shared core

`examples/04_loop_prototype.py` reruns his experiment (20 µm field, 64 s,
12,800 samples, 1,280 µm per path, lag → stage moves → drift → noise, eight features):

* coarse-stage move counts match his exactly for every family (63, 70, 83, 3, 3, 8, 33),
  so switching confidences match too, and the same path (his Hilbert curve) is chosen;
* RMS and feature errors agree within roughly 10–20 %; the random draws differ;
* his decision rule works through `grid_search`: on the hybrid ROI grid the best mean
  (roi_frac 0.20, −12.1 ± 6.2 nm) is correctly **not** accepted.

## Things to raise with him

1. **Rotation leaves the field.** Rotating a square path by 20–45° puts 11–18 % of its
   samples outside the 20 µm field, and rotation is the only free parameter for five of
   his seven families. The port shrinks a rotated path to fit (`serpentine`), which leaves
   the corners empty. Rotating the *field* with the path (as a real scan-rotation does)
   would need the sample to rotate too. Which do we want?
2. **Lissajous retracing.** His free grid (d = 1…6, phase 0…90°) includes non-coprime
   pairs; d = 2 gives 20/22, coverage falls from 0.77 to 0.29. A 90° phase also retraces
   when the x frequency is even. Restrict to coprime pairs and use a 30° phase.
3. **Constant speed is not neutral.** Walking every shape at the same speed matches length
   and time nicely, but a Lissajous walked at constant speed needs 27× the peak acceleration
   of the same curve driven by sinusoids. That removes the very property that makes
   Lissajous attractive. Matching *length and time* (mean speed), and reporting peak speed
   and acceleration as constraints, keeps each family's natural motion; the port does that
   for our own families and walks his shapes at constant speed as he intended.
4. **Drift dominates the objective.** With 2 nm/s over 64 s, drift alone gives 70–80 nm
   of the ~75 nm feature-position error. The optimizer is therefore mostly choosing *when*
   each feature is visited. That's physically real but worth knowing before reading
   anything into the ranking.
5. **Multiple comparisons.** With nine grid points tested at 2 s.e., roughly one run in
   five shows a "significant" point by chance (roi_frac 0.30 above). His confirm-on-fresh-
   seeds step is the right guard; keep it in every study.
6. **Sparse has no length knob.** Its dwell time sets the sample count in his notebook;
   in ours the budget is time, so its travelled length (≈170 µm) can't be matched.

## Still open (team decisions)

* **Run log.** His YAML log + `.npz` arrays is a good provenance record and a clean file
  interface for closing the loop. Proposal: a small `scanpath.log` module that writes the
  same sections (`run`, `stage_1_paths`, `stage_2_instrument`, `stage_3_evaluation`,
  `stage_4_optimization`); decide YAML (needs PyYAML) or JSON (no new dependency).
* **Weighted objective.** His `objective()` (weights in %, each metric relative to a
  reference path) is a clear, explicit way to combine metrics. It can become the `score`
  argument of `grid_search` once we agree on weights.
* **Free-parameter declarations.** His `free=` grids are useful defaults for optimizers;
  they could live next to `knob` in `register_path`.

## Files

Everything is merged into the existing stage files; there is no separate plug-in module.

* `scanpath/core.py`: `Budget.length_um`, `knob`, `seed`, `rerun`, `Sample.features`,
  `Sample.point_features` (his test image)
* `scanpath/optimisation.py`: `grid_search` with his paired-seed rule (`seeds`, `baseline`)
* `scanpath/paths.py`: his families (serpentine, radial, hilbert, sparse, hybrid);
  knobs for `raster` (lines) and `spiral` (turns)
* `scanpath/instruments.py`: his stages (drift, noise, stage_moves, creep) and `chain`
* `scanpath/metrics.py`: his quality functions (local misalignment, placement error,
  noise sensitivity, feature-position error, stage moves, switching confidence)
* `examples/04_loop_prototype.py`: his experiment on the shared core (+ figure)
* `tests/test_contract.py`: contract tests, now covering stages, chain, seeds, length
  matching and paired-seed search (11 tests)
