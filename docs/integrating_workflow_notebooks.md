# Integrating `Scan_Path_Workflow.ipynb` and `Optimization_Benchmark.ipynb`

The two notebooks are a careful, self-contained run of the whole loop: equal time, equal
measurements and equal dose for eight path families, an exact lag-plus-drift scanner, a
combined loss with hard motion limits, a feasibility-first choice of the best candidate,
and an equal-budget comparison of three search methods. This note records what was taken
into the package, what was already there, what was left out and why, and what the shared
code now finds. Both notebooks are kept unchanged in `notebooks/`.

The experiment itself runs on the package as `examples/06_constrained_benchmark.py`.

## What maps to what

| In the notebooks | In `scanpath` | Status |
|---|---|---|
| `Trajectory(xy, t, acquire, exposure)`, validated, read-only arrays | `Trajectory(t, xy, info, acquire, dose)`: validated on creation, frozen, read-only arrays; `.events` gives the measured positions | **core change** (optional fields; old code unchanged) |
| `exposure_budget(generator, count, dose)` wrapper | `Budget(acquisitions=..., dose=...)`: every path gets the same number of events, spread evenly over the samples where it may measure, sharing the same total dose | **core change** |
| `ScanBudget(duration, samples)` with validation | `Budget` already had time and samples; it now validates its inputs too | merged |
| generator classes `StraightStrokes`, `ClosedLoops`, `Flower` | `strokes`, `closed_loops`, `rose` in `paths.py` | **new**; same formulas (agree to 1e-14) |
| `RadialRays` (radius, orientation) | `radial` gained `radius` and `angle_deg` | merged |
| `Raster` (serpentine, rotated to fit) | already `serpentine(lines, angle_deg)`; same path | duplicate, not added |
| `PulseLattice` (dwell at lattice sites, transfers blanked) | `pulse_lattice` in `paths.py`; shares a `_dwell` helper with `sparse` | **new**; travel time ∝ hop length (his: equal blocks per site) |
| `Spiral` (constant angular rate, radius, orientation) | our `spiral` is constant area rate; gained `radius` | kept ours; different spiral |
| `Lissajous(cycles, ratio, phase)` | `lissajous(nx, ny, phase_deg)`; nx = cycles, ny = cycles × ratio | duplicate, not added |
| `Flower` = Kalinin's rose curve | `rose`; tested equal to `k_rose` when his repo is present | native, so it works without his repo |
| `LagInstrument(tau, drift)` exact for piecewise-linear commands | `lowpass` now uses the same exact response (`hold="linear"`), or the exact response to a DAC staircase (`hold="step"`); drift stays a separate model, combined with `chain` | **fix** (see below) |
| `IdealInstrument` | `ideal` | duplicate |
| `distortion` → rms_distortion | `rms_error_nm` | duplicate |
| `Coverage(query_bins, radius)` | `coverage` (radius = one pixel); his 30 nm / 48-bin version is registered from outside the package in example 06 | example of a plug-in |
| `motion`: peak speed and acceleration, commanded and realized | `speed_peak_um_s`, `accel_peak_um_s2`, `realized_speed_peak_um_s`, `realized_accel_peak_um_s2`; plus `acquisitions`, `total_dose` | **new metrics** |
| `evaluate(params, generator, instrument, metrics, objective, constraints)` | every search takes `score` = a metric name **or** a function `scores -> loss`, and `constraints` | **core change** |
| `comparison_rules`: equality checks on time, events, dose; motion limits | equalities hold by construction (`Budget`); motion limits via `sp.limits(at_most=..., at_least=...)` | merged |
| `best_trial`: feasible first, then loss | the `best` of every search; if nothing is feasible, the least-violating row with `feasible: False` | **core change** |
| `grid_search(grid, **components)`, lists vs tuples | one search-space format for every search: lists are choices, `(low, high)` ranges, plus `"int"` / `"log"` | merged |
| benchmark: grid refinement, random, differential evolution | `pattern_search`, `random_search`, `de_search`, alongside `grid_search` and `gp_search`; all in `sp.SEARCHES` with the same arguments | **new** |
| equal budget per method, duplicates counted | `n_evals` counts pipeline runs; a repeated point is not re-run, so each run is a different point; unspent budget goes to random points marked `fallback` | merged (slightly different accounting) |
| Pareto front over three objectives | `sp.pareto_front(rows, objectives)` | **new** |
| `history` list with `family`, `role`, `method` | every row carries `family`, `step`, `phase`, `loss`, `feasible`, `violation(s)`, `best_so_far` | merged |
| contract checks (exposure fairness, blanking, analytic lag, infeasible winner) | `tests/test_contract.py` (now 29 tests) | ported |
| loading definitions by `exec` of tagged cells from the other notebook | `import scanpath` | replaced (see below) |
| winner validation at finer instrument and coverage resolution (`substeps`, `query_bins=192`) | not ported: the lag response is exact at any sampling; coverage resolution is a property of the metric | left out |

## Check: his experiment on the shared package

Same configuration (1 field unit = 1 µm, 10 s, 1,000 samples, 200 events, dose 1,
τ = 80 ms, drift 1 nm/s, his loss and limits):

* For the families that are the same curve (strokes, serpentine/raster, closed loops,
  radial, rose/flower), RMS distortion agrees to four decimals, coverage to about 1 % and length to
  about 0.3 %. The small differences come from the time axis: his includes the end time, ours
  stops one sample before it.
* On the three-level grid the best losses match his: strokes 1.4697 (his 1.4696), rose
  1.4482 (1.4482), closed loops 1.7717 (1.7712), serpentine 1.6953 (1.6962), radial
  1.7045 (1.7030); feasible counts match for four of the five.
* The equal-budget comparison reaches his conclusion: at 48 runs every method lands within
  about 6 % of the grid's answer. Pattern search (his grid refinement) and GP are the most
  consistent; random and DE trail on the families with several discrete choices.

## Things to raise with him

1. **The loss rewards paths that don't cover the field.** With distortion in units of
   50 nm and coverage counting only 1 per field, the lowest loss belongs to a single flower
   (coverage 0.17) or a line scanned back and forth (strokes). Requiring coverage ≥ 0.4
   as one more constraint (`at_least={"coverage_30nm": 0.4}`) makes the spiral the winner
   and rules out five of the eight families. We should agree on the weights or the floor
   before reading anything into the ranking.
2. **Lag discretisation.** His `LagInstrument` is exact. Our old `lowpass` (and the loop
   prototype's lag) used a filter that lagged half a sample *less* than the true scanner.
   With 50 µs samples and τ = 106 µs that was a 25 % under-estimate. `lowpass` is now
   exact; see "What changed in our numbers".
3. **Pulse-lattice timing.** His sites get equal time blocks; ours spend travel time in
   proportion to hop length, so the long row-change hops are not rushed. Which matches
   the instrument?
4. **Spiral.** His spiral turns at a constant angular rate (dense centre, sparse edge);
   ours sweeps area at a constant rate (even density). Both are reasonable; if we want
   both, add his as a parameter of `spiral` rather than a second family.
5. **Duplicates in the benchmark.** He counts a repeated candidate as a call; we run each
   point once and spend the budget on new points. Both are fair, but on families with
   few discrete choices the two accountings can rank the methods slightly differently.

## What changed in our numbers

Making `lowpass` exact changes every result that uses it (other models are unchanged):

* raster on the 1.5 kHz scanner: image NRMSE 0.199 → 0.224 at the start point; the
  optimum moves from about 62 lines to about 55, with flyback still about 0.5 (0.136 →
  0.162);
* best Lissajous (ny = nx + 1) on 1.5 kHz: nx 29 → 23;
* loop-prototype experiment (example 04): RMS errors rise 25–40 %. On the hybrid ROI grid,
  0.20 and 0.30 are still about equal (−11 to −13 nm). The best mean is now 0.30 instead of
  0.20; it passes the two-standard-error rule, and it holds on 16 fresh seeds (−10.8 ± 2.3 nm).

## Why `import` instead of `exec`

The benchmark notebook runs the workflow notebook's tagged cells with `exec`, which keeps
the two notebooks in step without copying code. That is better than copying. A package
does the same job more safely: a name clash or an edited cell can't silently change the
other notebook, the code is tested, and a third notebook or a script can use it too. With
the definitions in `scanpath`, each notebook is one `import` plus its experiment.
