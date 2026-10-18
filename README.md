# scanpath — minimal scan-path pipeline

**Path generator → instrument model → evaluator → optimiser**, each replaceable by a
one-function plug-in or a one-line change. Needs only numpy, scipy and matplotlib.

**Start here:** `notebooks/full_pipeline.ipynb` (runs end to end in about 20 s) or
`examples/05_full_pipeline.py` (every choice in one `CONFIG` block, about 5 s).

| Stage | File | What you choose |
|---|---|---|
| 1 path generation | `paths.py` | family + its explicit parameters (`sp.parameters("raster")` → `{'lines': 48, 'flyback': 0.1}`) |
| 2 instrument model | `instruments.py` | **case A, path → path** (`lowpass`, `resonant`, `drift`, `chain`, …) or **case B, image → image** (`drift_warp`, `line_jitter`, `probe_blur`, `detector_noise`, `affine`, `from_path`) |
| 3 evaluation | `metrics.py` | any metric by name; image metrics work with both cases, path metrics need case A |
| 4 optimisation | `optimisation.py` | `grid_search`, `random_search`, `pattern_search`, `de_search` or `gp_search`: same arguments, same output; a metric or your own loss as the score; optional constraints (`sp.limits`) |

What's in it:

* **our core**, one file per stage (`paths.py`, `instruments.py`, `metrics.py`,
  `optimisation.py`) plus the shared `core.py`: raster,
  spiral and Lissajous; ideal, low-pass, resonant and `chain` instruments; trajectory,
  coverage and image metrics; seven image-to-image models; grid search with optional
  paired seeds, and GP-based search;
* **the loop prototype** (from the team notebook in `notebooks/`), merged into the same
  stage files: serpentine, radial, Hilbert, sparse and hybrid paths in `paths.py`;
  drift, noise, coarse-stage moves and creep in `instruments.py`; feature-position error,
  noise sensitivity and four more metrics in `metrics.py`. How it was integrated:
  `docs/integrating_loop_prototype.md`;
* **the workflow notebooks** (`Scan_Path_Workflow.ipynb`, `Optimization_Benchmark.ipynb`,
  in `notebooks/`): measurement events and dose in every `Trajectory`, equal-dose budgets,
  strokes, closed loops, rose and pulse-lattice paths, an exact lag model, motion metrics,
  constraints with feasibility-first selection, and three more searches. How it was
  integrated: `docs/integrating_workflow_notebooks.md`;
* **Kalinin's package** through a bridge (`scanpath/kalinin.py`): 37 more path families,
  his SPM scanner model and two metrics, read from a clone of his GitHub repo that is
  used as a reference and never modified.

```python
import scanpath as sp

budget = sp.Budget(fov_nm=100, duration_s=1.0, rate_hz=20_000)
sample = sp.Sample.synthetic(budget.fov_nm)

commanded = sp.make_path("lissajous", budget, nx=23, ny=24)       # 1 generator
realized  = sp.simulate(commanded, "lowpass", bandwidth_hz=1500)   # 2 instrument
scores    = sp.evaluate(commanded, realized, sample)               # 3 evaluator
image     = sp.make_image(sample, commanded, "drift_warp", vx_nm_s=5)  # 2, case B instead

# stages 1-3 in one call; the route (case A or B) follows from the model's name
r = sp.run(("raster", {"lines": 48}), ("lowpass", {"bandwidth_hz": 1500}), ["image_nrmse"], budget, sample)

# stage 4: GP search (25 runs); any of sp.SEARCHES takes the same arguments
rows, best = sp.gp_search("raster", {"lines": (16, 128, "int"), "flyback": (0.05, 0.8)},
                          "image_nrmse", ("lowpass", {"bandwidth_hz": 1500}), n_evals=25,
                          constraints=sp.limits(at_most={"realized_speed_peak_um_s": 15}))
print(sp.describe())      # every registered path, model and metric with defaults
```

## Run it

```bash
git clone https://github.com/1996suryamandalreddy-maker/scanpath-core.git
cd scanpath-core
pip install -e .            # numpy, scipy, matplotlib
pytest -q tests             # contract tests
```

Then, from the repository folder:

```powershell
python examples\01_run_one.py         # one path, scored      (~5 s)
python examples\02_lissajous_grid.py  # grid + plot          (~2 min)
python examples\03_swap_modules.py    # swap / add modules   (~10 s)
python examples\04_loop_prototype.py  # loop prototype       (~10 s)
python examples\05_full_pipeline.py   # full pipeline, GP    (~5 s)
python examples\06_constrained_benchmark.py  # equal dose, limits, 5 searches (~30 s)
python examples\07_add_a_path.py      # add your own path; spiral in/out (~5 s)
python tests\test_contract.py         # contract tests
```

Any Python ≥ 3.10 with numpy, scipy and matplotlib works the same way (`python examples/01_run_one.py`).
Or install it: `pip install -e .`. Figures and the grid CSV go to `figures/`.

`notebooks/full_pipeline.ipynb` is the end-to-end demonstration: path parameters, both kinds of
instrument model, swappable metrics, every search on the same budget, constraints.
`notebooks/scan_path_walkthrough.ipynb` walks through every stage with our and Kalinin's 49 path families
(about two minutes). It expects Kalinin's repo cloned as `ScanPath` next to
`microscopy scan path`; change `REFERENCE` in its first code cell if yours is elsewhere.

## What the examples show

| Example | Shows | Result (100 nm field, 1 s, 20 k samples) |
|---|---|---|
| `01_run_one.py` | the three stages, and three families at the same budget | path lengths 9.1–9.6 µm. On a 1.5 kHz scanner: image NRMSE raster 0.22, Lissajous 0.29, spiral 0.45 (the spiral leaves the corners empty). |
| `02_lissajous_grid.py` | optimizing within a path class on a grid | best Lissajous moves from nx ≈ 35 (3 kHz) to 23 (1.5 kHz) to 17 (0.8 kHz): more cycles cover better but drive the scanner harder. |
| `03_swap_modules.py` | instrument swapped by one string; a metric added from outside the package | resonant vs lowpass moves the optimum from nx = 23 to 47. An affine-corrected error shows the spiral's distortion is ⅔ correctable by calibration; raster's and Lissajous's are not. |
| `04_loop_prototype.py` | the team loop prototype on the shared core: length-matched budget, a four-stage instrument chain, feature-position error, paired-seed search (20 µm field, 64 s) | coarse-stage move counts match the original notebook exactly and the same path (Hilbert) is chosen; on the hybrid ROI grid roi_frac 0.30 beats 0.40 by 2 s.e. and holds on 16 fresh seeds (−10.8 ± 2.3 nm). |
| `05_full_pipeline.py` | the four stages from one `CONFIG` block; swap any line (path, case-A or case-B model, metrics, search, constraints) | raster on a 1.5 kHz scanner: GP finds 55 lines, flyback 0.52 in 25 runs, image NRMSE 0.224 → 0.162 (a 240-point grid finds 56 lines, 0.55). With `drift_warp` instead, path metrics are skipped and the pipeline runs unchanged. |
| `06_constrained_benchmark.py` | the workflow notebooks' experiment: 8 families with equal time, 200 measurements and equal dose; lag + drift; a combined loss; speed and acceleration limits; 4 searches on 48 runs each | best losses match the original notebook (e.g. rose/flower 1.4482, strokes 1.4697). Its loss favours paths that cover little of the field (rose at 17 %); a coverage floor of 0.4 makes the spiral the winner. Pattern and GP search are the most consistent at equal budget. |
| `07_add_a_path.py` | a new path added from a script in four lines; the pixel-list scans of Palos et al. 2026 (spiral in/out, N-sequential, random) on a scanner that steps pixel to pixel | spiral in/out ≈ raster ≪ N = 2 < N = 4 < random in image error, as large probe jumps cost settling (image NRMSE 0.24, 0.29, 0.44, 0.65, 1.05) |

Since the workflow integration, `lowpass` is the exact first-order response; it used to
lag half a sample too little, so numbers involving it are higher than in earlier versions
(details in `docs/integrating_workflow_notebooks.md`).

The interfaces and the open decisions are in **[INTERFACES.md](INTERFACES.md)**. New to the
code? `docs/clean_code.md` explains the habits the package follows, with examples, and
`docs/adding_a_path.md` shows how to add a scan path.
Instrument parameters are illustrative, not calibrated.

## License

MIT — see [LICENSE](LICENSE).
