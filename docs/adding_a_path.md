# Adding a new scan path

A path family is **one function plus one decorator**. You never edit the generator, the
instrument models, the metrics or the optimisers: once the function is registered, all of
them work with it. `examples/07_add_a_path.py` runs everything on this page.

## 1. What the function must do

```python
import numpy as np
import scanpath as sp

@sp.register_path("spiral_in_out", knob="turns")       # the name you will use everywhere
def spiral_in_out(s, turns=20, radius=1.0):           # keyword defaults = the parameters
    """Archimedean spiral from the edge to the centre, then back out to the edge."""
    r = radius * np.abs(1 - 2 * s)                     # 1 -> 0 -> 1 over the scan
    angle = 2 * np.pi * turns * 2 * s
    return np.column_stack((r * np.cos(angle), r * np.sin(angle)))
```

The contract (the package checks it for you):

| | |
|---|---|
| input `s` | normalised time, `t / duration`, from 0 to just below 1, one value per sample |
| output | an `(N, 2)` array of x, y positions in the unit square **[-1, 1]**: -1 is one edge of the field, +1 the other. Scaling to nm, the time axis and equal budgets are done for you |
| parameters | keyword arguments with defaults. They are what `sp.parameters(name)` lists and what an optimiser can tune. Put units in the name (`angle_deg`) |
| `knob` (optional) | the parameter that sets how densely the path fills the field (lines, turns, pixels…), so `Budget(length_um=…)` can match path lengths |
| probe moves without measuring? | return `(xy, acquire)`, with `acquire` an `(N,)` True/False mask (see `pulse_lattice`) |

Because you write the path as a function of *time*, you also decide how fast it moves. A
smooth function of `s` gives smooth motion; positions that hold still and then change in
one sample give a step (see the pixel-list scans below).

## 2. Check it

```python
sp.parameters("spiral_in_out")          # {'turns': 20, 'radius': 1.0}
path = sp.make_path("spiral_in_out", sp.Budget(), turns=30)
plt.plot(*path.xy.T)                    # look at it before trusting any number
sp.run(("spiral_in_out", {"turns": 30}), "lowpass", ["image_nrmse", "coverage"])
```

Then run the tests: `python tests/test_contract.py`. The contract tests loop over
**every** registered family, so a new path is checked (stays in the field, keeps the time
axis, gets equal measurements and dose, works with every metric) without you writing
those tests.

## 3. Use it like any other path

```python
sp.grid_search("spiral_in_out", {"turns": [10, 20, 30, 40]}, "image_nrmse", "lowpass")
sp.gp_search("spiral_in_out", {"turns": (5, 60, "int"), "radius": (0.7, 1.0)},
             "image_nrmse", "lowpass", n_evals=25)
```

A parameter does not have to be a number: `grid_spiral(direction="in" | "out")` is
searched with a list of choices, `{"direction": ["in", "out"]}`.

## 4. Where to put it

* **Trying an idea:** write it in your notebook or script, as above. Nothing in the
  package changes, and nobody else is affected.
* **Keeping it for the team:** move the function into `scanpath/paths.py`, in the section
  that matches it (driven curves, constant-speed shapes, dwell paths, pixel-list scans),
  using `register_path` from `.core`. Add a test with a known answer to
  `tests/test_contract.py`, e.g. the visiting order from a paper's figure. Update the
  family list in `INTERFACES.md`.

If two families share a step (e.g. "visit these pixels in this order"), write that step
once as a helper starting with `_` and call it from both, as `_step_through` is used by
three families.

## Worked example: the scans of Palos et al. (2026)

*Programable Beam Control for EELS and Ptychography*, Microsc. Microanal. 32, ozag080,
compares raster, inward and outward rectangular spirals, N-sequential (N = 2, 4) and
random pixel orders. Their scan engine plays a list of pixel coordinates and jumps
straight from one to the next, with no flyback. In `scanpath/paths.py`:

| Family | Parameters | What it does |
|---|---|---|
| `grid_spiral` | `pixels=64`, `direction="in"` | "in" starts top-left and winds clockwise to the centre; "out" is the same spiral backwards, centre to edge |
| `n_sequential` | `pixels=64`, `n=2` | N × N passes, each visiting every Nth pixel in each direction |
| `random_order` | `pixels=64`, `seed=0` | every pixel once, in a random order |

All three are about ten lines each. They produce a list of `(row, col)` in visiting
order and hand it to the shared helper `_step_through`, which gives each pixel an equal
time and steps between them. A test checks the orders against the 6 × 6 grids in the
paper's Fig. 1b.

To mimic their engine, pair them with a scanner that sees a stepped command:
`("lowpass", {"bandwidth_hz": 1500, "hold": "step"})`. On that scanner (64 × 64 pixels,
1 s) the large-jump orders come out worst, as in their Fig. 4:

| scan | image NRMSE | RMS position error (nm) |
|---|---|---|
| raster (continuous, 64 lines) | 0.29 | 2.7 |
| spiral in / out | 0.24 / 0.24 | 1.0 / 1.0 |
| N = 2 | 0.44 | 10.1 |
| N = 4 | 0.65 | 14.3 |
| random | 1.05 | 31.6 |

Our instrument model is illustrative, not calibrated to their microscope. Spiral in and
out score the same here because a deterministic lag treats a path and its reverse alike.
The paper's in/out difference (only the outward cryo dataset converged in ptychography)
it attributes partly to drift and instabilities; reproducing it would need random drift
or a sample-response model, which is an open item for the team.
