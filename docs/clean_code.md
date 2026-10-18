# Clean code: habits worth copying, with examples from this repository

The workflow notebooks are good code to learn from. This note picks out the habits that
make them easy to trust, shows where the package now uses the same habits, and points out
a few places where his code is harder to change than it needs to be, and what to do
instead. Each point has a small example from our own files.

## 1. Check inputs where they enter, and say exactly what is wrong

His `Trajectory` refuses bad data the moment it is created:

```python
if np.any(np.diff(t) <= 0):
    raise ValueError('Positions and times must be finite; times strictly increasing')
```

A mistake then fails *where it is made*, with a message that says what to fix, instead
of producing a strange number three stages later. `scanpath/core.py` now does the same
for `Trajectory` and `Budget`:

```python
>>> sp.Budget(duration_s=-1)
ValueError: duration_s must be a finite positive number, got -1
```

**Where to check:** at the boundaries, meaning records, configuration and anything a
user types. Inside the package, once data has passed those checks, trust it. Checking
the same thing in every function buries the logic.

## 2. Make shared data impossible to change by accident

His records are `@dataclass(frozen=True)` and their arrays are read-only
(`xy.setflags(write=False)`). When the commanded path is handed to the instrument model,
the model *cannot* modify it in place and silently corrupt the commanded path that the
metrics will later compare against. Our `Trajectory` now works the same way:

```python
>>> path.xy[0, 0] = 1.0
ValueError: assignment destination is read-only
```

To "change" a frozen record, make a new one: `dataclasses.replace(path, acquire=...)`.

## 3. Put every choice in one place

His notebook has one configuration cell: change it, run all, and every number and plot
comes from that one experiment. Our examples 05 and 06 do the same with a `CONFIG` dict
or a block of capitalised constants at the top. Rule of thumb: **if you would change it
between experiments, it belongs in the configuration, not in the middle of a function.**

## 4. Keep "what to measure", "how to weigh it" and "what is allowed" separate

His `evaluate` takes `metrics` (numbers), `objective` (one loss from those numbers) and
`constraints` (rules that make a candidate infeasible). Because they are separate, you can
change the weights without touching a metric, or add a speed limit without touching the
loss. The searches in `scanpath/optimisation.py` now take the same three things:

```python
sp.gp_search("raster", space, score=my_loss, metrics=[...],
             constraints=sp.limits(at_most={"realized_speed_peak_um_s": 15}))
```

## 5. Test against answers you know

His benchmark checks the lag model against the exact formula for a ramp:

```python
np.testing.assert_allclose(actual, t - .1 * (1 - np.exp(-t / .1)), atol=1e-13)
```

That one test found a real error in our old `lowpass` (it lagged half a sample too little).
Whenever a piece of physics has a known answer (a ramp, a step, a circle), write a test
that compares against it. Ours are in `tests/test_contract.py`.

## 6. Make failures visible, never quietly "best"

`best_trial` returns `None` when no candidate is feasible, so an infeasible point can never
be reported as the winner. Ours returns the least-bad row, marked `feasible: False`, so
you can see how far off it was. Either way the rule is the same: **an error or a broken
rule should be impossible to miss.**

---

## Where his code is harder to change, and what to do instead

### a. Code that lives in a notebook

The benchmark notebook gets its definitions by running the other notebook's tagged cells
with `exec`. It works, and it is better than copying, but:

* an edited cell in one notebook can silently change the other;
* nothing can `import` it (a script, a test, a third notebook);
* tools can't check it (no linting or tests run on notebook cells).

**Instead:** definitions go in `.py` files (the `scanpath` package); notebooks `import`
them and only *run experiments and draw plots*.

### b. Parameters hidden inside the function body

```python
def __call__(self, params):
    turns = positive(params.get('turns', 8), 'turns')
    radius = positive(params.get('radius', .5), 'radius')
```

To find out what a generator accepts, you have to read its body. A misspelled key
(`{'radious': .4}`) is silently ignored and the default is used.

**Instead:** parameters as keyword arguments with defaults:

```python
@register_path("spiral", knob="turns")
def spiral(s, turns=58, radius=1.0):
```

Now `sp.parameters("spiral")` lists them, the optimiser knows them, and a misspelled
name is an error (`TypeError: spiral() got an unexpected keyword argument 'radious'`).

### c. The same idea written twice

* `Raster` repeats the arc-length sampling that `polyline` already does.
* `Spiral` checks its radius by hand although `radius_parameter` exists for exactly that.

Duplicates drift apart: fix a bug in one and the other keeps it. **Instead:** when you
write something the second time, make it a helper and call it from both places. Our
`sparse` and `pulse_lattice` share `_dwell`; all five searches share `_Runner`.

### d. One function doing many jobs

`search` is about 100 lines and holds three search methods, a nested exception class and
two nested functions. To change one method you have to read all of them.

**Instead:** one small function per method, plus one shared piece for what they have in
common. In `optimisation.py`, `_Runner` runs a candidate, judges it and records it, and
each search (`random_search`, `pattern_search`, …) is 10–20 lines of "what to try next".

### e. Unexplained numbers and implicit units

`1e-10`, `.5 + np.arctan(loss) / np.pi`, `2. + sum(violations)`: each has a reason, but
the reader has to guess it. Positions are in "FOV units", which are written down nowhere
in the code.

**Instead:** give the number a name or a one-line comment, and put units in names
(`fov_nm`, `speed_peak_um_s`, `angle_deg`). Compare `merit()` in `optimisation.py`,
whose docstring says why infeasible points get "worst + 1 + violation".

---

## A short checklist before you share code

1. Can someone see every tunable parameter without reading the body? (keyword defaults)
2. Does bad input fail immediately, with a message that says what to fix?
3. Did I write anything twice? Make it a helper.
4. Is each function one job, short enough to read on one screen?
5. Are units in the names, and every unusual number explained?
6. Is there a test with a known answer for the physics I just added?
7. Are the definitions in a `.py` file, with the notebook only running and plotting?

Reading order for the package, if you want to see these habits in practice:
`core.py` (records, checks, registries) → `paths.py` → `instruments.py` → `metrics.py`
→ `optimisation.py` → `tests/test_contract.py` → `examples/05_full_pipeline.py`.
