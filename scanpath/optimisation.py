"""Stage 4: optimisation -- choose path parameters from the scores.

Every search runs the same loop -- propose parameters, run stages 1-3, judge the scores,
propose again -- and only the proposing differs. So all searches take the same inputs
and return the same ``(rows, best)``, and one can replace another by name (``SEARCHES``):

    grid_search     every point of a grid; shows the whole landscape for 1-2 parameters
    random_search   uniform random points; the honest baseline for the others
    pattern_search  steps along each parameter around the best point, halving the step
                    when a sweep finds nothing better (the workflow notebook's
                    grid refinement)
    de_search       differential evolution (scipy)
    gp_search       a Gaussian process models the loss; each new point maximises the
                    expected improvement (Bayesian optimisation)

What "best" means
    score        a metric name (optimised in that metric's own direction), or a function
                 ``scores -> loss`` for a combined objective (lower is better).
    constraints  optional function ``scores -> {rule: violation >= 0}``; ``limits(...)``
                 builds the usual one. A point is feasible when every violation is 0.
                 The best row is the feasible one with the lowest loss; when nothing is
                 feasible it is the row with the smallest total violation, and
                 ``best["feasible"]`` is False -- check it.

Search space, the same for every search
    (low, high)          a real parameter
    (low, high, "int")   an integer parameter
    (low, high, "log")   a real parameter searched on a log scale
    [a, b, c]            a fixed set of choices
grid_search also takes a dict of lists (every combination) or a list of parameter
dicts, and gives each range ``levels`` points.

Budgets: ``n_evals`` counts pipeline runs. A search that proposes a point it has
already run gets the stored result for free (runs are deterministic for a fixed
``run_seed``), so every run is a different point. If a search stops early, the rest
of its budget goes to random points, marked ``phase="fallback"``.

With a random instrument, ``grid_search(..., seeds=...)`` runs every point on the same
seeds and accepts the best only if it beats the baseline by more than two standard
errors, so the loop does not chase noise. Confirm a winner on fresh seeds.
"""
from __future__ import annotations

import inspect
import itertools
import math

import numpy as np
from scipy.linalg import cho_factor, cho_solve, solve_triangular
from scipy.optimize import differential_evolution
from scipy.stats import norm

from .core import METRICS, Budget, Sample, _get, run


# ------------------------------------------------------------ objectives and constraints
def limits(at_most: dict | None = None, at_least: dict | None = None):
    """Constraints for the searches: ``metric <= value`` and ``metric >= value`` rules.

    Each violation is the relative miss (0.1 = 10 % past the limit), so rules in
    different units can be summed. Example::

        limits(at_most={"realized_speed_peak_um_s": 60, "realized_accel_peak_um_s2": 2e5},
               at_least={"coverage": 0.5})
    """
    at_most, at_least = dict(at_most or {}), dict(at_least or {})
    for metric, value in {**at_most, **at_least}.items():
        _get(METRICS, metric, "metric")
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"limit for {metric!r} must be finite and positive, got {value}")

    def check(scores):
        violations = {}
        for metric, value in at_most.items():
            violations[f"{metric} <= {value:g}"] = _miss(scores[metric] / value - 1)
        for metric, value in at_least.items():
            violations[f"{metric} >= {value:g}"] = _miss(1 - scores[metric] / value)
        return violations

    check.metrics = [*at_most, *at_least]        # so the searches compute them
    return check


def _miss(excess):
    """Violation from a relative excess; a score that could not be computed fails."""
    return max(0.0, float(excess)) if np.isfinite(excess) else 1.0


def pareto_front(rows: list, objectives: list) -> list:
    """Feasible rows that no other feasible row beats on every objective at once.

    ``objectives`` are metric names, each taken in its own direction (coverage up,
    error down). Use it when the objectives compete and no weighting is agreed yet.
    """
    feasible = [r for r in rows if r.get("feasible", True)]
    signs = np.array([-1.0 if _get(METRICS, m, "metric").higher_is_better else 1.0
                      for m in objectives])
    values = np.array([[r[m] for m in objectives] for r in feasible]).reshape(-1, len(objectives))
    values = values * signs                              # now lower is better everywhere
    keep = [i for i, v in enumerate(values)
            if not np.any(np.all(values <= v, axis=1) & np.any(values < v, axis=1))]
    return [feasible[i] for i in keep]


# --------------------------------------------------------------------- search space
class _Space:
    """The parameters being searched, mapped to and from the unit cube [0, 1]^d."""

    KINDS = ("float", "int", "log")

    def __init__(self, space: dict):
        if not space:
            raise ValueError("the search space is empty")
        self.names, self.specs = list(space), []
        for name, spec in space.items():
            if isinstance(spec, list):
                if not spec:
                    raise ValueError(f"no choices given for {name!r}")
                self.specs.append(("choice", list(spec)))
                continue
            low, high, *kind = spec
            kind = kind[0] if kind else "float"
            if kind not in self.KINDS:
                raise ValueError(f"{name!r}: kind must be one of {self.KINDS}, got {kind!r}")
            if not low < high or (kind == "log" and low <= 0):
                raise ValueError(f"{name!r}: need low < high (and low > 0 for 'log')")
            self.specs.append((kind, float(low), float(high)))

    @property
    def dim(self) -> int:
        return len(self.names)

    def point(self, u) -> dict:
        """Parameters at unit-cube position u."""
        params = {}
        for name, spec, ui in zip(self.names, self.specs, np.clip(u, 0.0, 1.0)):
            kind = spec[0]
            if kind == "choice":
                choices = spec[1]
                params[name] = choices[min(int(ui * len(choices)), len(choices) - 1)]
            elif kind == "log":
                params[name] = float(math.exp(math.log(spec[1])
                                              + ui * (math.log(spec[2]) - math.log(spec[1]))))
            else:
                value = spec[1] + ui * (spec[2] - spec[1])
                params[name] = int(round(value)) if kind == "int" else float(value)
        return params

    def unit(self, params: dict) -> np.ndarray:
        """Unit-cube position of a parameter set (the inverse of ``point``)."""
        u = []
        for name, spec in zip(self.names, self.specs):
            kind, value = spec[0], params[name]
            if kind == "choice":
                u.append((spec[1].index(value) + 0.5) / len(spec[1]))
            elif kind == "log":
                u.append((math.log(value) - math.log(spec[1]))
                         / (math.log(spec[2]) - math.log(spec[1])))
            else:
                u.append((float(value) - spec[1]) / (spec[2] - spec[1]))
        return np.array(u)

    def centre(self) -> dict:
        return self.point(np.full(self.dim, 0.5))

    def grid(self, levels: int) -> list:
        """Every combination, with ``levels`` points across each range."""
        axes = []
        for spec in self.specs:
            kind = spec[0]
            if kind == "choice":
                axes.append(spec[1])
            elif kind == "log":
                axes.append([float(v) for v in np.geomspace(spec[1], spec[2], levels)])
            elif kind == "int":
                axes.append(sorted({int(round(v)) for v in np.linspace(spec[1], spec[2], levels)}))
            else:
                axes.append([float(v) for v in np.linspace(spec[1], spec[2], levels)])
        return [dict(zip(self.names, values)) for values in itertools.product(*axes)]


# -------------------------------------------------------- the loop shared by every search
class _Runner:
    """Runs stages 1-3 for a parameter set, judges the scores and keeps every row.

    Every search calls this one object, so scoring, constraints, the record of what
    was tried and the choice of the best are written once.
    """

    def __init__(self, family, score, instrument, fixed, metrics, constraints, budget,
                 sample, run_seed):
        self.family, self.score, self.instrument = family, score, instrument
        self.fixed, self.constraints = dict(fixed or {}), constraints
        self.budget, self.run_seed = budget, run_seed
        self.sample = Sample.synthetic(budget.fov_nm) if sample is None else sample
        if isinstance(score, str):
            self.sign = -1.0 if _get(METRICS, score, "metric").higher_is_better else 1.0
            wanted = [score, *(metrics or [])]
        elif callable(score):
            self.sign = 1.0
            wanted = None if metrics is None else list(metrics)
        else:
            raise TypeError("score must be a metric name or a function scores -> loss")
        if wanted is not None and constraints is not None:
            wanted += getattr(constraints, "metrics", [])
        self.metrics = None if wanted is None else list(dict.fromkeys(wanted))
        self.rows, self._seen = [], {}

    @property
    def count(self) -> int:
        """Pipeline runs so far (each a different point)."""
        return len(self.rows)

    def scores(self, params, seed=None) -> dict:
        result = run((self.family, {**self.fixed, **params}), self.instrument, self.metrics,
                     self.budget, self.sample, self.run_seed if seed is None else seed)
        return result.scores

    def judge(self, scores) -> dict:
        """Loss, violations and feasibility of one set of scores."""
        loss = (self.sign * scores[self.score] if isinstance(self.score, str)
                else float(self.score(scores)))
        violations = {} if self.constraints is None else dict(self.constraints(scores))
        if not np.isfinite(loss):
            violations["score could not be computed"] = 1.0
        if any(not np.isfinite(v) or v < 0 for v in violations.values()):
            raise ValueError(f"violations must be finite and >= 0, got {violations}")
        violation = float(sum(violations.values()))
        return {"loss": float(loss), "feasible": violation == 0, "violation": violation,
                "violations": violations}

    def __call__(self, params: dict, phase: str) -> dict:
        """The row for ``params``: run it if new, otherwise the stored row (free)."""
        key = tuple(sorted(params.items()))
        if key not in self._seen:
            scores = self.scores(params)
            row = {"family": self.family, **self.fixed, **params, **scores,
                   **self.judge(scores), "step": self.count, "phase": phase}
            self.rows.append(row)
            best = self.best()
            row["best_so_far"] = (self.sign * best["loss"] if best["feasible"]
                                  else float("nan"))
            self._seen[key] = row
        return self._seen[key]

    @staticmethod
    def rank(row):
        """Feasible first, by loss; then infeasible, by total violation."""
        return (0, row["loss"]) if row["feasible"] else (1, row["violation"])

    def best(self) -> dict:
        return dict(min(self.rows, key=self.rank))

    def merit(self, row) -> float:
        """One number per row for the model-based searches: the loss when feasible;
        when not, worse than every feasible row seen so far, more so the bigger the
        violation."""
        if row["feasible"]:
            return row["loss"]
        feasible = [r["loss"] for r in self.rows if r["feasible"]]
        worst = max(feasible) if feasible else 0.0
        return worst + 1.0 + row["violation"]

    def fill(self, space, n_evals, rng, phase="fallback"):
        """Spend what is left of the budget on random points."""
        for _ in range(100 * n_evals):
            if self.count >= n_evals:
                return
            self(space.point(rng.random(space.dim)), phase)


def _check_budget(n_evals):
    if isinstance(n_evals, bool) or not isinstance(n_evals, (int, np.integer)) or n_evals < 1:
        raise ValueError(f"n_evals must be a positive integer, got {n_evals!r}")


# --------------------------------------------------------------------------- searches
def grid_search(family: str, grid, score, instrument="ideal", *, levels: int = 5,
                fixed: dict | None = None, metrics=None, constraints=None,
                budget: Budget = Budget(), sample: Sample | None = None, run_seed=None,
                seeds=None, baseline: dict | None = None):
    """Evaluate every point of a parameter grid.

    grid: {param: [values]} for every combination (a (low, high[, kind]) range gets
    ``levels`` points), or a list of parameter dicts for explicit (e.g. coupled) points.

    For a random instrument, pass ``seeds`` (e.g. ``range(16)``). Every point is then run
    on the same seeds and compared with ``baseline`` (default: the first point) seed by
    seed. Rows hold the mean scores and gain ``change`` and ``change_sem`` (paired
    difference in `score`) and ``significant``; ``best["accepted"]`` is True only if the
    best point beats the baseline by more than two standard errors.
    """
    points = grid if isinstance(grid, list) else _Space(grid).grid(levels)
    runner = _Runner(family, score, instrument, fixed, metrics, constraints, budget,
                     sample, run_seed)
    if seeds is None:
        for point in points:
            runner(point, "grid")
        return runner.rows, runner.best()

    if not isinstance(score, str):
        raise ValueError("paired seeds need score to be a metric name")
    seeds = list(seeds)

    def per_seed(point):
        runs = [runner.scores(point, s) for s in seeds]
        return {m: np.array([r[m] for r in runs]) for m in runs[0]}

    base = per_seed(points[0] if baseline is None else baseline)[score]
    rows = []
    for step, point in enumerate(points):
        values = per_seed(point)
        means = {m: float(v.mean()) for m, v in values.items()}
        diff = values[score] - base
        sem = float(diff.std(ddof=1) / np.sqrt(len(diff))) if len(diff) > 1 else 0.0
        spread = (float(values[score].std(ddof=1) / np.sqrt(len(seeds)))
                  if len(seeds) > 1 else 0.0)
        rows.append({"family": family, **runner.fixed, **point, **means, **runner.judge(means),
                     "step": step, "phase": "grid", f"{score}_sem": spread,
                     "change": float(diff.mean()), "change_sem": sem,
                     "significant": bool(runner.sign * diff.mean() + 2 * sem < 0)})
    best = dict(min(rows, key=_Runner.rank))
    best["accepted"] = best["significant"] and best["feasible"]
    return rows, best


def random_search(family: str, space: dict, score, instrument="ideal", *, n_evals: int = 50,
                  seed: int = 0, fixed: dict | None = None, metrics=None, constraints=None,
                  budget: Budget = Budget(), sample: Sample | None = None, run_seed=None):
    """Uniform random points (the first is the centre of the space)."""
    _check_budget(n_evals)
    space, rng = _Space(space), np.random.default_rng(seed)
    runner = _Runner(family, score, instrument, fixed, metrics, constraints, budget,
                     sample, run_seed)
    runner(space.centre(), "start")
    runner.fill(space, n_evals, rng, "random")
    return runner.rows, runner.best()


def pattern_search(family: str, space: dict, score, instrument="ideal", *, n_evals: int = 50,
                   seed: int = 0, step: float = 0.25, fixed: dict | None = None,
                   metrics=None, constraints=None, budget: Budget = Budget(),
                   sample: Sample | None = None, run_seed=None):
    """Start at the centre; try one step up and down each parameter around the best point
    so far; halve the step whenever a whole sweep finds nothing better.

    ``step`` is a fraction of each range. Deterministic: ``seed`` only drives the random
    fallback if the step becomes too small before the budget is spent.
    """
    _check_budget(n_evals)
    space, rng = _Space(space), np.random.default_rng(seed)
    runner = _Runner(family, score, instrument, fixed, metrics, constraints, budget,
                     sample, run_seed)
    runner(space.centre(), "start")
    while runner.count < n_evals and step > 1e-3:
        incumbent = runner.best()
        centre = space.unit(incumbent)
        improved = False
        for axis, direction in itertools.product(range(space.dim), (-1, 1)):
            if runner.count >= n_evals:
                break
            u = centre.copy()
            u[axis] = np.clip(u[axis] + direction * step, 0.0, 1.0)
            row = runner(space.point(u), "pattern")
            improved |= _Runner.rank(row) < _Runner.rank(incumbent)
        if not improved:
            step /= 2
    runner.fill(space, n_evals, rng)
    return runner.rows, runner.best()


class _BudgetSpent(Exception):
    pass


def de_search(family: str, space: dict, score, instrument="ideal", *, n_evals: int = 50,
              seed: int = 0, popsize: int = 5, fixed: dict | None = None, metrics=None,
              constraints=None, budget: Budget = Budget(), sample: Sample | None = None,
              run_seed=None):
    """Differential evolution (scipy), stopped after ``n_evals`` runs.

    The population has ``popsize`` x (number of parameters) members and starts with the
    centre of the space. No final polishing, so every run counts against the budget.
    """
    _check_budget(n_evals)
    space, rng = _Space(space), np.random.default_rng(seed)
    runner = _Runner(family, score, instrument, fixed, metrics, constraints, budget,
                     sample, run_seed)

    def objective(u):
        if runner.count >= n_evals:
            raise _BudgetSpent
        return runner.merit(runner(space.point(u), "de"))

    population = rng.random((max(5, popsize * space.dim), space.dim))
    population[0] = 0.5
    random_arg = ("rng" if "rng" in inspect.signature(differential_evolution).parameters
                  else "seed")
    try:
        differential_evolution(objective, [(0.0, 1.0)] * space.dim, init=population,
                               maxiter=n_evals, tol=0, atol=0, polish=False,
                               updating="immediate", **{random_arg: rng})
    except _BudgetSpent:
        pass
    runner.fill(space, n_evals, rng)
    return runner.rows, runner.best()


# ------------------------------------------------------------- GP-based optimisation
def _rbf(a, b, length):
    d2 = ((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)
    return np.exp(-0.5 * d2 / length**2)


def _fit_gp(X, y):
    """Zero-mean GP on standardised y; length scale and noise by marginal likelihood."""
    mean, scale = y.mean(), y.std() or 1.0
    z = (y - mean) / scale
    best = None
    for length in (0.05, 0.1, 0.2, 0.35, 0.6, 1.0):
        for noise in (1e-6, 1e-3, 1e-2, 1e-1):
            K = _rbf(X, X, length) + (noise + 1e-10) * np.eye(len(X))
            try:
                factor = cho_factor(K, lower=True)
            except np.linalg.LinAlgError:
                continue
            alpha = cho_solve(factor, z)
            evidence = -0.5 * z @ alpha - np.log(np.diag(factor[0])).sum()
            if best is None or evidence > best[0]:
                best = (evidence, length, factor, alpha)
    _, length, factor, alpha = best

    def predict(Xs):
        Ks = _rbf(Xs, X, length)
        v = solve_triangular(factor[0], Ks.T, lower=True)
        sd = np.sqrt(np.maximum(1.0 - (v**2).sum(0), 1e-12))
        return mean + scale * (Ks @ alpha), scale * sd
    return predict


def _expected_improvement(mu, sd, best):
    gain = best - mu                            # minimising
    z = gain / sd
    return gain * norm.cdf(z) + sd * norm.pdf(z)


def gp_search(family: str, space: dict, score, instrument="ideal", *, n_evals: int = 28,
              n_init: int = 8, seed: int = 0, n_candidates: int = 2000,
              fixed: dict | None = None, metrics=None, constraints=None,
              budget: Budget = Budget(), sample: Sample | None = None, run_seed=None):
    """Bayesian optimisation: ``n_init`` space-filling points (a Latin hypercube), then
    each further point is the candidate with the highest expected improvement under a
    Gaussian-process model of the loss, until ``n_evals`` runs.

    Infeasible points are modelled as worse than every feasible one, so the search
    learns to stay inside the constraints. Rows are in evaluation order.
    """
    _check_budget(n_evals)
    space, rng = _Space(space), np.random.default_rng(seed)
    runner = _Runner(family, score, instrument, fixed, metrics, constraints, budget,
                     sample, run_seed)
    d = space.dim
    lhs = (np.argsort(rng.random((n_init, d)), axis=0) + rng.random((n_init, d))) / n_init
    for u in lhs[: n_evals]:
        runner(space.point(u), "initial")

    while runner.count < n_evals:
        X = np.array([space.unit(r) for r in runner.rows])
        y = np.array([runner.merit(r) for r in runner.rows])
        predict = _fit_gp(X, y)
        candidates = np.vstack((rng.random((n_candidates, d)),
                                np.clip(X[np.argmin(y)] + rng.normal(0, 0.05, (n_candidates // 4, d)),
                                        0, 1)))
        mu, sd = predict(candidates)
        before = runner.count
        for i in np.argsort(-_expected_improvement(mu, sd, y.min()))[:200]:
            runner(space.point(candidates[i]), "gp")
            if runner.count > before:
                break
        if runner.count == before:               # every promising candidate already run
            break
    runner.fill(space, n_evals, rng)
    return runner.rows, runner.best()


#: every search by name, for configs that pick one with a string
SEARCHES = {"grid": grid_search, "random": random_search, "pattern": pattern_search,
            "de": de_search, "gp": gp_search}
