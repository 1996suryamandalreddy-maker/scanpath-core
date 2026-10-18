"""Shared records, the three plug-in registries, and the pipeline functions.

Data flow -- every arrow carries one of the records defined here:

    family + params + Budget --[path]---------------------> Trajectory (commanded)
    stage 2, case A  path -> path:
        Trajectory (commanded) --[instrument]--------------> Trajectory (realized)
        realized + Sample --[image formation]--------------> image (as the user sees it)
    stage 2, case B  image -> image:
        Sample + Trajectory (commanded) --[image model]----> image (as the user sees it)
    commanded, realized, Sample --[path metric]------------> float
    image, Sample --[image metric]-------------------------> float
    scores over parameters --[optimisation.grid_search / gp_search]--> best params

Stages 1-4 live in paths.py, instruments.py, metrics.py and optimisation.py; this
module holds only what they share.

Units everywhere: nanometres and seconds. Positions are (x, y) with the origin at the
centre of the field of view. Images are indexed [row, col] = [y, x].

Records check their inputs when they are created and cannot be changed afterwards
(frozen dataclasses with read-only arrays), so a bad value fails where it is made, not
three stages later.
"""
from __future__ import annotations

import inspect
from dataclasses import dataclass, field, replace

import numpy as np
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree


# --------------------------------------------------------------------------- records
def _require_positive(value, name):
    if isinstance(value, bool) or not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive number, got {value!r}")


def _frozen_array(value, dtype):
    """A read-only copy, so no stage can change a record another stage is holding."""
    array = np.array(value, dtype=dtype, copy=True)
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class Budget:
    """What every candidate path gets, so families are compared like for like.

    Same field of view, same scan time, same number of position updates. By default
    path length and speed are *not* forced equal; they are reported as metrics (a soft
    constraint). Set ``length_um`` to also match path length (and so mean speed): every
    family that declares a ``knob`` is then tuned until it travels that far.

    By default every sample is a measurement. Set ``acquisitions`` to give every path the
    same number of measurement events (spread evenly over the samples where it may
    measure), and ``dose`` to give every path the same total dose.
    """
    fov_nm: float = 100.0      # side of the square field of view
    duration_s: float = 1.0    # total scan time
    rate_hz: float = 20_000.0  # position-update / sampling rate (uniform)
    length_um: float | None = None   # optional: commanded path length every family must hit
    acquisitions: int | None = None  # optional: number of measurement events every path gets
    dose: float | None = None        # optional: total dose, shared equally by the events

    def __post_init__(self):
        for name in ("fov_nm", "duration_s", "rate_hz"):
            _require_positive(getattr(self, name), name)
        for name in ("length_um", "dose"):
            if getattr(self, name) is not None:
                _require_positive(getattr(self, name), name)
        if self.acquisitions is not None and (
                isinstance(self.acquisitions, bool)
                or not isinstance(self.acquisitions, (int, np.integer))
                or not 1 <= self.acquisitions <= self.n):
            raise ValueError(f"acquisitions must be an integer from 1 to {self.n}")
        if self.n < 2:
            raise ValueError("duration_s * rate_hz must give at least two samples")

    @property
    def n(self) -> int:
        return int(round(self.duration_s * self.rate_hz))

    @property
    def t(self) -> np.ndarray:
        return np.arange(self.n) / self.rate_hz


@dataclass(frozen=True)
class Trajectory:
    """Probe positions over time. The same record holds commanded and realized paths.

    ``acquire`` marks the samples where a measurement is made (default: all of them);
    the probe still moves through the others, e.g. blanked transfers between dwell
    sites. ``dose`` is the dose delivered at each sample: positive exactly where
    ``acquire`` is True (default 1 per measurement).
    """
    t: np.ndarray                              # (N,) seconds, strictly increasing
    xy: np.ndarray                             # (N, 2) nm
    info: dict = field(default_factory=dict)   # family, params, instrument, budget ...
    acquire: np.ndarray | None = None          # (N,) bool
    dose: np.ndarray | None = None             # (N,) dose units

    def __post_init__(self):
        t, xy = _frozen_array(self.t, float), _frozen_array(self.xy, float)
        if t.ndim != 1 or len(t) < 2 or xy.shape != (len(t), 2):
            raise ValueError(f"expected t of shape (N,) with N >= 2 and xy of shape (N, 2); "
                             f"got {t.shape} and {xy.shape}")
        if not (np.isfinite(t).all() and np.isfinite(xy).all()):
            raise ValueError("times and positions must be finite")
        if np.any(np.diff(t) <= 0):
            raise ValueError("times must be strictly increasing")
        acquire = np.ones(len(t), bool) if self.acquire is None else self.acquire
        acquire = _frozen_array(acquire, bool)
        if acquire.shape != t.shape or not acquire.any():
            raise ValueError("acquire must have shape (N,) and mark at least one sample")
        dose = _frozen_array(acquire if self.dose is None else self.dose, float)
        if dose.shape != t.shape or not np.isfinite(dose).all():
            raise ValueError("dose must be finite with shape (N,)")
        if np.any(dose[~acquire] != 0) or np.any(dose[acquire] <= 0):
            raise ValueError("dose must be positive exactly where acquire is True")
        for name, value in (("t", t), ("xy", xy), ("acquire", acquire), ("dose", dose)):
            object.__setattr__(self, name, value)

    @property
    def events(self) -> np.ndarray:
        """(M, 2) positions of the measurement events."""
        return self.xy[self.acquire]


@dataclass
class Sample:
    """Ideal (ground-truth) image filling the field of view.

    ``features`` (optional) lists known feature centres in nm, for metrics that ask
    "did this feature end up in the right place?".
    """
    image: np.ndarray   # (H, W)
    fov_nm: float
    features: np.ndarray | None = None   # (K, 2) nm
    feature_size_nm: float | None = None

    def values_at(self, xy: np.ndarray) -> np.ndarray:
        """Bilinear image value at positions xy (nm). Outside the field: edge value."""
        h, w = self.image.shape
        col = (xy[:, 0] / self.fov_nm + 0.5) * w - 0.5
        row = (xy[:, 1] / self.fov_nm + 0.5) * h - 0.5
        return map_coordinates(self.image, [row, col], order=1, mode="nearest")

    def pixel_xy(self) -> np.ndarray:
        """(H*W, 2) pixel-centre positions in nm, row-major."""
        h, w = self.image.shape
        xs = ((np.arange(w) + 0.5) / w - 0.5) * self.fov_nm
        ys = ((np.arange(h) + 0.5) / h - 0.5) * self.fov_nm
        x, y = np.meshgrid(xs, ys)
        return np.column_stack((x.ravel(), y.ravel()))

    @classmethod
    def synthetic(cls, fov_nm: float = 100.0, size: int = 128, features: int = 70,
                  seed: int = 0) -> "Sample":
        """Random Gaussian particles (1-4 nm sigma) on a flat background."""
        rng = np.random.default_rng(seed)
        centres = rng.uniform(-0.45, 0.45, (features, 2)) * fov_nm
        sigma = rng.uniform(1.0, 4.0, features)
        height = rng.uniform(0.4, 1.0, features)
        px = cls(np.zeros((size, size)), fov_nm).pixel_xy()
        d2 = ((px[:, None, :] - centres[None]) ** 2).sum(-1)
        image = (height * np.exp(-d2 / (2 * sigma**2))).sum(1).reshape(size, size)
        return cls(image, fov_nm, centres, float(sigma.mean()))

    @classmethod
    def point_features(cls, fov_nm: float = 20_000.0, n: int = 8, size_nm: float = 300.0,
                       seed: int = 1, pixels: int = 256) -> "Sample":
        """n small Gaussian features at random known positions (the loop prototype's
        test image), for feature-position metrics."""
        rng = np.random.default_rng(seed)
        centres = rng.uniform(-0.4, 0.4, (n, 2)) * fov_nm
        px = cls(np.zeros((pixels, pixels)), fov_nm).pixel_xy()
        d2 = ((px[:, None, :] - centres[None]) ** 2).sum(-1)
        image = np.exp(-d2 / (2 * size_nm**2)).sum(1).reshape(pixels, pixels)
        return cls(image, fov_nm, centres, size_nm)


@dataclass
class Result:
    """One run of the pipeline. ``realized`` is None for image-to-image models; ``image``
    is the image the user would see (set whenever an image metric or model needed it)."""
    commanded: Trajectory
    realized: Trajectory | None
    scores: dict
    image: np.ndarray | None = None


# ------------------------------------------------------------------- image formation
def reconstruct(xy, values, sample, k=6):
    """Image on the sample's pixel grid from readings ``values`` drawn at positions ``xy``.

    Inverse-distance interpolation of the k nearest readings; robust to any path geometry
    (retraced, collinear or co-circular samples), which the optimiser will propose.
    """
    k = min(k, len(xy))
    distance, index = cKDTree(xy).query(sample.pixel_xy(), k=k)
    if k == 1:
        distance, index = distance[:, None], index[:, None]
    weight = 1.0 / np.maximum(distance, 1e-6) ** 2
    image = (weight * values[index]).sum(1) / weight.sum(1)
    return image.reshape(sample.image.shape)


def image_as_acquired(commanded, realized, sample, positions="commanded"):
    """The image a user would get from a path-to-path run.

    The signal is read where the probe really was (realized) and drawn at the commanded
    positions (what the instrument believes, the default) or at the realized ones
    (positions="realized", only possible in simulation).
    """
    where = {"commanded": commanded, "realized": realized}[positions].events
    return reconstruct(where, sample.values_at(realized.events), sample)


# ------------------------------------------------------------------------ registries
PATHS: dict = {}
INSTRUMENTS: dict = {}       # stage 2, case A: path -> path
IMAGE_MODELS: dict = {}      # stage 2, case B: image -> image
METRICS: dict = {}
METRIC_KINDS = ("path", "image")


def register_path(name, knob=None):
    """Decorator. fn(s, **params) -> (N, 2) array in [-1, 1]; s = t / duration in [0, 1).

    A family whose probe moves without measuring (blanked transfers) returns
    ``(xy, acquire)`` instead, with ``acquire`` an (N,) boolean mask.

    knob (optional): the parameter that sets how densely the family fills the field
    (lines, turns, spokes ...). It lets ``Budget(length_um=...)`` match path lengths.
    """
    def add(fn):
        fn.knob = knob
        PATHS[name] = fn
        return fn
    return add


def _unique(name):
    if name in INSTRUMENTS or name in IMAGE_MODELS:
        raise ValueError(f"{name!r} is already an instrument or image model name")


def register_instrument(name):
    """Decorator for a path-to-path model (stage 2, case A).

    fn(commanded: Trajectory, **params) -> realized Trajectory on the same time axis.
    """
    def add(fn):
        _unique(name)
        INSTRUMENTS[name] = fn
        return fn
    return add


def register_image_model(name):
    """Decorator for an image-to-image model (stage 2, case B).

    fn(sample: Sample, commanded: Trajectory, **params) -> image (same shape as
    sample.image): the image the user would see. The commanded path supplies where and
    when each part of the image is scanned.
    """
    def add(fn):
        _unique(name)
        IMAGE_MODELS[name] = fn
        return fn
    return add


def register_metric(name, higher_is_better=False, kind="path"):
    """Decorator for a quality function (stage 3).

    kind="path":  fn(commanded, realized, sample) -> float. Needs a path-to-path model.
                  A metric that needs repeat runs adds a 4th argument ``rerun(seed)``.
    kind="image": fn(image, sample) -> float. Works with either kind of instrument model:
                  the image comes from an image model, or is formed from a realized path.
    """
    if kind not in METRIC_KINDS:
        raise ValueError(f"kind must be one of {METRIC_KINDS}")

    def add(fn):
        fn.higher_is_better = higher_is_better
        fn.kind = kind
        METRICS[name] = fn
        return fn
    return add


def defaults(fn) -> dict:
    """Keyword defaults of a plug-in: these are its tunable parameters."""
    return {k: p.default for k, p in inspect.signature(fn).parameters.items()
            if p.default is not inspect.Parameter.empty}


def parameters(name: str) -> dict:
    """The explicit parameters (with defaults) of any registered component.

    Works for path families, both kinds of instrument model and metrics, e.g.
    ``parameters("raster") -> {"lines": 48, "flyback": 0.1}``.
    """
    for registry in (PATHS, INSTRUMENTS, IMAGE_MODELS, METRICS):
        if name in registry:
            return defaults(registry[name])
    raise KeyError(f"nothing registered as {name!r}")


def describe() -> str:
    """Everything registered, with parameters and defaults."""
    lines = []
    for title, reg in (("paths (stage 1)", PATHS),
                       ("path-to-path models (stage 2, case A)", INSTRUMENTS),
                       ("image-to-image models (stage 2, case B)", IMAGE_MODELS)):
        lines.append(f"{title}:")
        lines += [f"  {k:16s} {defaults(f)}"
                  + (f"   knob: {f.knob}" if getattr(f, "knob", None) else "")
                  for k, f in reg.items()]
    lines.append("metrics (stage 3):")
    lines += [f"  {k:26s} {f.kind:5s}  ({'higher' if f.higher_is_better else 'lower'} is better)"
              for k, f in METRICS.items()]
    return "\n".join(lines)


def _get(registry, name, kind):
    if name not in registry:
        raise KeyError(f"unknown {kind} {name!r}; registered: {sorted(registry)}")
    return registry[name]


def _spec(x):
    """'name' or ('name', {params}) -> (name, params)."""
    return (x, {}) if isinstance(x, str) else (x[0], dict(x[1]))


# ------------------------------------------------------------------- public pipeline
def _draw(fn, family, budget, params):
    """Unit-square positions and the acquisition mask of one family."""
    out = fn(budget.t / budget.duration_s, **params)
    unit, acquire = out if isinstance(out, tuple) else (out, np.ones(budget.n, bool))
    unit = np.asarray(unit, float)
    if unit.shape != (budget.n, 2) or np.abs(unit).max() > 1 + 1e-9:
        raise ValueError(f"path {family!r} must return ({budget.n}, 2) values in [-1, 1]")
    return unit, np.asarray(acquire, bool)


def _events(acquire, budget):
    """Measurement mask and per-sample dose under the budget.

    With ``budget.acquisitions`` set, that many events are spread evenly over the
    samples where the family allows a measurement. With ``budget.dose`` set, the dose
    is shared equally between the events. So every family gets the same number of
    measurements and the same total dose.
    """
    if budget.acquisitions is not None:
        eligible = np.flatnonzero(acquire)
        if len(eligible) < budget.acquisitions:
            raise ValueError(f"only {len(eligible)} samples can measure; budget asks for "
                             f"{budget.acquisitions} acquisitions")
        chosen = eligible[np.linspace(0, len(eligible) - 1, budget.acquisitions).astype(int)]
        acquire = np.zeros_like(acquire)
        acquire[chosen] = True
    dose = None if budget.dose is None else acquire * (budget.dose / acquire.sum())
    return acquire, dose


def _length_um(unit, budget):
    return np.linalg.norm(np.diff(unit, axis=0), axis=1).sum() * budget.fov_nm / 2 / 1000


def make_path(family: str, budget: Budget = Budget(), **params) -> Trajectory:
    """Stage 1: commanded trajectory for one path family under the shared budget.

    With ``budget.length_um`` set, the family's knob is tuned (unless given explicitly)
    until the path is that long. Families without a knob are drawn as they are.
    """
    fn = _get(PATHS, family, "path")
    knob = getattr(fn, "knob", None)
    if budget.length_um and knob and knob not in params:
        value, best = float(defaults(fn)[knob]), None
        for _ in range(8):
            drawn = _draw(fn, family, budget, {**params, knob: value})
            length = _length_um(drawn[0], budget)
            miss = abs(length / budget.length_um - 1)
            if best is None or miss < best[0]:
                best = (miss, value, drawn)
            if miss < 0.002 or length == 0:
                break
            value *= budget.length_um / length
        params = {**params, knob: best[1]}
        unit, acquire = best[2]
    else:
        unit, acquire = _draw(fn, family, budget, params)
    acquire, dose = _events(acquire, budget)
    return Trajectory(budget.t, unit * budget.fov_nm / 2,
                      {"family": family, "params": {**defaults(fn), **params},
                       "budget": budget}, acquire, dose)


def _accepts(fn, name):
    return name in inspect.signature(fn).parameters


def simulate(commanded: Trajectory, model: str = "ideal", seed=None, **params) -> Trajectory:
    """Stage 2: what the instrument actually does with the commanded trajectory.

    ``seed`` is passed to models that are random (they declare a ``seed`` parameter) and
    ignored by deterministic ones. Whatever the model reports in ``realized.info`` (a
    count of stage moves, say) is kept. Instruments move the probe; they do not decide
    when it measures, so the commanded acquisition mask and dose are carried over.
    """
    fn = _get(INSTRUMENTS, model, "instrument")
    if seed is not None and _accepts(fn, "seed"):
        params = {**params, "seed": seed}
    realized = fn(commanded, **params)
    if realized.xy.shape != commanded.xy.shape or not np.array_equal(realized.t, commanded.t):
        raise ValueError(f"instrument {model!r} must keep the commanded time axis")
    return replace(realized, acquire=commanded.acquire, dose=commanded.dose,
                   info={**commanded.info, **realized.info, "instrument": model,
                         "instrument_params": {**defaults(fn), **params}})


def make_image(sample: Sample, commanded: Trajectory, model: str, seed=None,
               **params) -> np.ndarray:
    """Stage 2, case B: the distorted image from an image-to-image model."""
    fn = _get(IMAGE_MODELS, model, "image model")
    if seed is not None and _accepts(fn, "seed"):
        params = {**params, "seed": seed}
    image = np.asarray(fn(sample, commanded, **params), float)
    if image.shape != sample.image.shape or not np.isfinite(image).all():
        raise ValueError(f"image model {model!r} must return a finite {sample.image.shape} image")
    return image


def _evaluate(commanded, realized, sample, metrics, rerun, image):
    """Scores, plus the image if one was needed (so run() can keep it)."""
    if metrics is None:                      # "everything" = everything we can compute
        names = [m for m, fn in METRICS.items()
                 if (fn.kind == "image" or realized is not None)
                 and not (_accepts(fn, "rerun") and rerun is None)]
    else:
        names = list(metrics)
    scores = {}
    for m in names:
        fn = _get(METRICS, m, "metric")
        if fn.kind == "image":
            if image is None:
                if realized is None:
                    raise ValueError(f"metric {m!r} needs an image or a realized path")
                image = image_as_acquired(commanded, realized, sample)
            scores[m] = float(fn(image, sample))
        elif realized is None:
            raise ValueError(f"metric {m!r} compares paths, so it needs a path-to-path "
                             "model; use an image metric with an image-to-image model")
        elif _accepts(fn, "rerun"):
            if rerun is None:
                raise ValueError(f"metric {m!r} needs repeat runs; call it through run()")
            scores[m] = float(fn(commanded, realized, sample, rerun))
        else:
            scores[m] = float(fn(commanded, realized, sample))
    return scores, image


def evaluate(commanded: Trajectory, realized: Trajectory | None, sample: Sample,
             metrics=None, rerun=None, image=None) -> dict:
    """Stage 3: named scores. metrics=None evaluates every metric that applies.

    Path metrics compare ``commanded`` and ``realized``; image metrics score ``image``
    (formed from the realized path when not given). ``rerun(seed)`` is handed to metrics
    that ask for it; ``run`` builds it.
    """
    return _evaluate(commanded, realized, sample, metrics, rerun, image)[0]


def run(path, instrument="ideal", metrics=None, budget: Budget = Budget(),
        sample: Sample | None = None, seed=None) -> Result:
    """Stages 1-3 in one call: path -> instrument model -> metrics.

    ``instrument`` is 'name' or ('name', {params}) of either a path-to-path model or an
    image-to-image model; the pipeline picks the right route. ``seed`` goes to random
    models.
    """
    family, p_params = _spec(path)
    model, i_params = _spec(instrument)
    sample = Sample.synthetic(budget.fov_nm) if sample is None else sample
    commanded = make_path(family, budget, **p_params)
    if model in IMAGE_MODELS:
        image = make_image(sample, commanded, model, seed, **i_params)
        scores, image = _evaluate(commanded, None, sample, metrics, None, image)
        return Result(commanded, None, scores, image)
    if model not in INSTRUMENTS:
        raise KeyError(f"unknown instrument {model!r}; path-to-path: {sorted(INSTRUMENTS)}; "
                       f"image-to-image: {sorted(IMAGE_MODELS)}")
    realized = simulate(commanded, model, seed, **i_params)
    rerun = lambda s: simulate(commanded, model, s, **i_params)
    scores, image = _evaluate(commanded, realized, sample, metrics, rerun, None)
    return Result(commanded, realized, scores, image)
