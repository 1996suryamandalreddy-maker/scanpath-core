"""Stage 1 plug-ins: path families.

Each family is a function of normalized time s = t / duration in [0, 1) returning
(N, 2) positions in [-1, 1]. Because every family is sampled on the same time axis,
scan time and point count are equal by construction (see core.Budget). Defaults are
chosen so raster, spiral and Lissajous also have about the same path length
(~9.2 um at 100 nm); ``Budget(length_um=...)`` matches lengths exactly for every
family that declares a knob.

Four kinds of family live here:

* driven curves: positions written directly as functions of time, so each keeps its
  natural velocity profile (a Lissajous stays sinusoidal): raster, spiral, lissajous,
  and from the workflow notebook closed_loops and rose;
* shapes walked at constant speed: a polyline is drawn and traversed at constant speed:
  serpentine, radial, hilbert, hybrid (loop-prototype notebook) and strokes (workflow
  notebook);
* dwell paths: the probe parks at sites and travels between them: sparse (jittered
  grid, measures throughout) and pulse_lattice (regular lattice, transfers blanked);
* pixel-list scans: the beam steps straight from pixel to pixel of a square grid, in a
  programmed order, with no flyback (Palos et al., Microsc. Microanal. 32, ozag080,
  2026): grid_spiral (inward or outward), n_sequential (every Nth pixel per pass),
  random_order.

``radius`` is always a fraction of the half-field (1 = reaches the edge) and angles are
in degrees (``angle_deg``). Rotations shrink a shape just enough to stay in the field.
"""
import math

import numpy as np

from .core import register_path


# --- driven curves --------------------------------------------------------------------
@register_path("raster", knob="lines")
def raster(s, lines=48, flyback=0.1):
    """Unidirectional raster; `flyback` is the fraction of each line spent returning."""
    k = np.minimum((s * lines).astype(int), lines - 1)
    u = s * lines - k                                    # position within the line
    trace = 1.0 - flyback
    x = np.where(u < trace, -1 + 2 * u / trace, 1 - 2 * (u - trace) / max(flyback, 1e-12))
    y = -1 + (2 * k + 1) / lines
    return np.column_stack((x, y))


@register_path("spiral", knob="turns")
def spiral(s, turns=58, radius=1.0):
    """Archimedean spiral, centre outwards, constant area rate; covers a disk of `radius`."""
    r = np.sqrt(s)
    angle = 2 * np.pi * turns * r
    return radius * np.column_stack((r * np.cos(angle), r * np.sin(angle)))


@register_path("lissajous")
def lissajous(s, nx=30, ny=31, phase_deg=30.0):
    """x completes nx cycles and y completes ny cycles per frame.

    Keep nx, ny coprime. phase_deg=30 never retraces for coprime pairs; the common
    90 deg choice retraces itself (half the samples wasted) whenever nx is even.
    """
    x = np.sin(2 * np.pi * nx * s + np.deg2rad(phase_deg))
    y = np.sin(2 * np.pi * ny * s)
    return np.column_stack((x, y))


@register_path("closed_loops", knob="loops")
def closed_loops(s, loops=6, radius=1.0, axis_ratio=1.0, angle_deg=0.0):
    """The same circle (axis_ratio 1) or ellipse traced `loops` times."""
    theta = 2 * np.pi * loops * s
    xy = radius * np.column_stack((np.cos(theta), axis_ratio * np.sin(theta)))
    return _rotate_to_fit(xy, angle_deg)


@register_path("rose", knob="cycles")
def rose(s, order=3, cycles=3, radius=1.0, angle_deg=0.0):
    """Polar rose r = radius cos(order theta), traced `cycles` times ("flower").

    Odd order gives `order` petals, even order 2 x `order`. The same curve as Kalinin's
    k_rose (k = order, n_turns = cycles).
    """
    theta = 2 * np.pi * cycles * s
    r = radius * np.cos(order * theta)
    return _rotate_to_fit(r[:, None] * np.column_stack((np.cos(theta), np.sin(theta))),
                          angle_deg)


# --- shapes walked at constant speed --------------------------------------------------
def _walk(px, py, s):
    """Position at fraction s of a polyline's length: the shape walked at constant speed."""
    px, py = np.asarray(px, float), np.asarray(py, float)
    arc = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(px), np.diff(py)))])
    target = np.asarray(s) * arc[-1]
    return np.column_stack((np.interp(target, arc, px), np.interp(target, arc, py)))


def _serpentine_corners(lines, half=1.0):
    n = max(2, int(round(lines)))
    ys = np.linspace(-half, half, n)
    xs = np.ravel([(-half, half) if i % 2 == 0 else (half, -half) for i in range(n)])
    return xs, np.repeat(ys, 2)


def _rotate_to_fit(xy, angle_deg):
    """Rotate, then shrink just enough to stay inside the field.

    His notebook rotated without shrinking, so up to 18% of samples left the field at
    45 deg. Shrinking keeps every sample inside but leaves the corners unscanned; which
    behaviour we want is a team decision.
    """
    if not angle_deg:
        return xy
    c, s = math.cos(math.radians(angle_deg)), math.sin(math.radians(angle_deg))
    out = xy @ np.array([[c, s], [-s, c]])
    return out / max(1.0, np.abs(out).max())


@register_path("strokes", knob="strokes")
def strokes(s, strokes=8, radius=1.0, angle_deg=0.0):
    """Back and forth along one centred line, `strokes` passes: a 1-D reference path."""
    n = max(1, int(round(strokes)))
    px = radius * np.where(np.arange(n + 1) % 2 == 0, -1.0, 1.0)
    return _rotate_to_fit(_walk(px, np.zeros(n + 1), s), angle_deg)


@register_path("serpentine", knob="lines")
def serpentine(s, lines=62, angle_deg=0.0):
    """Bidirectional raster (no flyback), walked at constant speed."""
    return _rotate_to_fit(_walk(*_serpentine_corners(lines), s), angle_deg)


@register_path("radial", knob="spokes")
def radial(s, spokes=64, radius=1.0, angle_deg=0.0):
    """Spokes from the centre out to `radius` and back, walked at constant speed."""
    n = max(1, int(round(spokes)))
    a = 2 * np.pi * np.arange(n) / n + np.radians(angle_deg)
    px = radius * np.ravel(np.column_stack((0 * a, np.cos(a), 0 * a)))
    py = radius * np.ravel(np.column_stack((0 * a, np.sin(a), 0 * a)))
    return _walk(px, py, s)


@register_path("hybrid", knob="fine_lines")
def hybrid(s, fine_lines=74, roi_frac=0.4, coarse_lines=32):
    """Coarse serpentine over the whole field, then a fine one over a central ROI."""
    cx, cy = _serpentine_corners(coarse_lines)
    fx, fy = _serpentine_corners(fine_lines, roi_frac)
    return _walk(np.concatenate((cx, fx)), np.concatenate((cy, fy)), s)


@register_path("hilbert")
def hilbert(s, order=6):
    """Hilbert curve over a 2**order x 2**order grid of cells, walked at constant speed.

    His notebook's own construction (density 50 -> order 6 at his budget).
    """
    p = np.array([[0.5, 0.5]])
    for _ in range(max(1, int(round(order)))):
        p = p / 2
        p = np.concatenate([p[:, ::-1], p + [0, 0.5], p + [0.5, 0.5], [1, 0.5] - p[:, ::-1]])
    return _walk(*(p * 2 - 1).T, s)


# --- dwell paths -------------------------------------------------------------------------
def _dwell(sites, s, dwell_fraction):
    """Park at each site in turn, travelling between them at constant speed.

    ``dwell_fraction`` of the scan time is spent parked, shared equally between sites;
    the rest is travel, shared in proportion to hop length. Returns the (N, 2) positions
    and an (N,) mask that is True while parked.
    """
    if not 0 < dwell_fraction < 1:
        raise ValueError(f"dwell_fraction must be between 0 and 1, got {dwell_fraction}")
    hops = np.hypot(*np.diff(sites, axis=0).T)
    dwell = dwell_fraction / len(sites)
    travel = (1 - dwell_fraction) * hops / max(hops.sum(), 1e-12)
    starts = np.concatenate(([0.0], np.cumsum(dwell + travel)))     # when each park begins
    times = np.ravel(np.column_stack((starts, starts + dwell)))      # park start, park end
    knots = np.repeat(sites, 2, axis=0)
    xy = np.column_stack([np.interp(s, times, knots[:, i]) for i in (0, 1)])
    site = np.searchsorted(starts, s, side="right") - 1
    parked = s <= starts[site] + dwell + 1e-12
    return xy, parked


def _serpentine_order(xs, ys):
    """Grid sites row by row, alternating direction."""
    gx, gy = np.meshgrid(xs, ys)
    gx[1::2] = gx[1::2, ::-1]
    return np.column_stack((gx.ravel(), gy.ravel()))


@register_path("sparse")
def sparse(s, n_points=64, dwell_fraction=0.875, seed=0):
    """Park at jittered grid points (serpentine order), measuring all the time.

    His loop-prototype notebook spent about 87.5% of the time parked.
    """
    k = max(1, int(round(math.sqrt(n_points))))
    centres = (np.arange(k) + 0.5) / k * 2 - 1
    p = _serpentine_order(centres, centres)
    p += np.random.default_rng(seed).uniform(-0.35, 0.35, p.shape) * (2 / k)
    return _dwell(np.clip(p, -1, 1), s, dwell_fraction)[0]


@register_path("pulse_lattice")
def pulse_lattice(s, rows=6, columns=6, dwell_fraction=0.6, radius=1.0):
    """Measure only while parked on a regular rows x columns lattice; the beam is
    blanked during the transfers between sites (they are not measurements)."""
    if min(rows, columns) < 2:
        raise ValueError("pulse_lattice needs at least 2 rows and 2 columns")
    sites = _serpentine_order(np.linspace(-radius, radius, int(round(columns))),
                              np.linspace(-radius, radius, int(round(rows))))
    return _dwell(sites, s, dwell_fraction)


# --- pixel-list scans (programmable beam control) ----------------------------------------
def _pixel_centres(pixels):
    """(pixels,) centre coordinates of a square pixel grid in [-1, 1]."""
    n = int(round(pixels))
    if n < 2:
        raise ValueError(f"pixels must be at least 2, got {pixels}")
    return (np.arange(n) + 0.5) / n * 2 - 1


def _step_through(order, pixels, s):
    """Visit grid pixels in ``order`` (a list of (row, col)), an equal time on each.

    The beam jumps to the next pixel in one step, as a scan engine does when it plays a
    list of coordinates. Row 0 is the top of the field, so the scan starts top-left.
    """
    centres = _pixel_centres(pixels)
    order = np.asarray(order)
    if len(s) < len(order):
        raise ValueError(f"{len(order)} pixels need at least {len(order)} samples; "
                         f"the budget has {len(s)}")
    k = np.minimum((np.asarray(s) * len(order)).astype(int), len(order) - 1)
    rows, cols = order[k, 0], order[k, 1]
    return np.column_stack((centres[cols], -centres[rows]))


def _spiral_inward(n):
    """(row, col) of an n x n grid, clockwise from the top-left corner to the centre."""
    order, top, bottom, left, right = [], 0, n - 1, 0, n - 1
    while top <= bottom and left <= right:
        order += [(top, c) for c in range(left, right + 1)]
        order += [(r, right) for r in range(top + 1, bottom + 1)]
        if top < bottom:
            order += [(bottom, c) for c in range(right - 1, left - 1, -1)]
        if left < right:
            order += [(r, left) for r in range(bottom - 1, top, -1)]
        top, bottom, left, right = top + 1, bottom - 1, left + 1, right - 1
    return order


@register_path("grid_spiral", knob="pixels")
def grid_spiral(s, pixels=64, direction="in"):
    """Rectangular spiral over a pixels x pixels grid: "in" starts at the top-left corner
    and winds to the centre; "out" is the same spiral run backwards, centre to edge."""
    if direction not in ("in", "out"):
        raise ValueError(f"direction must be 'in' or 'out', got {direction!r}")
    order = _spiral_inward(len(_pixel_centres(pixels)))
    return _step_through(order if direction == "in" else order[::-1], pixels, s)


@register_path("n_sequential", knob="pixels")
def n_sequential(s, pixels=64, n=2):
    """N x N interleaved scan: each of the N*N passes visits every Nth pixel in each
    direction (row by row), so neighbouring pixels are measured in different passes."""
    size, n = len(_pixel_centres(pixels)), int(round(n))
    if not 1 <= n <= size:
        raise ValueError(f"n must be between 1 and pixels, got {n}")
    order = [(r, c) for dy in range(n) for dx in range(n)
             for r in range(dy, size, n) for c in range(dx, size, n)]
    return _step_through(order, pixels, s)


@register_path("random_order")
def random_order(s, pixels=64, seed=0):
    """Every pixel once, in a random order (the same order for the same seed)."""
    size = len(_pixel_centres(pixels))
    grid = [(r, c) for r in range(size) for c in range(size)]
    order = np.random.default_rng(seed).permutation(len(grid))
    return _step_through([grid[i] for i in order], pixels, s)
