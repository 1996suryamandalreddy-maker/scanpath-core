"""Stage 3 plug-ins: quality metrics.

Two kinds, declared when registering:

* **path metrics** ``f(commanded, realized, sample) -> float`` compare the commanded and
  realized trajectories, so they need a path-to-path instrument model. One that needs
  repeat runs (noise sensitivity) takes a fourth argument, ``rerun(seed)``.
* **image metrics** ``f(image, sample) -> float`` score the image the user would see
  against the ideal image. They work with *either* kind of instrument model.

Every metric is kept separate; combining them into one objective is an explicit choice
(see INTERFACES.md). To use a different evaluation function, pass a different metric
name to ``run`` / ``grid_search`` / ``gp_search``.

Groups: trajectory distortion · coverage · image quality · task (feature positions) ·
repeatability · coarse-stage switching · motion limits (speed, acceleration) ·
measurement budget (acquisitions, dose).

Measurements happen only at the samples marked in ``Trajectory.acquire``; coverage and
image metrics use those events, trajectory and motion metrics use every sample.
"""
import math

import numpy as np
from scipy.spatial import cKDTree

from .core import reconstruct, register_metric
from .instruments import move_points


def _error(commanded, realized):
    return np.linalg.norm(realized.xy - commanded.xy, axis=1)


def _gaps(realized, sample):
    """Distance from every pixel centre to the nearest realized measurement, nm."""
    return cKDTree(realized.events).query(sample.pixel_xy())[0]


def _nrmse(image, truth):
    return np.sqrt(np.mean((image - truth) ** 2)) / truth.std()


# --- trajectory distortion (no sample needed) -----------------------------------------
@register_metric("rms_error_nm")
def rms_error_nm(commanded, realized, sample):
    """RMS distance between commanded and realized positions."""
    return np.sqrt(np.mean(_error(commanded, realized) ** 2))


@register_metric("max_error_nm")
def max_error_nm(commanded, realized, sample):
    """Worst-case distance between commanded and realized positions."""
    return _error(commanded, realized).max()


@register_metric("local_misalignment_nm")
def local_misalignment_nm(commanded, realized, sample):
    """RMS left after the best overall shift and rotation (shape error), nm."""
    P = commanded.xy - commanded.xy.mean(0)
    Q = realized.xy - realized.xy.mean(0)
    U, _, Vt = np.linalg.svd(Q.T @ P)
    D = np.diag([1.0, np.sign(np.linalg.det(U @ Vt))])   # a rotation, never a mirror
    return float(np.sqrt(np.mean(np.sum((Q @ U @ D @ Vt - P) ** 2, axis=1))))


@register_metric("placement_error_p99_nm")
def placement_error_p99_nm(commanded, realized, sample):
    """99th percentile of |realized - commanded|: the worst-placed samples, nm."""
    return float(np.percentile(_error(commanded, realized), 99))


# --- coverage -------------------------------------------------------------------------
@register_metric("coverage", higher_is_better=True)
def coverage(commanded, realized, sample):
    """Fraction of image pixels with a realized measurement within one pixel pitch."""
    pitch = sample.fov_nm / sample.image.shape[1]
    return np.mean(_gaps(realized, sample) <= pitch)


@register_metric("max_gap_nm")
def max_gap_nm(commanded, realized, sample):
    """Largest distance from any pixel to the nearest realized measurement."""
    return _gaps(realized, sample).max()


# --- image quality --------------------------------------------------------------------
@register_metric("image_nrmse", kind="image")
def image_nrmse(image, sample):
    """Image error as the user sees it, relative to the ideal image's contrast.
    Includes sampling gaps and every distortion."""
    return _nrmse(image, sample.image)


@register_metric("image_correlation", higher_is_better=True, kind="image")
def image_correlation(image, sample):
    """Pearson correlation between the image the user sees and the ideal image."""
    return np.corrcoef(image.ravel(), sample.image.ravel())[0, 1]


@register_metric("image_nrmse_oracle")
def image_nrmse_oracle(commanded, realized, sample):
    """Image error if the true probe positions were known: sampling error only."""
    values = sample.values_at(realized.events)
    return _nrmse(reconstruct(realized.events, values, sample), sample.image)


# --- task: features in the right place ------------------------------------------------
@register_metric("feature_position_error_nm")
def feature_position_error_nm(commanded, realized, sample):
    """How far each known feature appears from where a perfect scan would put it, nm.

    Needs ``sample.features``; features the path never visits are skipped (NaN if none).
    """
    if sample.features is None:
        return float("nan")
    size, errs = sample.feature_size_nm, []
    told, really = commanded.events, realized.events
    for f in sample.features:
        w0 = np.exp(-np.sum((told - f) ** 2, axis=1) / (2 * size**2))    # seen here
        w1 = np.exp(-np.sum((really - f) ** 2, axis=1) / (2 * size**2))  # really seen
        if w0.sum() > 1e-3 and w1.sum() > 1e-3:
            a0 = w0 @ told / w0.sum()
            a1 = w1 @ told / w1.sum()               # ...but recorded at the commanded spot
            errs.append(np.hypot(*(a1 - a0)))
    return float(np.mean(errs)) if errs else float("nan")


# --- repeatability (needs repeat runs) ------------------------------------------------
@register_metric("noise_sensitivity_nm")
def noise_sensitivity_nm(commanded, realized, sample, rerun):
    """RMS change in the realized path between repeat runs of the same command, nm."""
    return float(np.mean([np.sqrt(np.mean(_error(realized, rerun(s)) ** 2)) for s in (101, 102)]))


# --- coarse-stage switching -----------------------------------------------------------
@register_metric("stage_moves")
def stage_moves_count(commanded, realized, sample):
    """Coarse stage moves the path needs (reported by the instrument, or a 5 µm window)."""
    if "stage_moves" in realized.info:
        return float(realized.info["stage_moves"])
    return float(move_points(commanded.xy, 5000.0).size)


@register_metric("switch_confidence", higher_is_better=True)
def switch_confidence(commanded, realized, sample):
    """Chance that no coarse move lands more than 100 nm off target: (1 - p_bad)^moves."""
    sd = realized.info.get("landing_sd_nm", 50.0)
    p_bad = math.exp(-100.0**2 / (2 * sd**2))
    return float((1 - p_bad) ** stage_moves_count(commanded, realized, sample))


# --- comparability and motion limits (report, or use as constraints) ------------------
@register_metric("path_length_um")
def path_length_um(commanded, realized, sample):
    """Commanded path length, micrometres."""
    return np.linalg.norm(np.diff(commanded.xy, axis=0), axis=1).sum() / 1000


def _velocity(path):
    """(N-1, 2) velocity between samples, nm/s."""
    return np.diff(path.xy, axis=0) / np.diff(path.t)[:, None]


def _acceleration(path):
    """(N-2, 2) acceleration at interior samples, nm/s^2."""
    dt = np.diff(path.t)
    return np.diff(_velocity(path), axis=0) / ((dt[:-1] + dt[1:]) / 2)[:, None]


def _p99_um(vectors):
    return float(np.percentile(np.linalg.norm(vectors, axis=1), 99)) / 1000


def _peak_um(vectors):
    return float(np.linalg.norm(vectors, axis=1).max()) / 1000


@register_metric("speed_p99_um_s")
def speed_p99_um_s(commanded, realized, sample):
    """99th-percentile commanded speed, um/s (ignores single-sample line jumps)."""
    return _p99_um(_velocity(commanded))


@register_metric("speed_peak_um_s")
def speed_peak_um_s(commanded, realized, sample):
    """Largest commanded speed between samples, um/s (includes line jumps)."""
    return _peak_um(_velocity(commanded))


@register_metric("accel_peak_um_s2")
def accel_peak_um_s2(commanded, realized, sample):
    """Largest commanded acceleration between samples, um/s^2: how hard the command
    drives the scanner at its sharpest corner (a sample-to-sample figure)."""
    return _peak_um(_acceleration(commanded))


@register_metric("realized_speed_peak_um_s")
def realized_speed_peak_um_s(commanded, realized, sample):
    """Largest speed the probe really reaches, um/s."""
    return _peak_um(_velocity(realized))


@register_metric("realized_accel_peak_um_s2")
def realized_accel_peak_um_s2(commanded, realized, sample):
    """Largest acceleration the probe really undergoes, um/s^2."""
    return _peak_um(_acceleration(realized))


# --- measurement budget (equal by construction when Budget sets them) -----------------
@register_metric("acquisitions")
def acquisitions(commanded, realized, sample):
    """Number of measurement events."""
    return float(realized.acquire.sum())


@register_metric("total_dose")
def total_dose(commanded, realized, sample):
    """Total dose delivered over the scan (dose units of the Budget)."""
    return float(realized.dose.sum())
