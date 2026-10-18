"""Adapters: Kalinin's `instrument_transfer` pieces as scanpath plug-ins.

Importing this module registers, without changing either package:

* every path family in his catalog, as ``k_<name>`` (37 families),
* his SPM transfer function, as the instrument ``spm_hysteresis``,
* two of his metrics, ``k_distortion_roughness`` and ``k_coverage_mismatch``.

His generators draw a path in the unit box with their own sample count. The adapter
treats his sample index as time and resamples it onto the shared Budget, so every
family gets the same field of view, scan time and number of samples.

His package is used as a read-only reference: nothing here writes to it. Make it
importable in one of two ways before importing this module:

* put its ``src`` folder on ``sys.path`` (the walkthrough notebook does this), or
* set the environment variable ``SCANPATH_REFERENCE_SRC`` to that ``src`` folder.
"""
import functools
import os
import sys

import numpy as np

_reference = os.environ.get("SCANPATH_REFERENCE_SRC")
if _reference and _reference not in sys.path:
    sys.path.append(_reference)
_bytecode = sys.dont_write_bytecode
sys.dont_write_bytecode = True             # read the reference repo, never write into it
try:
    from instrument_transfer import metrics as _km
    from instrument_transfer.instruments import (build_spm_params_hysteresis_dominated,
                                                 spm_transfer_function)
    from instrument_transfer.trajectories import get_scan_path_catalog, list_scan_paths
except ImportError as err:                 # say what to do, not just what failed
    raise ImportError(
        "scanpath.kalinin needs Kalinin's instrument_transfer package "
        "(github.com/SergeiVKalinin/ScanPath). Add its src folder to sys.path, e.g.\n"
        "    sys.path.insert(0, r'C:\\...\\ScanPath\\src')\n"
        "or set the environment variable SCANPATH_REFERENCE_SRC to that folder.") from err
finally:
    sys.dont_write_bytecode = _bytecode

from .core import Trajectory, register_instrument, register_metric, register_path

_CATALOG = get_scan_path_catalog()

#: ``k_<name>`` -> Kalinin's path class ("0", "0+0", "1", "1+0", "2")
KALININ_CLASS = {}


def _wrap(generator):
    @functools.wraps(generator)               # describe() shows his parameters
    def path(s, **params):
        native, _line_ids = generator(**params)
        u = np.linspace(0.0, 1.0, len(native))
        xy = np.column_stack([np.interp(s, u, native[:, i]) for i in (0, 1)])
        return np.clip(2.0 * xy - 1.0, -1.0, 1.0)
    return path


for _name in list_scan_paths():
    register_path("k_" + _name)(_wrap(_CATALOG[_name]["generator"]))
    KALININ_CLASS["k_" + _name] = _CATALOG[_name]["class"]


@register_instrument("spm_hysteresis")
def spm_hysteresis(commanded, hysteresis_strength=2.0, other_distortion_scale=0.0, seed=0):
    """Kalinin's SPM scanner: affine + nonlinear + hysteresis + creep + drift.

    His model works in the unit box with one time step per sample (dt = 1).
    """
    fov = commanded.info["budget"].fov_nm
    params = build_spm_params_hysteresis_dominated(hysteresis_strength, other_distortion_scale)
    real = spm_transfer_function(commanded.xy / fov + 0.5, dt=1.0, params=params, seed=seed)
    return Trajectory(commanded.t, (real - 0.5) * fov)


def _unit(commanded, realized):
    fov = commanded.info["budget"].fov_nm
    return commanded.xy / fov + 0.5, realized.xy / fov + 0.5


@register_metric("k_distortion_roughness")
def k_distortion_roughness(commanded, realized, sample):
    """Non-affine local roughness of the displacement field (Kalinin)."""
    return _km.evaluate_smooth_distortion(*_unit(commanded, realized))["distortion_roughness_rms"]


@register_metric("k_coverage_mismatch")
def k_coverage_mismatch(commanded, realized, sample):
    """Smoothed coverage density, realized vs commanded (Kalinin)."""
    return _km.evaluate_smooth_coverage(*_unit(commanded, realized))["coverage_mismatch"]
