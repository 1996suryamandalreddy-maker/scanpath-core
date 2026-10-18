"""Stage 2 plug-ins: instrument models, in two clearly separate cases.

CASE A -- path-to-path models   ``f(commanded, **params) -> realized Trajectory``
    What the instrument does to the probe's motion. The image is then formed by reading
    the signal where the probe really was and drawing it where it was told to be
    (``core.image_as_acquired``). Every path metric and every image metric applies.

    * scanner response: ideal, lowpass (first-order lag, exact), resonant (mass-spring)
    * sample and stage effects: drift, noise, stage_moves (coarse stage + fine piezo
      window), creep
    * composition: chain (several models in order, e.g. lowpass -> stage_moves ->
      drift -> noise, the loop-prototype instrument)

CASE B -- image-to-image models   ``f(sample, commanded, **params) -> image``
    Go straight from the ideal image to the image the user sees, without modelling a
    realized trajectory. The commanded path still says where and when each part of the
    image is scanned, so the result depends on the path. Only image metrics apply.

    * sampled (no distortion), drift_warp, line_jitter, probe_blur, detector_noise,
      affine (calibration error)
    * from_path: any case-A model, expressed as an image model -- the bridge between
      the two cases

Random models take a ``seed`` and repeat exactly for the same seed. Whatever a case-A
model reports (e.g. the number of stage moves) goes in ``realized.info``.
"""
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import bilinear, lfilter, lfilter_zi

from .core import (Trajectory, image_as_acquired, reconstruct, register_image_model,
                   register_instrument, simulate)


def _apply(b, a, commanded, delay_s):
    dt = commanded.t[1] - commanded.t[0]
    x = commanded.xy
    shift = int(round(delay_s / dt))
    if shift:
        x = np.vstack((np.repeat(x[:1], shift, axis=0), x[:-shift]))
    zi = lfilter_zi(b, a)
    out = np.column_stack([lfilter(b, a, x[:, i], zi=zi * x[0, i])[0] for i in range(2)])
    return Trajectory(commanded.t, out)


# ======================================================================================
# CASE A: path-to-path models
# ======================================================================================
# --- scanner response -----------------------------------------------------------------
@register_instrument("ideal")
def ideal(commanded):
    """Perfect tracking: realized = commanded."""
    return Trajectory(commanded.t, commanded.xy.copy())


@register_instrument("lowpass")
def lowpass(commanded, bandwidth_hz=1500.0, delay_s=0.0, hold="linear"):
    """First-order scanner (-3 dB at bandwidth_hz, time constant tau = 1 / 2 pi
    bandwidth) plus a pure delay: lag and rounded turns.

    The response is exact, not an approximation that improves with finer sampling:
    hold="linear" for a command that moves in straight lines between samples (the path
    as drawn; a ramp lags by exactly tau), hold="step" for a command held constant
    between samples, as from a DAC (a ramp lags by tau + dt / 2).
    """
    dt = commanded.t[1] - commanded.t[0]
    tau = 1.0 / (2 * np.pi * bandwidth_hz)
    alpha = np.exp(-dt / tau)
    if hold == "linear":                    # exact for piecewise-linear input
        g = 1.0 - tau * (1.0 - alpha) / dt
        b = [g, 1.0 - alpha - g]
    elif hold == "step":                    # exact for zero-order-hold input
        b = [0.0, 1.0 - alpha]
    else:
        raise ValueError(f"hold must be 'linear' or 'step', got {hold!r}")
    return _apply(b, [1.0, -alpha], commanded, delay_s)


@register_instrument("resonant")
def resonant(commanded, resonance_hz=1500.0, damping=0.15, delay_s=0.0):
    """Second-order (mass-spring) scanner, e.g. a piezo tube: overshoot and ringing."""
    w0 = 2 * np.pi * resonance_hz
    dt = commanded.t[1] - commanded.t[0]
    b, a = bilinear([w0**2], [1, 2 * damping * w0, w0**2], fs=1 / dt)
    return _apply(b, a, commanded, delay_s)


# --- sample and stage effects ---------------------------------------------------------
@register_instrument("drift")
def drift(commanded, vx_nm_s=2.0, vy_nm_s=1.0):
    """Linear drift of the sample under the probe."""
    t = commanded.t[:, None] - commanded.t[0]
    return Trajectory(commanded.t, commanded.xy + t * [vx_nm_s, vy_nm_s])


@register_instrument("noise")
def noise(commanded, sd_nm=2.0, seed=0):
    """White position noise."""
    rng = np.random.default_rng(seed)
    return Trajectory(commanded.t, commanded.xy + rng.normal(0, sd_nm, commanded.xy.shape))


def move_points(xy, r_fine_nm):
    """Indices where the path leaves the fine (piezo) window, so the coarse stage moves."""
    idx, lo, hi = [], xy[0].copy(), xy[0].copy()
    for i in range(1, len(xy)):
        lo, hi = np.minimum(lo, xy[i]), np.maximum(hi, xy[i])
        if (hi - lo).max() > 2 * r_fine_nm:
            idx.append(i)
            lo, hi = xy[i].copy(), xy[i].copy()
    return np.array(idx, int)


@register_instrument("stage_moves")
def stage_moves(commanded, r_fine_nm=5000.0, landing_sd_nm=50.0, seed=0):
    """Coarse stage + fine piezo: each coarse move lands with a random offset that stays.

    Reports the number of coarse moves in ``info["stage_moves"]``.
    """
    rng = np.random.default_rng(seed)
    idx = move_points(commanded.xy, r_fine_nm)
    landing = np.vstack(([0.0, 0.0], rng.normal(0, landing_sd_nm, (idx.size, 2))))
    offset = landing[np.searchsorted(idx, np.arange(len(commanded.xy)), side="right")]
    return Trajectory(commanded.t, commanded.xy + offset,
                      {"stage_moves": int(idx.size), "landing_sd_nm": landing_sd_nm})


@register_instrument("creep")
def creep(commanded, amount=0.01, tau_s=10.0):
    """Piezo creep: a slow extra extension, a fraction of the distance from the start."""
    g = amount * (1 - np.exp(-(commanded.t - commanded.t[0]) / tau_s))[:, None]
    return Trajectory(commanded.t, commanded.xy + g * (commanded.xy - commanded.xy[0]))


# --- composition ----------------------------------------------------------------------
@register_instrument("chain")
def chain(commanded, stages=(), seed=0):
    """Several models applied in order, e.g. [("lowpass", {...}), ("drift", {...})].

    Each stage is any registered instrument, so every model is also a building block.
    Random stages get their own seed (seed, seed + 1, ...); what each stage reports in
    ``info`` is collected.
    """
    current, info = commanded, {}
    for i, stage in enumerate(stages):
        name, params = (stage, {}) if isinstance(stage, str) else stage
        current = simulate(current, name, seed + i, **params)
        info.update({k: v for k, v in current.info.items()
                     if k not in commanded.info and k not in ("instrument", "instrument_params")})
    info["stages"] = [s if isinstance(s, str) else s[0] for s in stages]
    return Trajectory(commanded.t, current.xy, info)


# ======================================================================================
# CASE B: image-to-image models
# ======================================================================================
def _scan(sample, commanded, read_at=None, image=None, noise_sd=0.0, rng=None):
    """Read an image at the commanded measurement events and rebuild it there.

    read_at: (N, 2) positions where the signal is really read, one per sample (default:
    the commanded positions); only the measurement events are used.
    image: the image being read (default: the ideal one).
    """
    source = sample if image is None else type(sample)(image, sample.fov_nm)
    read_at = commanded.xy if read_at is None else read_at
    values = source.values_at(read_at[commanded.acquire])
    if noise_sd:
        values = values + rng.normal(0.0, noise_sd, values.shape)
    return reconstruct(commanded.events, values, sample)


@register_image_model("sampled")
def sampled(sample, commanded):
    """No distortion: the ideal image as sampled along the commanded path."""
    return _scan(sample, commanded)


@register_image_model("from_path")
def from_path(sample, commanded, path_model="lowpass", path_params=None, seed=None):
    """Any path-to-path model (case A) used as an image model: realize the path, then
    form the image. Gives exactly the image that the case-A route produces."""
    realized = simulate(commanded, path_model, seed, **(path_params or {}))
    return image_as_acquired(commanded, realized, sample)


@register_image_model("drift_warp")
def drift_warp(sample, commanded, vx_nm_s=2.0, vy_nm_s=1.0):
    """The sample drifts under the probe: content scanned at time t comes from v * t
    further along. A raster turns this into a shear; a spiral into a swirl."""
    t = (commanded.t - commanded.t[0])[:, None]
    return _scan(sample, commanded, read_at=commanded.xy + t * [vx_nm_s, vy_nm_s])


@register_image_model("line_jitter")
def line_jitter(sample, commanded, sd_nm=1.0, segments=48, seed=0):
    """Each of ``segments`` equal slices of scan time (lines, for a raster) is shifted
    sideways by its own random offset, as with line-to-line jitter in SPM images."""
    rng = np.random.default_rng(seed)
    offsets = rng.normal(0.0, sd_nm, segments)
    s = (commanded.t - commanded.t[0]) / (commanded.t[-1] - commanded.t[0] + 1e-300)
    which = np.minimum((s * segments).astype(int), segments - 1)
    read_at = commanded.xy + np.column_stack((offsets[which], np.zeros(len(which))))
    return _scan(sample, commanded, read_at=read_at)


@register_image_model("probe_blur")
def probe_blur(sample, commanded, sigma_nm=1.0):
    """A finite probe: the ideal image blurred by a Gaussian of width sigma_nm."""
    pixel_nm = sample.fov_nm / sample.image.shape[1]
    return _scan(sample, commanded, image=gaussian_filter(sample.image, sigma_nm / pixel_nm))


@register_image_model("detector_noise")
def detector_noise(sample, commanded, snr=20.0, seed=0):
    """White noise on every reading, at signal-to-noise ratio ``snr`` (image std / noise)."""
    return _scan(sample, commanded, noise_sd=sample.image.std() / snr,
                 rng=np.random.default_rng(seed))


@register_image_model("affine")
def affine(sample, commanded, scale_x=1.0, scale_y=1.0, shear=0.0, rotation_deg=0.0):
    """Scanner calibration error: the probe reads at A @ commanded position."""
    c, s = np.cos(np.radians(rotation_deg)), np.sin(np.radians(rotation_deg))
    A = np.array([[c, -s], [s, c]]) @ np.array([[scale_x, shear], [0.0, scale_y]])
    return _scan(sample, commanded, read_at=commanded.xy @ A.T)
