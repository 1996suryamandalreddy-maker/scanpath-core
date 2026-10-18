"""Contract tests: every registered plug-in -- including ones added later -- is checked
automatically. Run with `python -m pytest tests` or `python tests/test_contract.py`.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import scanpath as sp  # noqa: E402

BUDGET = sp.Budget(fov_nm=50, duration_s=0.1, rate_hz=50_000)   # 5,000 samples
SAMPLE = sp.Sample.synthetic(BUDGET.fov_nm, size=48, features=15, seed=1)


def test_every_path_fits_the_budget():
    for family in sp.PATHS:
        path = sp.make_path(family, BUDGET)
        assert path.xy.shape == (BUDGET.n, 2), family
        assert np.abs(path.xy).max() <= BUDGET.fov_nm / 2 + 1e-9, family
        assert np.allclose(path.t, BUDGET.t), family


def test_every_instrument_keeps_time_axis_and_starts_settled():
    """A parked beam stays (nearly) where it was parked: no model may fling it away.

    Drift and noise move it a little by design, so the check is a tolerance, not equality.
    """
    parked = sp.make_path("raster", BUDGET)
    parked = sp.Trajectory(parked.t, np.tile([[3.0, -2.0]], (BUDGET.n, 1)), parked.info)
    for model in sp.INSTRUMENTS:
        realized = sp.simulate(parked, model)
        assert np.array_equal(realized.t, parked.t), model
        assert np.abs(realized.xy - parked.xy).max() < 0.25 * BUDGET.fov_nm, model


def test_random_instruments_repeat_with_the_same_seed():
    path = sp.make_path("lissajous", BUDGET)
    for model in ("noise", "stage_moves"):
        a, b = sp.simulate(path, model, seed=3), sp.simulate(path, model, seed=3)
        c = sp.simulate(path, model, seed=4)
        assert np.array_equal(a.xy, b.xy), model
        assert not np.array_equal(a.xy, c.xy) or model == "stage_moves", model


def test_chain_applies_stages_in_order_and_keeps_their_reports():
    path = sp.make_path("serpentine", BUDGET)
    stages = [("lowpass", {"bandwidth_hz": 1000}), ("stage_moves", {"r_fine_nm": 10.0})]
    r = sp.simulate(path, "chain", seed=0, stages=stages)
    assert r.info["stages"] == ["lowpass", "stage_moves"]
    assert r.info["stage_moves"] > 0


def test_length_budget_matches_every_family_with_a_knob():
    budget = sp.Budget(fov_nm=BUDGET.fov_nm, duration_s=BUDGET.duration_s,
                       rate_hz=BUDGET.rate_hz, length_um=2.0)
    for family, fn in sp.PATHS.items():
        if getattr(fn, "knob", None):
            length = sp.run(family, "ideal", ["path_length_um"], budget, SAMPLE).scores["path_length_um"]
            assert abs(length / 2.0 - 1) < 0.05, (family, length)


def test_ideal_instrument_has_zero_trajectory_error():
    result = sp.run("lissajous", "ideal", ["rms_error_nm", "max_error_nm"], BUDGET, SAMPLE)
    assert result.scores == {"rms_error_nm": 0.0, "max_error_nm": 0.0}


def test_every_metric_returns_a_finite_number():
    for family in ("raster", "lissajous", "serpentine", "hybrid", "hilbert"):
        scores = sp.run(family, "lowpass", None, BUDGET, SAMPLE).scores
        assert set(scores) == set(sp.METRICS)
        assert all(np.isfinite(v) for v in scores.values()), (family, scores)


def test_paired_seed_grid_search_reports_change_against_baseline():
    noisy = ("chain", {"stages": [("noise", {"sd_nm": 1.0})]})
    rows, best = sp.grid_search("lissajous", {"nx": [5, 7, 9]}, "rms_error_nm", instrument=noisy,
                                budget=BUDGET, sample=SAMPLE, seeds=range(4))
    assert rows[0]["change"] == 0 and "accepted" in best
    assert all("change_sem" in r and "significant" in r for r in rows)


def test_plugins_can_be_added_without_editing_the_package():
    @sp.register_path("_test_circle")
    def circle(s, radius=0.5):
        return radius * np.column_stack((np.cos(2 * np.pi * s), np.sin(2 * np.pi * s)))

    @sp.register_instrument("_test_offset")
    def offset(commanded, dx_nm=1.0):
        return sp.Trajectory(commanded.t, commanded.xy + [dx_nm, 0.0])

    @sp.register_metric("_test_mean_x")
    def mean_x(commanded, realized, sample):
        return realized.xy[:, 0].mean()

    try:
        r = sp.run(("_test_circle", {"radius": 0.4}), ("_test_offset", {"dx_nm": 2.0}),
                   ["_test_mean_x", "rms_error_nm"], BUDGET, SAMPLE)
        assert abs(r.scores["_test_mean_x"] - 2.0) < 1e-6
        assert abs(r.scores["rms_error_nm"] - 2.0) < 1e-9
    finally:
        for registry, key in ((sp.PATHS, "_test_circle"), (sp.INSTRUMENTS, "_test_offset"),
                              (sp.METRICS, "_test_mean_x")):
            registry.pop(key, None)


def test_grid_search_picks_the_best_row():
    rows, best = sp.grid_search("lissajous", {"nx": [3, 7, 11], "ny": [8]}, "coverage",
                                budget=BUDGET, sample=SAMPLE)
    assert len(rows) == 3 and best["coverage"] == max(r["coverage"] for r in rows)


def test_unknown_names_fail_loudly():
    for call in (lambda: sp.make_path("nope", BUDGET),
                 lambda: sp.run("raster", "nope", None, BUDGET, SAMPLE),
                 lambda: sp.run("raster", "ideal", ["nope"], BUDGET, SAMPLE)):
        try:
            call()
        except KeyError:
            continue
        raise AssertionError("unknown plug-in name was accepted")


# --- case B: image-to-image models, metric kinds, parameters, GP search -------------------

def test_every_image_model_returns_a_finite_image_of_the_sample_shape():
    path = sp.make_path("raster", BUDGET)
    for model in sp.IMAGE_MODELS:
        image = sp.make_image(SAMPLE, path, model, seed=0)
        assert image.shape == SAMPLE.image.shape and np.isfinite(image).all(), model


def test_instrument_and_image_model_names_never_clash():
    assert not set(sp.INSTRUMENTS) & set(sp.IMAGE_MODELS)


def test_from_path_image_model_equals_the_path_to_path_route():
    path = sp.make_path("raster", BUDGET)
    realized = sp.simulate(path, "lowpass", bandwidth_hz=1500)
    case_a = sp.image_as_acquired(path, realized, SAMPLE)
    case_b = sp.make_image(SAMPLE, path, "from_path", path_model="lowpass",
                           path_params={"bandwidth_hz": 1500})
    assert np.allclose(case_a, case_b)


def test_run_with_an_image_model_reports_only_image_metrics():
    r = sp.run("raster", ("drift_warp", {"vx_nm_s": 5.0}), None, BUDGET, SAMPLE)
    assert r.realized is None and r.image.shape == SAMPLE.image.shape
    assert set(r.scores) == {m for m, f in sp.METRICS.items() if f.kind == "image"}


def test_path_metric_with_an_image_model_fails_with_a_clear_message():
    try:
        sp.run("raster", "probe_blur", ["rms_error_nm"], BUDGET, SAMPLE)
    except ValueError as err:
        assert "path-to-path" in str(err)
    else:
        raise AssertionError("path metric was accepted without a realized path")


def test_image_metrics_work_with_both_kinds_of_model():
    for model in ("lowpass", "drift_warp"):
        r = sp.run("lissajous", model, ["image_nrmse", "image_correlation"], BUDGET, SAMPLE)
        assert all(np.isfinite(v) for v in r.scores.values()), model


def test_parameters_lists_the_keyword_defaults_of_any_component():
    assert sp.parameters("raster") == {"lines": 48, "flyback": 0.1}
    assert "bandwidth_hz" in sp.parameters("lowpass")
    assert "vx_nm_s" in sp.parameters("drift_warp")


def test_gp_search_stays_in_bounds_and_respects_integer_parameters():
    rows, best = sp.gp_search("raster", {"lines": (8, 40, "int"), "flyback": (0.05, 0.6)},
                              "image_nrmse", instrument="lowpass", n_init=4, n_evals=8,
                              budget=BUDGET, sample=SAMPLE)
    assert len(rows) == 8
    for r in rows:
        assert isinstance(r["lines"], int) and 8 <= r["lines"] <= 40
        assert 0.05 <= r["flyback"] <= 0.6
    assert best["image_nrmse"] == min(r["image_nrmse"] for r in rows)


# --- measurement events, dose, exact lag, constraints and the search family -------------
# (several of these port the contract checks of the workflow notebooks)

def test_trajectory_rejects_bad_input_and_cannot_be_changed():
    t = np.arange(4.0)
    for bad in (dict(t=[0, 1, 1, 2], xy=np.zeros((4, 2))),          # time not increasing
                dict(t=t, xy=[[0, 0], [np.nan, 0], [0, 0], [0, 0]]),  # not finite
                dict(t=t, xy=np.zeros((3, 2))),                       # wrong shape
                dict(t=t, xy=np.zeros((4, 2)), acquire=[1, 0, 0, 0], dose=[1, 1, 0, 0])):
        try:
            sp.Trajectory(**bad)
        except ValueError:
            continue
        raise AssertionError(f"accepted {bad}")
    path = sp.Trajectory(t, np.zeros((4, 2)))
    try:
        path.xy[0, 0] = 1.0
    except ValueError:
        return
    raise AssertionError("trajectory arrays can be modified")


def test_budget_gives_every_family_the_same_events_and_dose():
    budget = sp.Budget(fov_nm=BUDGET.fov_nm, duration_s=BUDGET.duration_s,
                       rate_hz=BUDGET.rate_hz, acquisitions=200, dose=1.0)
    for family in sp.PATHS:
        scores = sp.run(family, "lowpass", ["acquisitions", "total_dose"], budget, SAMPLE).scores
        assert scores["acquisitions"] == 200, family
        assert abs(scores["total_dose"] - 1.0) < 1e-12, family


def test_pulse_lattice_measures_only_while_parked():
    raw = sp.make_path("pulse_lattice", BUDGET)
    budgeted = sp.make_path("pulse_lattice", sp.Budget(BUDGET.fov_nm, BUDGET.duration_s,
                                                       BUDGET.rate_hz, acquisitions=100))
    assert 0 < raw.acquire.mean() < 1
    assert np.all(raw.acquire[budgeted.acquire])          # events only at parked samples
    both = raw.acquire[1:] & raw.acquire[:-1]             # consecutive measurements...
    assert np.all(raw.xy[1:][both] == raw.xy[:-1][both])  # ...never move the probe


def test_instruments_keep_the_measurement_events():
    path = sp.make_path("pulse_lattice", BUDGET)
    for model in sp.INSTRUMENTS:
        realized = sp.simulate(path, model)
        assert np.array_equal(realized.acquire, path.acquire), model
        assert np.array_equal(realized.dose, path.dose), model


def test_lowpass_is_the_exact_first_order_response():
    tau, t = 0.1, np.linspace(0.0, 1.0, 101)
    ramp = sp.Trajectory(t, np.column_stack((t, 0 * t)))
    bandwidth = 1 / (2 * np.pi * tau)
    linear = sp.simulate(ramp, "lowpass", bandwidth_hz=bandwidth).xy[:, 0]
    np.testing.assert_allclose(linear, t - tau * (1 - np.exp(-t / tau)), atol=1e-13)
    step = sp.simulate(ramp, "lowpass", bandwidth_hz=bandwidth, hold="step").xy[:, 0]
    assert abs((t - step)[-1] - (tau + 0.005)) < 1e-4    # DAC hold lags half a sample more


def test_every_search_spends_its_budget_on_distinct_points_inside_the_space():
    space = {"lines": (8, 40, "int"), "flyback": (0.05, 0.6)}
    for name, search in sp.SEARCHES.items():
        options = {"levels": 3} if name == "grid" else {"n_evals": 9}
        rows, best = search("raster", space, "image_nrmse", "lowpass", budget=BUDGET,
                            sample=SAMPLE, **options)
        assert len(rows) == 9, name
        assert len({(r["lines"], r["flyback"]) for r in rows}) == 9, name
        assert all(8 <= r["lines"] <= 40 and 0.05 <= r["flyback"] <= 0.6 for r in rows), name
        assert best["image_nrmse"] == min(r["image_nrmse"] for r in rows), name


def test_constraints_decide_feasibility_and_the_best_row():
    speed = sp.limits(at_most={"speed_p99_um_s": 1e-6})       # nothing can be this slow
    rows, best = sp.grid_search("lissajous", {"nx": [5, 7]}, "coverage", budget=BUDGET,
                                sample=SAMPLE, constraints=speed)
    assert not any(r["feasible"] for r in rows) and best["feasible"] is False
    assert best["violation"] == min(r["violation"] for r in rows)
    loose = sp.limits(at_least={"coverage": 1e-6})
    rows, best = sp.grid_search("lissajous", {"nx": [5, 7]}, "coverage", budget=BUDGET,
                                sample=SAMPLE, constraints=loose)
    assert all(r["feasible"] for r in rows) and best["coverage"] == max(r["coverage"] for r in rows)


def test_a_function_can_be_the_score():
    loss = lambda m: m["rms_error_nm"] + (1 - m["coverage"])
    rows, best = sp.random_search("lissajous", {"nx": (4, 12, "int")}, loss, "lowpass",
                                  metrics=["rms_error_nm", "coverage"], n_evals=5,
                                  budget=BUDGET, sample=SAMPLE)
    assert all(abs(r["loss"] - loss(r)) < 1e-12 for r in rows)
    assert best["loss"] == min(r["loss"] for r in rows)


def test_pareto_front_keeps_only_undominated_rows():
    rows = [{"rms_error_nm": 1, "coverage": 0.5}, {"rms_error_nm": 2, "coverage": 0.9},
            {"rms_error_nm": 2, "coverage": 0.4}, {"rms_error_nm": 0.5, "coverage": 0.3}]
    front = sp.pareto_front(rows, ["rms_error_nm", "coverage"])
    assert rows[2] not in front and len(front) == 3


def test_pixel_list_scans_visit_every_pixel_once_in_the_published_order():
    """Orders from Palos et al. 2026, Fig. 1b (6 x 6 grid, numbered in visiting order)."""
    budget = sp.Budget(fov_nm=6, duration_s=1.0, rate_hz=36)          # one sample per pixel

    def visiting_order(family, **params):
        xy = sp.make_path(family, budget, pixels=6, **params).xy
        col, row = np.round(xy[:, 0] + 2.5).astype(int), np.round(2.5 - xy[:, 1]).astype(int)
        grid = np.zeros((6, 6), int)
        grid[row, col] = np.arange(1, 37)
        return grid

    assert visiting_order("n_sequential", n=2)[:2].tolist() == [[1, 10, 2, 11, 3, 12],
                                                                [19, 28, 20, 29, 21, 30]]
    assert visiting_order("grid_spiral", direction="in")[:3].tolist() == [
        [1, 2, 3, 4, 5, 6], [20, 21, 22, 23, 24, 7], [19, 32, 33, 34, 25, 8]]
    assert visiting_order("grid_spiral", direction="out")[3, 2] == 1     # starts at the centre
    for family in ("grid_spiral", "n_sequential", "random_order"):
        assert sorted(visiting_order(family).ravel()) == list(range(1, 37)), family


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for test in tests:
        test()
        print(f"ok  {test.__name__}")
    print(f"{len(tests)} passed")
