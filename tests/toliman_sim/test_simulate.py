import copy
import numpy as onp
import pytest

from toliman_sim import simulate

from .small_config import SMALL_CONFIG


def test_shapes_and_flux():
    result = simulate(SMALL_CONFIG)
    assert result["data"].shape == (3, 32, 32)
    assert result["expected"].shape == (3, 32, 32)
    # Most of the detected photons land on the 32x32 detector, plus 1 e- of dark per pixel
    detected = float(result["telescope"].source.raw_fluxes.sum())
    assert 0.5 * detected < result["expected"][0].sum() < detected + 32 * 32


def test_seed_is_reproducible():
    a = simulate(SMALL_CONFIG)["data"]
    b = simulate(SMALL_CONFIG)["data"]
    onp.testing.assert_array_equal(a, b)


def test_unimplemented_effect_raises():
    config = copy.deepcopy(SMALL_CONFIG)
    config["miscellaneous_effects"] = {"ghosts": {"enabled": True}}
    with pytest.raises(NotImplementedError):
        simulate(config)


def test_coefficient_count_checked():
    config = copy.deepcopy(SMALL_CONFIG)
    config["optics"].update(radial_orders=[2], coefficients_m=[1e-9, 1e-9])
    with pytest.raises(ValueError, match="3 Zernikes"):
        simulate(config)


def test_parameter_draws_and_truth():
    config = copy.deepcopy(SMALL_CONFIG)
    config["observations"] = {
        "n_observations": 4,
        "parameters": {
            "separation_arcsec": {"std": 0.1},
            "x_position_arcsec": {"mean": 1.0, "std": 0.0},
            "position_angle_deg": {"distribution": "uniform", "low": 0, "high": 360},
        },
    }
    result = simulate(config)
    truth = result["truth"]
    assert truth["separation_arcsec"].shape == (4,)
    assert onp.std(truth["separation_arcsec"]) > 0  # drawn
    onp.testing.assert_allclose(truth["x_position_arcsec"], 1.0)  # mean overrides config
    onp.testing.assert_allclose(truth["y_position_arcsec"], 0.0)  # not drawn: recorded
    assert ((0 <= truth["position_angle_deg"]) & (truth["position_angle_deg"] < 360)).all()
    # Different parameters give different expected images
    assert not onp.allclose(result["expected"][0], result["expected"][1])


def test_noise_flags():
    config = copy.deepcopy(SMALL_CONFIG)
    config["noise"] = {"poisson": False, "read_noise": False}
    result = simulate(config)
    onp.testing.assert_array_equal(result["data"], result["expected"])

    config["noise"] = {"poisson": False, "read_noise": True, "read_noise_electrons": 3.0}
    residual = (lambda r: r["data"] - r["expected"])(simulate(config))
    assert abs(residual.std() - 3.0) < 0.2

    # Poisson only: integer counts with variance equal to the mean
    config["noise"] = {"poisson": True, "read_noise": False}
    config["observations"] = {"n_observations": 200}
    result = simulate(config)
    onp.testing.assert_array_equal(result["data"], onp.round(result["data"]))
    bright = result["expected"][0] > 100
    variance = result["data"][:, bright].var(0)
    ratio = (variance / result["expected"][0][bright]).mean()
    assert abs(ratio - 1) < 0.05


def test_writes_npy(tmp_path):
    from toliman_sim.io import write_outputs

    result = simulate(SMALL_CONFIG)
    config = copy.deepcopy(SMALL_CONFIG) | {"output": {"directory": str(tmp_path)}}
    write_outputs(result, config, {"pixel_scale_arcsec": 0.375})
    data = onp.load(tmp_path / "test_data.npy")
    truth = onp.load(tmp_path / "test_truth.npz")
    assert data.shape == (3, 32, 32)
    assert truth["separation_arcsec"].shape == (3,)
