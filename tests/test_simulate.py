import copy
import numpy as onp
import pytest

from toliman_sim import simulate

SMALL_CONFIG = {
    "run_id": "test",
    "seed": 0,
    "run_type": "non_temporal",
    "exposure_time_s": 0.01,
    "output": {"format": "npz", "n_frames": 3},
    "optics": {"wf_npixels": 64, "psf_npixels": 32, "oversample": 2},
    "source": {
        "separation_arcsec": 2.0,
        "spectra": {
            "n_wavels": 20,
            "n_coarse": 4,
            "throughput": {"components": [{"tophat_nm": [530, 640]}]},
            "A": {"sed": "blackbody", "teff_k": 5790, "v_mag": 0.01},
            "B": {"sed": "blackbody", "teff_k": 5260, "v_mag": 1.33},
        },
    },
    "detector": {
        "layers": [
            {"name": "high_frequency_jitter", "kernel": "gaussian", "sigma_mas": 50.0},
            {"name": "pixel_response", "flat_sigma": 0.0},
            {"name": "dark_current_mean", "mean_e_per_pixel": 1.0},
        ]
    },
    "noise_sampling": {
        "sequence": [
            {"name": "shot_noise"},
            {"name": "read_noise", "sigma_electrons": 5.0},
        ]
    },
}


def test_shapes_and_flux():
    result = simulate(SMALL_CONFIG)
    assert result["frames"].shape == (3, 32, 32)
    assert result["expected"].shape == (32, 32)
    # Most of the detected photons land on the 32x32 detector, plus 1 e- of dark per pixel
    detected = float(result["telescope"].source.raw_fluxes.sum())
    assert 0.5 * detected < result["expected"].sum() < detected + 32 * 32


def test_seed_is_reproducible():
    a = simulate(SMALL_CONFIG)["frames"]
    b = simulate(SMALL_CONFIG)["frames"]
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
