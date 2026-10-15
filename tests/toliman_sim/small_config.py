"""A tiny, fast simulation config shared by the toliman_sim tests."""

SMALL_CONFIG = {
    "run_id": "test",
    "seed": 0,
    "exposure_time_s": 0.01,
    "output": {"format": "npy"},
    "observations": {"n_observations": 3},
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
            {"name": "dark_current", "rate_e_per_s_per_pixel": 100.0},  # 1 e- at 0.01 s
        ]
    },
    "noise": {"poisson": True, "read_noise": True, "read_noise_electrons": 5.0},
}
