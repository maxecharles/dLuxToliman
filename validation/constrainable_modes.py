"""
Number of constrainable spectral modes K for Alpha Cen A and B, to set n_coarse ~ 2K.

    python validation/constrainable_modes.py configs/example.json

Uses the spectral Fisher tools of dluxtras. For each star the
Fisher is taken over incident photons per wavelength bin, with the other star, dark
current and read noise contributing variance, summed over n_frames identical frames
(the Hermite model error is the same in every frame, so it adds coherently: K must be
computed at the TOTAL photon count that will be fit, not per frame).
"""

import argparse
import copy

import jax

jax.config.update("jax_enable_x64", True)

import numpy as onp
from dluxtras.spectra import recommend_wavelength_sampling
from dluxtras.spectral_eigenmodes import (
    constrainable_modes,
    mode_criterion,
    spectral_eigenmodes,
)

import dLux.utils as dlu
from toliman_sim import build_telescope, load_config
from toliman_sim.noise import read_noise_sigma


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("config")
    parser.add_argument("--n-wavels", type=int, default=100, help="Fisher bins (>= ~100)")
    parser.add_argument("--n-frames", type=int, help="Frames fit together (default: observations.n_observations)")
    args = parser.parse_args()

    config = load_config(args.config)
    config["source"]["spectra"]["n_wavels"] = args.n_wavels
    config["source"]["spectra"]["n_coarse"] = None  # exact propagation for the reference
    n_frames = args.n_frames or config.get("observations", {}).get("n_observations", 1)

    telescope = build_telescope(config, jax.random.PRNGKey(config.get("seed", 0)))
    source = telescope.source
    read_sigma = read_noise_sigma(config)

    stars = {"A": (source.spectrum_A, 0, 1), "B": (source.spectrum_B, 1, 0)}
    for name, (spectrum, this, other) in stars.items():
        # Image of the other star alone (plus dark), for its Poisson variance
        other_image = telescope.multiply(f"source.spectrum_{name}.basis_weights", 0.0).model()

        spectrum_args = dict(
            wavels=spectrum.wavelengths,
            filt_weights=spectrum.filt_weights,
            basis=onp.eye(args.n_wavels),
            basis_weights=onp.asarray(spectrum.spec_weights()) * float(source.exposure_time),
        )
        position = dlu.rad2arcsec(source.xy_positions[this])
        modes = spectral_eigenmodes(
            telescope,
            spectrum_args,
            k=args.n_wavels,
            position_arcsec=onp.asarray(position),
            noise="poisson",
            variance_fn=lambda mu: (mu + other_image + read_sigma**2) / n_frames,
        )
        K = constrainable_modes(modes)
        detected = float(source.raw_fluxes[this]) * n_frames
        print(f"\nAlpha Cen {name}: K = {K} at {detected:.2e} detected photons "
              f"({n_frames} frames)")
        print("  criterion of first modes:", onp.round(mode_criterion(modes)[: K + 3], 3))
        recommend_wavelength_sampling(max(K, 1), detected)


if __name__ == "__main__":
    main()
