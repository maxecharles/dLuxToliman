"""
How the choice of reference point (centre_weight) affects inference of the separation.

    python validation/reference_point_fisher.py configs/example.json

Builds the expected Fisher matrix (Poisson + read noise) for
    (x, y, separation, position_angle, flux_A, flux_B)
with flux_A / flux_B multiplicative scales on each star's spectrum, and reports for each
centre_weight the marginal separation error, the correlations of the separation with the
other parameters, and the condition number of the correlation matrix.
"""

import argparse

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as np
import numpy as onp
import dLuxToliman as dlT

from toliman_sim import build_telescope, load_config
from toliman_sim.noise import read_noise_sigma


def fisher(telescope, centre_weight, read_sigma):
    source = telescope.source
    A, B = onp.degrees(onp.asarray(source.xy_positions)) * 3600  # arcsec
    ref = A + centre_weight * (B - A)
    # centre_weight is static, so rebuild the source rather than .set() it
    telescope = telescope.set("source", dlT.SpectralAlphaCen(
        source.spectrum_A, source.spectrum_B, source.exposure_time, source.separation,
        source.position_angle, ref[0], ref[1], centre_weight=centre_weight,
    ))

    def model(theta):
        x, y, sep, pa, fa, fb = theta
        t = telescope.set(
            ["source.x_position", "source.y_position", "source.separation",
             "source.position_angle"],
            [x, y, sep, pa],
        )
        t = t.multiply("source.spectrum_A.basis_weights", fa)
        t = t.multiply("source.spectrum_B.basis_weights", fb)
        return t.model()

    theta = np.array([ref[0], ref[1], source.separation, source.position_angle, 1.0, 1.0])
    mu = model(theta)
    J = jax.jacfwd(model)(theta).reshape(-1, len(theta))
    var = (mu + read_sigma**2).reshape(-1)
    return onp.asarray(J.T @ (J / var[:, None]))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("config")
    parser.add_argument("--weights", type=float, nargs="*")
    args = parser.parse_args()

    config = load_config(args.config)
    telescope = build_telescope(config, jax.random.PRNGKey(config.get("seed", 0)))
    read_sigma = read_noise_sigma(config)
    f_A, f_B = onp.asarray(telescope.source.raw_fluxes)
    photocentre = f_B / (f_A + f_B)
    weights = args.weights or [0.0, photocentre, 0.5, 1.0]

    print(f"photocentre weight F_B / (F_A + F_B) = {photocentre:.3f}")
    print(f"{'weight':>8} {'sigma_sep [uas]':>16} {'corr(sep,x)':>12} {'corr(sep,y)':>12} "
          f"{'max|corr| with sep':>19} {'cond(corr)':>11}")
    for w in weights:
        F = fisher(telescope, w, read_sigma)
        cov = onp.linalg.inv(F)
        sig = onp.sqrt(onp.diag(cov))
        corr = cov / onp.outer(sig, sig)
        others = onp.delete(onp.abs(corr[2]), 2)
        print(f"{w:8.3f} {sig[2] * 1e6:16.3f} {corr[2, 0]:12.3f} {corr[2, 1]:12.3f} "
              f"{others.max():19.3f} {onp.linalg.cond(corr):11.1f}")


if __name__ == "__main__":
    main()
