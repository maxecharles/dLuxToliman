"""
Accuracy and speed of Hermite wavelength sparsity against exact propagation.

    python validation/hermite_accuracy.py configs/example.json [--nodes 8 12 16 24 32]

Models the nominal telescope (no parameter draws) three ways:
  - exact: every one of the n_wavels wavelengths propagated (n_coarse = None),
  - as configured: the n_coarse set per star in the config,
  - a scan: the same n_coarse for both stars, for each value in --nodes,
and reports the model error as
    delta chi^2 = n_observations * sum((hermite - exact)^2 / (exact + read_noise^2)),
i.e. the chi^2 a perfect fit of n_observations frames would pay for the Hermite error
(the error is the same in every frame, so it adds up coherently), plus the run time.
"""

import argparse
import copy
import time

import equinox as eqx
import jax
import numpy as onp

from toliman_sim import build_telescope, load_config
from toliman_sim.noise import read_noise_sigma


@eqx.filter_jit
def _model(telescope):
    return telescope.model()


def timed_model(config):
    telescope = build_telescope(config, jax.random.PRNGKey(config.get("seed", 0)))
    _model(telescope).block_until_ready()  # compile
    start = time.time()
    image = onp.asarray(_model(telescope).block_until_ready())
    return image, time.time() - start


def with_nodes(config, nodes):
    config = copy.deepcopy(config)
    spectra = config["source"]["spectra"]
    spectra["n_coarse"] = nodes
    for star in ("A", "B"):
        spectra[star].pop("n_coarse", None)
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("config")
    parser.add_argument("--nodes", type=int, nargs="*", default=[8, 12, 16, 24, 32])
    parser.add_argument("--n-observations", type=int,
                        help="Frames fit together (default: observations.n_observations)")
    args = parser.parse_args()

    config = load_config(args.config)
    n = args.n_observations or config.get("observations", {}).get("n_observations", 1)
    variance_floor = read_noise_sigma(config) ** 2

    exact, t_exact = timed_model(with_nodes(config, None))

    def report(label, image, seconds):
        chi2 = n * ((image - exact) ** 2 / (exact + variance_floor)).sum()
        print(f"{label:>28}  delta chi2 = {chi2:10.3g}   {seconds:6.2f} s  "
              f"({t_exact / seconds:4.1f}x faster)")

    spectra = config["source"]["spectra"]
    configured = [spectra[s].get("n_coarse", spectra.get("n_coarse")) for s in ("A", "B")]
    print(f"exact: {spectra.get('n_wavels', 300)} wavelengths per star, {t_exact:.2f} s; "
          f"delta chi2 over {n} observations\n")
    report(f"configured A/B = {configured[0]}/{configured[1]}", *timed_model(config))
    for nodes in args.nodes:
        report(f"n_coarse = {nodes}", *timed_model(with_nodes(config, nodes)))


if __name__ == "__main__":
    main()
