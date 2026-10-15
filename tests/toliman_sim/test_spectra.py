import copy
import numpy as onp

import dLuxToliman as dlT
from toliman_sim import build_optics, build_source
from toliman_sim.spectra import V_ZERO_POINT, blackbody_sed

from .small_config import SMALL_CONFIG


def _source(config):
    optics = build_optics(config)
    return optics, build_source(config, optics)


def test_blackbody_normalised_at_550nm():
    assert onp.isclose(blackbody_sed(550.0, 5790, 0.0), V_ZERO_POINT)
    assert onp.isclose(blackbody_sed(550.0, 5790, 2.5), V_ZERO_POINT / 10)


def test_exposure_time_is_a_multiplier():
    optics, source = _source(SMALL_CONFIG)
    longer = source.set("exposure_time", 3 * source.exposure_time)
    onp.testing.assert_allclose(longer.model(optics), 3 * source.model(optics), rtol=1e-10)


def test_A_and_B_have_their_own_spectra():
    _, source = _source(SMALL_CONFIG)
    a, b = source.spectrum_A, source.spectrum_B
    # B is fainter and redder than A
    assert source.raw_fluxes[1] < source.raw_fluxes[0]
    assert (b.weights[-1] / b.weights[0]) > (a.weights[-1] / a.weights[0])


def test_hermite_matches_full_propagation():
    config = copy.deepcopy(SMALL_CONFIG)
    config["source"]["spectra"]["n_coarse"] = None
    optics, exact = _source(config)
    _, hermite = _source(SMALL_CONFIG)  # n_coarse = 4
    exact, hermite = exact.model(optics), hermite.model(optics)
    assert onp.abs(hermite - exact).max() < 1e-3 * exact.max()


def test_spectrum_flux_conventions():
    _, source = _source(SMALL_CONFIG)
    spectrum = source.spectrum_A
    assert isinstance(spectrum, dlT.CombinedBasisSpectrum)
    onp.testing.assert_allclose(spectrum.weights.sum(), 1.0)
    # Top-hat throughput of 1: detected = incident
    onp.testing.assert_allclose(spectrum.proper_flux(), spectrum.flux)


def test_contrast_and_log_flux_from_spectra():
    _, source = _source(SMALL_CONFIG)
    flux_A, flux_B = source.raw_fluxes
    onp.testing.assert_allclose(source.contrast, flux_A / flux_B)
    onp.testing.assert_allclose(10**source.log_flux, flux_A + flux_B)


def test_centre_weight_only_moves_the_reference_point():
    import dLux.utils as dlu

    _, source = _source(SMALL_CONFIG)  # centre_weight = 0.5 (midpoint)
    A, B = dlu.rad2arcsec(source.xy_positions)
    # position angle is the direction of B relative to A
    d = source.separation * onp.array([onp.sin(onp.radians(source.position_angle)),
                                       onp.cos(onp.radians(source.position_angle))])
    onp.testing.assert_allclose(B - A, d, atol=1e-12)

    # Re-referencing on A, with the reference moved onto A, gives the same stars
    on_A = dlT.SpectralAlphaCen(
        source.spectrum_A, source.spectrum_B, source.exposure_time,
        source.separation, source.position_angle, *A, centre_weight=0.0,
    )
    onp.testing.assert_allclose(dlu.rad2arcsec(on_A.xy_positions), onp.array([A, B]), atol=1e-12)


def test_real_throughput_and_spectra_files():
    from pathlib import Path
    from toliman_sim.spectra import system_throughput

    repo = Path(__file__).parents[2]
    grid, throughput = system_throughput(
        {"components": [{"path": "data/throughput/bandpass.csv"},
                        {"path": "data/throughput/qe_mono.csv"}]},
        base=repo,
    )
    peak = grid[throughput.argmax()]
    assert 520 < peak < 650 and 0.5 < throughput.max() <= 1

    config = copy.deepcopy(SMALL_CONFIG)
    config["_config_path"] = str(repo / "configs" / "test.json")
    config["source"]["spectra"]["A"] = {"sed": "table", "path": "../data/spectra/alpha_cen_A.csv"}
    config["source"]["spectra"]["B"] = {"sed": "table", "path": "../data/spectra/alpha_cen_B.csv"}
    _, source = _source(config)
    # Same 530-640 nm top-hat as the blackbody test: PHOENIX contrast is close to 3.2
    assert 3.0 < float(source.contrast) < 3.5
