"""
Builds dLux/dLuxToliman objects (optics, source, detector, telescope) from a
simulation config.

Detector layers are looked up by their "name" in DETECTOR_LAYERS. Each entry
says which grid the layer acts on:
- "oversampled": applied to the oversampled PSF (e.g. jitter, sub-pixel geometry)
- "detector":    applied after downsampling to detector pixels (e.g. flat, dark)
The final detector is [oversampled layers] + Downsample(oversample) + [detector
layers], keeping the config order within each group.
"""

from __future__ import annotations
import math
import numpy as onp
import jax.numpy as np
import jax.random as jr
from jax import Array
import dLux
import dLux.layers as dll
import dLuxToliman as dlT

from .config import config_dir, strip_comments, to_kwargs
from .spectra import build_spectra, resolve


def build_optics(config: dict) -> dlT.TolimanOpticalSystem:
    kwargs = to_kwargs(config["optics"], dlT.TolimanOpticalSystem, "optics")

    # Catch the most common config mistake with a clearer error than JAX gives
    radial_orders = kwargs.get("radial_orders")
    coefficients = kwargs.get("coefficients")
    if radial_orders is not None and coefficients is not None:
        n_zernikes = sum(order + 1 for order in radial_orders)
        if len(coefficients) != n_zernikes:
            raise ValueError(
                f"[optics] radial_orders {radial_orders} gives {n_zernikes} "
                f"Zernikes, but {len(coefficients)} coefficients were given."
            )

    for key in ("coefficients", "mask"):
        if kwargs.get(key) is not None:
            kwargs[key] = np.asarray(kwargs[key], dtype=float)
    return dlT.TolimanOpticalSystem(**kwargs)


def build_source(
    config: dict, optics: dlT.TolimanOpticalSystem
) -> dlT.SpectralAlphaCen:
    source = dict(config["source"])
    spectra = build_spectra(source.pop("spectra"), optics, config_dir(config))
    kwargs = to_kwargs(source, dlT.SpectralAlphaCen, "source")
    return dlT.SpectralAlphaCen(
        spectrum_A=spectra["A"],
        spectrum_B=spectra["B"],
        exposure_time=config["exposure_time_s"],
        **kwargs,
    )


# --- Detector layers ---------------------------------------------------------


def _odd(n: float) -> int:
    n = int(math.ceil(n))
    return n + 1 - n % 2


def _jitter(spec, optics, config, key):
    # Kernels are built on the oversampled grid
    pixel_scale = optics.psf_pixel_scale / optics.oversample  # arcsec

    if spec.get("kernel", "gaussian") == "gaussian":
        sigma = spec["sigma_mas"] / 1e3  # arcsec, major axis
        shear = spec.get("shear", 0.0)
        # r = det(covariance) = (sigma_major * sigma_minor)^2 [arcsec^4], with
        # sigma_minor = (1 - shear) * sigma_major. The kernel_size padding should
        # span ~8 sigma_major.
        return dlT.GaussianJitter(
            r=(sigma**2 * (1 - shear)) ** 2,
            shear=shear,
            phi=spec.get("phi_deg", 0.0),
            kernel_size=max(11, _odd(8 * sigma / pixel_scale)),
        )

    if spec["kernel"] == "shm":
        amplitude = spec["amplitude_mas"] / 1e3  # arcsec
        return dlT.SHMJitter(
            A=amplitude,
            phi=spec.get("phi_deg", 0.0),
            kernel_size=max(11, _odd(2 * amplitude / pixel_scale)),
        )

    raise ValueError(f"Unknown jitter kernel '{spec['kernel']}'")


def _pixel_response(spec, optics, config, key):
    shape = (optics.psf_npixels, optics.psf_npixels)
    path = spec.get("pixel_response_map_path")
    if path is not None:
        response = onp.load(resolve(path, config_dir(config)))
    else:
        # No calibration map given: draw a random Gaussian flat field
        response = 1 + spec.get("flat_sigma", 0.0) * jr.normal(key, shape)
    if response.shape != shape:
        raise ValueError(
            f"[pixel_response] map has shape {response.shape}, expected {shape}"
        )
    return dll.ApplyPixelResponse(np.asarray(response))


def _dark_current(spec, optics, config, key):
    return dll.AddConstant(spec["rate_e_per_s_per_pixel"] * config["exposure_time_s"])


def _not_implemented(spec, optics, config, key):
    raise NotImplementedError(
        f"Detector layer '{spec['name']}' is not implemented yet (see the DEVELOPMENT.md "
        "checklist). Set \"enabled\": false to skip it."
    )


# name -> (grid, builder(spec, optics, config, key))
DETECTOR_LAYERS = {
    "high_frequency_jitter": ("oversampled", _jitter),
    "pixel_dislocations": ("oversampled", _not_implemented),
    "local_pixel_scale_variation": ("oversampled", _not_implemented),
    "pixel_response": ("detector", _pixel_response),
    "brighter_fatter_effect": ("detector", _not_implemented),
    "dark_current": ("detector", _dark_current),
}


def build_detector(
    config: dict, optics: dlT.TolimanOpticalSystem, key: Array
) -> dLux.LayeredDetector:
    oversampled, detector = [], []
    for i, spec in enumerate(config.get("detector", {}).get("layers", [])):
        spec = strip_comments(spec)
        if not spec.get("enabled", True):
            continue
        name = spec["name"]
        if name not in DETECTOR_LAYERS:
            raise ValueError(
                f"Unknown detector layer '{name}'. "
                f"Known layers: {list(DETECTOR_LAYERS)}"
            )
        grid, builder = DETECTOR_LAYERS[name]
        layer = (name, builder(spec, optics, config, jr.fold_in(key, i)))
        (oversampled if grid == "oversampled" else detector).append(layer)

    downsample = ("downsample", dll.Downsample(optics.oversample))
    return dLux.LayeredDetector(oversampled + [downsample] + detector)


def build_telescope(config: dict, key: Array) -> dlT.Toliman:
    """`key` is a jax.random key, used by random calibration maps (e.g. the flat field)."""
    optics = build_optics(config)
    source = build_source(config, optics)
    detector = build_detector(config, optics, key)
    return dlT.Toliman(optics, source, detector)
