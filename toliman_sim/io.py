"""
Writing simulated data to disk. Each run directory gets:

  npy (default):
    {run_id}_data.npy      noisy frames, (n_observations, ny, nx), photoelectrons
    {run_id}_expected.npy  noiseless expected images, same shape
    {run_id}_truth.npz     drawn/true parameters, one row per observation
  fits:
    {run_id}.fits          DATA cube, EXPECTED cube, TRUTH table
  always:
    config.json            the config that produced it

output.directory is relative to the config file (like every relative path in a config).
"""

from __future__ import annotations
import json
from pathlib import Path
import numpy as onp

from .config import config_dir
from .spectra import resolve


def _check_writable(paths, overwrite: bool):
    for path in paths:
        if path.exists() and not overwrite:
            raise FileExistsError(f"{path} exists; set output.overwrite to replace it.")


def write_npy(directory: Path, run_id: str, result: dict, meta: dict, overwrite: bool):
    paths = [directory / f"{run_id}_{name}" for name in ("data.npy", "expected.npy", "truth.npz")]
    _check_writable(paths, overwrite)
    onp.save(paths[0], result["data"].astype(onp.float32))
    onp.save(paths[1], result["expected"])
    onp.savez(paths[2], **result["truth"], **meta)
    return paths[0]


def write_fits(directory: Path, run_id: str, result: dict, meta: dict, overwrite: bool):
    from astropy.io import fits  # optional dependency, only needed for FITS output
    from astropy.table import Table

    path = directory / f"{run_id}.fits"
    _check_writable([path], overwrite)

    primary = fits.PrimaryHDU(result["data"].astype(onp.float32))
    primary.header["EXTNAME"] = "DATA"
    primary.header["BUNIT"] = "electron"
    for key, value in meta.items():
        primary.header[key[:8].upper()] = value

    expected = fits.ImageHDU(result["expected"], name="EXPECTED")
    expected.header["BUNIT"] = "electron"
    truth = fits.table_to_hdu(Table(result["truth"]))
    truth.name = "TRUTH"

    fits.HDUList([primary, expected, truth]).writeto(path, overwrite=overwrite)
    return path


WRITERS = {"npy": write_npy, "fits": write_fits}


def write_outputs(result: dict, config: dict, meta: dict) -> Path:
    output = config.get("output", {})
    fmt = output.get("format", "npy")
    if fmt not in WRITERS:
        raise ValueError(f"Unknown output format '{fmt}'. Known: {list(WRITERS)}")

    overwrite = output.get("overwrite", False)
    run_id = config["run_id"]
    directory = resolve(output.get("directory", f"simulated_data/{run_id}"), config_dir(config))
    directory.mkdir(parents=True, exist_ok=True)

    config_path = directory / "config.json"
    _check_writable([config_path], overwrite)
    path = WRITERS[fmt](directory, run_id, result, meta, overwrite)
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    return path
