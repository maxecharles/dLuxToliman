"""
Writing simulated data to disk.

Each run directory gets:
- the noisy frames, shape (n_frames, ny, nx), in electrons
- the noiseless expected image, for checking fits against truth
- a copy of the config that produced it
"""

from __future__ import annotations
import json
from pathlib import Path
import numpy as onp


def _check_writable(path: Path, overwrite: bool):
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; set output.overwrite to replace it.")


def write_fits(path: Path, frames, expected, config: dict, header: dict, overwrite):
    from astropy.io import fits  # optional dependency, only needed for FITS output

    primary = fits.PrimaryHDU(onp.asarray(frames, dtype=onp.float32))
    primary.header["EXTNAME"] = "FRAMES"
    primary.header["BUNIT"] = "electron"
    for key, (value, comment) in header.items():
        primary.header[key] = (value, comment)

    truth = fits.ImageHDU(onp.asarray(expected, dtype=onp.float64), name="EXPECTED")
    truth.header["BUNIT"] = "electron"
    truth.header["COMMENT"] = "Noiseless expected image (instrument.model())"

    fits.HDUList([primary, truth]).writeto(path, overwrite=overwrite)


def write_npz(path: Path, frames, expected, config: dict, header: dict, overwrite):
    onp.savez(
        path,
        frames=onp.asarray(frames, dtype=onp.float32),
        expected=onp.asarray(expected, dtype=onp.float64),
        **{key.lower(): value for key, (value, _) in header.items()},
    )


WRITERS = {"fits": (write_fits, ".fits"), "npz": (write_npz, ".npz")}


def write_outputs(frames, expected, config: dict, header: dict) -> Path:
    output = config.get("output", {})
    fmt = output.get("format", "npz")
    if fmt not in WRITERS:
        raise ValueError(f"Unknown output format '{fmt}'. Known: {list(WRITERS)}")
    writer, ext = WRITERS[fmt]

    overwrite = output.get("overwrite", False)
    directory = Path(output.get("directory", f"./simulated_data/{config['run_id']}"))
    directory.mkdir(parents=True, exist_ok=True)

    data_path = directory / f"{config['run_id']}{ext}"
    config_path = directory / "config.json"
    _check_writable(data_path, overwrite)
    _check_writable(config_path, overwrite)

    writer(data_path, frames, expected, config, header, overwrite)
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    return data_path
