"""
Loading simulation configs and mapping them onto constructor arguments.

Conventions (deliberately loose while the format is still in flux):
- Keys starting with "_" are comments/annotations and are ignored.
- Relative input paths (spectra, throughput, calibration maps) are relative to
  the config file. Output paths are relative to the working directory.
- Config keys may carry a unit suffix (e.g. "separation_arcsec", "m1_diameter_m").
  The suffix is stripped to find the matching constructor argument, so the unit
  in the key must match the unit the dLuxToliman constructor expects.
"""

from __future__ import annotations
import inspect
import json
import warnings
from pathlib import Path

UNIT_SUFFIXES = ("_arcsec", "_mas", "_deg", "_nm", "_m")


def load_config(path: str | Path) -> dict:
    """Reads a JSON config file and returns it as a dict."""
    path = Path(path)
    with open(path) as f:
        config = json.load(f)
    config["_config_path"] = str(path.resolve())
    return config


def config_dir(config: dict) -> Path | None:
    """Folder of the config file; relative input paths (data tables) resolve against it."""
    path = config.get("_config_path")
    return Path(path).parent if path is not None else None


def strip_comments(section: dict) -> dict:
    """Removes "_"-prefixed annotation keys from a config section."""
    return {k: v for k, v in section.items() if not k.startswith("_")}


def to_kwargs(section: dict, constructor, section_name: str) -> dict:
    """
    Maps a config section onto the keyword arguments of `constructor`, dropping
    unit suffixes where needed. Unknown keys raise a warning and are ignored.
    """
    accepted = set(inspect.signature(constructor).parameters) - {"self"}
    kwargs = {}
    for key, value in strip_comments(section).items():
        name = key
        if name not in accepted:
            for suffix in UNIT_SUFFIXES:
                if key.endswith(suffix) and key[: -len(suffix)] in accepted:
                    name = key[: -len(suffix)]
                    break
        if name not in accepted:
            warnings.warn(
                f"[{section_name}] ignoring unknown key '{key}' "
                f"(not an argument of {constructor.__name__})"
            )
            continue
        kwargs[name] = value
    return kwargs
