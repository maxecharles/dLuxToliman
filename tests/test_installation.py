import sys
import os
import re
import importlib.metadata
import pytest

platform: str = sys.platform
not_lin_err: str = """
    The installation test is designed to run on a linux machine. In particular,
    this test is designed to be run by github actions.
"""

if platform not in ["linux", "linux2"]:
    print(f"Error: {not_lin_err}")


def test_install_dLuxToliman():
    if platform not in ["linux", "linux2"]:
        pytest.skip("Installation smoke-test only runs on Linux CI.")

    install_ok = os.system("pip install --quiet .")

    assert install_ok == 0

    os.system("rm -r dLuxToliman.egg-info")
    os.system("rm -r build")


def test_import_dLuxToliman():
    import dLuxToliman

    dLuxToliman.optical_systems
    assert True


def test_declared_dependencies():
    import dLuxToliman

    try:
        version = importlib.metadata.version("dLuxToliman")
    except importlib.metadata.PackageNotFoundError:
        pytest.skip("dLuxToliman is not installed, so there is no metadata to check.")
    if version != dLuxToliman.__version__:
        pytest.skip(f"The installed dLuxToliman ({version}) is not this checkout.")

    # The 0.4.0 wheel was published without any of these
    requires = importlib.metadata.requires("dLuxToliman") or []
    names = {re.split(r"[\s<>=!~;\[]", req, maxsplit=1)[0].lower() for req in requires}
    assert {"dlux", "zodiax", "jax", "equinox"} <= names
