import jax
import pytest


@pytest.fixture(autouse=True)
def double_precision():
    """toliman_sim needs JAX x64; enable it per test only, so other tests are unaffected."""
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", previous)
