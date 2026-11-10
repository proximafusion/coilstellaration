"""Tests for the JAX persistent compilation-cache enabler."""

import jax
import pytest

from leaderboard.src.backend import config, jax_cache


@pytest.fixture(autouse=True)
def _reset_state():
    """Restore the global enable-once flag and jax cache-dir between tests."""
    prev_dir = jax.config.jax_compilation_cache_dir
    jax_cache._ENABLED = False
    yield
    jax_cache._ENABLED = False
    jax.config.update("jax_compilation_cache_dir", prev_dir)


def test_enable_points_jax_at_cache_dir(tmp_path, monkeypatch):
    cache = tmp_path / "cc"
    monkeypatch.setattr(config, "JAX_COMPILE_CACHE_DIR", str(cache))

    jax_cache.enable()

    assert cache.is_dir()
    assert jax.config.jax_compilation_cache_dir == str(cache)


def test_enable_is_idempotent(tmp_path, monkeypatch):
    cache = tmp_path / "cc"
    monkeypatch.setattr(config, "JAX_COMPILE_CACHE_DIR", str(cache))
    jax_cache.enable()

    # A second call must not re-run even if the configured dir changes.
    monkeypatch.setattr(config, "JAX_COMPILE_CACHE_DIR", str(tmp_path / "other"))
    jax_cache.enable()

    assert not (tmp_path / "other").exists()
    assert jax.config.jax_compilation_cache_dir == str(cache)


def test_empty_dir_disables_without_touching_config(monkeypatch):
    monkeypatch.setattr(config, "JAX_COMPILE_CACHE_DIR", "")
    before = jax.config.jax_compilation_cache_dir

    jax_cache.enable()

    assert jax.config.jax_compilation_cache_dir == before


def test_unwritable_dir_is_non_fatal(monkeypatch):
    monkeypatch.setattr(config, "JAX_COMPILE_CACHE_DIR", "/proc/cannot/create/here")
    # Must swallow the OSError and simply skip caching.
    jax_cache.enable()
