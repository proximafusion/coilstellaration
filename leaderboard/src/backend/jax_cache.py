"""Enable JAX's persistent (on-disk) compilation cache for evaluations.

Each model evaluation runs in its own short-lived process (see
`runner._run_eval`), so the XLA kernels behind the DESC physics are otherwise
recompiled from scratch on every run (~9s of pure warmup). Pointing JAX at a
stable on-disk cache lets the second run over the fixed benchmark instances
reuse the compiled executables. This only caches compiled binaries keyed by the
jax/jaxlib/XLA version and the HLO hash, so any mismatch is a cache miss that
recompiles -- the produced numerics are never affected.
"""
import logging
import os

from src.backend import config

logger = logging.getLogger(__name__)

_ENABLED = False


def enable() -> None:
    """Point JAX at the on-disk compilation cache (idempotent, best-effort).

    Must run before the first jitted physics call. A missing/unwritable cache
    directory is logged and ignored -- evaluation just falls back to compiling
    in-process, exactly as before.
    """
    global _ENABLED
    if _ENABLED:
        return
    _ENABLED = True

    cache_dir = config.JAX_COMPILE_CACHE_DIR
    if not cache_dir:
        return
    try:
        import jax

        os.makedirs(cache_dir, exist_ok=True)
        jax.config.update("jax_compilation_cache_dir", cache_dir)
        # Cache every kernel, not just the slow ones: the eval compiles many
        # small kernels whose combined warmup is what we want to amortize.
        jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
        logger.info("JAX persistent compilation cache enabled at %s", cache_dir)
    except Exception as exc:  # noqa: BLE001 - caching is an optimisation, never fatal
        logger.warning("Could not enable JAX compilation cache: %s", exc)
