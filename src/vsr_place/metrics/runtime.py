"""Timing utilities for experiment runtime tracking."""

import time
from contextlib import contextmanager


@contextmanager
def timer(name: str = "", results: dict | None = None):
    """Context manager for timing code blocks.

    Args:
        name: Name for this timing measurement.
        results: Optional dict to store the elapsed time in.

    Yields:
        None

    Example:
        times = {}
        with timer("sampling", times):
            result = model.sample(...)
        print(f"Sampling took {times['sampling']:.2f}s")
    """
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    if results is not None and name:
        results[name] = elapsed
