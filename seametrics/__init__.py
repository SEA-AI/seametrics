"""SeaMetrics.

Custom metrics for evaluating performance of A.I. pipelines at SEA.AI.
"""

try:
    from importlib.metadata import PackageNotFoundError, version
except ImportError:
    from importlib_metadata import PackageNotFoundError, version  # type: ignore[import-not-found]

try:
    __version__ = version("seametrics")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = ["__version__"]
