"""SeaMetrics.

Custom metrics for evaluating performance of A.I. pipelines at SEA.AI.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("seametrics")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = ["__version__"]
