import sys
from importlib.util import find_spec
import numpy as np

# motmetrics uses np.asfarray which was removed in NumPy 2.0
if not hasattr(np, "asfarray"):
    np.asfarray = lambda a, dtype=float: np.asarray(a, dtype=dtype)

def package_available(package_name: str) -> bool:
    """Check if a package is available in your environment.

    >>> package_available('os')
    True
    >>> package_available('bla')
    False 
    """
    try:
        return find_spec(package_name) is not None
    except ModuleNotFoundError:
        return False

_PYTHON_VERSION = ".".join(map(str, [sys.version_info.major, sys.version_info.minor, sys.version_info.micro]))
_MOTMETRICS_AVAILABLE = package_available("motmetrics")