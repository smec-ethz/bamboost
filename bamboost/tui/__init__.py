__author__ = "Flavio Lorez"
__version__ = "unknown"

import os
import sys

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib  # noqa: F401

os.environ["BAMBOOST_MPI"] = "0"

# Determine the version of the package
try:
    # If the package is installed, the version is stored in _version.py
    from bamboost._version import __version__
except ImportError:
    # This is necessary if the package is not installed
    from setuptools_scm import get_version

    __version__ = get_version(root="..", relative_to=__file__)
