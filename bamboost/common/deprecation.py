# This file is part of bamboost, a Python library built for datamanagement
# using the HDF5 file format.
#
# https://gitlab.ethz.ch/compmechmat/research/libs/dbmanager
#
# Copyright 2023 Flavio Lorez and contributors
#
# There is no warranty for this code
from functools import wraps
import logging

log = logging.getLogger(__name__)


def deprecated(version=None, alternative=None):
    """
    This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emitted
    when the function is used.

    Args:
        func: The function to be deprecated
        version: The version in which the function was deprecated
        alternative: The alternative function to use instead of the deprecated one
    """

    def decorator(func):
        message = f"Function/property `{func.__name__}` was deprecated."
        if version:
            message = f"Function/property `{func.__name__}` was deprecated in version {version})."
        if alternative:
            message += f" Use `{alternative}` instead."

        @wraps(func)
        def wrapper(*args, **kwargs):
            log.warning(message)
            return func(*args, **kwargs)

        docstring = f'{message}\n\n'
        docstring += wrapper.__doc__ or ""
        wrapper.__doc__ = docstring
        return wrapper

    return decorator
