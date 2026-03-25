"""Module for indexing BAMBOOST collections.

Uses a caching mechanism using SQLAlchemy and an SQLite database to store
information about collections and simulations.

Usage:
    Create an instance of the `Index` class and use its methods to interact
    with the index.

    >>> from bamboost.index import Index
    >>> index = Index()

    Scan for collections in known paths:

    >>> index.scan_for_collections()

    Resolve the path of a collection:

    >>> index.resolve_path(<collection-uid>)

    Get a simulation from its collection and simulation name:

    >>> index.get_simulation(<collection-uid>, <simulation-name>)

Classes:
    Index: API for indexing BAMBOOST collections and simulations.
"""

from .base import Index
from .scanner import (
    create_identifier_file,
    get_identifier_filename,
)
from .uids import (
    CollectionUID,
    SimulationName,
    SimulationUID,
)

__all__ = [
    "CollectionUID",
    "SimulationUID",
    "SimulationName",
    "Index",
    "create_identifier_file",
    "get_identifier_filename",
]
