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

from .base import CollectionUID as CollectionUID
from .base import Index as Index
from .base import create_identifier_file as create_identifier_file
from .base import get_identifier_filename as get_identifier_filename
