from __future__ import annotations

import fnmatch
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

from bamboost._config import config
from bamboost._logger import BAMBOOST_LOGGER
from bamboost._typing import StrPath

log = BAMBOOST_LOGGER.getChild("Scanner")

IDENTIFIER_PREFIX = ".bamboost-collection"
IDENTIFIER_SEPARATOR = "-"


def load_collection_metadata(path: Path, uid: str) -> dict[str, Any] | None:
    """Load the metadata of a collection from its identifier file.

    Args:
        path: Path to the collection directory
        uid: UID of the collection
    """
    metadata_file = Path(path).joinpath(get_identifier_filename(uid))
    if not metadata_file.exists():
        return None

    try:
        raw = yaml.safe_load(metadata_file.read_text())
    except FileNotFoundError:
        return None
    except Exception as exc:  # pragma: no cover - defensive logging
        log.debug(
            "Failed to load metadata for collection %s from %s: %s",
            uid,
            metadata_file,
            exc,
        )
        raise Exception(
            f"Failed to load metadata for collection {uid} (file: {metadata_file})"
        ) from exc

    if raw is None:
        raw = {}

    if not isinstance(raw, Mapping):
        log.debug("Unexpected metadata format for collection %s: %r", uid, raw)
        raise TypeError(
            f"Unexpected metadata format for collection {uid}: {type(raw)}. "
            "Revise the identifier/metadata file."
        )

    return normalize_collection_metadata(raw)


def normalize_collection_metadata(data: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize the metadata of a collection.

    This also handles backward compatibility for the "Date of creation" field.

    Args:
        data: The raw metadata dictionary.
    """
    metadata: dict[str, Any] = dict(**data)

    created_at_value = None
    # "Date of creation" is for backward compatibility only
    # we will write "created_at" from now on
    for key in ("created_at", "Date of creation"):
        if key in data and data[key] is not None:
            created_at_value = data[key]
            break

    parsed_created_at = _parse_datetime_value(created_at_value)
    if parsed_created_at is not None:
        metadata["created_at"] = parsed_created_at

    if tags := data.get("tags"):
        metadata["tags"] = deduplicate_sequence(tags)
    if aliases := data.get("aliases"):
        metadata["aliases"] = deduplicate_sequence(aliases, casefold=True)

    return metadata


def normalize_simulation_metadata(data: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize simulation metadata before writing to SQL."""
    metadata: dict[str, Any] = dict(**data)
    if (tags := data.get("tags")) is not None:
        metadata["tags"] = deduplicate_sequence(tags)
    return metadata


def deduplicate_sequence(values: Any, *, casefold: bool = False) -> list[str]:
    """Deduplicate a sequence of strings.

    Args:
        values: The sequence of strings to deduplicate.
        casefold: Whether to ignore case when deduplicating.
    """
    if values is None:
        iterable: Iterable[Any] = []
    elif isinstance(values, (str, bytes)):
        iterable = [values]
    else:
        try:
            iterable = list(values)
        except TypeError:
            iterable = [values]

    seen: set[str] = set()
    result: list[str] = []

    for value in iterable:
        text = str(value).strip()
        if not text:
            continue

        key = text.casefold() if casefold else text
        if key in seen:
            continue

        seen.add(key)
        result.append(text)

    return result


def _parse_datetime_value(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        candidate = value.strip()
        if not candidate:
            return None
        try:
            return datetime.fromisoformat(candidate)
        except ValueError:
            return None
    return None


def create_identifier_file(path: StrPath, uid: str) -> None:
    """Create an identifier file in the collection directory.

    Args:
        path: Path to the collection directory
        uid: UID of the collection
    """
    path = Path(path)
    with open(path.joinpath(get_identifier_filename(uid)), "w") as f:
        f.write("Date of creation: " + str(datetime.now()))


def get_identifier_filename(uid: str) -> str:
    return IDENTIFIER_PREFIX + IDENTIFIER_SEPARATOR + uid


def validate_path(path: Path, uid: str) -> bool:
    return path.is_dir() and path.joinpath(get_identifier_filename(uid)).is_file()


def find_uid_from_path(path: Path) -> str | None:
    try:
        return path.glob(f"{IDENTIFIER_PREFIX}*").__next__().name.rsplit("-", 1)[1]
    except StopIteration:
        return None


def find_collection(uid: str, root_dir: Path) -> tuple[Path, ...]:
    """Find the collection with UID under given root_dir.

    Args:
        uid: UID to search for
        root_dir: root directory for search
    """
    # First, try to find from identifier-file
    paths_by_cuid = tuple(
        Path(i).parent for i in find_files(get_identifier_filename(uid), root_dir)
    )

    if paths_by_cuid:
        return paths_by_cuid
    
    # Maybe the uid is an alias
    log.debug(f"No identifier-file found for uid='{uid}', now scanning for aliases.")
    norm_uid = uid.lower()

    all_colls = scan_directory_for_collections(root_dir)

    paths_by_alias = []
    for coll_uid, coll_path in all_colls:
        metadata_dict = load_collection_metadata(coll_path, coll_uid) or {}
        aliases = metadata_dict.get("aliases", [])

        if norm_uid in aliases:
            paths_by_alias.append(coll_path)
    
    return tuple(paths_by_alias)


def find_files(
    pattern: str,
    root_dir: str | os.PathLike,
    exclude: Iterable[str] | None = None,
) -> tuple[Path, ...]:
    """
    Locate every file matching *pattern* under *root_dir* while **pruning**
    directory names listed in *exclude* (exact-match on the final path part).

    Returns an immutable tuple of absolute paths (str) just like the POSIX helper.
    """
    root = Path(root_dir)
    hits: list[Path] = []

    if exclude is None:
        exclude = config.index.excludeDirs

    for base, dirnames, filenames in os.walk(root, topdown=True):
        # --- prune in–place so the walker never descends further ---
        dirnames[:] = [d for d in dirnames if d not in exclude]

        for fname in filenames:
            if fnmatch.fnmatch(fname, pattern):
                hits.append(Path(base, fname))

    return tuple(hits)


def scan_directory_for_collections(root_dir: Path) -> tuple[tuple[str, Path], ...]:
    """Scan the directory for collections.

    Args:
        root_dir: Directory to scan for collections

    Returns:
        Tuple of tuples with the UID and path of the collection
    """

    log.debug(f"Scanning {root_dir}")

    if not root_dir.exists():
        log.warning(f"Path does not exist: {root_dir}")
        return ()

    found_indicator_files = find_files(
        get_identifier_filename("*"), root_dir.as_posix()
    )

    if not found_indicator_files:
        log.info(f"No collections found in {root_dir}")
        return ()

    return tuple(
        (i.name.rsplit(IDENTIFIER_SEPARATOR, 1)[-1], i.parent)
        for i in found_indicator_files
    )
