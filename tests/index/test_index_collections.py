from datetime import datetime
from pathlib import Path

import pytest
import yaml

from bamboost.exceptions import InvalidCollectionError
from bamboost.index.base import (
    IDENTIFIER_PREFIX,
    IDENTIFIER_SEPARATOR,
    Index,
    _find_uid_from_path,
    _validate_path,
)


@pytest.fixture
def index(tmp_path):
    return Index(sql_file=":memory:", search_paths=[tmp_path])


def id_file_name(uid: str) -> str:
    return f"{IDENTIFIER_PREFIX}{IDENTIFIER_SEPARATOR}{uid}"


def test_find_uid_from_identifier(tmp_path):
    collection_path = tmp_path / "collection2"
    collection_path.mkdir()
    (collection_path / ".bamboost-collection-COLL456").touch()

    uid = _find_uid_from_path(collection_path)
    assert uid == "COLL456"


@pytest.mark.parametrize("uid,result", [("123", True), ("345", False)])
def test_validate_path(tmp_path: Path, uid: str, result: bool):
    collection_path = tmp_path / "collection"
    collection_path.mkdir()
    (collection_path / id_file_name("123")).touch()

    assert _validate_path(collection_path, uid) == result


def test_insert_collection(index: Index, tmp_path):
    uid = "COLL123"
    collection_path = tmp_path / "collection1"

    index.upsert_collection(uid, collection_path)
    resolved_path = index._get_collection_path(uid)
    assert resolved_path == collection_path


def test_upsert_collection(index: Index, tmp_path):
    uid = "COLL123"
    collection_path = tmp_path / "collection1"
    index.upsert_collection(uid, collection_path)
    assert index._get_collection_path(uid) == collection_path

    # Update the path
    new_collection_path = tmp_path / "collection2"
    index.upsert_collection(uid, new_collection_path)
    assert index._get_collection_path(uid) == new_collection_path
    assert len(index._get_collections()) == 1


def test_resolve_uid(index: Index, tmp_path: Path):
    uid = "COLL789"
    collection_path = tmp_path / "collection3"
    collection_path.mkdir()
    (collection_path / id_file_name(uid)).touch()

    resolved_uid = index.resolve_uid(collection_path)
    assert resolved_uid == uid


def test_resolve_uid_not_found(index: Index, tmp_path: Path):
    collection_path = tmp_path / "collection4"
    collection_path.mkdir()

    with pytest.raises(InvalidCollectionError):
        index.resolve_uid(collection_path)


def test_scan_for_collections(index, tmp_path):
    coll1 = tmp_path / "coll1"
    coll2 = tmp_path / "coll2"
    coll1.mkdir()
    coll2.mkdir()
    (coll1 / id_file_name("UID001")).touch()
    (coll2 / id_file_name("UID002")).touch()

    found = index.scan_for_collections()

    assert len(found) == 2
    assert set(uid for uid, _ in found) == {"UID001", "UID002"}

    stored_uids = {c.uid for c in index.all_collections}
    assert stored_uids == {"UID001", "UID002"}


def test_scan_persists_metadata(index: Index, tmp_path: Path):
    uid = "UIDMETA"
    coll_path = tmp_path / "with_meta"
    coll_path.mkdir()

    metadata_file = coll_path / id_file_name(uid)
    metadata = {
        "uid": uid,
        "created_at": datetime(2024, 5, 1, 12, 30, 0),
        "description": "Sample collection",
        "tags": ["alpha", "beta", "alpha"],
        "aliases": ["demo", "Demo"],
    }
    metadata_file.write_text(yaml.safe_dump(metadata, sort_keys=False))

    index.scan_for_collections()

    collection = index.collection(uid)
    assert collection is not None
    assert collection.description == "Sample collection"
    assert collection.tags == ["alpha", "beta"]
    assert collection.aliases == ["demo"]
    assert collection.created_at == datetime(2024, 5, 1, 12, 30, 0)


def test_resolve_path_with_alias(index: Index, tmp_path: Path):
    uid = "UIDALIAS"
    coll_path = tmp_path / "with_alias"
    coll_path.mkdir()

    metadata_file = coll_path / id_file_name(uid)
    metadata_file.write_text(
        yaml.safe_dump(
            {
                "uid": uid,
                "aliases": ["alias-one"],
            },
            sort_keys=False,
        )
    )

    index.scan_for_collections()

    resolved = index.resolve_path("alias-one")
    assert resolved == coll_path

    fetched = index.collection("ALIAS-ONE")
    assert fetched is not None
    assert fetched.uid == uid
