from pathlib import Path

import pytest

from bamboost.index.base import (
    IDENTIFIER_PREFIX,
    IDENTIFIER_SEPARATOR,
    Index,
    _find_uid_from_path,
    _validate_path,
)
from bamboost.exceptions import InvalidCollectionError


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
