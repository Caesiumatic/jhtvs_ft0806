from __future__ import annotations

from pathlib import Path

import pytest

from jhtvs_ft0806.provenance import canonical_json_bytes, content_hash, sha256_file
from jhtvs_ft0806.schemas import SchemaError, read_csv_rows, write_csv_deterministic


def test_deterministic_csv_is_sorted_and_uses_lf(tmp_path: Path) -> None:
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    rows = [{"id": "B", "value": 2}, {"id": "A", "value": 1}]

    write_csv_deterministic(first, ("id", "value"), rows, sort_by=("id",))
    write_csv_deterministic(second, ("id", "value"), reversed(rows), sort_by=("id",))

    assert first.read_bytes() == second.read_bytes() == b"id,value\nA,1\nB,2\n"
    assert read_csv_rows(first) == [{"id": "A", "value": "1"}, {"id": "B", "value": "2"}]
    assert sha256_file(first) == sha256_file(second)


def test_deterministic_csv_rejects_undeclared_fields(tmp_path: Path) -> None:
    with pytest.raises(SchemaError, match="undeclared fields"):
        write_csv_deterministic(
            tmp_path / "bad.csv",
            ("id",),
            [{"id": "A", "unexpected": "value"}],
        )


def test_canonical_json_hash_is_key_order_invariant() -> None:
    left = {"b": [2, 1], "a": "value"}
    right = {"a": "value", "b": [2, 1]}

    assert canonical_json_bytes(left) == canonical_json_bytes(right)
    assert content_hash(left) == content_hash(right)
