"""Content-addressed provenance helpers."""

from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path
from typing import Any


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def content_hash(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def csv_record_sha256(path: Path, *, key_field: str, key_value: str) -> str:
    """Hash the exact UTF-8 CSV record, including its line terminator."""

    with path.open("r", encoding="utf-8", newline="") as source:
        text = source.read()
    handle = io.StringIO(text, newline="")
    reader = csv.DictReader(handle)
    fieldnames = reader.fieldnames
    if fieldnames is None or key_field not in fieldnames:
        raise ValueError(f"CSV lacks key field {key_field!r}: {path}")
    matches: list[bytes] = []
    while True:
        record_start = handle.tell()
        try:
            row = next(reader)
        except StopIteration:
            break
        record_end = handle.tell()
        if row[key_field] == key_value:
            matches.append(text[record_start:record_end].encode("utf-8"))
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one {key_field}={key_value!r} row in {path}, "
            f"found {len(matches)}"
        )
    return sha256_bytes(matches[0])
