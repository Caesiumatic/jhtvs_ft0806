"""Shared typed records and deterministic serialization."""

from __future__ import annotations

import csv
import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence


class SchemaError(ValueError):
    """Raised when a supplied table does not match its declared schema."""


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    code: str
    message: str


@dataclass(slots=True)
class ValidationReport:
    spec_dir: str
    checks: dict[str, object] = field(default_factory=dict)
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.issues

    def to_dict(self) -> dict[str, object]:
        return {
            "status": "PASS" if self.ok else "FAIL",
            "spec_dir": self.spec_dir,
            "checks": self.checks,
            "issues": [asdict(issue) for issue in self.issues],
        }

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(), ensure_ascii=False, sort_keys=True, indent=2
        )


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise SchemaError(f"CSV has no header: {path}")
        rows = list(reader)
    if any(None in row for row in rows):
        raise SchemaError(f"CSV row has more fields than the header: {path}")
    return rows


def csv_fieldnames(path: Path) -> tuple[str, ...]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        try:
            return tuple(next(reader))
        except StopIteration as exc:
            raise SchemaError(f"CSV is empty: {path}") from exc


def write_csv_deterministic(
    path: Path,
    fieldnames: Sequence[str],
    rows: Iterable[Mapping[str, object]],
    *,
    sort_by: Sequence[str] = (),
) -> None:
    """Atomically write a UTF-8 CSV with stable columns, rows, and newlines."""

    materialized = [dict(row) for row in rows]
    expected = set(fieldnames)
    for index, row in enumerate(materialized, start=1):
        extras = set(row) - expected
        if extras:
            raise SchemaError(f"row {index} has undeclared fields: {sorted(extras)}")
    if sort_by:
        missing = set(sort_by) - expected
        if missing:
            raise SchemaError(f"sort fields are not in schema: {sorted(missing)}")
        materialized.sort(key=lambda row: tuple(str(row.get(key, "")) for key in sort_by))

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=list(fieldnames),
                extrasaction="raise",
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(materialized)
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
