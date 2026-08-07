"""Render native and explicit custom ORCA 6.1 SMD blocks."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Mapping


CUSTOM_SMD_FIELDS = (
    "epsilon",
    "refrac",
    "soln",
    "soln25",
    "sola",
    "solb",
    "solg",
    "solc",
    "solh",
)
CUSTOM_SMD_LIBRARY_SEED = "water"
_NATIVE_INPUT = re.compile(r"^native: SMD\((.+)\)$")


class SMDConfigurationError(ValueError):
    """Raised when a frozen registry row cannot produce an exact SMD block."""


def parse_custom_payload(payload: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for token in payload.split(";"):
        key, separator, value = token.strip().partition("=")
        if not separator or not key or not value or key in values:
            raise SMDConfigurationError(f"invalid custom SMD payload token: {token!r}")
        values[key] = value
    if set(values) != set(CUSTOM_SMD_FIELDS):
        raise SMDConfigurationError(
            f"custom SMD payload fields {sorted(values)} != {list(CUSTOM_SMD_FIELDS)}"
        )
    for field in CUSTOM_SMD_FIELDS:
        float(values[field])
    return values


def native_smd_name(row: Mapping[str, str]) -> str:
    match = _NATIVE_INPUT.fullmatch(row["orca_smd_input_from_source"])
    if match is None:
        raise SMDConfigurationError(
            f"{row['solvent_id']}: invalid native SMD input "
            f"{row['orca_smd_input_from_source']!r}"
        )
    return match.group(1)


def render_smd_block(row: Mapping[str, str]) -> str:
    mode = row["orca_smd_mode"]
    if mode == "native_orca_smd":
        return (
            "%cpcm\n"
            "  smd true\n"
            f'  SMDsolvent "{native_smd_name(row)}"\n'
            "end\n"
        )
    if mode == "custom_smd":
        values = parse_custom_payload(row["orca_parameter_payload_resolved"])
        rendered_values = "".join(
            f"  {field} {values[field]}\n" for field in CUSTOM_SMD_FIELDS
        )
        return (
            "%cpcm\n"
            "  smd true\n"
            f'  SMDsolvent "{CUSTOM_SMD_LIBRARY_SEED}"\n'
            f"{rendered_values}"
            "end\n"
        )
    raise SMDConfigurationError(
        f"{row['solvent_id']}: unsupported ORCA SMD mode {mode!r}"
    )


def smd_payload_sha256(row: Mapping[str, str]) -> str:
    if row["orca_smd_mode"] == "native_orca_smd":
        payload: object = {
            "mode": "native_orca_smd",
            "name": native_smd_name(row),
        }
    else:
        payload = {
            "mode": "custom_smd",
            "library_seed": CUSTOM_SMD_LIBRARY_SEED,
            "values": parse_custom_payload(row["orca_parameter_payload_resolved"]),
        }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()
