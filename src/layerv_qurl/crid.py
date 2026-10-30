"""CRID validation and delivered-key binding, matching qurl-go/crid.

A valid CRID is a locator, not authorization. Unknown nonzero versions pass
this local gate; the service decides whether they can be used. A checksum
only detects typos. Key matching binds the exact DER bytes, not a signed link.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import re


def _payload(value: str) -> bytes:
    if not isinstance(value, str) or not re.fullmatch(r"[a-z2-7]*", value):
        raise ValueError("crid: charset outside lowercase unpadded base32")
    if len(value) not in (47, 60):
        raise ValueError("crid: length must be 47 or 60")
    decoded = base64.b32decode(value.upper() + "=" * (-len(value) % 8))
    if base64.b32encode(decoded).decode().lower().rstrip("=") != value:
        raise ValueError("crid: non_canonical trailing bits")
    payload, checksum = decoded[:-4], decoded[-4:]
    # CRC32C (Castagnoli), not zlib's IEEE CRC32. Inputs are at most 33 bytes.
    crc = 0xFFFFFFFF
    for byte in payload:
        crc ^= byte
        for _ in range(8):
            crc = (crc >> 1) ^ (0x82F63B78 if crc & 1 else 0)
    if (crc ^ 0xFFFFFFFF).to_bytes(4, "big") != checksum:
        raise ValueError("crid: checksum mismatch")
    if payload[0] == 0:
        raise ValueError("crid: version zero is forbidden")
    return payload


def matches_shape(value: str) -> bool:
    """Dispatch predicate matching Go MatchesShape; never substitutes for validate."""
    return re.fullmatch(r"(?:[a-z2-7]{47}|[a-z2-7]{60})", value) is not None


def validate(value: str) -> str:
    """Return a locally valid CRID unchanged, or raise ValueError.

    Never trims or normalizes. Does not assert that the resource exists.
    """
    _payload(value)
    return value


def key_matches(value: str, der_spki: bytes) -> bool:
    """Check exact delivered DER SPKI bytes against an independently held CRID."""
    digest = _payload(value)[1:]
    # Match Go: the same binding rule applies to forwardable version bytes.
    expected = hashlib.sha256(b"NHP-QURL-CRID-V1\x00" + der_spki).digest()
    return hmac.compare_digest(digest, expected[:len(digest)])
