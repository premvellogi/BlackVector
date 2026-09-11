"""
SatQuery AI - B2 Deterministic Input Hasher
===========================================
Chunked, memory-safe cryptographic hashing for satellite raster files and payloads.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Union


def calculate_file_hash(file_path: Union[str, Path], chunk_size: int = 65536) -> str:
    """
    Computes a deterministic SHA-256 cryptographic digest of a file.
    Streams chunks to avoid loading multi-gigabyte satellite imagery into RAM.

    Parameters
    ----------
    file_path : str | Path
        Path to file on disk.
    chunk_size : int, default 64KB (65536 bytes)
        Buffer chunk size for streaming read.

    Returns
    -------
    str
        Hexadecimal SHA-256 digest string.
    """
    path = Path(file_path).resolve()
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Cannot calculate hash: File not found at '{file_path}'")

    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)

    return hasher.hexdigest()


def calculate_bytes_hash(data: bytes) -> str:
    """Computes SHA-256 hash of in-memory raw bytes."""
    return hashlib.sha256(data).hexdigest()


def calculate_dict_hash(data: Dict[str, Any]) -> str:
    """Computes deterministic SHA-256 hash of a JSON-serializable dictionary."""
    canonical_json = json.dumps(data, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


def verify_file_hash(file_path: Union[str, Path], expected_hash: str) -> bool:
    """
    Verifies that a file's SHA-256 digest matches an expected recorded hash.
    """
    try:
        actual_hash = calculate_file_hash(file_path)
        return actual_hash.lower() == expected_hash.strip().lower()
    except Exception:
        return False
