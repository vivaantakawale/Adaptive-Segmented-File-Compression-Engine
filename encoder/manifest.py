"""Manifest format connecting ML chunking/prediction step to encoder

Records chunk boundaries (as offsets into source file) and a predicted algorithm per chunk. 
Encoder re-validates source file's size/checksum and re-checksums every chunk before compressing it, 
so stale or hand edited manifest gets caught and output not corrupted
"""

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

MANIFEST_VERSION = 1


class ManifestError(ValueError):
    """Raised when manifest is malformed or fails validation against source file"""


@dataclass
class ChunkRecord:
    offset: int
    length: int
    algorithm: str
    checksum: str  # hex sha256 of chunk's original bytes


@dataclass
class Manifest:
    source_file: str
    source_size: int
    source_sha256: str
    chunks: list[ChunkRecord] = field(default_factory=list)
    version: int = MANIFEST_VERSION


def sha256_hex(data: bytes) -> str:
    """Compute a hex SHA-256 digest

    Args:
        data: Bytes to hash

    Returns:
        Hex encoded SHA-256 digest
    """
    return hashlib.sha256(data).hexdigest()


def sha256_file_hex(path: Path, buf_size: int = 1024 * 1024) -> str:
    """Stream file through SHA-256 without loading fully into memory

    Args:
        path: File to hash
        buf_size: Read buffer size in bytes

    Returns:
        Hex encoded SHA-256 digest of file's contents
    """
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(buf_size):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(source_file: Path, chunk_records: list[ChunkRecord]) -> Manifest:
    """Build Manifest, stamping source_size/source_sha256 from file on disk

    Args:
        source_file: Source file the chunks were taken from
        chunk_records: Chunk boundaries and algorithms, in order

    Returns:
        Manifest describing `source_file` and `chunk_records`
    """
    source_file = Path(source_file)
    return Manifest(
        source_file=str(source_file),
        source_size=source_file.stat().st_size,
        source_sha256=sha256_file_hex(source_file),
        chunks=chunk_records,
    )


def write_manifest(manifest: Manifest, out_path: Path) -> None:
    """Serialize Manifest to JSON on disk

    Args:
        manifest: Manifest to write
        out_path: Destination path
    """
    payload = asdict(manifest)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_manifest(path: Path) -> Manifest:
    """Load Manifest from JSON file

    Args:
        path: Manifest file to read

    Returns:
        Parsed Manifest

    Raises:
        ManifestError: If file's contents aren't a valid manifest
    """
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    try:
        chunks = [ChunkRecord(**c) for c in raw["chunks"]]
        return Manifest(
            source_file=raw["source_file"],
            source_size=raw["source_size"],
            source_sha256=raw["source_sha256"],
            chunks=chunks,
            version=raw.get("version", MANIFEST_VERSION),
        )
    except (KeyError, TypeError) as exc:
        raise ManifestError(f"malformed manifest: {exc}") from exc


def validate_against_source(manifest: Manifest, source_path: Path) -> None:
    """Validate if manifest matches the file it describes

    Args:
        manifest: Manifest to validate
        source_path: File the manifest claims to describe

    Raises:
        ManifestError: If manifest version, size, checksum, or chunk boundaries don't match file at `source_path`
    """
    source_path = Path(source_path)
    if manifest.version != MANIFEST_VERSION:
        raise ManifestError(
            f"unsupported manifest version {manifest.version} (expected {MANIFEST_VERSION})"
        )
    actual_size = source_path.stat().st_size
    if actual_size != manifest.source_size:
        raise ManifestError(
            f"source size mismatch: manifest says {manifest.source_size}, "
            f"file is {actual_size} bytes"
        )
    actual_sha256 = sha256_file_hex(source_path)
    if actual_sha256 != manifest.source_sha256:
        raise ManifestError(
            "source checksum mismatch: manifest does not match this file's contents"
        )
    end_of_file = 0
    for i, c in enumerate(manifest.chunks):
        if c.offset < 0 or c.length < 0:
            raise ManifestError(f"chunk {i}: negative offset/length")
        if c.offset != end_of_file:
            raise ManifestError(
                f"chunk {i}: offset {c.offset} does not follow previous chunk end {end_of_file}"
            )
        end_of_file = c.offset + c.length
    if end_of_file != manifest.source_size:
        raise ManifestError(
            f"chunks cover {end_of_file} bytes, expected {manifest.source_size}"
        )


__all__ = [
    "MANIFEST_VERSION",
    "ManifestError",
    "ChunkRecord",
    "Manifest",
    "sha256_hex",
    "sha256_file_hex",
    "build_manifest",
    "write_manifest",
    "load_manifest",
    "validate_against_source",
]
