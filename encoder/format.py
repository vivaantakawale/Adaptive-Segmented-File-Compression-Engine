"""Binary layout for encoder's archive container (SZE1)

Separate from src/archive/format.py's .szip (older pipeline)
Carries per-chunk checksum to decoder to verify each chunk independently

Layout:

    --- header (fixed size, see `_HEADER`) ---
    MAGIC (4 bytes)              b"SZE1"
    version (uint8)
    original_size (uint64, BE)   size of original (unchunked) file
    original_checksum (32 bytes) SHA-256 digest of original file
    chunk_count (uint32, BE)

    --- chunk metadata table (chunk_count fixed-size entries) ---
    per entry:
        algorithm_id (uint8)       index into src.compressors.registry.ALGORITHM_NAMES
        original_length (uint32, BE)
        compressed_length (uint32, BE)
        checksum (32 bytes)        SHA-256 of this chunk's original bytes
        payload_offset (uint64, BE) byte offset of this chunk's payload relative to start of payload blob

    --- payload blob ---
    concatenated compressed bytes of every chunk, in same order as metadata table

Metadata is fully separated from payload bytes so whole table can be read and validated before touching any payload bytes
"""

import struct
from dataclasses import dataclass

from src.compressors.registry import ALGORITHM_NAMES

MAGIC = b"SZE1"
VERSION = 1
CHECKSUM_SIZE = 32  # sha256 digest length


class ArchiveError(ValueError):
    """Raised when archive is malformed, truncated, or fails integrity verification (checksum mismatch or chunk fails to decompress)"""


# magic, version, original_size, original_checksum, chunk_count
_HEADER = struct.Struct(f">4sBQ{CHECKSUM_SIZE}sI")

# algorithm_id, original_length, compressed_length, checksum, payload_offset
_CHUNK_META = struct.Struct(f">BII{CHECKSUM_SIZE}sQ")


@dataclass
class ArchiveHeader:
    version: int
    original_size: int
    original_checksum: bytes
    chunk_count: int


@dataclass
class ChunkMeta:
    algorithm_id: int
    original_length: int
    compressed_length: int
    checksum: bytes
    payload_offset: int


def header_size() -> int:
    """Return fixed size of acked archive header in bytes"""
    return _HEADER.size


def chunk_meta_size() -> int:
    """Return fixed size of one packed chunk metadata entry in bytes"""
    return _CHUNK_META.size


def metadata_table_offset() -> int:
    """Return byte offset where chunk metadata table begins"""
    return header_size()


def payload_blob_offset(chunk_count: int) -> int:
    """Return byte offset where payload blob begins, given `chunk_count`"""
    return header_size() + chunk_count * chunk_meta_size()


def pack_header(original_size: int, original_checksum: bytes, chunk_count: int) -> bytes:
    """Serialize archive header
    Raises ValueError if `original_checksum` isn't CHECKSUM_SIZE bytes"""
    if len(original_checksum) != CHECKSUM_SIZE:
        raise ValueError(
            f"original_checksum must be {CHECKSUM_SIZE} bytes, got {len(original_checksum)}"
        )
    return _HEADER.pack(MAGIC, VERSION, original_size, original_checksum, chunk_count)


def unpack_header(buf: bytes, offset: int = 0) -> ArchiveHeader:
    """Parse an archive header from `buf` at `offset`
    Raises ArchiveError if truncated or magic doesn't match"""
    if len(buf) < offset + _HEADER.size:
        raise ArchiveError("truncated archive: header incomplete")
    magic, version, original_size, original_checksum, chunk_count = _HEADER.unpack_from(
        buf, offset
    )
    if magic != MAGIC:
        raise ArchiveError(f"not a recognized archive (bad magic: {magic!r})")
    return ArchiveHeader(
        version=version,
        original_size=original_size,
        original_checksum=original_checksum,
        chunk_count=chunk_count,
    )


def pack_chunk_meta(
    algorithm_id: int,
    original_length: int,
    compressed_length: int,
    checksum: bytes,
    payload_offset: int,
) -> bytes:
    """Serialize one chunk-metadata entry
    Raises ValueError if `checksum` isn't CHECKSUM_SIZE bytes"""
    if len(checksum) != CHECKSUM_SIZE:
        raise ValueError(f"checksum must be {CHECKSUM_SIZE} bytes, got {len(checksum)}")
    return _CHUNK_META.pack(
        algorithm_id, original_length, compressed_length, checksum, payload_offset
    )


def unpack_chunk_meta(buf: bytes, offset: int) -> ChunkMeta:
    """Parse one chunk metadata entry from `buf` at `offset`
    Raises ArchiveError if truncated"""
    if len(buf) < offset + _CHUNK_META.size:
        raise ArchiveError("truncated archive: chunk metadata incomplete")
    algorithm_id, original_length, compressed_length, checksum, payload_offset = (
        _CHUNK_META.unpack_from(buf, offset)
    )
    return ChunkMeta(
        algorithm_id=algorithm_id,
        original_length=original_length,
        compressed_length=compressed_length,
        checksum=checksum,
        payload_offset=payload_offset,
    )


def registry_index(algorithm_name: str) -> int:
    """Map algorithm name to its stable disk id"""
    return ALGORITHM_NAMES.index(algorithm_name)


def registry_name(algorithm_id: int) -> str:
    """Map disk algorithm id back to its name"""
    return ALGORITHM_NAMES[algorithm_id]


__all__ = [
    "MAGIC",
    "VERSION",
    "CHECKSUM_SIZE",
    "ArchiveError",
    "ArchiveHeader",
    "ChunkMeta",
    "header_size",
    "chunk_meta_size",
    "metadata_table_offset",
    "payload_blob_offset",
    "pack_header",
    "unpack_header",
    "pack_chunk_meta",
    "unpack_chunk_meta",
    "registry_index",
    "registry_name",
]
