"""Binary layout for .szip archive container format

Layout:

    --- archive header (fixed size, see `_HEADER`) ---
    MAGIC (4 bytes)              b"SZIP"
    version (uint8)
    original_size (uint64, BE)   size of original (unchunked) file
    checksum (32 bytes)          SHA-256 digest of original file
    chunk_count (uint32, BE)

    --- chunk metadata table (chunk_count fixed-size entries, see `_CHUNK_META`) ---
    per entry:
        algorithm_id (uint8)      index into src.compressors.registry.ALGORITHM_NAMES
        original_length (uint32, BE)
        compressed_length (uint32, BE)
        offset (uint64, BE)       byte offset of chunk's payload, relative to start of payload blob

    --- payload blob ---
    concatenated compressed bytes of every chunk, in same order as metadata table

Metadata fully separated from chunk payloads, so whole table can be read and validated in one shot before touching any payload bytes
"""

import hashlib
import struct
from dataclasses import dataclass

from src.compressors.registry import ALGORITHM_NAMES

MAGIC = b"SZIP"
VERSION = 1
CHECKSUM_SIZE = 32  # sha256 digest length


class ArchiveError(ValueError):
    """Raised when archive is malformed, truncated, or fails integrity verification 
    (checksum mismatch, chunk size mismatch, or chunk fails to decompress)"""

# magic, version, original_size, checksum, chunk_count
_HEADER = struct.Struct(f">4sBQ{CHECKSUM_SIZE}sI")

# algorithm_id, original_length, compressed_length, offset
_CHUNK_META = struct.Struct(">BIIQ")


@dataclass
class ArchiveHeader:
    version: int
    original_size: int
    checksum: bytes
    chunk_count: int


@dataclass
class ChunkMeta:
    algorithm_id: int
    original_length: int
    compressed_length: int
    offset: int


def compute_checksum(data: bytes) -> bytes:
    """Compute checksum stored in archive header

    Args:
        data: Original file bytes

    Returns:
        SHA-256 digest of `data`
    """
    return hashlib.sha256(data).digest()


def write_archive_header(original_size: int, checksum: bytes, chunk_count: int) -> bytes:
    """Serialize archive header

    Args:
        original_size: Size of original (unchunked) file in bytes
        checksum: 32-byte SHA-256 digest of original file
        chunk_count: Number of chunks in archive

    Returns:
        Packed header bytes

    Raises:
        ValueError: If `checksum` isn't CHECKSUM_SIZE bytes
    """
    if len(checksum) != CHECKSUM_SIZE:
        raise ValueError(f"checksum must be {CHECKSUM_SIZE} bytes, got {len(checksum)}")
    return _HEADER.pack(MAGIC, VERSION, original_size, checksum, chunk_count)


def read_archive_header(buf: bytes, offset: int = 0) -> ArchiveHeader:
    """Parse archive header

    Args:
        buf: Buffer containing archive
        offset: Byte offset where header starts

    Returns:
        Parsed ArchiveHeader

    Raises:
        ArchiveError: If buffer is truncated or magic doesn't match
    """
    if len(buf) < offset + _HEADER.size:
        raise ArchiveError("truncated archive: header incomplete")
    magic, version, original_size, checksum, chunk_count = _HEADER.unpack_from(buf, offset)
    if magic != MAGIC:
        raise ArchiveError(f"not szip archive (bad magic: {magic!r})")
    return ArchiveHeader(
        version=version,
        original_size=original_size,
        checksum=checksum,
        chunk_count=chunk_count,
    )


def header_size() -> int:
    """Return fixed size of packed archive header in bytes"""
    return _HEADER.size


def chunk_meta_size() -> int:
    """Return fixed size of one packed chunk metadata entry in bytes"""
    return _CHUNK_META.size


def write_chunk_meta(algorithm_id: int, original_length: int, compressed_length: int, offset: int) -> bytes:
    """Serialize one chunk metadata entry

    Args:
        algorithm_id: disk algorithm id (see `registry_index`)
        original_length: Chunk's uncompressed length in bytes
        compressed_length: Chunk's compressed length in bytes
        offset: Byte offset of chunk's payload relative to start of payload blob

    Returns:
        Packed chunk metadata bytes
    """
    return _CHUNK_META.pack(algorithm_id, original_length, compressed_length, offset)


def read_chunk_meta(buf: bytes, offset: int) -> ChunkMeta:
    """Parse one chunk metadata entry

    Args:
        buf: Buffer containing archive
        offset: Byte offset where metadata entry starts

    Returns:
        Parsed ChunkMeta

    Raises:
        ArchiveError: If buffer is truncated
    """
    if len(buf) < offset + _CHUNK_META.size:
        raise ArchiveError("truncated archive: chunk metadata incomplete")
    algorithm_id, original_length, compressed_length, chunk_offset = _CHUNK_META.unpack_from(
        buf, offset
    )
    return ChunkMeta(
        algorithm_id=algorithm_id,
        original_length=original_length,
        compressed_length=compressed_length,
        offset=chunk_offset,
    )


def metadata_table_offset() -> int:
    """Return byte offset where chunk metadata table begins"""
    return header_size()


def payload_blob_offset(chunk_count: int) -> int:
    """Return byte offset where payload blob begins

    Args:
        chunk_count: Number of chunks in archive

    Returns:
        Byte offset of start of payload blob
    """
    return header_size() + chunk_count * chunk_meta_size()


def registry_index(algorithm_name: str) -> int:
    """Map algorithm name to its stable disk id

    Args:
        algorithm_name: Algorithm name

    Returns:
        disk algorithm id
    """
    return ALGORITHM_NAMES.index(algorithm_name)


def registry_name(algorithm_id: int) -> str:
    """Map disk algorithm id back to name

    Args:
        algorithm_id: disk algorithm id

    Returns:
        Algorithm name
    """
    return ALGORITHM_NAMES[algorithm_id]
