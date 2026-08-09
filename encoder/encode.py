"""Encoder: manifest + source file -> compressed archive

Streams source file rather than loading whole 
holds at most one chunk's original and compressed bytes in memory at once
reading each chunk directly from offset/length manifest points to

Metadata table disk position depends on chunk_count
so this writes a placeholder table first, streams every chunk's payload right after it, 
then seeks back to fill in real table
"""

import os
from pathlib import Path
from typing import BinaryIO

from src.compressors import registry

from . import format
from .manifest import (
    ChunkRecord,
    Manifest,
    ManifestError,
    load_manifest,
    sha256_hex,
    validate_against_source,
)


class EncodeError(ValueError):
    """raised when a chunk fails to validate or compress"""


def _read_chunk(source: BinaryIO, record: ChunkRecord) -> bytes:
    """Read one chunk's bytes from source file

    Args:
        source: Open source file, positioned anywhere (seeks internally)
        record: Chunk's offset/length in source file

    Returns:
        Chunk's raw bytes

    Raises:
        EncodeError: If less than `record.length` bytes available
    """
    source.seek(record.offset)
    data = source.read(record.length)
    if len(data) != record.length:
        raise EncodeError(
            f"short read at offset {record.offset}: expected {record.length} bytes, got {len(data)}"
        )
    return data


def encode(manifest_path: Path, output_path: Path) -> None:
    """Encode manifest + source file into SZE1 archive on disk

    Args:
        manifest_path: Path to manifest produced by `encoder.manifest`
        output_path: archive output path - written atomically

    Raises:
        ManifestError: If manifest malformed or stale relative to its source file
        EncodeError: If chunk fails checksum validation or compression
    """
    manifest: Manifest = load_manifest(manifest_path)
    source_path = Path(manifest.source_file)
    validate_against_source(manifest, source_path)

    #Write to temp path and rename into place so failure mid write doesnt leave corrupt archive at output_path
    tmp_path = output_path.with_name(output_path.name + ".partial")
    try:
        with open(source_path, "rb") as source, open(tmp_path, "wb") as out:
            chunk_count = len(manifest.chunks)

            out.write(format.pack_header(
                original_size=manifest.source_size,
                original_checksum=bytes.fromhex(manifest.source_sha256),
                chunk_count=chunk_count,
            ))

            table_offset = out.tell()
            placeholder = b"\x00" * format.chunk_meta_size() * chunk_count
            out.write(placeholder)

            payload_start = out.tell()
            metas: list[bytes] = []

            for i, record in enumerate(manifest.chunks):
                data = _read_chunk(source, record)
                actual_checksum = sha256_hex(data)
                if actual_checksum != record.checksum:
                    raise EncodeError(
                        f"chunk {i} checksum mismatch: manifest says {record.checksum}, "
                        f"actual bytes hash to {actual_checksum} -- source file may have "
                        f"changed since the manifest was produced"
                    )

                compressed = registry.compress(data, record.algorithm)
                payload_offset = out.tell() - payload_start
                out.write(compressed)

                metas.append(format.pack_chunk_meta(
                    algorithm_id=format.registry_index(record.algorithm),
                    original_length=record.length,
                    compressed_length=len(compressed),
                    checksum=bytes.fromhex(record.checksum),
                    payload_offset=payload_offset,
                ))

            out.seek(table_offset)
            out.write(b"".join(metas))
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise

    os.replace(tmp_path, output_path)


__all__ = ["encode", "EncodeError"]
