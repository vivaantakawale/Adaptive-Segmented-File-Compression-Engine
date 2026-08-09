"""Decoder: archive -> restored original file

Reads header and metadata table (small, O(chunk count) not O(filesize)) in one shot,
Then streams payload blob: 
seek to each chunk, decompress, verify checksum, write restored bytes
Only one chunk's bytes are resident at once
"""

import hashlib
import os
from pathlib import Path

from src.compressors import registry

from encoder import format
from encoder.manifest import sha256_hex


class DecodeError(format.ArchiveError):
    """Raised when archive fails to decode: bad magic, truncated data, chunk fails checksum, or final size/checksum mismatch against header"""


def decode(archive_path: Path, output_path: Path) -> None:
    """Decode SZE1 archive on disk back into original file

    Args:
        archive_path: Archive to decode
        output_path: Restored file output path - written atomically

    Raises:
        DecodeError: If archive is malformed, truncated, or fails any checksum/size verification
    """
    with open(archive_path, "rb") as src:
        header_bytes = src.read(format.header_size())
        try:
            header = format.unpack_header(header_bytes)
        except format.ArchiveError as exc:
            raise DecodeError(str(exc)) from exc

        table_bytes = src.read(header.chunk_count * format.chunk_meta_size())
        if len(table_bytes) != header.chunk_count * format.chunk_meta_size():
            raise DecodeError("truncated archive: metadata table incomplete")
        try:
            metas = [
                format.unpack_chunk_meta(table_bytes, i * format.chunk_meta_size())
                for i in range(header.chunk_count)
            ]
        except format.ArchiveError as exc:
            raise DecodeError(str(exc)) from exc

        payload_start = src.tell()
        assert payload_start == format.payload_blob_offset(header.chunk_count)

        overall_hash = hashlib.sha256()
        total_written = 0

        tmp_path = output_path.with_name(output_path.name + ".partial")
        try:
            with open(tmp_path, "wb") as out:
                for i, meta in enumerate(metas):
                    src.seek(payload_start + meta.payload_offset)
                    compressed = src.read(meta.compressed_length)
                    if len(compressed) != meta.compressed_length:
                        raise DecodeError(f"chunk {i}: truncated payload")

                    algo_name = format.registry_name(meta.algorithm_id)
                    try:
                        restored = registry.decompress(compressed, algo_name)
                    except Exception as exc:
                        raise DecodeError(
                            f"chunk {i}: failed to decompress ({algo_name}): {exc}"
                        ) from exc

                    if len(restored) != meta.original_length:
                        raise DecodeError(
                            f"chunk {i}: decompressed length {len(restored)} != "
                            f"expected {meta.original_length}"
                        )
                    if sha256_hex(restored) != meta.checksum.hex():
                        raise DecodeError(f"chunk {i}: checksum mismatch after decompression")

                    out.write(restored)
                    overall_hash.update(restored)
                    total_written += len(restored)

            if total_written != header.original_size:
                raise DecodeError(
                    f"restored size {total_written} != header original_size {header.original_size}"
                )
            if overall_hash.digest() != header.original_checksum:
                raise DecodeError("restored file checksum doesnt match archive header")
        except BaseException:
            tmp_path.unlink(missing_ok=True)
            raise

        os.replace(tmp_path, output_path)


__all__ = ["decode", "DecodeError"]
