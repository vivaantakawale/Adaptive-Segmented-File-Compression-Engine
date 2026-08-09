"""Unpack .szip archive back into original file bytes

Reads metadata table, decompresses each chunk with its recorded algorithm, reassembles them in order, verifies the result against original size and SHA-256 checksum stored in header
Any corruption surfaces as `ArchiveError`
"""

import argparse
import os
from pathlib import Path

from src.archive import format as fmt
from src.compressors import registry


def unpack_bytes(archive: bytes) -> bytes:
    """Read .szip archive and reconstruct original data

    Args:
        archive: Serialized .szip archive bytes

    Returns:
        Reconstructed original bytes

    Raises:
        fmt.ArchiveError: If archive is malformed or fails integrity checks
    """
    header = fmt.read_archive_header(archive)

    meta_offset = fmt.metadata_table_offset()
    metas = []
    for _ in range(header.chunk_count):
        metas.append(fmt.read_chunk_meta(archive, meta_offset))
        meta_offset += fmt.chunk_meta_size()

    payload_start = fmt.payload_blob_offset(header.chunk_count)

    pieces = []
    for meta in metas:
        start = payload_start + meta.offset
        end = start + meta.compressed_length
        if end > len(archive):
            raise fmt.ArchiveError("truncated archive: chunk payload incomplete")
        compressed = archive[start:end]

        try:
            algo_name = fmt.registry_name(meta.algorithm_id)
        except (IndexError, KeyError) as e:
            raise fmt.ArchiveError(f"corrupt archive: unknown algorithm id {meta.algorithm_id}") from e

        try:
            original = registry.decompress(compressed, algo_name)
        except Exception as e:
            raise fmt.ArchiveError(f"corrupt archive: failed to decompress chunk ({e})") from e

        if len(original) != meta.original_length:
            raise fmt.ArchiveError(
                f"corrupt archive: chunk size mismatch "
                f"(expected {meta.original_length}, got {len(original)})"
            )
        pieces.append(original)

    data = b"".join(pieces)

    if len(data) != header.original_size:
        raise fmt.ArchiveError(
            f"corrupt archive: reconstructed size {len(data)} != recorded size {header.original_size}"
        )

    if fmt.compute_checksum(data) != header.checksum:
        raise fmt.ArchiveError("corrupt archive: checksum mismatch")

    return data


def unpack_file(input_path: Path, output_path: Path) -> None:
    """Unpack .szip archive on disk back into original file

    Args:
        input_path: Archive to unpack
        output_path: Where to write restored file - Written atomically

    Raises:
        fmt.ArchiveError: If the archive is malformed or fails integrity checks
    """
    archive = input_path.read_bytes()
    data = unpack_bytes(archive)
    tmp_path = output_path.with_name(output_path.name + ".partial")
    try:
        tmp_path.write_bytes(data)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
    os.replace(tmp_path, output_path)


def main() -> None:
    """CLI entry point. Parses sys.argv and runs `unpack_file`"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    unpack_file(args.input, args.output)


if __name__ == "__main__":
    main()
