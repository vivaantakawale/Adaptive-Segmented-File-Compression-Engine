"""Tests for encoder/encode.py

Build manifest by hand, then run the encoder against it and check resulting archive's structure and integrity guarantees
"""

import os

import pytest

from encoder import format
from encoder.encode import EncodeError, encode
from encoder.manifest import (
    ChunkRecord,
    ManifestError,
    build_manifest,
    sha256_hex,
    validate_against_source,
    write_manifest,
)


def _manifest_for(path, chunk_size, algorithm="store"):
    data = path.read_bytes()
    records = []
    for start in range(0, len(data), chunk_size):
        piece = data[start : start + chunk_size]
        records.append(ChunkRecord(
            offset=start, length=len(piece), algorithm=algorithm, checksum=sha256_hex(piece)
        ))
    return build_manifest(path, records)


def test_encode_roundtrip_structure(tmp_path):
    src = tmp_path / "input.bin"
    src.write_bytes(os.urandom(50_000))

    manifest = _manifest_for(src, chunk_size=4096, algorithm="zstd")
    manifest_path = tmp_path / "input.manifest.json"
    write_manifest(manifest, manifest_path)

    out_path = tmp_path / "out.archive"
    encode(manifest_path, out_path)

    raw = out_path.read_bytes()
    header = format.unpack_header(raw)
    assert header.original_size == src.stat().st_size
    assert header.chunk_count == len(manifest.chunks)

    table_offset = format.metadata_table_offset()
    payload_start = format.payload_blob_offset(header.chunk_count)
    metas = [
        format.unpack_chunk_meta(raw, table_offset + i * format.chunk_meta_size())
        for i in range(header.chunk_count)
    ]
    assert all(format.registry_name(m.algorithm_id) == "zstd" for m in metas)

    # every chunk's payload must round-trip through its own algorithm
    for record, meta in zip(manifest.chunks, metas):
        payload = raw[
            payload_start + meta.payload_offset : payload_start + meta.payload_offset + meta.compressed_length
        ]
        from src.compressors import registry as compressor_registry

        restored = compressor_registry.decompress(payload, "zstd")
        assert sha256_hex(restored) == record.checksum


def test_encode_rejects_stale_manifest(tmp_path):
    src = tmp_path / "input.bin"
    src.write_bytes(b"original content")
    manifest = _manifest_for(src, chunk_size=8)
    manifest_path = tmp_path / "input.manifest.json"
    write_manifest(manifest, manifest_path)

    # mutate source after manifest was produced
    src.write_bytes(b"totally different content, same length!")

    with pytest.raises(ManifestError):
        encode(manifest_path, tmp_path / "out.archive")


def test_validate_against_source_detects_size_mismatch(tmp_path):
    src = tmp_path / "input.bin"
    src.write_bytes(b"abc" * 100)
    manifest = _manifest_for(src, chunk_size=16)

    src.write_bytes(b"abc" * 200)
    with pytest.raises(ManifestError):
        validate_against_source(manifest, src)


def test_encode_rejects_tampered_chunk_checksum(tmp_path):
    src = tmp_path / "input.bin"
    src.write_bytes(os.urandom(1000))
    manifest = _manifest_for(src, chunk_size=100)
    manifest.chunks[2].checksum = "0" * 64  # corrupt one record's checksum
    manifest_path = tmp_path / "input.manifest.json"
    write_manifest(manifest, manifest_path)

    with pytest.raises(EncodeError):
        encode(manifest_path, tmp_path / "out.archive")


def test_encode_handles_variable_chunk_sizes(tmp_path):
    src = tmp_path / "input.bin"
    data = os.urandom(10_000)
    src.write_bytes(data)

    sizes = [100, 5000, 2000, 2900]
    assert sum(sizes) == len(data)
    records = []
    pos = 0
    for size, algo in zip(sizes, ["store", "gzip", "brotli", "store"]):
        piece = data[pos : pos + size]
        records.append(ChunkRecord(offset=pos, length=size, algorithm=algo, checksum=sha256_hex(piece)))
        pos += size
    manifest = build_manifest(src, records)
    manifest_path = tmp_path / "input.manifest.json"
    write_manifest(manifest, manifest_path)

    out_path = tmp_path / "out.archive"
    encode(manifest_path, out_path)

    raw = out_path.read_bytes()
    header = format.unpack_header(raw)
    assert header.chunk_count == 4
    payload_start = format.payload_blob_offset(header.chunk_count)
    total_payload = len(raw) - payload_start
    assert total_payload > 0
