"""Tests for decoder/decode.py

Builds manifest by hand, runs it through the encoder, then verifies decoder reverses that exactly and detects corruption
"""

import os

import pytest

from decoder.decode import DecodeError, decode
from encoder import format
from encoder.encode import encode
from encoder.manifest import ChunkRecord, build_manifest, sha256_hex, write_manifest


def _manifest_for(path, chunk_size, algorithm):
    data = path.read_bytes()
    records = []
    for start in range(0, len(data), chunk_size):
        piece = data[start : start + chunk_size]
        records.append(ChunkRecord(
            offset=start, length=len(piece), algorithm=algorithm, checksum=sha256_hex(piece)
        ))
    return build_manifest(path, records)


def _make_archive(tmp_path, data: bytes, chunk_size: int, algorithm: str):
    src = tmp_path / "input.bin"
    src.write_bytes(data)
    manifest = _manifest_for(src, chunk_size, algorithm)
    manifest_path = tmp_path / "input.manifest.json"
    write_manifest(manifest, manifest_path)
    archive_path = tmp_path / "input.archive"
    encode(manifest_path, archive_path)
    return src, archive_path


@pytest.mark.parametrize("algorithm", ["store", "gzip", "zstd", "brotli", "bzip2", "lzma"])
def test_roundtrip_byte_identical(tmp_path, algorithm):
    data = os.urandom(3000) + b"repetitive text " * 500
    src, archive_path = _make_archive(tmp_path, data, chunk_size=777, algorithm=algorithm)

    restored_path = tmp_path / "restored.bin"
    decode(archive_path, restored_path)

    assert restored_path.read_bytes() == src.read_bytes()


def test_roundtrip_empty_file(tmp_path):
    src, archive_path = _make_archive(tmp_path, b"", chunk_size=4096, algorithm="store")
    restored_path = tmp_path / "restored.bin"
    decode(archive_path, restored_path)
    assert restored_path.read_bytes() == b""


def test_roundtrip_variable_chunk_sizes(tmp_path):
    data = os.urandom(20_000)
    src = tmp_path / "input.bin"
    src.write_bytes(data)

    sizes = [100, 9000, 3000, 7900]
    algos = ["store", "gzip", "brotli", "zstd"]
    records = []
    pos = 0
    for size, algo in zip(sizes, algos):
        piece = data[pos : pos + size]
        records.append(ChunkRecord(offset=pos, length=size, algorithm=algo, checksum=sha256_hex(piece)))
        pos += size
    manifest = build_manifest(src, records)
    manifest_path = tmp_path / "input.manifest.json"
    write_manifest(manifest, manifest_path)
    archive_path = tmp_path / "input.archive"
    encode(manifest_path, archive_path)

    restored_path = tmp_path / "restored.bin"
    decode(archive_path, restored_path)
    assert restored_path.read_bytes() == data


def test_decode_rejects_bad_magic(tmp_path):
    _, archive_path = _make_archive(tmp_path, b"hello world", chunk_size=4096, algorithm="store")
    raw = bytearray(archive_path.read_bytes())
    raw[0:4] = b"XXXX"
    archive_path.write_bytes(bytes(raw))

    with pytest.raises(format.ArchiveError):
        decode(archive_path, tmp_path / "restored.bin")


def test_decode_rejects_truncated_payload(tmp_path):
    _, archive_path = _make_archive(tmp_path, os.urandom(10_000), chunk_size=1000, algorithm="zstd")
    raw = archive_path.read_bytes()
    archive_path.write_bytes(raw[: len(raw) - 50])  # chop off tail of payload blob

    with pytest.raises(DecodeError):
        decode(archive_path, tmp_path / "restored.bin")


def test_decode_rejects_tampered_payload_bytes(tmp_path):
    _, archive_path = _make_archive(tmp_path, b"consistent content " * 1000, chunk_size=4096, algorithm="store")
    raw = bytearray(archive_path.read_bytes())
    # flip a byte well past header+metadata table, inside payload blob
    flip_at = len(raw) - 10
    raw[flip_at] ^= 0xFF
    archive_path.write_bytes(bytes(raw))

    with pytest.raises(DecodeError):
        decode(archive_path, tmp_path / "restored.bin")
