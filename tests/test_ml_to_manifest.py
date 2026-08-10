"""Tests for ML-box -> encoder connector (scripts/ml_to_manifest.py)"""

import os

import pytest

from encoder import format
from encoder.encode import encode
from encoder.manifest import load_manifest, sha256_hex, validate_against_source, write_manifest
from src.chunking import content_aware
from scripts.ml_to_manifest import OutOfDistributionChunkWarning, build_manifest_from_file

pytest.importorskip("sklearn")

MODEL_PATH_MISSING = "models/algo_selector.joblib not present in this checkout"


def _has_shipped_model() -> bool:
    from pathlib import Path

    return Path("models/algo_selector.joblib").exists()


@pytest.mark.skipif(not _has_shipped_model(), reason=MODEL_PATH_MISSING)
def test_connector_output_is_valid_manifest(tmp_path):
    src = tmp_path / "sample.txt"
    src.write_text("hello world " * 2000)

    manifest = build_manifest_from_file(src, chunk_size=512)

    assert manifest.source_size == src.stat().st_size
    assert sum(c.length for c in manifest.chunks) == manifest.source_size
    for c in manifest.chunks:
        chunk_bytes = src.read_bytes()[c.offset : c.offset + c.length]
        assert sha256_hex(chunk_bytes) == c.checksum

    validate_against_source(manifest, src)  # should not raise


@pytest.mark.skipif(not _has_shipped_model(), reason=MODEL_PATH_MISSING)
def test_connector_output_feeds_encoder_end_to_end(tmp_path):
    src = tmp_path / "sample.bin"
    src.write_bytes((b"abc123" * 5000) + bytes(range(256)) * 20)

    manifest = build_manifest_from_file(src, chunk_size=1024)
    manifest_path = tmp_path / "sample.manifest.json"
    write_manifest(manifest, manifest_path)

    reloaded = load_manifest(manifest_path)

    out_path = tmp_path / "sample.archive"
    encode(manifest_path, out_path)

    raw = out_path.read_bytes()
    header = format.unpack_header(raw)
    assert header.original_size == src.stat().st_size
    assert header.chunk_count == len(reloaded.chunks)


@pytest.mark.skipif(not _has_shipped_model(), reason=MODEL_PATH_MISSING)
def test_content_aware_streaming_matches_in_memory_chunker(tmp_path):
    """Streaming connector must produce byte-identical chunk boundaries to
    the in-memory content_aware.chunk(). Data includes a hex-encoded blob
    so the hex_ratio check is exercised too, not just printable/entropy."""
    hex_blob = os.urandom(3000).hex().encode("ascii")
    data = (
        (b"the quick brown fox " * 200)
        + os.urandom(3000)
        + hex_blob
        + (b"more english text here " * 200)
    )
    src = tmp_path / "mixed.bin"
    src.write_bytes(data)

    expected_chunks = list(content_aware.chunk(data))
    expected_offsets = []
    pos = 0
    for c in expected_chunks:
        expected_offsets.append((pos, len(c)))
        pos += len(c)

    manifest = build_manifest_from_file(src, chunker="content_aware", batch_size=4)

    actual_offsets = [(c.offset, c.length) for c in manifest.chunks]
    assert actual_offsets == expected_offsets
    assert sum(c.length for c in manifest.chunks) == len(data)

    for record, expected_chunk in zip(manifest.chunks, expected_chunks):
        assert record.checksum == sha256_hex(expected_chunk)


@pytest.mark.skipif(not _has_shipped_model(), reason=MODEL_PATH_MISSING)
def test_content_aware_warns_on_oversized_chunk(tmp_path):
    """A homogeneous text file collapses content_aware to one chunk far
    bigger than what the shipped model trained on -- must warn."""
    data = b"the quick brown fox jumps over the lazy dog. " * 3000  # ~140KB, all "text"
    src = tmp_path / "homogeneous.txt"
    src.write_bytes(data)

    with pytest.warns(OutOfDistributionChunkWarning):
        manifest = build_manifest_from_file(src, chunker="content_aware")

    assert len(manifest.chunks) == 1
    assert manifest.chunks[0].length == len(data)


@pytest.mark.skipif(not _has_shipped_model(), reason=MODEL_PATH_MISSING)
def test_fixed_size_does_not_warn(tmp_path, recwarn):
    """fixed_size chunks should never trigger the oversized-chunk warning."""
    src = tmp_path / "sample.bin"
    src.write_bytes(os.urandom(50_000))

    build_manifest_from_file(src, chunker="fixed_size", chunk_size=4096)

    assert not any(issubclass(w.category, OutOfDistributionChunkWarning) for w in recwarn.list)
