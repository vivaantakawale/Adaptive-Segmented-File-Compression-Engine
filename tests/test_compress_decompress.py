"""Tests for user-facing chain scripts (scripts/compress.py, scripts/decompress.py)"""

import os
from pathlib import Path

import pytest

from scripts.compress import compress
from scripts.decompress import decompress

pytest.importorskip("sklearn")


def _has_shipped_model() -> bool:
    return Path("models/algo_selector.joblib").exists()


pytestmark = pytest.mark.skipif(
    not _has_shipped_model(), reason="models/algo_selector.joblib not present in this checkout"
)


def test_compress_decompress_roundtrip(tmp_path):
    src = tmp_path / "input.txt"
    src.write_bytes(b"hello world, this is a test file. " * 500)

    archive_path = tmp_path / "input.archive"
    summary = compress(src, archive_path)

    assert Path(summary["output_path"]) == archive_path
    assert summary["original_size"] == src.stat().st_size
    assert summary["compressed_size"] == archive_path.stat().st_size
    assert summary["compressed_size"] > 0
    assert summary["ratio"] > 0
    assert summary["chunk_count"] == sum(summary["algorithm_counts"].values())
    assert summary["manifest_path"] is not None
    assert Path(summary["manifest_path"]).exists()

    restored_path = tmp_path / "restored.txt"
    decompress_summary = decompress(archive_path, restored_path)

    assert restored_path.read_bytes() == src.read_bytes()
    assert decompress_summary["restored_size"] == src.stat().st_size


def test_compress_discards_manifest_when_not_kept(tmp_path):
    src = tmp_path / "input.bin"
    src.write_bytes(os.urandom(20_000))
    archive_path = tmp_path / "input.archive"

    summary = compress(src, archive_path, keep_manifest=False)

    assert summary["manifest_path"] is None
    default_manifest_path = archive_path.with_suffix(archive_path.suffix + ".manifest.json")
    assert not default_manifest_path.exists()


def test_compress_default_manifest_path_lives_next_to_archive(tmp_path):
    src = tmp_path / "input.bin"
    src.write_bytes(os.urandom(5000))
    archive_path = tmp_path / "input.archive"

    summary = compress(src, archive_path)

    expected_manifest = archive_path.with_suffix(archive_path.suffix + ".manifest.json")
    assert Path(summary["manifest_path"]) == expected_manifest
    assert expected_manifest.exists()


def test_compress_respects_custom_manifest_path(tmp_path):
    src = tmp_path / "input.bin"
    src.write_bytes(os.urandom(5000))
    archive_path = tmp_path / "input.archive"
    custom_manifest = tmp_path / "custom.manifest.json"

    summary = compress(src, archive_path, manifest_path=custom_manifest)

    assert Path(summary["manifest_path"]) == custom_manifest
    assert custom_manifest.exists()


def test_roundtrip_variable_content_and_algorithms(tmp_path):
    # mix of text and random bytes so more than one algorithm gets predicted
    data = (b"repetitive english text here " * 1000) + os.urandom(10_000)
    src = tmp_path / "mixed.bin"
    src.write_bytes(data)
    archive_path = tmp_path / "mixed.archive"

    compress(src, archive_path, chunk_size=1024)

    restored_path = tmp_path / "mixed.restored"
    decompress(archive_path, restored_path)

    assert restored_path.read_bytes() == data


def test_decompress_raises_on_corrupt_archive(tmp_path):
    src = tmp_path / "input.txt"
    src.write_bytes(b"some content to compress " * 200)
    archive_path = tmp_path / "input.archive"
    compress(src, archive_path)

    raw = bytearray(archive_path.read_bytes())
    raw[0:4] = b"XXXX"  # corrupt the magic
    archive_path.write_bytes(bytes(raw))

    from encoder import format

    with pytest.raises(format.ArchiveError):
        decompress(archive_path, tmp_path / "restored.txt")
