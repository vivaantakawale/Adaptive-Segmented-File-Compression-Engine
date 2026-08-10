import os

import pytest

from src.archive import format as fmt
from src.archive.pack import pack_bytes
from src.archive.unpack import unpack_bytes
from src.model.predict import DEFAULT_MODEL_PATH

TEXT_DATA = b"the quick brown fox jumps over the lazy dog. " * 500
BINARY_DATA = bytes((i * 2654435761) % 256 for i in range(30_000))
RANDOM_DATA = os.urandom(30_000)
MIXED_DATA = TEXT_DATA + RANDOM_DATA + TEXT_DATA

_HAS_MODEL = DEFAULT_MODEL_PATH.exists()


@pytest.mark.parametrize(
    "data",
    [TEXT_DATA, BINARY_DATA, RANDOM_DATA, MIXED_DATA],
    ids=["text", "binary", "random", "mixed"],
)
def test_roundtrip_brute_force_default(data):
    archive = pack_bytes(data)
    restored = unpack_bytes(archive)
    assert restored == data


@pytest.mark.parametrize("algo", ["store", "gzip", "bzip2", "lzma", "zstd", "brotli"])
def test_roundtrip_fixed_algorithm(algo):
    archive = pack_bytes(MIXED_DATA, algorithm=algo)
    restored = unpack_bytes(archive)
    assert restored == MIXED_DATA


def test_roundtrip_fixed_size_chunker():
    archive = pack_bytes(MIXED_DATA, chunker="fixed_size")
    restored = unpack_bytes(archive)
    assert restored == MIXED_DATA


@pytest.mark.skipif(not _HAS_MODEL, reason="no trained model at models/algo_selector.joblib")
@pytest.mark.parametrize(
    "data",
    [TEXT_DATA, BINARY_DATA, RANDOM_DATA, MIXED_DATA],
    ids=["text", "binary", "random", "mixed"],
)
def test_roundtrip_model_mode(data):
    archive = pack_bytes(data, mode="model", chunker="fixed_size", chunk_size=4096)
    restored = unpack_bytes(archive)
    assert restored == data


@pytest.mark.skipif(not _HAS_MODEL, reason="no trained model at models/algo_selector.joblib")
def test_model_mode_defaults_to_brute_force_behavior_when_algorithm_given():
    archive = pack_bytes(MIXED_DATA, algorithm="store", mode="model")
    header = fmt.read_archive_header(archive)
    meta = fmt.read_chunk_meta(archive, fmt.metadata_table_offset())
    assert fmt.registry_name(meta.algorithm_id) == "store"
    assert unpack_bytes(archive) == MIXED_DATA


def test_roundtrip_empty_file():
    archive = pack_bytes(b"")
    restored = unpack_bytes(archive)
    assert restored == b""


def test_archive_header_has_correct_metadata():
    archive = pack_bytes(TEXT_DATA)
    header = fmt.read_archive_header(archive)
    assert header.version == fmt.VERSION
    assert header.original_size == len(TEXT_DATA)
    assert header.checksum == fmt.compute_checksum(TEXT_DATA)
    assert header.chunk_count > 0


def test_brute_force_never_larger_than_store_per_chunk():
    archive = pack_bytes(RANDOM_DATA)  # near-incompressible input
    store_archive = pack_bytes(RANDOM_DATA, algorithm="store")
    assert len(archive) <= len(store_archive)


# orruption handling

def test_corrupted_payload_byte_raises_archive_error():
    archive = bytearray(pack_bytes(TEXT_DATA, algorithm="lzma"))
    payload_start = fmt.payload_blob_offset(fmt.read_archive_header(bytes(archive)).chunk_count)
    # flip a byte well inside compressed payload
    corrupt_index = payload_start + 5
    archive[corrupt_index] ^= 0xFF

    with pytest.raises(fmt.ArchiveError):
        unpack_bytes(bytes(archive))


def test_corrupted_checksum_raises_archive_error():
    archive = bytearray(pack_bytes(TEXT_DATA, algorithm="store"))
    payload_start = fmt.payload_blob_offset(fmt.read_archive_header(bytes(archive)).chunk_count)
    archive[payload_start] ^= 0xFF

    with pytest.raises(fmt.ArchiveError):
        unpack_bytes(bytes(archive))


def test_bad_magic_raises_archive_error():
    archive = bytearray(pack_bytes(TEXT_DATA))
    archive[0:4] = b"NOPE"
    with pytest.raises(fmt.ArchiveError):
        unpack_bytes(bytes(archive))


def test_truncated_archive_raises_archive_error():
    archive = pack_bytes(TEXT_DATA)
    truncated = archive[: len(archive) // 2]
    with pytest.raises(fmt.ArchiveError):
        unpack_bytes(truncated)


def test_corruption_does_not_silently_return_wrong_bytes():
    archive = bytearray(pack_bytes(TEXT_DATA, algorithm="brotli"))
    payload_start = fmt.payload_blob_offset(fmt.read_archive_header(bytes(archive)).chunk_count)
    archive[payload_start + 10] ^= 0xFF

    try:
        result = unpack_bytes(bytes(archive))
    except fmt.ArchiveError:
        pass
    else:
        assert result == TEXT_DATA
