import os

import pytest

from src.compressors import registry

TEXT_SAMPLE = b"the quick brown fox jumps over the lazy dog " * 200
BINARY_SAMPLE = bytes((i * 2654435761) % 256 for i in range(20_000))
RANDOM_SAMPLE = os.urandom(20_000)
EMPTY_SAMPLE = b""
ALL_SAMPLES = [TEXT_SAMPLE, BINARY_SAMPLE, RANDOM_SAMPLE, EMPTY_SAMPLE]


def test_list_algorithms_default_excludes_gzip():
    names = registry.list_algorithms()
    for expected in ("store", "bzip2", "lzma", "zstd", "brotli"):
        assert expected in names
    assert "gzip" not in names


def test_list_algorithms_include_excluded_still_has_gzip():
    names = registry.list_algorithms(include_excluded=True)
    for expected in ("store", "gzip", "bzip2", "lzma", "zstd", "brotli"):
        assert expected in names


# Verified for every registered algorithm, including ones excluded from default search set (currently gzip)
@pytest.mark.parametrize("algo", registry.list_algorithms(include_excluded=True))
@pytest.mark.parametrize("data", ALL_SAMPLES, ids=["text", "binary", "random", "empty"])
def test_roundtrip_exact_bytes(algo, data):
    compressed = registry.compress(data, algo)
    restored = registry.decompress(compressed, algo)
    assert restored == data


@pytest.mark.parametrize(
    "algo", [n for n in registry.list_algorithms(include_excluded=True) if n != "store"]
)
def test_compresses_redundant_data(algo):
    compressed = registry.compress(TEXT_SAMPLE, algo)
    assert len(compressed) < len(TEXT_SAMPLE)


def test_store_is_identity():
    compressed = registry.compress(TEXT_SAMPLE, "store")
    assert compressed == TEXT_SAMPLE


def test_unknown_algorithm_raises_on_compress():
    with pytest.raises(KeyError):
        registry.compress(TEXT_SAMPLE, "does-not-exist")


def test_unknown_algorithm_raises_on_decompress():
    with pytest.raises(KeyError):
        registry.decompress(TEXT_SAMPLE, "does-not-exist")


def test_unknown_algorithm_raises_on_get():
    with pytest.raises(KeyError):
        registry.get("does-not-exist")
