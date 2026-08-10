import os

import pytest

from src.chunking import content_aware, fixed_size


def test_fixed_size_chunk_sizes():
    data = bytes(range(256)) * 10  # 2560 bytes
    chunks = list(fixed_size.chunk(data, chunk_size=100))
    assert all(len(c) == 100 for c in chunks[:-1])
    assert len(chunks[-1]) == len(data) % 100 or len(chunks[-1]) == 100
    assert b"".join(chunks) == data


def test_fixed_size_default_chunk_size_is_16kb():
    assert fixed_size.DEFAULT_CHUNK_SIZE == 16 * 1024


def test_fixed_size_last_chunk_may_be_smaller():
    data = b"x" * 250
    chunks = list(fixed_size.chunk(data, chunk_size=100))
    assert [len(c) for c in chunks] == [100, 100, 50]
    assert b"".join(chunks) == data


def test_fixed_size_rejects_non_positive_size():
    with pytest.raises(ValueError):
        list(fixed_size.chunk(b"abc", chunk_size=0))


def test_fixed_size_empty_input():
    assert list(fixed_size.chunk(b"")) == []


def test_content_aware_reconstructs_original_mixed_content():
    text_run = b"the quick brown fox jumps over the lazy dog. " * 100
    binary_run = os.urandom(5000)
    more_text = b"and here is some more plain english text.\n" * 100
    data = text_run + binary_run + more_text
    chunks = list(content_aware.chunk(data))
    assert b"".join(chunks) == data


def test_content_aware_reconstructs_pure_text():
    data = b"all printable ascii text, nothing binary here at all.\n" * 200
    chunks = list(content_aware.chunk(data))
    assert b"".join(chunks) == data


def test_content_aware_reconstructs_pure_binary():
    data = os.urandom(10_000)
    chunks = list(content_aware.chunk(data))
    assert b"".join(chunks) == data


def test_content_aware_separates_text_and_binary_runs():
    text_run = b"a" * 4000  # printable well above threshold
    binary_run = bytes(range(256)) * 20  # includes many non printable bytes
    data = text_run + binary_run
    chunks = list(content_aware.chunk(data, min_chunk_size=512))
    assert len(chunks) >= 2
    # text run should be classified separately from binary run
    assert content_aware.printable_ratio(chunks[0]) >= content_aware.PRINTABLE_THRESHOLD
    assert content_aware.printable_ratio(chunks[-1]) < content_aware.PRINTABLE_THRESHOLD


def test_content_aware_empty_input():
    assert list(content_aware.chunk(b"")) == []


def test_content_aware_min_chunk_size_merges_short_runs():
    # rapid alternation between short text/binary bursts should not produce a flood of tiny chunks once min_chunk_size is applied
    text_burst = b"hello " * 10
    binary_burst = bytes(range(200, 256))
    data = (text_burst + binary_burst) * 50
    chunks = list(content_aware.chunk(data, min_chunk_size=2048))
    assert all(len(c) >= 2048 for c in chunks[:-1])
    assert b"".join(chunks) == data


@pytest.mark.parametrize(
    "kwargs",
    [
        {"window_size": 0},
        {"printable_threshold": 1.5},
        {"printable_threshold": -0.1},
        {"min_chunk_size": 0},
    ],
)
def test_content_aware_rejects_invalid_params(kwargs):
    with pytest.raises(ValueError):
        list(content_aware.chunk(b"some data", **kwargs))
