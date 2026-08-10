import os

import numpy as np

from src.features.extract import (
    BIGRAM_FEATURE_NAMES,
    FEATURE_NAMES,
    HISTOGRAM_FEATURE_NAMES,
    SCALAR_FEATURE_NAMES,
    byte_histogram,
    byte_mean_variance,
    extract,
    extract_vector,
    printable_ratio,
    shannon_entropy,
    top_k_bigram_freqs,
)

ALL_ZERO = b"\x00" * 20_000
RANDOM = os.urandom(20_000)
REPEATED_PATTERN = b"xy" * 10_000
ASCII_TEXT = (b"the quick brown fox jumps over the lazy dog. " * 400)[:20_000]


# Shannon entropy: sanity checks against known ordering


def test_entropy_all_zero_is_zero():
    assert shannon_entropy(ALL_ZERO) == 0.0


def test_entropy_empty_is_zero():
    assert shannon_entropy(b"") == 0.0


def test_entropy_random_is_near_max():
    assert shannon_entropy(RANDOM) > 7.9


def test_entropy_ordering_random_text_repeated_zero():
    e_random = shannon_entropy(RANDOM)
    e_text = shannon_entropy(ASCII_TEXT)
    e_repeated = shannon_entropy(REPEATED_PATTERN)
    e_zero = shannon_entropy(ALL_ZERO)
    assert e_random > e_text > e_repeated > e_zero


# printable ratio

def test_printable_ratio_text_is_one():
    assert printable_ratio(ASCII_TEXT) == 1.0


def test_printable_ratio_zero_bytes_is_zero():
    assert printable_ratio(ALL_ZERO) == 0.0


def test_printable_ratio_random_is_lower_than_text():
    assert printable_ratio(RANDOM) < printable_ratio(ASCII_TEXT)


# histogram


def test_histogram_sums_to_one():
    for data in (ALL_ZERO, RANDOM, REPEATED_PATTERN, ASCII_TEXT):
        hist = byte_histogram(data)
        assert hist.shape == (256,)
        assert np.isclose(hist.sum(), 1.0)


def test_histogram_all_zero_concentrated_in_one_bin():
    hist = byte_histogram(ALL_ZERO)
    assert hist[0] == 1.0
    assert hist.sum() - hist[0] == 0.0


def test_histogram_empty_is_all_zero_bins():
    hist = byte_histogram(b"")
    assert np.array_equal(hist, np.zeros(256))


def test_histogram_random_is_roughly_uniform():
    hist = byte_histogram(RANDOM)
    # 256 bins over 20000 samples -> expected ~0.0039/bin; no single bin
    # should dominate the way it does for zero/repeated data.
    assert hist.max() < 0.02


# mean / variance


def test_mean_variance_all_zero():
    mean, var = byte_mean_variance(ALL_ZERO)
    assert mean == 0.0
    assert var == 0.0


def test_mean_variance_random_is_high_variance():
    mean, var = byte_mean_variance(RANDOM)
    assert 100 < mean < 155
    assert var > 1000  # uniform 0-255 has variance ~5461


def test_mean_variance_empty():
    assert byte_mean_variance(b"") == (0.0, 0.0)


# bigram frequencies


def test_bigram_freqs_repeated_pattern_dominated_by_top_bigram():
    freqs = top_k_bigram_freqs(REPEATED_PATTERN)
    assert len(freqs) == 10
    assert freqs[0] > 0.4  # "xy"/"yx" alternate ~50/50


def test_bigram_freqs_random_much_flatter_than_repeated():
    random_freqs = top_k_bigram_freqs(RANDOM)
    repeated_freqs = top_k_bigram_freqs(REPEATED_PATTERN)
    assert random_freqs[0] < repeated_freqs[0]


def test_bigram_freqs_short_input_padded_with_zero():
    assert top_k_bigram_freqs(b"a") == [0.0] * 10
    assert top_k_bigram_freqs(b"") == [0.0] * 10


# full extract() / extract_vector()


def test_feature_names_composition():
    assert FEATURE_NAMES == SCALAR_FEATURE_NAMES + HISTOGRAM_FEATURE_NAMES + BIGRAM_FEATURE_NAMES
    assert len(FEATURE_NAMES) == 5 + 256 + 10


def test_extract_returns_all_expected_keys():
    features = extract(ASCII_TEXT)
    assert set(features.keys()) == set(FEATURE_NAMES)


def test_extract_length_matches_input():
    assert extract(ASCII_TEXT)["length"] == len(ASCII_TEXT)
    assert extract(b"")["length"] == 0


def test_extract_empty_chunk_does_not_raise():
    features = extract(b"")
    assert features["length"] == 0
    assert features["entropy"] == 0.0


def test_extract_vector_matches_extract_dict_order():
    vec = extract_vector(ASCII_TEXT)
    d = extract(ASCII_TEXT)
    assert isinstance(vec, np.ndarray)
    assert vec.shape == (len(FEATURE_NAMES),)
    assert np.allclose(vec, np.array([d[name] for name in FEATURE_NAMES]))
