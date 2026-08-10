"""Feature extraction for single chunk of bytes

Computes feature vector used by algorithm selection model:
Shannon entropy, normalized 256-bin byte-value histogram, printable-ASCII ratio, byte mean/variance, top-K byte bigram frequencies, chunk length
"""

import numpy as np

HISTOGRAM_BINS = 256
TOP_K_BIGRAMS = 10
BIGRAM_ALPHABET = HISTOGRAM_BINS * HISTOGRAM_BINS  # 65536 possible byte pairs

_PRINTABLE = frozenset([0x09, 0x0A, 0x0D]) | frozenset(range(0x20, 0x7F))
PRINTABLE_LUT = np.array(
    [1.0 if b in _PRINTABLE else 0.0 for b in range(HISTOGRAM_BINS)], dtype=np.float64
)

SCALAR_FEATURE_NAMES = ["length", "entropy", "printable_ratio", "byte_mean", "byte_variance"]
HISTOGRAM_FEATURE_NAMES = [f"hist_{i}" for i in range(HISTOGRAM_BINS)]
BIGRAM_FEATURE_NAMES = [f"bigram_top{i}" for i in range(TOP_K_BIGRAMS)]
FEATURE_NAMES = SCALAR_FEATURE_NAMES + HISTOGRAM_FEATURE_NAMES + BIGRAM_FEATURE_NAMES


def byte_counts(data: bytes) -> np.ndarray:
    """Compute raw byte value counts

    Args:
        data: Bytes to count

    Returns:
        256 element array of unnormalized byte value counts
    """
    return np.bincount(np.frombuffer(data, dtype=np.uint8), minlength=HISTOGRAM_BINS)


def entropy_from_counts(counts: np.ndarray, total: int) -> float:
    """Compute Shannon entropy from precomputed byte value counts

    Args:
        counts: 256 element byte value counts returned by `byte_counts`
        total: Total number of bytes counts were computed over

    Returns:
        Entropy in bits per byte in [0, 8], 0.0 if `total` is 0
    """
    if total == 0:
        return 0.0
    counts = counts.astype(np.float64)
    probs = counts[counts > 0] / total
    return float(-np.sum(probs * np.log2(probs)))


def printable_ratio_from_counts(counts: np.ndarray, total: int) -> float:
    """Compute printable byte ratio from precomputed byte value counts

    Args:
        counts: 256 element byte value counts returned by `byte_counts`
        total: Total number of bytes the counts were computed over

    Returns:
        Fraction (0.0-1.0) of bytes that are printable ASCII or common whitespace, 0.0 if `total` is 0
    """
    if total == 0:
        return 0.0
    return float(np.dot(counts.astype(np.float64), PRINTABLE_LUT) / total)


def shannon_entropy(data: bytes) -> float:
    """Compute Shannon entropy of byte string

    Args:
        data: Bytes to measure

    Returns:
        Entropy in bits per byte in [0, 8], 0.0 for empty input
    """
    return entropy_from_counts(byte_counts(data), len(data)) if data else 0.0


def byte_histogram(data: bytes) -> np.ndarray:
    """Compute normalized byte value histogram

    Args:
        data: Bytes to histogram

    Returns:
        256 element array of byte value frequencies, summing to 1.0, All zero for empty input
    """
    if not data:
        return np.zeros(HISTOGRAM_BINS, dtype=np.float64)
    return byte_counts(data).astype(np.float64) / len(data)


def printable_ratio(data: bytes) -> float:
    """Compute fraction of printable ASCII/whitespace bytes

    Args:
        data: Bytes to check

    Returns:
        Fraction (0.0-1.0) of bytes that are printable ASCII (0x20-0x7e) or common whitespace, 0.0 for empty input
    """
    return printable_ratio_from_counts(byte_counts(data), len(data)) if data else 0.0


def byte_mean_variance(data: bytes) -> tuple[float, float]:
    """Compute mean and variance of raw byte values

    Args:
        data: Bytes to measure

    Returns:
        (mean, population variance) of byte values, (0.0, 0.0) for empty input
    """
    if not data:
        return 0.0, 0.0
    arr = np.frombuffer(data, dtype=np.uint8).astype(np.float64)
    return float(arr.mean()), float(arr.var())


def top_k_bigram_freqs(data: bytes, k: int = TOP_K_BIGRAMS) -> list[float]:
    """Compute frequencies of the k most common byte bigrams

    Args:
        data: Bytes to scan
        k: Number of top bigram frequencies to return

    Returns:
        List of k frequencies (fraction of all bigrams), sorted descending and zero-padded if fewer than k distinct bigrams are present
    """
    if len(data) < 2:
        return [0.0] * k
    arr = np.frombuffer(data, dtype=np.uint8).astype(np.uint32)
    # Pack each byte pair into a single int in [0, 65536) for bincount.
    pair_ids = arr[:-1] * HISTOGRAM_BINS + arr[1:]
    total = len(pair_ids)
    counts = np.bincount(pair_ids, minlength=BIGRAM_ALPHABET)
    nonzero_counts = counts[counts > 0]
    top = np.sort(nonzero_counts)[::-1][:k]
    freqs = (top / total).tolist()
    freqs += [0.0] * (k - len(freqs))
    return freqs


def extract(data: bytes) -> dict:
    """Compute full feature dict for single chunk

    Args:
        data: Chunk bytes to extract features from

    Returns:
        Dict keyed by `FEATURE_NAMES`, values are the corresponding features
    """
    total = len(data)
    counts = byte_counts(data) if data else np.zeros(HISTOGRAM_BINS, dtype=np.int64)
    mean, variance = byte_mean_variance(data)
    values: dict = {
        "length": total,
        "entropy": entropy_from_counts(counts, total),
        "printable_ratio": printable_ratio_from_counts(counts, total),
        "byte_mean": mean,
        "byte_variance": variance,
    }
    hist = counts.astype(np.float64) / total if total else counts.astype(np.float64)
    for name, v in zip(HISTOGRAM_FEATURE_NAMES, hist):
        values[name] = float(v)
    for name, v in zip(BIGRAM_FEATURE_NAMES, top_k_bigram_freqs(data)):
        values[name] = v
    return values


def extract_vector(data: bytes) -> np.ndarray:
    """Compute full feature vector for single chunk

    Args:
        data: Chunk bytes to extract features from

    Returns:
        1-D array of the same features as `extract`, ordered by FEATURE_NAMES
    """
    values = extract(data)
    return np.array([values[name] for name in FEATURE_NAMES], dtype=np.float64)
