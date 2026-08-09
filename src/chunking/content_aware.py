"""Content-aware chunking: split data into runs of text vs. binary content

Slides fixed size window across data and cuts boundary wherever text/binary classification flips
Window is "text" if it's printable, low-entropy, and not mostly hex digits
Short runs are merged into a neighbor to avoid fragmenting into many tiny chunks
"""

from collections.abc import Iterator

import numpy as np

from src.features.extract import (
    HISTOGRAM_BINS,
    byte_counts,
    entropy_from_counts,
    printable_ratio_from_counts,
)

WINDOW_SIZE = 512
PRINTABLE_THRESHOLD = 0.85
ENTROPY_THRESHOLD = 5.0  # bits/byte; separates prose/code (~4.2-4.3) from base64 (~5.9)
HEX_RATIO_THRESHOLD = 0.95  # fraction of [0-9a-fA-F] bytes
MIN_CHUNK_SIZE = 1024

_HEX_DIGITS = frozenset(b"0123456789abcdefABCDEF")
_HEX_LUT = np.array(
    [1.0 if b in _HEX_DIGITS else 0.0 for b in range(HISTOGRAM_BINS)], dtype=np.float64
)


def hex_ratio(window: bytes) -> float:
    """Compute fraction of hex digit bytes in window

    Args:
        window: Bytes to check

    Returns:
        Fraction (0.0-1.0) of bytes in `window` that are [0-9a-fA-F]
        0.0 for empty input
    """
    if not window:
        return 0.0
    counts = byte_counts(window).astype(np.float64)
    return float(np.dot(counts, _HEX_LUT) / len(window))


def printable_ratio(window: bytes) -> float:
    """Compute fraction of printable/whitespace bytes in window

    Args:
        window: Bytes to check

    Returns:
        Fraction (0.0-1.0) of bytes in `window` that are printable ASCII or common whitespace
        0.0 for empty input
    """
    return printable_ratio_from_counts(byte_counts(window), len(window)) if window else 0.0


def _classify_windows(
    data: bytes,
    window_size: int,
    threshold: float,
    entropy_threshold: float,
    hex_ratio_threshold: float,
) -> list[str]:
    labels = []
    for start in range(0, len(data), window_size):
        window = data[start : start + window_size]
        counts = byte_counts(window)
        total = len(window)
        is_text = (
            printable_ratio_from_counts(counts, total) >= threshold
            and entropy_from_counts(counts, total) < entropy_threshold
            and float(np.dot(counts.astype(np.float64), _HEX_LUT) / total) < hex_ratio_threshold
        )
        labels.append("text" if is_text else "binary")
    return labels


def _boundaries_from_labels(labels: list[str], window_size: int, total_size: int) -> list[int]:
    """Find byte offsets where text/binary classification changes

    Args:
        labels: Per window "text"/"binary" labels
        window_size: Size of each window in bytes
        total_size: Total size of underlying data in bytes

    Returns:
        Sorted byte offsets always including 0 and total_size
    """
    boundaries = [0]
    for i in range(1, len(labels)):
        if labels[i] != labels[i - 1]:
            boundaries.append(i * window_size)
    boundaries.append(total_size)
    return boundaries


def _merge_small_runs(boundaries: list[int], min_chunk_size: int) -> list[int]:
    """Drop interior boundaries that would create undersized run

    Args:
        boundaries: Sorted byte offsets as returned by `_boundaries_from_labels`
        min_chunk_size: Minimum run length in bytes

    Returns:
        `boundaries` with any interior boundary that would create run shorter than `min_chunk_size` removed (merged into predecessor)
    """
    if len(boundaries) <= 2:
        return boundaries
    merged = [boundaries[0]]
    for b in boundaries[1:-1]:
        if b - merged[-1] < min_chunk_size:
            continue
        merged.append(b)
    merged.append(boundaries[-1])
    return merged


def chunk(
    data: bytes,
    window_size: int = WINDOW_SIZE,
    printable_threshold: float = PRINTABLE_THRESHOLD,
    entropy_threshold: float = ENTROPY_THRESHOLD,
    hex_ratio_threshold: float = HEX_RATIO_THRESHOLD,
    min_chunk_size: int = MIN_CHUNK_SIZE,
) -> Iterator[bytes]:
    """Yield chunks of `data` split at text/binary transitions

    Window counts as "text" only if it clears `printable_threshold`, stays under `entropy_threshold`, and stays under `hex_ratio_threshold`

    Args:
        data: Bytes to split
        window_size: Size of sliding classification window in bytes
        printable_threshold: Minimum fraction (0.0-1.0) of printable bytes for window to count as "text"
        entropy_threshold: Maximum Shannon entropy (bits/byte, 0.0-8.0) for window to count as "text"
        hex_ratio_threshold: Maximum fraction (0.0-1.0) of hex digit bytes for window to count as "text"
        min_chunk_size: Minimum chunk size in bytes, shorter runs are merged into the previous run

    Yields:
        Chunks of `data`
        Concatenating them reproduces `data`

    Raises:
        ValueError: If threshold or size argument is out of range
    """
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    if not 0.0 <= printable_threshold <= 1.0:
        raise ValueError("printable_threshold must be in [0, 1]")
    if not 0.0 <= entropy_threshold <= 8.0:
        raise ValueError("entropy_threshold must be in [0, 8]")
    if not 0.0 <= hex_ratio_threshold <= 1.0:
        raise ValueError("hex_ratio_threshold must be in [0, 1]")
    if min_chunk_size <= 0:
        raise ValueError("min_chunk_size must be positive")

    if not data:
        return

    labels = _classify_windows(
        data, window_size, printable_threshold, entropy_threshold, hex_ratio_threshold
    )
    boundaries = _boundaries_from_labels(labels, window_size, len(data))
    boundaries = _merge_small_runs(boundaries, min_chunk_size)

    for start, end in zip(boundaries, boundaries[1:]):
        yield data[start:end]
