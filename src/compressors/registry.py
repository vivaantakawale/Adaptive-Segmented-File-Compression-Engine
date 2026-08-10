"""Central registry mapping algorithm names to compress/decompress functions

Public interface:

    compress(data: bytes, algo: str) -> bytes
    decompress(data: bytes, algo: str) -> bytes
    list_algorithms(include_excluded: bool = False) -> list[str]

Every per-algorithm wrapper module exposes same underlying API:

    NAME: str
    compress(data: bytes, level: int = ...) -> bytes
    decompress(data: bytes) -> bytes

`store` is a pseudo-algorithm (no-op) used for chunks that don't compress
well (e.g. already-compressed or encrypted data), so callers always have a
"give up" option that's guaranteed not to expand the chunk
"""

from collections.abc import Callable
from dataclasses import dataclass

from src.compressors import (
    brotli_wrapper,
    bzip2_wrapper,
    gzip_wrapper,
    lzma_wrapper,
    zstd_wrapper,
)


def _store_compress(data: bytes, level: int = 0) -> bytes:
    return data


def _store_decompress(data: bytes) -> bytes:
    return data


@dataclass(frozen=True)
class Algorithm:
    name: str
    compress: Callable[..., bytes]
    decompress: Callable[[bytes], bytes]


_ALGORITHMS: dict[str, Algorithm] = {
    "store": Algorithm("store", _store_compress, _store_decompress),
    gzip_wrapper.NAME: Algorithm(
        gzip_wrapper.NAME, gzip_wrapper.compress, gzip_wrapper.decompress
    ),
    bzip2_wrapper.NAME: Algorithm(
        bzip2_wrapper.NAME, bzip2_wrapper.compress, bzip2_wrapper.decompress
    ),
    lzma_wrapper.NAME: Algorithm(
        lzma_wrapper.NAME, lzma_wrapper.compress, lzma_wrapper.decompress
    ),
    zstd_wrapper.NAME: Algorithm(
        zstd_wrapper.NAME, zstd_wrapper.compress, zstd_wrapper.decompress
    ),
    brotli_wrapper.NAME: Algorithm(
        brotli_wrapper.NAME, brotli_wrapper.compress, brotli_wrapper.decompress
    ),
}

# disk algorithm ids (archive format + model class labels) are indexes into this list 
# Order must never change
# appending allowed, removing or reordering breaks every existing archive and trained model
ALGORITHM_NAMES: list[str] = list(_ALGORITHMS.keys())

# Excluded from default brute force/prediction candidate set 
# still usable via explicit compress/decompress/get calls
# gzip strictly dominated by zstd/brotli at this project chunk size
DEFAULT_EXCLUDED_FROM_SEARCH: frozenset[str] = frozenset({"gzip"})


def get(name: str) -> Algorithm:
    """Look up Algorithm by name

    Args:
        name: Algorithm name

    Returns:
        Matching Algorithm

    Raises:
        KeyError: If `name` not registered
    """
    return _ALGORITHMS[name]


def compress(data: bytes, algo: str, level: int | None = None) -> bytes:
    """Compress `data` with named algorithm

    Args:
        data: Raw bytes to compress
        algo: Algorithm name (see `list_algorithms`)
        level: Optional algorithm-specific compression level
                Uses algorithm's own default if omitted.

    Returns:
        Compressed bytes
    """
    algorithm = get(algo)
    return algorithm.compress(data) if level is None else algorithm.compress(data, level)


def decompress(data: bytes, algo: str) -> bytes:
    """Decompress `data` with named algorithm.

    Args:
        data: Compressed bytes produced by `compress` with same `algo`
        algo: Algorithm name bytes were compressed with

    Returns:
        Original uncompressed bytes
    """
    return get(algo).decompress(data)


def list_algorithms(include_excluded: bool = False) -> list[str]:
    """List candidate algorithms for brute force search / model training

    Args:
        include_excluded: If True, also include algorithms in `DEFAULT_EXCLUDED_FROM_SEARCH` (currently just gzip)

    Returns:
        Algorithm names in stable `ALGORITHM_NAMES` order
    """
    if include_excluded:
        return list(ALGORITHM_NAMES)
    return [name for name in ALGORITHM_NAMES if name not in DEFAULT_EXCLUDED_FROM_SEARCH]
