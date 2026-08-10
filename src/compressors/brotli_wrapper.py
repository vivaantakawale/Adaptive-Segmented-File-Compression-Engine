"""Brotli compressor conforming to common wrapper API

Requires `brotli` package (see requirements.txt)
"""

import brotli

NAME = "brotli"
DEFAULT_LEVEL = 9


def compress(data: bytes, level: int = DEFAULT_LEVEL) -> bytes:
    """Compress bytes with brotli

    Args:
        data: Raw bytes to compress
        level: Quality level, 0-11 (higher = smaller output, slower)

    Returns:
        Compressed bytes
    """
    return brotli.compress(data, quality=level)


def decompress(data: bytes) -> bytes:
    """Decompress brotli-compressed bytes

    Args:
        data: Compressed bytes produced by `compress`

    Returns:
        Original uncompressed bytes
    """
    return brotli.decompress(data)
