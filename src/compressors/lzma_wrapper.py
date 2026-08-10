"""LZMA (xz) compressor conforming to common wrapper API"""

import lzma

NAME = "lzma"
DEFAULT_LEVEL = 6


def compress(data: bytes, level: int = DEFAULT_LEVEL) -> bytes:
    """Compress bytes with LZMA

    Args:
        data: Raw bytes to compress
        level: Preset level, 0-9 (higher = smaller output, slower)

    Returns:
        Compressed bytes
    """
    return lzma.compress(data, preset=level)


def decompress(data: bytes) -> bytes:
    """Decompress LZMA-compressed bytes

    Args:
        data: Compressed bytes produced by `compress`

    Returns:
        Original uncompressed bytes
    """
    return lzma.decompress(data)
