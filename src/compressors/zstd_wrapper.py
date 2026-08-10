"""Zstandard compressor conforming to common wrapper API

Requires `zstandard` package (see requirements.txt)
"""

import zstandard as zstd

NAME = "zstd"
DEFAULT_LEVEL = 9


def compress(data: bytes, level: int = DEFAULT_LEVEL) -> bytes:
    """Compress bytes with Zstandard

    Args:
        data: Raw bytes to compress
        level: Compression level 1-22 (higher = smaller output, slower)

    Returns:
        Compressed bytes
    """
    return zstd.ZstdCompressor(level=level).compress(data)


def decompress(data: bytes) -> bytes:
    """Decompress Zstandard compressed bytes

    Args:
        data: Compressed bytes produced by `compress`

    Returns:
        Original uncompressed bytes
    """
    return zstd.ZstdDecompressor().decompress(data)
