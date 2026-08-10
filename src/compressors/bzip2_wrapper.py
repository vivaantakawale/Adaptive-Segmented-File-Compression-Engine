"""bzip2 compressor conforming to common wrapper API"""

import bz2

NAME = "bzip2"
DEFAULT_LEVEL = 9


def compress(data: bytes, level: int = DEFAULT_LEVEL) -> bytes:
    """Compress bytes with bzip2

    Args:
        data: Raw bytes to compress
        level: Compression level, 1-9 (higher = smaller output, slower)

    Returns:
        Compressed bytes
    """
    return bz2.compress(data, compresslevel=level)


def decompress(data: bytes) -> bytes:
    """Decompress bzip2-compressed bytes

    Args:
        data: Compressed bytes produced by `compress`

    Returns:
        Original uncompressed bytes
    """
    return bz2.decompress(data)
