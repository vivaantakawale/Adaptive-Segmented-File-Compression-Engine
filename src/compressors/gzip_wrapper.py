"""gzip (zlib/DEFLATE) compressor conforming to common wrapper API"""

import gzip

NAME = "gzip"
DEFAULT_LEVEL = 6


def compress(data: bytes, level: int = DEFAULT_LEVEL) -> bytes:
    """Compress bytes with gzip

    Args:
        data: Raw bytes to compress
        level: Compression level, 0-9 (higher = smaller output, slower)

    Returns:
        Compressed bytes.
    """
    return gzip.compress(data, compresslevel=level)


def decompress(data: bytes) -> bytes:
    """Decompress gzip-compressed bytes

    Args:
        data: Compressed bytes produced by `compress`

    Returns:
        Original uncompressed bytes
    """
    return gzip.decompress(data)
