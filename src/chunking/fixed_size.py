"""Fixed size chunking: split  byte string into equal sized blocks"""

from collections.abc import Iterator

DEFAULT_CHUNK_SIZE = 16 * 1024  # 16 KiB


def chunk(data: bytes, chunk_size: int = DEFAULT_CHUNK_SIZE) -> Iterator[bytes]:
    """Yield successive `chunk_size` - byte slices of `data`

    Args:
        data: Bytes to split
        chunk_size: Size of each slice in bytes 

    Yields:
        Chunks of `data`

    Raises:
        ValueError: If `chunk_size` is not positive
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    for start in range(0, len(data), chunk_size):
        yield data[start : start + chunk_size]
