"""Pack file into .szip archive

Chunks input and picks compression algorithm per chunk according to `mode`:
  - "brute_force" (default): try every registered algorithm and keep smallest
    Always optimal, always slow
  - "model": predict best algorithm from chunk's features (src.model.predict) and compress
    Much faster; ratio is only as good as prediction

`algorithm` overrides both modes, forcing every chunk to one algorithm

Default chunker is `fixed_size` at DEFAULT_CHUNK_SIZE (4KB)
`content_aware` is opt-in and hasn't been benchmarked with mode="model" (see README)

Actual algorithm used per chunk is always recorded in archive metadata, 
so a bad prediction only costs ratio, never correctness
"""

import argparse
import os
from pathlib import Path

from src.archive import format as fmt
from src.chunking import content_aware, fixed_size
from src.compressors import registry

# Matches chunk granularity models/algo_selector.joblib was trained on
DEFAULT_CHUNK_SIZE = 4096


def _brute_force_smallest(chunk_bytes: bytes) -> tuple[str, bytes]:
    """Try every registered algorithm and return smallest result

    Args:
        chunk_bytes: Chunk bytes to compress

    Returns:
        (algorithm_name, compressed_bytes) for smallest output 
        Ties prefer `store`
    """
    best_name = None
    best_compressed = None
    for name in registry.list_algorithms():
        compressed = registry.compress(chunk_bytes, name)
        if best_compressed is None or len(compressed) < len(best_compressed) or (
            len(compressed) == len(best_compressed) and name == "store"
        ):
            best_name, best_compressed = name, compressed
    return best_name, best_compressed


def _chunk_data(data: bytes, chunker: str, chunk_size: int) -> list[bytes]:
    """Split `data` into chunks using named chunker

    Args:
        data: Bytes to split
        chunker: "fixed_size" or "content_aware"
        chunk_size: Chunk size in bytes (fixed_size only)

    Returns:
        List of chunk byte strings

    Raises:
        ValueError: If `chunker` is unrecognized
    """
    if chunker == "content_aware":
        return list(content_aware.chunk(data))
    if chunker == "fixed_size":
        return list(fixed_size.chunk(data, chunk_size=chunk_size))
    raise ValueError(f"unknown chunker: {chunker}")


def pack_bytes(
    data: bytes,
    algorithm: str | None = None,
    chunker: str = "fixed_size",
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    mode: str = "brute_force",
    model_path: Path | None = None,
) -> bytes:
    """Chunk `data`, compress each chunk, and serialize to .szip binary format

    Args:
        data: Bytes to pack
        algorithm: If given, every chunk compressed with this one algorithm, overriding `mode`
        chunker: "fixed_size" or "content_aware"
        chunk_size: Chunk size in bytes (fixed_size only)
        mode: "brute_force" or "model" (ignored if `algorithm` is given);
            see module docstring.
        model_path: Path to trained model, used only when mode="model"

    Returns:
        Serialized .szip archive bytes

    Raises:
        ValueError: If `mode` is unrecognized
    """
    if algorithm is None and mode not in ("brute_force", "model"):
        raise ValueError(f"unknown mode: {mode}")

    chunks = _chunk_data(data, chunker, chunk_size)

    predictor = None
    if algorithm is None and mode == "model":
        from src.model.predict import DEFAULT_MODEL_PATH, AlgorithmPredictor

        predictor = AlgorithmPredictor.load(model_path or DEFAULT_MODEL_PATH)

    # Batch predictions across all chunks
    # per-chunk .predict() calls pay sklearn's dispatch overhead repeatedly and can dominate runtime
    predicted_algos = (
        predictor.predict_chunks(chunks) if algorithm is None and mode == "model" else None
    )

    metas = bytearray()
    payload = bytearray()
    for i, chunk_bytes in enumerate(chunks):
        if algorithm is not None:
            algo_name = algorithm
            compressed = registry.compress(chunk_bytes, algo_name)
        elif mode == "model":
            algo_name = predicted_algos[i]
            compressed = registry.compress(chunk_bytes, algo_name)
        else:  # brute_force
            algo_name, compressed = _brute_force_smallest(chunk_bytes)

        algo_id = fmt.registry_index(algo_name)
        offset = len(payload)
        payload += compressed
        metas += fmt.write_chunk_meta(algo_id, len(chunk_bytes), len(compressed), offset)

    checksum = fmt.compute_checksum(data)
    header = fmt.write_archive_header(len(data), checksum, len(chunks))
    return header + bytes(metas) + bytes(payload)


def pack_file(
    input_path: Path,
    output_path: Path,
    algorithm: str | None = None,
    chunker: str = "fixed_size",
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    mode: str = "brute_force",
    model_path: Path | None = None,
) -> None:
    """Pack file into .szip archive on disk

    Args:
        input_path: File to pack
        output_path: Where to write archive - Written atomically
        algorithm: See `pack_bytes`
        chunker: See `pack_bytes`
        chunk_size: See `pack_bytes`
        mode: See `pack_bytes`
        model_path: See `pack_bytes`
    """
    data = input_path.read_bytes()
    archive = pack_bytes(
        data,
        algorithm=algorithm,
        chunker=chunker,
        chunk_size=chunk_size,
        mode=mode,
        model_path=model_path,
    )
    tmp_path = output_path.with_name(output_path.name + ".partial")
    try:
        tmp_path.write_bytes(archive)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
    os.replace(tmp_path, output_path)


def main() -> None:
    """CLI entry point. Parses sys.argv and runs `pack_file`."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--algorithm",
        choices=registry.list_algorithms(include_excluded=True),
        default=None,
        help="Fixed algorithm for every chunk (overrides --mode)",
    )
    parser.add_argument("--mode", choices=["brute_force", "model"], default="brute_force")
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument(
        "--chunker",
        choices=["fixed_size", "content_aware"],
        default="fixed_size",
        help="content_aware is opt-in only; not benchmarked with --mode model (see module docstring)",
    )
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    args = parser.parse_args()

    pack_file(
        args.input,
        args.output,
        algorithm=args.algorithm,
        chunker=args.chunker,
        chunk_size=args.chunk_size,
        mode=args.mode,
        model_path=args.model_path,
    )


if __name__ == "__main__":
    main()
