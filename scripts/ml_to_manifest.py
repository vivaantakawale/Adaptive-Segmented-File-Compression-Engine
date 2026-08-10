"""Chunks file, predicts per-chunk algorithm, and writes manifest for encoder to consume
Both chunkers stream from disk in bounded batches rather than loading whole file into memory
"""

import argparse
import hashlib
import warnings
from pathlib import Path

from src.chunking.content_aware import (
    ENTROPY_THRESHOLD,
    HEX_RATIO_THRESHOLD,
    MIN_CHUNK_SIZE,
    PRINTABLE_THRESHOLD,
    WINDOW_SIZE,
    hex_ratio,
    printable_ratio,
)
from src.features.extract import shannon_entropy
from src.model.predict import DEFAULT_MODEL_PATH, AlgorithmPredictor

from encoder.manifest import ChunkRecord, Manifest, sha256_hex, write_manifest

DEFAULT_CHUNK_SIZE = 4096  # matches chunk size models/algo_selector.joblib was trained on
DEFAULT_BATCH_SIZE = 256  # chunks held in memory per prediction batch
LARGE_CHUNK_WARN_MULTIPLE = 8  # warn if content_aware chunk exceeds this many x DEFAULT_CHUNK_SIZE


class OutOfDistributionChunkWarning(UserWarning):
    """A content_aware chunk is far larger than what the shipped model trained on."""


def _warn_if_oversized_chunks(chunk_lengths: list[int], chunk_size_trained_on: int) -> None:
    """Emit OutOfDistributionChunkWarning if any of `chunk_lengths` is far
    larger than `chunk_size_trained_on`"""
    threshold = chunk_size_trained_on * LARGE_CHUNK_WARN_MULTIPLE
    oversized = [n for n in chunk_lengths if n > threshold]
    if not oversized:
        return
    biggest = max(oversized)
    warnings.warn(
        f"{len(oversized)} content_aware chunk(s) exceed {threshold} bytes "
        f"({LARGE_CHUNK_WARN_MULTIPLE}x the {chunk_size_trained_on}-byte chunks the shipped "
        f"model trained on) -- largest is {biggest} bytes. Predictions for these chunks "
        f"may be unreliable; correctness is unaffected either way since the encoder "
        f"always records the actual algorithm used",
        OutOfDistributionChunkWarning,
        stacklevel=2,
    )


def _predict_batch(
    batch: list[tuple[int, bytes]], predictor: AlgorithmPredictor
) -> list[ChunkRecord]:
    """Turn a batch of (offset, chunk_bytes) pairs into ChunkRecords, predicting once for whole batch rather than once per chunk"""
    chunk_bytes_list = [b for _, b in batch]
    algos = predictor.predict_chunks(chunk_bytes_list)
    return [
        ChunkRecord(offset=offset, length=len(data), algorithm=algo, checksum=sha256_hex(data))
        for (offset, data), algo in zip(batch, algos)
    ]


def _build_manifest_fixed_size_streaming(
    input_path: Path,
    chunk_size: int,
    predictor: AlgorithmPredictor,
    batch_size: int,
) -> Manifest:
    """Build manifest for `input_path` using fixed-size chunking, streaming from disk in batches of `batch_size` chunks"""
    overall_hash = hashlib.sha256()
    records: list[ChunkRecord] = []
    offset = 0
    batch: list[tuple[int, bytes]] = []

    with open(input_path, "rb") as f:
        while data := f.read(chunk_size):
            overall_hash.update(data)
            batch.append((offset, data))
            offset += len(data)
            if len(batch) >= batch_size:
                records.extend(_predict_batch(batch, predictor))
                batch = []
        if batch:
            records.extend(_predict_batch(batch, predictor))

    return Manifest(
        source_file=str(input_path),
        source_size=offset,
        source_sha256=overall_hash.hexdigest(),
        chunks=records,
    )


def _build_manifest_content_aware_streaming(
    input_path: Path,
    predictor: AlgorithmPredictor,
    batch_size: int,
    window_size: int = WINDOW_SIZE,
    printable_threshold: float = PRINTABLE_THRESHOLD,
    entropy_threshold: float = ENTROPY_THRESHOLD,
    hex_ratio_threshold: float = HEX_RATIO_THRESHOLD,
    min_chunk_size: int = MIN_CHUNK_SIZE,
) -> Manifest:
    """Build manifest for `input_path` using content-aware chunking, streaming from disk per window
    Mirrors content_aware.chunk()'s boundary logic
    (see that function for what the threshold args mean)"""
    overall_hash = hashlib.sha256()
    records: list[ChunkRecord] = []
    pending: list[tuple[int, bytes]] = []  # completed chunks awaiting prediction

    def flush_batch() -> None:
        nonlocal pending
        if pending:
            records.extend(_predict_batch(pending, predictor))
            pending = []

    run_start = 0
    prev_label: str | None = None
    buffer = bytearray()
    pos = 0

    with open(input_path, "rb") as f:
        while window := f.read(window_size):
            overall_hash.update(window)
            buffer.extend(window)
            is_text = (
                printable_ratio(window) >= printable_threshold
                and shannon_entropy(window) < entropy_threshold
                and hex_ratio(window) < hex_ratio_threshold
            )
            label = "text" if is_text else "binary"

            if prev_label is not None and label != prev_label:
                run_len = pos - run_start
                if run_len >= min_chunk_size:
                    finalized = bytes(buffer[:run_len])
                    pending.append((run_start, finalized))
                    if len(pending) >= batch_size:
                        flush_batch()
                    del buffer[:run_len]
                    run_start = pos
                # else: run too short, keep accumulating past this window

            prev_label = label
            pos += len(window)

    if buffer:
        pending.append((run_start, bytes(buffer)))
    flush_batch()

    _warn_if_oversized_chunks([r.length for r in records], DEFAULT_CHUNK_SIZE)

    return Manifest(
        source_file=str(input_path),
        source_size=pos,
        source_sha256=overall_hash.hexdigest(),
        chunks=records,
    )


def build_manifest_from_file(
    input_path: Path,
    chunker: str = "fixed_size",
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    model_path: Path | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> Manifest:
    """Chunk `input_path`, predict a per-chunk algorithm, and build manifest
    `chunker` is "fixed_size" or "content_aware"
    raises ValueError otherwise"""
    predictor = AlgorithmPredictor.load(model_path or DEFAULT_MODEL_PATH)

    if chunker == "fixed_size":
        return _build_manifest_fixed_size_streaming(input_path, chunk_size, predictor, batch_size)
    if chunker == "content_aware":
        return _build_manifest_content_aware_streaming(input_path, predictor, batch_size)
    raise ValueError(f"unknown chunker: {chunker}")


def main() -> None:
    """CLI entry point. Parses sys.argv and runs `build_manifest_from_file`"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("manifest_out", type=Path)
    parser.add_argument(
        "--chunker", choices=["fixed_size", "content_aware"], default="fixed_size"
    )
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="chunks held in memory per prediction batch (fixed_size chunker only)",
    )
    args = parser.parse_args()

    manifest = build_manifest_from_file(
        args.input,
        chunker=args.chunker,
        chunk_size=args.chunk_size,
        model_path=args.model_path,
        batch_size=args.batch_size,
    )
    write_manifest(manifest, args.manifest_out)
    print(f"wrote manifest for {len(manifest.chunks)} chunks -> {args.manifest_out}")


if __name__ == "__main__":
    main()
