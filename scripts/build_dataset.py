#!/usr/bin/env python
"""Build labeled training dataset from corpus of files

For every file under --corpus: 
chunk it, extract features per chunk, brute force label each chunk with best performing algorithm, 
write results to labeled dataset (--out)

Default chunker is fixed_size; 
content_aware is available via --chunker for corpora with mixed content types within files

Resumable: each file's rows written to own shard under --shard-dir once fully labeled (atomic write), so crash only costs the file in flight 
Re-running skips files that already have a shard
"""

import argparse
import os
import time
from pathlib import Path

import pandas as pd

from src.chunking import content_aware, fixed_size
from src.features.extract import extract
from src.labeling.brute_force_label import label_chunk

VALID_SUFFIXES = (".parquet", ".csv")


def shard_path_for(source_file: Path, corpus_dir: Path, shard_dir: Path, suffix: str) -> Path:
    """Compute shard file path for source file

    Args:
        source_file: File the shard covers
        corpus_dir: Root corpus directory `source_file` lives under
        shard_dir: Directory shards are written to
        suffix: Shard file extension, ".parquet" or ".csv"

    Returns:
        Shard path, unique per source file
    """
    rel = source_file.relative_to(corpus_dir)
    safe_name = str(rel).replace(os.sep, "__")
    return shard_dir / f"{safe_name}{suffix}"


def _chunk_data(data: bytes, chunker: str, chunk_size: int) -> list[bytes]:
    if chunker == "fixed_size":
        return list(fixed_size.chunk(data, chunk_size=chunk_size))
    if chunker == "content_aware":
        return list(content_aware.chunk(data))
    raise ValueError(f"unknown chunker: {chunker}")


def build_rows_for_file(
    file_path: Path, chunker: str = "fixed_size", chunk_size: int = fixed_size.DEFAULT_CHUNK_SIZE
) -> list[dict]:
    """Chunk and brute force label every chunk of one file

    Args:
        file_path: File to process
        chunker: "fixed_size" or "content_aware"
        chunk_size: Chunk size in bytes (fixed_size only)

    Returns:
        One row (dict) per chunk: features, brute-force label, per-algorithm sizes/times, source file path
    """
    data = file_path.read_bytes()
    rows = []
    for chunk_bytes in _chunk_data(data, chunker, chunk_size):
        if not chunk_bytes:
            continue
        features = extract(chunk_bytes)
        label = label_chunk(chunk_bytes)
        row = dict(features)
        row["best_algorithm"] = label.best_algorithm
        row["best_size"] = label.best_size
        for algo, size in label.sizes.items():
            row[f"size_{algo}"] = size
        for algo, seconds in label.times.items():
            row[f"time_{algo}"] = seconds
        row["source_file"] = str(file_path)
        rows.append(row)
    return rows


def _write_table(df: pd.DataFrame, path: Path, suffix: str) -> None:
    if suffix == ".parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)


def _read_table(path: Path, suffix: str) -> pd.DataFrame:
    if suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def write_shard(rows: list[dict], shard_path: Path, suffix: str) -> None:
    """Write shard atomically: build under .tmp name then rename into place

    Args:
        rows: Rows to write, as produced by `build_rows_for_file`
        shard_path: Destination path
        suffix: File format - ".parquet" or ".csv"
    """
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    tmp_path = shard_path.with_name(shard_path.name + ".tmp")
    _write_table(df, tmp_path, suffix)
    os.replace(tmp_path, shard_path)  # atomic on POSIX and Windows (same filesystem)


def merge_shards(shard_dir: Path, suffix: str) -> pd.DataFrame:
    """Concatenate every shard in directory into one dataset

    Args:
        shard_dir: Directory containing shard files
        suffix: Shard file extension - ".parquet" or ".csv"

    Returns:
        Combined DataFrame of all shards' rows
        Empty if none found
    """
    shard_files = sorted(p for p in shard_dir.glob(f"*{suffix}") if not p.name.endswith(".tmp"))
    frames = [_read_table(p, suffix) for p in shard_files]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def build_dataset(
    corpus_dir: Path,
    out_path: Path,
    shard_dir: Path,
    force: bool = False,
    chunker: str = "fixed_size",
    chunk_size: int = fixed_size.DEFAULT_CHUNK_SIZE,
) -> pd.DataFrame:
    """Build or resume labeled dataset from corpus of files

    Args:
        corpus_dir: Directory of input files to label (recursive)
        out_path: Combined output dataset path - .parquet or .csv
        shard_dir: Per-file shard directory
        force: If True, reprocess files even if shard already exists
        chunker: "fixed_size" or "content_aware"
        chunk_size: Chunk size in bytes (fixed_size only)

    Returns:
        Combined labeled dataset (also written to `out_path`)

    Raises:
        ValueError: If `corpus_dir` contains no files
    """
    suffix = out_path.suffix if out_path.suffix in VALID_SUFFIXES else ".parquet"
    files = sorted(p for p in corpus_dir.rglob("*") if p.is_file() and p.name != ".gitkeep")
    if not files:
        raise ValueError(f"no files found under {corpus_dir}")

    skipped = 0
    processed = 0
    failed: list[tuple[Path, str]] = []
    for i, file_path in enumerate(files, 1):
        shard_path = shard_path_for(file_path, corpus_dir, shard_dir, suffix)
        if shard_path.exists() and not force:
            print(f"[{i}/{len(files)}] skip {file_path.name} (already labeled)")
            skipped += 1
            continue
        print(f"[{i}/{len(files)}] labeling {file_path.name} ...")
        start = time.perf_counter()
        try:
            rows = build_rows_for_file(file_path, chunker=chunker, chunk_size=chunk_size)
            write_shard(rows, shard_path, suffix)
        except Exception as e:
            print(f"    FAILED: {e}")
            failed.append((file_path, str(e)))
            continue
        elapsed = time.perf_counter() - start
        print(f"    {len(rows)} chunks in {elapsed:.2f}s -> {shard_path}")
        processed += 1

    df = merge_shards(shard_dir, suffix)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _write_table(df, out_path, suffix)
    print(
        f"Done: {processed} file(s) processed, {skipped} skipped (already labeled), "
        f"{len(failed)} failed. Wrote {len(df)} labeled chunks to {out_path}"
    )
    if failed:
        print("Failed files:")
        for path, err in failed:
            print(f"  {path}: {err}")
    return df


def main() -> None:
    """CLI entry point. Parses sys.argv and runs `build_dataset`"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=Path("data/raw"), help="Directory of input files")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/labeled/dataset.parquet"),
        help="Combined output dataset path - .parquet or .csv",
    )
    parser.add_argument(
        "--shard-dir",
        type=Path,
        default=None,
        help="Per-file shard directory (default: <out dir>/.shards)",
    )
    parser.add_argument(
        "--force", action="store_true", help="Reprocess files even if shard already exists"
    )
    parser.add_argument(
        "--chunker", choices=["fixed_size", "content_aware"], default="fixed_size"
    )
    parser.add_argument(
        "--chunk-size", type=int, default=fixed_size.DEFAULT_CHUNK_SIZE, help="fixed_size chunker only"
    )
    args = parser.parse_args()

    shard_dir = args.shard_dir or (args.out.parent / ".shards")
    build_dataset(
        args.corpus,
        args.out,
        shard_dir,
        force=args.force,
        chunker=args.chunker,
        chunk_size=args.chunk_size,
    )


if __name__ == "__main__":
    main()
