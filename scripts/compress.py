"""User-facing entry point: file -> manifest -> archive, in one command

Chains scripts.ml_to_manifest and encoder.encode, then prints a summary (sizes, ratio, per-algorithm breakdown)
Manifest is kept on disk next to archive by default; 
pass --no-keep-manifest to discard it
"""

import argparse
import os
from collections import Counter
from pathlib import Path

from encoder.encode import encode
from encoder.manifest import write_manifest
from scripts.ml_to_manifest import DEFAULT_BATCH_SIZE, DEFAULT_CHUNK_SIZE, build_manifest_from_file


def compress(
    input_path: Path,
    output_path: Path,
    chunker: str = "fixed_size",
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    model_path: Path | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    manifest_path: Path | None = None,
    keep_manifest: bool = True,
) -> dict:
    """Run the full compress pipeline: file -> manifest -> archive

    `manifest_path` defaults to `output_path` with ".manifest.json" appended; 
    pass `keep_manifest=False` to delete it after encoding

    Returns:
        Summary dict: input_path, output_path, manifest_path, original_size, compressed_size, ratio, chunk_count, algorithm_counts
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    manifest_path = Path(manifest_path) if manifest_path else output_path.with_suffix(
        output_path.suffix + ".manifest.json"
    )

    manifest = build_manifest_from_file(
        input_path,
        chunker=chunker,
        chunk_size=chunk_size,
        model_path=model_path,
        batch_size=batch_size,
    )
    write_manifest(manifest, manifest_path)

    try:
        encode(manifest_path, output_path)
    except BaseException:
        if not keep_manifest:
            manifest_path.unlink(missing_ok=True)
        raise

    if not keep_manifest:
        os.remove(manifest_path)
        manifest_path = None

    original_size = manifest.source_size
    compressed_size = output_path.stat().st_size
    algorithm_counts = Counter(c.algorithm for c in manifest.chunks)

    return {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "manifest_path": str(manifest_path) if manifest_path else None,
        "original_size": original_size,
        "compressed_size": compressed_size,
        "ratio": (original_size / compressed_size) if compressed_size else float("inf"),
        "chunk_count": len(manifest.chunks),
        "algorithm_counts": dict(algorithm_counts),
    }


def _print_summary(summary: dict) -> None:
    """Print a human-readable summary from `compress`'s return value."""
    print(f"compressed {summary['input_path']} -> {summary['output_path']}")
    print(f"  original size:    {summary['original_size']:,} bytes")
    print(f"  compressed size:  {summary['compressed_size']:,} bytes")
    print(f"  ratio:            {summary['ratio']:.3f}x")
    print(f"  chunks:           {summary['chunk_count']}")
    print("  algorithm breakdown:")
    for algo, count in sorted(summary["algorithm_counts"].items(), key=lambda kv: -kv[1]):
        pct = 100 * count / summary["chunk_count"]
        print(f"    {algo:8s} {count:6d} chunks ({pct:5.1f}%)")
    if summary["manifest_path"]:
        print(f"  manifest saved:   {summary['manifest_path']}")


def main() -> None:
    """CLI entry point. Parses sys.argv and runs `compress`"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--chunker", choices=["fixed_size", "content_aware"], default="fixed_size")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--manifest-path", type=Path, default=None)
    parser.add_argument(
        "--no-keep-manifest",
        action="store_false",
        dest="keep_manifest",
        help="delete the intermediate manifest after encoding instead of keeping it on disk",
    )
    args = parser.parse_args()

    summary = compress(
        args.input,
        args.output,
        chunker=args.chunker,
        chunk_size=args.chunk_size,
        model_path=args.model_path,
        batch_size=args.batch_size,
        manifest_path=args.manifest_path,
        keep_manifest=args.keep_manifest,
    )
    _print_summary(summary)


if __name__ == "__main__":
    main()
