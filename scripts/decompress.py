"""User-facing entry point: archive -> original file, in one command

Thin wrapper around decoder.decode(), printing a summary (restored size, confirmation integrity checks passed)
All verification happens inside decoder.decode()
this only reports result
"""

import argparse
from pathlib import Path

from decoder.decode import decode


def decompress(archive_path: Path, output_path: Path) -> dict:
    """Decompress `archive_path` into `output_path`
    Returns a summary dict (archive_path, output_path, restored_size)
    Raises DecodeError if archive fails any integrity check"""
    archive_path = Path(archive_path)
    output_path = Path(output_path)

    decode(archive_path, output_path)

    return {
        "archive_path": str(archive_path),
        "output_path": str(output_path),
        "restored_size": output_path.stat().st_size,
    }


def _print_summary(summary: dict) -> None:
    """Print a human-readable summary from `decompress`'s return value."""
    print(f"decompressed {summary['archive_path']} -> {summary['output_path']}")
    print(f"  restored size: {summary['restored_size']:,} bytes")
    print("  integrity verified: per-chunk checksums + whole-file checksum matched")


def main() -> None:
    """CLI entry point. Parses sys.argv and runs `decompress`"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    summary = decompress(args.archive, args.output)
    _print_summary(summary)


if __name__ == "__main__":
    main()
