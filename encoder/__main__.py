"""CLI entry point: python -m encoder <manifest.json> <output.archive>"""

import sys
from pathlib import Path

from .encode import encode


def main(argv: list[str]) -> int:
    """
    Run the encoder CLI

    Args:
        argv: Command-line arguments (excluding program name):
            [manifest_path, output_path]

    Returns:
        Exit code: 0 on success, 2 on usage error
    """
    if len(argv) != 2:
        print("usage: python -m encoder <manifest.json> <output.archive>", file=sys.stderr)
        return 2
    manifest_path, output_path = Path(argv[0]), Path(argv[1])
    encode(manifest_path, output_path)
    print(f"wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
