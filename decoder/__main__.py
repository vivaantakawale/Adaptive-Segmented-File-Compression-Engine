"""CLI entry point: python -m decoder <archive> <output.file>"""

import sys
from pathlib import Path

from .decode import decode


def main(argv: list[str]) -> int:
    """Run decoder CLI

    Args:
        argv: Command line arguments:
            [archive_path, output_path]

    Returns:
        Exit code: 0 on success, 2 on usage error
    """
    if len(argv) != 2:
        print("usage: python -m decoder <archive> <output.file>", file=sys.stderr)
        return 2
    archive_path, output_path = Path(argv[0]), Path(argv[1])
    decode(archive_path, output_path)
    print(f"wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
