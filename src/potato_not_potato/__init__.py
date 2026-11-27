import sys

from potato_not_potato.cli import main as _cli_main


def main() -> None:
    sys.exit(_cli_main())


__all__ = ["main"]
