"""CLI entry point: python -m corp_profile build <ISIN>."""

from __future__ import annotations

import argparse
import sys

from .profile import build_context_document, build_profile


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="corp-profile",
        description="Build rich company context documents from corp-graph Postgres",
    )
    sub = parser.add_subparsers(dest="command")

    build_cmd = sub.add_parser("build", help="Build a company profile by ISIN")
    build_cmd.add_argument("isin", help="ISIN to look up")

    args = parser.parse_args()

    if args.command == "build":
        try:
            profile = build_profile(args.isin)
        except LookupError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
        print(build_context_document(profile))
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
