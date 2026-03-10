"""CLI entry point: python -m corp_profile build <ISIN>."""

from __future__ import annotations

import argparse
import sys

from .profile import (
    build_context_document,
    build_profile,
    build_profile_from_file,
    save_profile,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="corp-profile",
        description="Build rich company context documents from corp-graph Postgres",
    )
    sub = parser.add_subparsers(dest="command")

    build_cmd = sub.add_parser("build", help="Build a company profile by ISIN")
    build_source = build_cmd.add_mutually_exclusive_group(required=True)
    build_source.add_argument("--isin", help="ISIN to look up in corp-graph DB")
    build_source.add_argument("--from-file", help="Load profile from JSON file")
    build_cmd.add_argument(
        "-o", "--output", help="Save profile JSON to file instead of printing"
    )

    args = parser.parse_args()

    if args.command == "build":
        try:
            if args.from_file:
                profile = build_profile_from_file(args.from_file)
            else:
                profile = build_profile(args.isin)
        except LookupError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

        if args.output:
            save_profile(profile, args.output)
            print(f"Profile saved to {args.output}", file=sys.stderr)
        else:
            print(build_context_document(profile))
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
