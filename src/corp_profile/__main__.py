"""CLI entry point: python -m corp_profile."""

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

    # build command
    build_cmd = sub.add_parser("build", help="Build a company profile")
    build_source = build_cmd.add_mutually_exclusive_group(required=True)
    build_source.add_argument("--isin", help="ISIN to look up in corp-graph DB")
    build_source.add_argument("--from-file", help="Load profile from JSON file")
    build_cmd.add_argument(
        "-o", "--output", help="Save profile JSON to file instead of printing"
    )
    build_cmd.add_argument(
        "--enrich", action="store_true", help="Run LLM enrichment on the profile"
    )

    # enrich command
    enrich_cmd = sub.add_parser("enrich", help="Enrich an existing profile JSON")
    enrich_cmd.add_argument("file", help="Path to profile JSON file")
    enrich_cmd.add_argument(
        "-o", "--output", help="Save enriched profile to file (default: print)"
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

        if args.enrich:
            profile = _run_enrich(profile)

        if args.output:
            save_profile(profile, args.output)
            print(f"Profile saved to {args.output}", file=sys.stderr)
        else:
            print(build_context_document(profile))

    elif args.command == "enrich":
        profile = build_profile_from_file(args.file)
        profile = _run_enrich(profile)

        if args.output:
            save_profile(profile, args.output)
            print(f"Enriched profile saved to {args.output}", file=sys.stderr)
        else:
            print(build_context_document(profile))

    else:
        parser.print_help()
        sys.exit(1)


def _run_enrich(profile):
    """Run enrichment and print changes to stderr."""
    from .enrich import EnrichConfig, enrich_profile

    try:
        config = EnrichConfig.from_env()
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    profile, changes = enrich_profile(profile, config)
    if changes:
        print("Enrichment changes:", file=sys.stderr)
        for c in changes:
            print(f"  - {c}", file=sys.stderr)
    return profile


if __name__ == "__main__":
    main()
