"""CLI entry point: python -m corp_profile."""

from __future__ import annotations

import argparse
import sys

from dotenv import load_dotenv

load_dotenv()

from .profile import (
    build_profile,
    build_profile_from_file,
    save_profile,
    save_profile_markdown,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="corp-profile",
        description="Build rich company context documents from corp-graph or JSON files",
    )
    sub = parser.add_subparsers(dest="command")

    # build command
    build_cmd = sub.add_parser("build", help="Build a company profile")
    build_source = build_cmd.add_mutually_exclusive_group(required=True)
    build_source.add_argument(
        "identifier", nargs="?", default=None,
        help="ISIN, LEI, issuer_id, or company name to look up in corp-graph DB",
    )
    build_source.add_argument("--from-file", help="Load profile from JSON file")
    build_cmd.add_argument(
        "-o", "--output", help="Also save profile JSON to this path"
    )
    build_cmd.add_argument(
        "--llm", action="store_true", help="Run LLM enrichment on the profile"
    )
    build_cmd.add_argument(
        "--web", action="store_true",
        help="Enable web search during LLM enrichment (implies --llm)",
    )

    args = parser.parse_args()

    if args.command == "build":
        # Apply config.toml [profile] defaults — CLI flags override
        from .enrich import load_config

        profile_cfg = load_config().get("profile", {})
        if profile_cfg.get("llm", False):
            args.llm = True
        if profile_cfg.get("web", False):
            args.web = True

        # --web implies --llm
        if args.web:
            args.llm = True

        try:
            if args.from_file:
                profile = build_profile_from_file(args.from_file)
            else:
                profile = build_profile(args.identifier)
        except LookupError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

        if args.llm:
            profile = _run_enrich(profile, web_search=args.web)

        if args.output:
            save_profile(profile, args.output)
            print(f"Profile JSON saved to {args.output}", file=sys.stderr)

        # Always save markdown — this is what downstream LLMs consume
        out_path = save_profile_markdown(profile)
        print(f"Saved to {out_path}", file=sys.stderr)

    else:
        parser.print_help()
        sys.exit(1)


def _run_enrich(profile, *, web_search: bool = False):
    """Run enrichment and print changes to stderr."""
    from .enrich import EnrichConfig, enrich_profile

    try:
        config = EnrichConfig.load()
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # CLI --web flag overrides config
    if web_search:
        config = config.model_copy(update={"web_search": True})

    try:
        profile, changes = enrich_profile(profile, config)
    except Exception as e:
        print(f"Enrichment failed: {e}", file=sys.stderr)
        sys.exit(1)
    if changes:
        print("Enrichment changes:", file=sys.stderr)
        for c in changes:
            print(f"  - {c}", file=sys.stderr)
    return profile


if __name__ == "__main__":
    main()
