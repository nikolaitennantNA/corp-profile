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
        "--enrich", action="store_true",
        help="Run LLM enrichment on the profile",
    )
    build_cmd.add_argument(
        "--web", action="store_true",
        help="Enable web search during enrichment (implies --enrich)",
    )

    # research command
    research_cmd = sub.add_parser("research", help="Research a company via web search (no DB needed)")
    research_source = research_cmd.add_mutually_exclusive_group(required=True)
    research_source.add_argument(
        "identifier", nargs="?", default=None,
        help="ISIN, LEI, issuer_id, or company name",
    )
    research_source.add_argument("--seed", help="Partial JSON file to seed research")
    research_cmd.add_argument("--name", help="Company name to help search accuracy")
    research_cmd.add_argument(
        "-o", "--output", help="Also save profile JSON to this path"
    )

    args = parser.parse_args()

    if args.command == "build":
        from .config import PipelineConfig

        pipeline_cfg = PipelineConfig.load()
        if pipeline_cfg.enrich:
            args.enrich = True
        if pipeline_cfg.web:
            args.web = True

        # --web implies --enrich
        if args.web:
            args.enrich = True

        try:
            if args.from_file:
                profile = build_profile_from_file(args.from_file)
            else:
                profile = build_profile(args.identifier)
        except LookupError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

        if args.enrich:
            profile = _run_enrich(profile, web=args.web)

        if args.output:
            save_profile(profile, args.output)
            print(f"Profile JSON saved to {args.output}", file=sys.stderr)

        out_path = save_profile_markdown(profile)
        print(f"Saved to {out_path}", file=sys.stderr)

    elif args.command == "research":
        from .config import ResearchConfig
        from .research import research_profile

        try:
            config = ResearchConfig.load()
        except RuntimeError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

        seed = None
        identifier = args.identifier
        if args.seed:
            seed = build_profile_from_file(args.seed)
            if not identifier:
                identifier = seed.issuer_id or (seed.isin_list[0] if seed.isin_list else None)

        try:
            profile, changes = research_profile(
                identifier=identifier,
                name=args.name,
                seed=seed,
                config=config,
            )
        except Exception as e:
            print(f"Research failed: {e}", file=sys.stderr)
            sys.exit(1)

        if changes:
            print("Research findings:", file=sys.stderr)
            for c in changes:
                print(f"  - {c}", file=sys.stderr)

        if args.output:
            save_profile(profile, args.output)
            print(f"Profile JSON saved to {args.output}", file=sys.stderr)

        out_path = save_profile_markdown(profile)
        print(f"Saved to {out_path}", file=sys.stderr)

    else:
        parser.print_help()
        sys.exit(1)


def _run_enrich(profile, *, web: bool = False):
    """Run enrichment and print changes to stderr."""
    from .config import EnrichConfig, WebConfig
    from .enrich import enrich_profile

    try:
        enrich_config = EnrichConfig.load()
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    web_config = None
    if web:
        try:
            web_config = WebConfig.load()
        except RuntimeError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

    try:
        profile, changes = enrich_profile(profile, enrich_config, web_config=web_config)
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
