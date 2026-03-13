"""Pipeline configuration — per-stage config classes with config.toml + env var resolution."""

from __future__ import annotations

import os
import warnings
from pathlib import Path

from pydantic import BaseModel


def load_config() -> dict:
    """Load config.toml from CWD or package root."""
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib  # type: ignore[no-redef]

    for candidate in [Path("config.toml"), Path(__file__).resolve().parents[3] / "config.toml"]:
        if candidate.exists():
            with open(candidate, "rb") as f:
                return tomllib.load(f)
    return {}


def _resolve_str(env_key: str, section: dict, toml_key: str, default: str) -> str:
    return os.environ.get(env_key) or section.get(toml_key, default)


def _resolve_bool(env_key: str, section: dict, toml_key: str, default: bool) -> bool:
    env_val = os.environ.get(env_key)
    if env_val is not None:
        return env_val.lower() in ("true", "1", "yes")
    toml_val = section.get(toml_key)
    if toml_val is not None:
        return bool(toml_val)
    return default


class PipelineConfig(BaseModel):
    """Top-level pipeline toggles (maps to [pipeline] in config.toml)."""

    enrich: bool = False
    web: bool = False

    @classmethod
    def load(cls) -> PipelineConfig:
        cfg = load_config().get("pipeline", {})
        return cls(
            enrich=_resolve_bool("CORPPROFILE_ENRICH", cfg, "enrich", False),
            web=_resolve_bool("CORPPROFILE_WEB", cfg, "web", False),
        )


class EnrichConfig(BaseModel):
    """Config for clean/enrich/refine stages (maps to [enrich] in config.toml)."""

    model: str
    aws_region: str | None = None
    aws_profile: str | None = None

    @classmethod
    def load(cls) -> EnrichConfig:
        cfg = load_config()
        enrich = cfg.get("enrich", {})
        aws = cfg.get("aws", {})

        # Support deprecated CORPPROFILE_LLM_MODEL env var
        model = os.environ.get("CORPPROFILE_ENRICH_MODEL")
        if not model:
            deprecated = os.environ.get("CORPPROFILE_LLM_MODEL")
            if deprecated:
                warnings.warn(
                    "CORPPROFILE_LLM_MODEL is deprecated, use CORPPROFILE_ENRICH_MODEL",
                    DeprecationWarning,
                    stacklevel=2,
                )
                model = deprecated
        if not model:
            model = enrich.get("model")
        if not model:
            raise RuntimeError(
                "Enrich model not configured. Set model in [enrich] in config.toml, "
                "or set CORPPROFILE_ENRICH_MODEL env var."
            )

        return cls(
            model=model,
            aws_region=_resolve_str("AWS_DEFAULT_REGION", aws, "region", "") or None,
            aws_profile=_resolve_str("AWS_PROFILE", aws, "profile", "") or None,
        )


class ResearchConfig(BaseModel):
    """Config for the research command (maps to [research] in config.toml)."""

    model: str
    provider: str = "auto"
    aws_region: str | None = None
    aws_profile: str | None = None

    @classmethod
    def load(cls) -> ResearchConfig:
        cfg = load_config()
        research = cfg.get("research", {})
        aws = cfg.get("aws", {})

        model = _resolve_str("CORPPROFILE_RESEARCH_MODEL", research, "model", "")
        if not model:
            raise RuntimeError(
                "Research model not configured. Set model in [research] in config.toml, "
                "or set CORPPROFILE_RESEARCH_MODEL env var."
            )

        return cls(
            model=model,
            provider=_resolve_str("CORPPROFILE_RESEARCH_PROVIDER", research, "provider", "auto"),
            aws_region=_resolve_str("AWS_DEFAULT_REGION", aws, "region", "") or None,
            aws_profile=_resolve_str("AWS_PROFILE", aws, "profile", "") or None,
        )


class WebConfig(BaseModel):
    """Config for web search enrichment stage (maps to [web] in config.toml)."""

    model: str
    provider: str = "auto"
    aws_region: str | None = None
    aws_profile: str | None = None

    @classmethod
    def load(cls) -> WebConfig:
        cfg = load_config()
        web = cfg.get("web", {})
        aws = cfg.get("aws", {})

        model = _resolve_str("CORPPROFILE_WEB_MODEL", web, "model", "")
        if not model:
            raise RuntimeError(
                "Web search model not configured. Set model in [web] in config.toml, "
                "or set CORPPROFILE_WEB_MODEL env var."
            )

        return cls(
            model=model,
            provider=_resolve_str("CORPPROFILE_WEB_PROVIDER", web, "provider", "auto"),
            aws_region=_resolve_str("AWS_DEFAULT_REGION", aws, "region", "") or None,
            aws_profile=_resolve_str("AWS_PROFILE", aws, "profile", "") or None,
        )
