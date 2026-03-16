"""Postgres connection helpers for corp-graph database."""

from __future__ import annotations

import json
import os
from typing import Any

import psycopg
import psycopg.rows
from dotenv import load_dotenv


def get_connection() -> psycopg.Connection:
    """Return a psycopg connection to the corp-graph database.

    Reads CORPGRAPH_DB_URL from environment (via .env file).
    """
    load_dotenv()
    db_url = os.environ.get("CORPGRAPH_DB_URL")
    if not db_url:
        raise RuntimeError(
            "CORPGRAPH_DB_URL not set. Copy .env.example to .env and configure it."
        )
    return psycopg.connect(db_url, row_factory=psycopg.rows.dict_row)


def _ensure_profile_cache_table(conn: psycopg.Connection) -> None:
    """Create profile_cache table if it doesn't exist."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS profile_cache (
            issuer_id TEXT PRIMARY KEY,
            profile_json JSONB NOT NULL,
            enriched_at TIMESTAMPTZ DEFAULT NOW()
        )
    """)
    conn.commit()


def save_cached_profile(conn: psycopg.Connection, issuer_id: str, profile_data: dict[str, Any]) -> None:
    """Cache an enriched profile to the database."""
    _ensure_profile_cache_table(conn)
    conn.execute(
        """INSERT INTO profile_cache (issuer_id, profile_json, enriched_at)
           VALUES (%s, %s, NOW())
           ON CONFLICT (issuer_id) DO UPDATE SET
             profile_json = EXCLUDED.profile_json,
             enriched_at = EXCLUDED.enriched_at""",
        (issuer_id, json.dumps(profile_data, default=str)),
    )
    conn.commit()


def load_cached_profile(conn: psycopg.Connection, issuer_id: str) -> dict[str, Any] | None:
    """Load a cached enriched profile from the database. Returns None if not found."""
    _ensure_profile_cache_table(conn)
    row = conn.execute(
        "SELECT profile_json FROM profile_cache WHERE issuer_id = %s",
        (issuer_id,),
    ).fetchone()
    if row:
        return row["profile_json"]
    return None
