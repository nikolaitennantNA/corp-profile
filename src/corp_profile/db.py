"""Postgres connection helpers for corp-graph database."""

from __future__ import annotations

import os

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
