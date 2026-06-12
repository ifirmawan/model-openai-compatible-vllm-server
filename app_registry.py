"""SQLite-backed API key registry for allowed client applications."""

from __future__ import annotations

import hashlib
import os
import secrets
import sqlite3
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path


DEFAULT_DB_PATH = "app_registry.sqlite3"


@dataclass(frozen=True)
class RegisteredApp:
    name: str
    api_key: str
    expires_at: str


def get_db_path(db_path: str | None = None) -> str:
    """Resolve the registry DB path from CLI input, env, or local default."""
    return db_path or os.getenv("APP_REGISTRY_DB", DEFAULT_DB_PATH)


def connect(db_path: str | None = None) -> sqlite3.Connection:
    """Open the SQLite DB and ensure its parent directory exists."""
    resolved = Path(get_db_path(db_path))
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(resolved)


def init_db(db_path: str | None = None) -> None:
    """Create the app registry table if it does not already exist."""
    with connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS app_clients (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                api_key_hash TEXT NOT NULL UNIQUE,
                expires_at TEXT NOT NULL,
                created_at TEXT NOT NULL,
                revoked_at TEXT
            )
            """
        )


def generate_api_key() -> str:
    """Create a high-entropy API key suitable for the X-API-Key header."""
    return f"vllm_{secrets.token_urlsafe(32)}"


def hash_api_key(api_key: str) -> str:
    """Store only a SHA-256 hash of the API key."""
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()


def parse_expiry(expires_at: str) -> date:
    """Validate YYYY-MM-DD expiry strings used by the CLI and server."""
    return datetime.strptime(expires_at, "%Y-%m-%d").date()


def register_app(
    *,
    name: str,
    expires_at: str,
    db_path: str | None = None,
    api_key: str | None = None,
) -> RegisteredApp:
    """Create or rotate an app API key and return the raw key once."""
    clean_name = name.strip()
    if not clean_name:
        raise ValueError("App name is required.")
    parse_expiry(expires_at)
    raw_key = api_key or generate_api_key()
    now = datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
    init_db(db_path)
    with connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO app_clients (name, api_key_hash, expires_at, created_at, revoked_at)
            VALUES (?, ?, ?, ?, NULL)
            ON CONFLICT(name) DO UPDATE SET
                api_key_hash = excluded.api_key_hash,
                expires_at = excluded.expires_at,
                created_at = excluded.created_at,
                revoked_at = NULL
            """,
            (clean_name, hash_api_key(raw_key), expires_at, now),
        )
    return RegisteredApp(name=clean_name, api_key=raw_key, expires_at=expires_at)


def verify_api_key(api_key: str, *, db_path: str | None = None, today: date | None = None) -> bool:
    """Return true when an API key exists, is not revoked, and has not expired."""
    if not api_key:
        return False
    init_db(db_path)
    current = today or date.today()
    with connect(db_path) as conn:
        row = conn.execute(
            """
            SELECT expires_at, revoked_at
            FROM app_clients
            WHERE api_key_hash = ?
            """,
            (hash_api_key(api_key),),
        ).fetchone()
    if row is None:
        return False
    expires_at, revoked_at = row
    if revoked_at:
        return False
    return parse_expiry(expires_at) >= current
