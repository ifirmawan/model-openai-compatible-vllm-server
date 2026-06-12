"""Register allowed client applications for the Modal vLLM backend.

Usage:
    python -m register_apps --name="vibe2blog" --expired=2027-01-01
"""

from __future__ import annotations

import argparse

from app_registry import get_db_path, register_app


def main() -> None:
    parser = argparse.ArgumentParser(description="Register an app API key for X-API-Key auth.")
    parser.add_argument("--name", required=True, help="Client application name, e.g. vibe2blog")
    parser.add_argument("--expired", required=True, help="Expiry date in YYYY-MM-DD format")
    parser.add_argument("--db", default=None, help="SQLite DB path. Defaults to APP_REGISTRY_DB or app_registry.sqlite3")
    parser.add_argument("--api-key", default=None, help="Optional explicit key for rotation/import workflows")
    args = parser.parse_args()

    app = register_app(
        name=args.name,
        expires_at=args.expired,
        db_path=args.db,
        api_key=args.api_key,
    )
    print(f"registered={app.name}")
    print(f"expires_at={app.expires_at}")
    print(f"db={get_db_path(args.db)}")
    print(f"api_key={app.api_key}")
    print("Store this key securely. It is shown only by this command.")


if __name__ == "__main__":
    main()
