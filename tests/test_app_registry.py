from __future__ import annotations

from datetime import date

from app_registry import register_app, verify_api_key


def test_register_app_returns_key_and_allows_valid_request(tmp_path):
    db = tmp_path / "registry.sqlite3"

    app = register_app(name="vibe2blog", expires_at="2027-01-01", db_path=str(db))

    assert app.name == "vibe2blog"
    assert app.api_key.startswith("vllm_")
    assert verify_api_key(app.api_key, db_path=str(db), today=date(2026, 6, 12))


def test_expired_key_is_rejected(tmp_path):
    db = tmp_path / "registry.sqlite3"

    app = register_app(name="old-app", expires_at="2025-01-01", db_path=str(db))

    assert not verify_api_key(app.api_key, db_path=str(db), today=date(2026, 6, 12))


def test_registering_same_app_rotates_key(tmp_path):
    db = tmp_path / "registry.sqlite3"

    first = register_app(name="vibe2blog", expires_at="2027-01-01", db_path=str(db))
    second = register_app(name="vibe2blog", expires_at="2027-01-01", db_path=str(db))

    assert first.api_key != second.api_key
    assert not verify_api_key(first.api_key, db_path=str(db), today=date(2026, 6, 12))
    assert verify_api_key(second.api_key, db_path=str(db), today=date(2026, 6, 12))
