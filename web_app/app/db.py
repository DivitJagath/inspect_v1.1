"""Small local SQLite store for web accounts, inspection snapshots and audit."""

from contextlib import contextmanager
from datetime import datetime, timezone
from hashlib import pbkdf2_hmac
from hmac import compare_digest
import json
import os
from pathlib import Path
import sqlite3
import threading
from secrets import token_bytes

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DATA_DIR / "inspect_web.db"
_db_lock = threading.Lock()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH), timeout=10)
    conn.row_factory = sqlite3.Row
    return conn


@contextmanager
def get_conn():
    conn = _connect()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def hash_password(password: str) -> str:
    rounds = 310_000
    salt = token_bytes(16)
    digest = pbkdf2_hmac("sha256", password.encode("utf-8"), salt, rounds)
    return f"pbkdf2_sha256${rounds}${salt.hex()}${digest.hex()}"


def verify_password(password: str, encoded: str) -> bool:
    try:
        scheme, rounds_text, salt_hex, digest_hex = encoded.split("$")
        rounds = int(rounds_text)
        if scheme != "pbkdf2_sha256" or not 100_000 <= rounds <= 1_000_000:
            return False
        digest = pbkdf2_hmac(
            "sha256", password.encode("utf-8"), bytes.fromhex(salt_hex), rounds
        )
        return compare_digest(digest.hex(), digest_hex)
    except (ValueError, TypeError):
        return False


def init_db() -> None:
    with _db_lock, get_conn() as conn:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL,
                created_at TEXT NOT NULL
            )"""
        )
        conn.execute(
            """CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )"""
        )
        conn.execute(
            """CREATE TABLE IF NOT EXISTS detection_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                counts_json TEXT NOT NULL,
                expected_json TEXT NOT NULL,
                stable INTEGER NOT NULL,
                result TEXT NOT NULL,
                note TEXT
            )"""
        )
        conn.execute(
            """CREATE TABLE IF NOT EXISTS audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                username TEXT NOT NULL,
                action TEXT NOT NULL,
                details TEXT
            )"""
        )


def seed_defaults() -> None:
    password = os.environ.get("INSPECT_WEB_DEFAULT_PASSWORD", "")
    if len(password) < 12:
        raise RuntimeError(
            "Set INSPECT_WEB_DEFAULT_PASSWORD to a password of at least "
            "12 characters before web-app startup."
        )
    with _db_lock, get_conn() as conn:
        row = conn.execute(
            "SELECT password_hash FROM users WHERE username = ?", ("admin",)
        ).fetchone()
        if not row:
            conn.execute(
                "INSERT INTO users(username, password_hash, role, created_at) "
                "VALUES(?,?,?,?)",
                ("admin", hash_password(password), "admin", utc_now()),
            )
        elif not verify_password(password, row["password_hash"]):
            conn.execute(
                "UPDATE users SET password_hash = ? WHERE username = ?",
                (hash_password(password), "admin"),
            )


def get_user(username: str):
    with _db_lock, get_conn() as conn:
        return conn.execute(
            "SELECT * FROM users WHERE username = ?", (username,)
        ).fetchone()


def add_detection_event(counts: dict, expected: dict, stable: bool,
                        result: str, note: str = "") -> None:
    with _db_lock, get_conn() as conn:
        conn.execute(
            "INSERT INTO detection_history(ts, counts_json, expected_json, stable, result, note) "
            "VALUES(?,?,?,?,?,?)",
            (utc_now(), json.dumps(counts), json.dumps(expected), int(stable), result, note),
        )


def list_detection_events(limit: int = 100):
    with _db_lock, get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM detection_history ORDER BY id DESC LIMIT ?",
            (max(1, min(500, int(limit))),),
        ).fetchall()
    return [{
        "id": row["id"],
        "ts": row["ts"],
        "counts": json.loads(row["counts_json"]),
        "expected": json.loads(row["expected_json"]),
        "stable": bool(row["stable"]),
        "result": row["result"],
        "note": row["note"] or "",
    } for row in rows]


def add_audit(username: str, action: str, details: str = "") -> None:
    with _db_lock, get_conn() as conn:
        conn.execute(
            "INSERT INTO audit_log(ts, username, action, details) VALUES(?,?,?,?)",
            (utc_now(), username, action, details),
        )
