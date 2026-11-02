import json
import os
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from hashlib import pbkdf2_hmac
from hmac import compare_digest
from secrets import token_bytes

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)
DB_PATH = os.path.join(DATA_DIR, "inspect_web.db")

_db_lock = threading.Lock()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


@contextmanager
def get_conn():
    conn = _connect()
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def hash_password(password: str) -> str:
    salt = token_bytes(16)
    digest = pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 120_000)
    return f"pbkdf2_sha256$120000${salt.hex()}${digest.hex()}"


def verify_password(password: str, encoded: str) -> bool:
    try:
        scheme, rounds, salt_hex, digest_hex = encoded.split("$")
        if scheme != "pbkdf2_sha256":
            return False
        digest = pbkdf2_hmac(
            "sha256", password.encode("utf-8"), bytes.fromhex(salt_hex), int(rounds)
        )
        return compare_digest(digest.hex(), digest_hex)
    except Exception:
        return False


def init_db() -> None:
    with _db_lock, get_conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS detection_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                counts_json TEXT NOT NULL,
                expected_json TEXT NOT NULL,
                stable INTEGER NOT NULL,
                result TEXT NOT NULL,
                note TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                username TEXT NOT NULL,
                action TEXT NOT NULL,
                details TEXT
            )
            """
        )


def seed_defaults() -> None:
    default_expected = {"A": 5, "B": 0, "C": 0, "D": 0, "E": 2, "F": 10}
    with _db_lock, get_conn() as conn:
        row = conn.execute("SELECT 1 FROM users WHERE username = ?", ("admin",)).fetchone()
        if not row:
            conn.execute(
                "INSERT INTO users(username, password_hash, role, created_at) VALUES(?,?,?,?)",
                ("admin", hash_password(os.environ.get("INSPECT_WEB_DEFAULT_PASSWORD", "admin123")), "admin", utc_now()),
            )
        for key, value in {
            "expected_bom": default_expected,
            "camera_urls": {"cam1": "rtsp://192.168.1.10/profile1", "cam2": "rtsp://192.168.1.20/profile1"},
            "pipeline_mode": os.environ.get("INSPECT_WEB_PIPELINE_MODE", "demo"),
        }.items():
            found = conn.execute("SELECT 1 FROM settings WHERE key = ?", (key,)).fetchone()
            if not found:
                conn.execute(
                    "INSERT INTO settings(key, value, updated_at) VALUES(?,?,?)",
                    (key, json.dumps(value), utc_now()),
                )


def get_user(username: str):
    with _db_lock, get_conn() as conn:
        return conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()


def get_setting_json(key: str, default):
    with _db_lock, get_conn() as conn:
        row = conn.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
    if not row:
        return default
    try:
        return json.loads(row["value"])
    except Exception:
        return default


def set_setting_json(key: str, value) -> None:
    payload = json.dumps(value)
    now = utc_now()
    with _db_lock, get_conn() as conn:
        exists = conn.execute("SELECT 1 FROM settings WHERE key = ?", (key,)).fetchone()
        if exists:
            conn.execute("UPDATE settings SET value = ?, updated_at = ? WHERE key = ?", (payload, now, key))
        else:
            conn.execute("INSERT INTO settings(key, value, updated_at) VALUES(?,?,?)", (key, payload, now))


def add_detection_event(counts: dict, expected: dict, stable: bool, result: str, note: str = "") -> None:
    with _db_lock, get_conn() as conn:
        conn.execute(
            "INSERT INTO detection_history(ts, counts_json, expected_json, stable, result, note) VALUES(?,?,?,?,?,?)",
            (utc_now(), json.dumps(counts), json.dumps(expected), int(stable), result, note),
        )


def list_detection_events(limit: int = 100):
    with _db_lock, get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM detection_history ORDER BY id DESC LIMIT ?", (int(limit),)
        ).fetchall()
    out = []
    for row in rows:
        out.append(
            {
                "id": row["id"],
                "ts": row["ts"],
                "counts": json.loads(row["counts_json"]),
                "expected": json.loads(row["expected_json"]),
                "stable": bool(row["stable"]),
                "result": row["result"],
                "note": row["note"] or "",
            }
        )
    return out


def add_audit(username: str, action: str, details: str = "") -> None:
    with _db_lock, get_conn() as conn:
        conn.execute(
            "INSERT INTO audit_log(ts, username, action, details) VALUES(?,?,?,?)",
            (utc_now(), username, action, details),
        )
