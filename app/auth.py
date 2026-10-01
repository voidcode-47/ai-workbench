"""轻量用户系统与知识库元数据：SQLite 存储 + PBKDF2 密码哈希 + HMAC 签名 token。

设计说明（不加外部依赖）：
- 用户表存 username / password_hash / salt / role(admin|employee)；
- 密码用 PBKDF2-HMAC-SHA256（12 万次迭代）+ 随机盐，绝不存明文；
- 登录令牌为自签 HMAC token（payload + 签名），含过期时间，服务端不存会话；
- 知识库元数据（owner）单独建表，供 RAG 权限判断使用。
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import re
import secrets
import sqlite3
import threading
import time
from datetime import datetime
from pathlib import Path

from app.config import get_settings
from app.exceptions import AppError

_DB_LOCK = threading.Lock()
_TOKEN_TTL_SECONDS = 7 * 24 * 3600  # 7 天
_USERNAME_RE = re.compile(r"^[A-Za-z0-9_.-]{2,32}$")


class UserExistsError(AppError):
    status_code = 400
    code = "USER_EXISTS"


class UserNotFoundError(AppError):
    status_code = 404
    code = "USER_NOT_FOUND"


class InvalidCredentialsError(AppError):
    status_code = 401
    code = "INVALID_CREDENTIALS"


def _db_path() -> Path:
    return Path(get_settings().data_dir) / "app.db"


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(_db_path(), timeout=10)
    conn.row_factory = sqlite3.Row
    return conn


# ---------- 初始化 ----------
def init_auth_db() -> None:
    """建表并确保初始管理员存在（缺省 admin/admin123，登录后请尽快改密）。"""
    with _DB_LOCK:
        conn = _connect()
        try:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS users ("
                " username TEXT PRIMARY KEY,"
                " password_hash TEXT NOT NULL,"
                " salt TEXT NOT NULL,"
                " role TEXT NOT NULL DEFAULT 'employee',"
                " created_at TEXT)"
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS kb_meta ("
                " kb_id TEXT PRIMARY KEY,"
                " name TEXT,"
                " owner TEXT NOT NULL,"
                " created_at TEXT)"
            )
            conn.commit()
        finally:
            conn.close()

    settings = get_settings()
    admin_name = settings.admin_username
    with _DB_LOCK:
        conn = _connect()
        try:
            row = conn.execute("SELECT 1 FROM users WHERE username=?", (admin_name,)).fetchone()
            if row is None:
                digest, salt = _hash_password(settings.admin_password)
                conn.execute(
                    "INSERT INTO users (username, password_hash, salt, role, created_at) VALUES (?,?,?,?,?)",
                    (admin_name, digest, salt, "admin", _now()),
                )
                conn.commit()
        finally:
            conn.close()


# ---------- 密码 ----------
def _hash_password(password: str, salt: str | None = None) -> tuple[str, str]:
    salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 120_000)
    return digest.hex(), salt


def _verify_password(password: str, password_hash: str, salt: str) -> bool:
    digest, _ = _hash_password(password, salt)
    return hmac.compare_digest(digest, password_hash)


# ---------- Token ----------
def _token_secret() -> bytes:
    path = Path(get_settings().data_dir) / "secret.key"
    if path.exists():
        return path.read_bytes()
    key = secrets.token_bytes(32)
    path.write_bytes(key)
    return key


def create_token(username: str, role: str, ttl: int = _TOKEN_TTL_SECONDS) -> str:
    payload = json.dumps({"u": username, "r": role, "exp": int(time.time()) + ttl}).encode()
    sig = hmac.new(_token_secret(), payload, hashlib.sha256).hexdigest()
    return base64.urlsafe_b64encode(payload).decode() + "." + sig


def verify_token(token: str) -> dict | None:
    """校验 token，成功返回 {"username","role"}，失败返回 None。"""
    try:
        payload_b64, sig = token.split(".")
        payload = base64.urlsafe_b64decode(payload_b64.encode())
        expected = hmac.new(_token_secret(), payload, hashlib.sha256).hexdigest()
        if not hmac.compare_digest(sig, expected):
            return None
        data = json.loads(payload)
        if data.get("exp", 0) < time.time():
            return None
        return {"username": data["u"], "role": data["r"]}
    except Exception:
        return None


# ---------- 用户 CRUD ----------
def authenticate(username: str, password: str) -> dict:
    with _DB_LOCK:
        conn = _connect()
        try:
            row = conn.execute("SELECT * FROM users WHERE username=?", (username,)).fetchone()
        finally:
            conn.close()
    if row is None or not _verify_password(password, row["password_hash"], row["salt"]):
        raise InvalidCredentialsError("用户名或密码错误")
    return {"username": row["username"], "role": row["role"]}


def create_user(username: str, password: str, role: str = "employee") -> dict:
    if not _USERNAME_RE.fullmatch(username):
        raise AppError("用户名只能包含字母、数字、点、下划线、中划线（2-32 字符）")
    if len(password) < 6:
        raise AppError("密码长度至少 6 位")
    if role not in {"admin", "employee"}:
        raise AppError("角色必须是 admin 或 employee")
    digest, salt = _hash_password(password)
    with _DB_LOCK:
        conn = _connect()
        try:
            try:
                conn.execute(
                    "INSERT INTO users (username, password_hash, salt, role, created_at) VALUES (?,?,?,?,?)",
                    (username, digest, salt, role, _now()),
                )
                conn.commit()
            except sqlite3.IntegrityError as exc:
                raise UserExistsError(f"用户已存在: {username}") from exc
        finally:
            conn.close()
    return {"username": username, "role": role}


def list_users() -> list[dict]:
    with _DB_LOCK:
        conn = _connect()
        try:
            rows = conn.execute(
                "SELECT username, role, created_at FROM users ORDER BY created_at"
            ).fetchall()
        finally:
            conn.close()
    return [{"username": r["username"], "role": r["role"], "created_at": r["created_at"]} for r in rows]


def delete_user(username: str) -> dict:
    """删除用户（仅限普通员工；管理员账号不可删除）。

    名下知识库的元数据一并清除；向量数据由调用方（路由层）级联删除。
    """
    if username == get_settings().admin_username:
        raise AppError("不能删除管理员账号")
    with _DB_LOCK:
        conn = _connect()
        try:
            row = conn.execute("SELECT username, role FROM users WHERE username=?", (username,)).fetchone()
            if row is None:
                raise UserNotFoundError("用户不存在")
            role = row["role"]
            if role == "admin":
                raise AppError("不能删除管理员账号")
            # 名下知识库元数据级联清除（员工库随账号删除）
            conn.execute("DELETE FROM kb_meta WHERE owner=?", (username,))
            conn.execute("DELETE FROM users WHERE username=?", (username,))
            conn.commit()
        finally:
            conn.close()
    return {"username": username, "role": role}


def change_password(username: str, old_password: str, new_password: str) -> None:
    if len(new_password) < 6:
        raise AppError("新密码长度至少 6 位")
    with _DB_LOCK:
        conn = _connect()
        try:
            row = conn.execute("SELECT * FROM users WHERE username=?", (username,)).fetchone()
            if row is None:
                raise UserNotFoundError("用户不存在")
            if not _verify_password(old_password, row["password_hash"], row["salt"]):
                raise InvalidCredentialsError("原密码错误")
            digest, salt = _hash_password(new_password)
            conn.execute(
                "UPDATE users SET password_hash=?, salt=? WHERE username=?",
                (digest, salt, username),
            )
            conn.commit()
        finally:
            conn.close()


# ---------- 知识库元数据（所有权） ----------
def kb_set_meta(kb_id: str, name: str | None, owner: str) -> None:
    with _DB_LOCK:
        conn = _connect()
        try:
            conn.execute(
                "INSERT OR REPLACE INTO kb_meta (kb_id, name, owner, created_at) VALUES (?,?,?,?)",
                (kb_id, name, owner, _now()),
            )
            conn.commit()
        finally:
            conn.close()


def kb_get_meta(kb_id: str) -> dict | None:
    with _DB_LOCK:
        conn = _connect()
        try:
            row = conn.execute("SELECT * FROM kb_meta WHERE kb_id=?", (kb_id,)).fetchone()
        finally:
            conn.close()
    return dict(row) if row else None


def kb_list_meta() -> list[dict]:
    with _DB_LOCK:
        conn = _connect()
        try:
            rows = conn.execute("SELECT * FROM kb_meta ORDER BY created_at").fetchall()
        finally:
            conn.close()
    return [dict(r) for r in rows]


def kb_delete_meta(kb_id: str) -> None:
    with _DB_LOCK:
        conn = _connect()
        try:
            conn.execute("DELETE FROM kb_meta WHERE kb_id=?", (kb_id,))
            conn.commit()
        finally:
            conn.close()
