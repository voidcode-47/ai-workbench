"""app.auth 单元测试：密码哈希、HMAC token 往返/过期/篡改、用户与知识库元数据 CRUD。

全部使用 tmp_path 下的临时 SQLite 与密钥文件，不触碰真实 data/ 目录。
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from app import auth
from app.exceptions import AppError
from app.auth import (
    InvalidCredentialsError,
    UserExistsError,
    UserNotFoundError,
    _hash_password,
    _verify_password,
    authenticate,
    change_password,
    create_token,
    create_user,
    delete_user,
    init_auth_db,
    kb_delete_meta,
    kb_get_meta,
    kb_list_meta,
    kb_set_meta,
    list_users,
    verify_token,
)


@pytest.fixture
def auth_env(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    fake_settings = SimpleNamespace(
        data_dir=str(data_dir),
        admin_username="admin",
        admin_password="admin123",
    )
    monkeypatch.setattr(auth, "get_settings", lambda: fake_settings)
    return data_dir


class TestPasswordHashing:
    def test_hash_is_salted_and_verifiable(self):
        digest, salt = _hash_password("s3cret!")
        assert digest != "s3cret!"
        assert _verify_password("s3cret!", digest, salt) is True
        assert _verify_password("wrong", digest, salt) is False

    def test_same_password_produces_different_hashes(self):
        digest1, salt1 = _hash_password("same-password")
        digest2, salt2 = _hash_password("same-password")
        assert salt1 != salt2
        assert digest1 != digest2


class TestToken:
    def test_create_and_verify_roundtrip(self, auth_env):
        token = create_token("alice", "employee", token_version=3)
        payload = verify_token(token)
        assert payload == {"username": "alice", "role": "employee", "token_version": 3}

    def test_token_version_defaults_to_zero(self, auth_env):
        assert verify_token(create_token("alice", "employee"))["token_version"] == 0

    def test_expired_token_is_rejected(self, auth_env):
        token = create_token("alice", "employee", ttl=-10)
        assert verify_token(token) is None

    def test_tampered_signature_is_rejected(self, auth_env):
        token = create_token("alice", "employee")
        payload_b64, sig = token.split(".")
        forged = payload_b64 + "." + ("0" * len(sig))
        assert verify_token(forged) is None

    def test_malformed_token_returns_none(self, auth_env):
        assert verify_token("not-a-token") is None
        assert verify_token("a.b.c") is None
        assert verify_token("") is None


class TestUserCRUD:
    def test_init_creates_admin_and_authenticate_succeeds(self, auth_env):
        init_auth_db()
        user = authenticate("admin", "admin123")
        assert user["username"] == "admin"
        assert user["role"] == "admin"
        assert user["token_version"] == 0

    def test_change_password_bumps_token_version(self, auth_env):
        """改密必须递增 token_version，否则旧的（含被盗的）token 仍然有效。"""
        init_auth_db()
        create_user("alice", "old-pass-1")
        before = auth.get_user("alice")["token_version"]

        change_password("alice", "old-pass-1", "new-pass-2")

        assert auth.get_user("alice")["token_version"] == before + 1

    def test_get_user_returns_none_for_unknown(self, auth_env):
        init_auth_db()
        assert auth.get_user("ghost") is None
        assert auth.get_user("admin")["role"] == "admin"

    def test_authenticate_with_wrong_password_raises(self, auth_env):
        init_auth_db()
        with pytest.raises(InvalidCredentialsError):
            authenticate("admin", "wrong-password")

    def test_authenticate_unknown_user_raises(self, auth_env):
        init_auth_db()
        with pytest.raises(InvalidCredentialsError):
            authenticate("ghost", "whatever")

    def test_create_user_then_authenticate_and_list(self, auth_env):
        init_auth_db()
        create_user("alice", "secret123", "employee")
        assert authenticate("alice", "secret123")["role"] == "employee"
        users = list_users()
        assert {u["username"] for u in users} == {"admin", "alice"}

    def test_duplicate_user_raises(self, auth_env):
        init_auth_db()
        create_user("alice", "secret123")
        with pytest.raises(UserExistsError):
            create_user("alice", "another456")

    def test_invalid_username_and_short_password_are_rejected(self, auth_env):
        init_auth_db()
        with pytest.raises(AppError, match="用户名"):
            create_user("a", "secret123")  # 少于 2 字符
        with pytest.raises(AppError, match="用户名"):
            create_user("bad name", "secret123")  # 含空格
        with pytest.raises(AppError, match="密码"):
            create_user("bob", "12345")

    def test_delete_admin_is_forbidden(self, auth_env):
        init_auth_db()
        with pytest.raises(AppError, match="管理员"):
            delete_user("admin")

    def test_delete_unknown_user_raises(self, auth_env):
        init_auth_db()
        with pytest.raises(UserNotFoundError):
            delete_user("ghost")

    def test_delete_employee_removes_user(self, auth_env):
        init_auth_db()
        create_user("alice", "secret123")
        result = delete_user("alice")
        assert result["username"] == "alice"
        with pytest.raises(InvalidCredentialsError):
            authenticate("alice", "secret123")

    def test_change_password_flow(self, auth_env):
        init_auth_db()
        create_user("alice", "old-pass-1")
        change_password("alice", "old-pass-1", "new-pass-2")
        assert authenticate("alice", "new-pass-2")["username"] == "alice"
        with pytest.raises(InvalidCredentialsError):
            authenticate("alice", "old-pass-1")

    def test_change_password_rejects_wrong_old_password(self, auth_env):
        init_auth_db()
        create_user("alice", "old-pass-1")
        with pytest.raises(InvalidCredentialsError, match="原密码错误"):
            change_password("alice", "wrong-old", "new-pass-2")


class TestKbMeta:
    def test_set_get_list_delete_roundtrip(self, auth_env):
        init_auth_db()
        kb_set_meta("kb1", "公共库", "admin")
        kb_set_meta("kb2", "私有库", "alice")
        assert kb_get_meta("kb1")["owner"] == "admin"
        assert {m["kb_id"] for m in kb_list_meta()} == {"kb1", "kb2"}
        kb_delete_meta("kb1")
        assert kb_get_meta("kb1") is None
        assert {m["kb_id"] for m in kb_list_meta()} == {"kb2"}

    def test_set_meta_replaces_existing_row(self, auth_env):
        init_auth_db()
        kb_set_meta("kb1", "旧名", "admin")
        kb_set_meta("kb1", "新名", "alice")
        meta = kb_get_meta("kb1")
        assert meta["name"] == "新名"
        assert meta["owner"] == "alice"
        assert len(kb_list_meta()) == 1
