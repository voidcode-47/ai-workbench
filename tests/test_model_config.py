"""app.model_config 单元测试：Base URL 规范化、配置解析优先级、热切换与 Key 保护。

全部使用 tmp_path 下的临时配置目录，并替换 .env 基线，不触碰真实 data/ 与 .env。
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from app import model_config as mc


def _make_seed(data_dir: str, **overrides) -> SimpleNamespace:
    values = dict(
        data_dir=data_dir,
        openai_api_key="sk-env-key",
        openai_base_url="https://api.openai.com/v1",
        llm_model="env-model",
        embedding_model="env-embed",
        embedding_base_url="",
        vision_model="env-vision",
    )
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.fixture
def env(tmp_path, monkeypatch):
    """隔离环境：临时 data 目录 + 可控 .env 基线 + 无副作用的缓存清理。

    runtime 与 seed 刻意用两个独立对象：生产环境中 get_settings() 返回全局单例
    （会被热切换改写），而 .env 基线是另建的 Settings() 实例。
    两者若共用同一对象，切换提供商就会把上一个提供商的配置当成「默认值」。
    """
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    runtime = _make_seed(str(data_dir))
    baseline = _make_seed(str(data_dir))
    monkeypatch.setattr(mc, "get_settings", lambda: runtime)
    monkeypatch.setattr(mc, "_seed", baseline)
    monkeypatch.setattr(mc, "_invalidate_runtime_caches", lambda: None)
    return SimpleNamespace(runtime=runtime, seed=baseline, data_dir=data_dir)


def _config_file(env):
    return env.data_dir / "model_config.json"


class TestNormalizeBaseUrl:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("https://x.com", "https://x.com/v1"),
            ("https://x.com/", "https://x.com/v1"),
            ("https://api.openai.com/v1", "https://api.openai.com/v1"),
            ("https://x.com/v1/chat/completions", "https://x.com/v1"),
            ("https://x.com/v1/completions", "https://x.com/v1"),
            ("http://127.0.0.1:1234", "http://127.0.0.1:1234/v1"),
            ("http://localhost:11434/v1", "http://localhost:11434/v1"),
            ("https://open.bigmodel.cn/api/paas/v4", "https://open.bigmodel.cn/api/paas/v4"),
            ("https://dashscope.aliyuncs.com/compatible-mode/v1",
             "https://dashscope.aliyuncs.com/compatible-mode/v1"),
            # 主机名以 v 开头不应被误判为已带版本段
            ("https://vserver.com", "https://vserver.com/v1"),
            ("", ""),
            ("   ", ""),
        ],
    )
    def test_normalizes(self, raw, expected):
        assert mc.normalize_base_url(raw) == expected


class TestResolveFallback:
    """未配置时行为必须与引入本模块之前一致（完全沿用 .env）。"""

    def test_without_config_file_uses_env(self, env):
        effective = mc.resolve_config()
        assert effective.provider == ""
        assert effective.base_url == "https://api.openai.com/v1"
        assert effective.api_key == "sk-env-key"
        assert effective.llm_model == "env-model"
        assert effective.embedding_model == "env-embed"
        assert effective.vision_model == "env-vision"

    def test_corrupted_config_file_falls_back(self, env):
        _config_file(env).write_text("{ not json", encoding="utf-8")
        assert mc.resolve_config().provider == ""

    def test_unknown_provider_in_file_falls_back(self, env):
        _config_file(env).write_text(
            '{"provider": "nonexistent", "providers": {}}', encoding="utf-8"
        )
        assert mc.resolve_config().provider == ""


class TestSaveAndResolve:
    def test_saved_profile_wins_and_unset_fields_use_catalog(self, env):
        mc.save_profile("lmstudio", {"llm_model": "qwen2.5-7b"})
        effective = mc.resolve_config()
        assert effective.provider == "lmstudio"
        assert effective.llm_model == "qwen2.5-7b"
        # 未提交的字段回落到提供商默认值
        assert effective.base_url == "http://localhost:1234/v1"

    def test_empty_catalog_embedding_model_falls_back_to_env(self, env):
        # LM Studio 目录里没写死 embedding 模型（需从已加载模型里挑），
        # 此时回落到 .env 的 EMBEDDING_MODEL，而不是留空导致调用必然失败
        mc.save_profile("lmstudio", {"llm_model": "qwen2.5-7b"})
        assert mc.resolve_config().embedding_model == "env-embed"

    def test_provider_catalog_embedding_model_wins_over_env(self, env):
        mc.save_profile("qwen", {"llm_model": "qwen-plus"})
        assert mc.resolve_config().embedding_model == "text-embedding-v3"

    def test_base_url_is_normalized_on_save(self, env):
        mc.save_profile("custom", {"base_url": "http://127.0.0.1:8000", "llm_model": "m"})
        assert mc.load_config()["providers"]["custom"]["base_url"] == "http://127.0.0.1:8000/v1"

    def test_blank_api_key_does_not_clear_saved_key(self, env):
        mc.save_profile("deepseek", {"api_key": "sk-secret-1234", "llm_model": "deepseek-chat"})
        mc.save_profile("deepseek", {"api_key": "", "llm_model": "deepseek-reasoner"})

        saved = mc.load_config()["providers"]["deepseek"]
        assert saved["api_key"] == "sk-secret-1234"
        assert saved["llm_model"] == "deepseek-reasoner"  # 其他字段照常更新

    def test_providers_keep_independent_profiles(self, env):
        mc.save_profile("lmstudio", {"llm_model": "local-a", "base_url": "http://127.0.0.1:1234"})
        mc.save_profile("deepseek", {"llm_model": "deepseek-chat", "api_key": "sk-d"})

        # 切回 lmstudio 应恢复它自己的参数，而不是继承 deepseek
        mc.save_profile("lmstudio", {})
        effective = mc.resolve_config()
        assert effective.provider == "lmstudio"
        assert effective.llm_model == "local-a"
        assert effective.api_key == ""  # 本地服务无 Key，且不回落到 deepseek 的 Key

    def test_empty_string_clears_field_to_default(self, env):
        mc.save_profile("ollama", {"llm_model": "custom-model"})
        assert mc.resolve_config().llm_model == "custom-model"
        mc.save_profile("ollama", {"llm_model": ""})
        # 空串表示「恢复默认」，回落到目录里的默认模型
        assert mc.resolve_config().llm_model == mc.PROVIDER_CATALOG["ollama"]["default_model"]

    def test_openai_provider_respects_env_base_url_override(self, env):
        # 选了 openai 但没改地址时，应尊重 .env 里的 OPENAI_BASE_URL 覆盖
        env.seed.openai_base_url = "https://gateway.internal/v1"
        mc.save_profile("openai", {"llm_model": "gpt-4o-mini"})
        assert mc.resolve_config().base_url == "https://gateway.internal/v1"

    def test_unknown_provider_rejected(self, env):
        with pytest.raises(ValueError):
            mc.save_profile("nonexistent", {"llm_model": "x"})

    def test_clear_config_returns_to_env(self, env):
        mc.save_profile("deepseek", {"api_key": "sk-x", "llm_model": "deepseek-chat"})
        assert mc.resolve_config().provider == "deepseek"
        mc.clear_config()
        assert mc.resolve_config().provider == ""


class TestSecrets:
    def test_public_view_never_exposes_plaintext_key(self, env):
        mc.save_profile("deepseek", {"api_key": "sk-super-secret-9876", "llm_model": "deepseek-chat"})
        effective = mc.resolve_config()

        assert effective.api_key == "sk-super-secret-9876"  # 内部仍用明文调用
        public = effective.public()
        assert "sk-super-secret-9876" not in str(public)
        assert public["has_api_key"] is True
        assert public["api_key_hint"] == "****9876"

    def test_short_key_hint_is_fully_masked(self, env):
        mc.save_profile("deepseek", {"api_key": "abc", "llm_model": "m"})
        assert mc.resolve_config().public()["api_key_hint"] == "****"

    def test_saved_profiles_public_hides_key(self, env):
        mc.save_profile("deepseek", {"api_key": "sk-hidden-1111", "llm_model": "m"})
        saved = mc.saved_profiles_public()["deepseek"]
        assert saved["has_api_key"] is True
        assert "sk-hidden-1111" not in str(saved)


class TestApplyRuntimeConfig:
    def test_writes_through_to_settings(self, env):
        mc.save_profile("lmstudio", {
            "base_url": "http://127.0.0.1:1234",
            "llm_model": "qwen2.5-7b",
            "embedding_model": "bge-m3",
            "vision_model": "qwen2-vl",
        })
        effective = mc.apply_runtime_config()

        assert effective.provider == "lmstudio"
        assert env.runtime.openai_base_url == "http://127.0.0.1:1234/v1"
        assert env.runtime.llm_model == "qwen2.5-7b"
        assert env.runtime.embedding_model == "bge-m3"
        assert env.runtime.vision_model == "qwen2-vl"

    def test_hot_swap_does_not_pollute_env_baseline(self, env):
        """热切换只改运行期 Settings，.env 基线必须保持干净。

        否则「恢复默认」会把当前提供商的配置当成默认值，再也回不去。
        """
        baseline = (
            env.seed.llm_model, env.seed.openai_base_url,
            env.seed.embedding_model, env.seed.vision_model,
        )
        mc.save_profile("deepseek", {"llm_model": "deepseek-chat", "api_key": "sk-x"})
        mc.apply_runtime_config()
        assert env.runtime.llm_model == "deepseek-chat"
        assert (
            env.seed.llm_model, env.seed.openai_base_url,
            env.seed.embedding_model, env.seed.vision_model,
        ) == baseline

    def test_reset_restores_env_values(self, env):
        mc.save_profile("deepseek", {"llm_model": "deepseek-chat", "api_key": "sk-x"})
        mc.apply_runtime_config()
        assert env.runtime.llm_model == "deepseek-chat"

        mc.clear_config()
        mc.apply_runtime_config()
        assert env.runtime.llm_model == env.seed.llm_model == "env-model"
        assert env.runtime.openai_base_url == env.seed.openai_base_url


class TestCatalog:
    def test_every_provider_has_required_fields(self):
        for key, meta in mc.PROVIDER_CATALOG.items():
            assert meta["label"], key
            assert isinstance(meta["models"], list), key
            assert "local" in meta, key

    def test_catalog_public_flags(self, env):
        catalog = {item["key"]: item for item in mc.catalog_public()}
        assert catalog["lmstudio"]["local"] is True
        assert catalog["ollama"]["local"] is True
        assert catalog["deepseek"]["local"] is False
        # .env 里只提供了 OpenAI 的 Key
        assert catalog["openai"]["has_env_key"] is True
        assert catalog["deepseek"]["has_env_key"] is False
