"""模型接入配置：多提供商 / 多模型的运行时选择与热切换。

优先级：
    界面保存的配置（data/model_config.json） > .env / 环境变量默认值

配置只决定「连哪个端点、用哪个模型」，不改变应用行为。保存后立即生效、无需重启：
apply_runtime_config() 把解析结果写回全局 Settings 单例（get_settings() 全局共享同一
实例），并清掉 OpenAI 客户端缓存、Embedding 单例与 Agent 状态图缓存。

安全约定：API Key 只落盘在 data/model_config.json（已 gitignore），
任何接口返回给前端的配置都不包含明文 Key，只给出 has_api_key 与末四位。
"""
from __future__ import annotations

import json
import logging
import re
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from app.config import Settings, get_settings
from app.exceptions import mask_secrets

logger = logging.getLogger(__name__)

# ---------- 提供商目录 ----------
# 全部按 OpenAI 兼容协议接入（应用统一使用 openai SDK / ChatOpenAI）。
# env_key 指向 Settings 中的字段名：留空表示该提供商没有对应的 .env 变量，
# API Key 需在界面填写；本地服务两种方式都不需要。
PROVIDER_CATALOG: dict[str, dict[str, Any]] = {
    "openai": {
        "label": "OpenAI",
        "base_url": "https://api.openai.com/v1",
        "env_key": "openai_api_key",
        "local": False,
        "supports_vision": True,
        "models": ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "gpt-4.1"],
        "default_model": "gpt-4o-mini",
        "embedding_model": "text-embedding-3-small",
        "vision_model": "gpt-4o",
    },
    "deepseek": {
        "label": "DeepSeek",
        "base_url": "https://api.deepseek.com/v1",
        "env_key": None,
        "local": False,
        "supports_vision": False,
        "models": ["deepseek-chat", "deepseek-reasoner"],
        "default_model": "deepseek-chat",
        "embedding_model": "",  # DeepSeek 无 embedding 服务，需另配端点
        "vision_model": "",
    },
    "zhipu": {
        "label": "智谱 GLM",
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "env_key": None,
        "local": False,
        "supports_vision": True,
        "models": ["glm-4-flash", "glm-4-plus", "glm-4v-flash"],
        "default_model": "glm-4-flash",
        "embedding_model": "embedding-3",
        "vision_model": "glm-4v-flash",
    },
    "qwen": {
        "label": "通义千问（DashScope）",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "env_key": None,
        "local": False,
        "supports_vision": True,
        "models": ["qwen-plus", "qwen-turbo", "qwen-max", "qwen-vl-plus"],
        "default_model": "qwen-plus",
        "embedding_model": "text-embedding-v3",
        "vision_model": "qwen-vl-plus",
    },
    "moonshot": {
        "label": "月之暗面 Kimi",
        "base_url": "https://api.moonshot.cn/v1",
        "env_key": None,
        "local": False,
        "supports_vision": True,
        "models": ["moonshot-v1-8k", "moonshot-v1-32k", "moonshot-v1-128k"],
        "default_model": "moonshot-v1-8k",
        "embedding_model": "",
        "vision_model": "moonshot-v1-8k-vision-preview",
    },
    "ollama": {
        "label": "Ollama（本地）",
        "base_url": "http://localhost:11434/v1",
        "env_key": None,
        "local": True,
        "supports_vision": True,
        "models": ["qwen2.5:7b", "llama3.1:8b", "qwen2.5vl:7b"],
        "default_model": "qwen2.5:7b",
        "embedding_model": "nomic-embed-text",
        "vision_model": "qwen2.5vl:7b",
    },
    "lmstudio": {
        "label": "LM Studio（本地）",
        "base_url": "http://localhost:1234/v1",
        "env_key": None,
        "local": True,
        "supports_vision": True,
        # 模型由 LM Studio 动态加载，留空表示运行时通过 /v1/models 自动发现
        "models": [],
        "default_model": "",
        "embedding_model": "",
        "vision_model": "",
    },
    "custom": {
        "label": "自定义 OpenAI 兼容端点",
        "base_url": "",
        "env_key": None,
        "local": True,  # 视为免 Key：允许 vLLM / One-API / 自建网关等
        "supports_vision": True,
        "models": [],
        "default_model": "",
        "embedding_model": "",
        "vision_model": "",
    },
}

# 只在这些主机上必须提供 API Key；其余端点（本地/自建网关）允许免 Key
_PUBLIC_HOSTS = ("api.openai.com", "api.deepseek.com", "open.bigmodel.cn",
                 "dashscope.aliyuncs.com", "api.moonshot.cn")

_CONFIG_FILENAME = "model_config.json"
_CONFIG_LOCK = threading.Lock()

# .env 基线：只在首次使用时读取一次。运行期热切换会覆盖 Settings 单例上的字段，
# 若每次都从单例取「默认值」，第二次解析就会把上一个提供商的配置当成默认值。
_seed: Settings | None = None


def _seed_settings() -> Settings:
    global _seed
    if _seed is None:
        _seed = Settings()
    return _seed


@dataclass
class EffectiveConfig:
    """解析后的实际生效配置。"""

    provider: str = ""  # 空串 = 未通过界面配置，完全沿用 .env
    base_url: str = ""
    api_key: str = ""
    llm_model: str = ""
    embedding_model: str = ""
    embedding_base_url: str = ""
    vision_model: str = ""

    @property
    def has_api_key(self) -> bool:
        return bool(self.api_key)

    @property
    def api_key_hint(self) -> str:
        """Key 末四位（前端展示用，不泄露完整 Key）。"""
        if not self.api_key:
            return ""
        return f"****{self.api_key[-4:]}" if len(self.api_key) > 4 else "****"

    def public(self) -> dict:
        """对外（前端）可见的字段，不含明文 Key。"""
        return {
            "provider": self.provider,
            "provider_label": PROVIDER_CATALOG.get(self.provider, {}).get("label", "未配置"),
            "base_url": self.base_url,
            "llm_model": self.llm_model,
            "embedding_model": self.embedding_model,
            "embedding_base_url": self.embedding_base_url,
            "vision_model": self.vision_model,
            "has_api_key": self.has_api_key,
            "api_key_hint": self.api_key_hint,
        }


# ---------- Base URL 规范化 ----------
_VERSION_SEGMENT_RE = re.compile(r"v\d+")


def normalize_base_url(url: str) -> str:
    """补齐 OpenAI 兼容端点：去掉多余路径，缺少 /v1 时补上。

    https://x.com                        -> https://x.com/v1
    https://x.com/v1/chat/completions    -> https://x.com/v1
    https://open.bigmodel.cn/api/paas/v4 -> 保持不变（非 /v1 结尾的合法端点）
    """
    base = (url or "").strip().rstrip("/")
    if not base:
        return ""
    for suffix in ("/chat/completions", "/completions", "/chat"):
        if base.endswith(suffix):
            base = base[: -len(suffix)].rstrip("/")
            break
    # 末段已是版本号（/v1、/v4 …）的端点不再追加
    if not _VERSION_SEGMENT_RE.fullmatch(base.rsplit("/", 1)[-1]):
        base += "/v1"
    return base


# ---------- 配置文件读写 ----------
def _config_path() -> Path:
    return Path(get_settings().data_dir) / _CONFIG_FILENAME


def load_config() -> dict:
    """读取模型接入配置（文件缺失或损坏时返回空配置，等价于「完全沿用 .env」）。"""
    path = _config_path()
    if not path.exists():
        return {"provider": "", "providers": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("模型配置读取失败，回退 .env 默认值: %s", exc)
        return {"provider": "", "providers": {}}
    if not isinstance(data, dict):
        return {"provider": "", "providers": {}}
    providers = data.get("providers")
    return {
        "provider": data.get("provider") or "",
        "providers": providers if isinstance(providers, dict) else {},
    }


def _atomic_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def save_profile(provider: str, profile: dict) -> dict:
    """保存某提供商的接入参数并置为当前生效提供商（None 值表示不修改该字段）。

    api_key 传空串 = 不修改已保存的 Key（前端不回显 Key，留空保存不应清空它）。
    """
    if provider not in PROVIDER_CATALOG:
        raise ValueError(f"未知提供商: {provider}，可选: {sorted(PROVIDER_CATALOG)}")

    with _CONFIG_LOCK:
        cfg = load_config()
        merged = dict(cfg["providers"].get(provider) or {})
        for key, value in profile.items():
            if value is None:
                continue
            if key == "api_key":
                if not str(value).strip():
                    continue  # 留空 = 不修改
                merged[key] = str(value).strip()
                continue
            if key == "base_url":
                merged[key] = normalize_base_url(str(value))
                continue
            merged[key] = str(value).strip()

        cfg["providers"][provider] = merged
        cfg["provider"] = provider
        cfg["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        _atomic_write(_config_path(), cfg)
    return cfg


def clear_config() -> None:
    """清除界面配置，回到 .env 默认值。"""
    with _CONFIG_LOCK:
        path = _config_path()
        if path.exists():
            path.unlink()


def saved_profiles_public() -> dict[str, dict]:
    """已保存的各提供商参数（剔除以明文保存的 API Key，只给出是否已配置）。"""
    cfg = load_config()
    return {
        provider: {
            "base_url": profile.get("base_url") or "",
            "has_api_key": bool(profile.get("api_key")),
            "llm_model": profile.get("llm_model") or "",
            "embedding_model": profile.get("embedding_model") or "",
            "embedding_base_url": profile.get("embedding_base_url") or "",
            "vision_model": profile.get("vision_model") or "",
        }
        for provider, profile in cfg["providers"].items()
        if isinstance(profile, dict)
    }


# ---------- 配置解析 ----------
def resolve_config() -> EffectiveConfig:
    """合并「界面配置 + 提供商默认值 + .env 基线」，得到实际生效的配置。"""
    seed = _seed_settings()
    cfg = load_config()
    provider = cfg.get("provider") or ""

    if not provider or provider not in PROVIDER_CATALOG:
        # 未通过界面配置：完全沿用 .env，行为与引入本模块之前一致
        return EffectiveConfig(
            provider="",
            base_url=seed.openai_base_url,
            api_key=seed.openai_api_key,
            llm_model=seed.llm_model,
            embedding_model=seed.embedding_model,
            embedding_base_url=seed.embedding_base_url,
            vision_model=seed.vision_model,
        )

    catalog = PROVIDER_CATALOG[provider]
    profile = cfg["providers"].get(provider) or {}

    api_key = str(profile.get("api_key") or "")
    if not api_key:
        env_field = catalog.get("env_key")
        api_key = getattr(seed, env_field, "") if env_field else ""
    if not api_key and provider == "openai":
        api_key = seed.openai_api_key

    base_url = profile.get("base_url") or catalog.get("base_url") or seed.openai_base_url
    # 选择 openai 提供商但未改地址时，尊重 .env 里的 OPENAI_BASE_URL 覆盖
    if provider == "openai" and not profile.get("base_url"):
        base_url = seed.openai_base_url or base_url

    return EffectiveConfig(
        provider=provider,
        base_url=base_url,
        api_key=api_key,
        llm_model=profile.get("llm_model") or catalog.get("default_model") or seed.llm_model,
        embedding_model=(
            profile.get("embedding_model")
            or catalog.get("embedding_model")
            or seed.embedding_model
        ),
        embedding_base_url=profile.get("embedding_base_url") or "",
        vision_model=profile.get("vision_model") or catalog.get("vision_model") or seed.vision_model,
    )


# ---------- 运行时生效 ----------
def apply_runtime_config() -> EffectiveConfig:
    """把当前配置写回全局 Settings 单例并清理缓存 —— 换模型无需重启。"""
    settings = get_settings()
    effective = resolve_config()

    settings.openai_base_url = effective.base_url
    settings.openai_api_key = effective.api_key
    settings.llm_model = effective.llm_model
    settings.embedding_model = effective.embedding_model
    settings.embedding_base_url = effective.embedding_base_url
    settings.vision_model = effective.vision_model

    _invalidate_runtime_caches()

    if effective.provider:
        logger.info(
            "模型接入已生效 provider=%s base_url=%s model=%s embedding=%s key=%s",
            effective.provider, effective.base_url, effective.llm_model,
            effective.embedding_model or "(未配置)", effective.api_key_hint or "(无)",
        )
    return effective


def _invalidate_runtime_caches() -> None:
    """清理在构造时固化了模型/端点的缓存对象。"""
    # 延迟导入：app.dependencies 反向依赖 app.config，模块级导入会成环
    from app.dependencies import get_agent_executor, get_openai_client
    from rag.embeddings import reset_embedder

    # OpenAI 客户端把 base_url / api_key 固化在实例上
    get_openai_client.cache_clear()
    # Embedding 单例持有旧端点与模型（RAGService 持有的是转发代理，无需重建）
    reset_embedder()
    # Agent 状态图在构建时就把 ChatOpenAI 实例绑定了模型名
    try:
        get_agent_executor().reset_graph()
    except Exception as exc:  # 图尚未构建或依赖缺失时无需重建
        logger.debug("重置 Agent 状态图跳过: %s", exc)


# ---------- 探测与测试（不写入全局配置） ----------
def _build_probe_client(provider: str, profile: dict | None = None):
    """按给定参数构造一次性 OpenAI 兼容客户端，用于模型探测与连接测试。"""
    from openai import OpenAI

    profile = profile or {}
    catalog = PROVIDER_CATALOG.get(provider, {})
    cfg = load_config()
    saved = cfg["providers"].get(provider) or {}

    base_url = profile.get("base_url") or saved.get("base_url") or catalog.get("base_url") or ""
    base_url = normalize_base_url(base_url)
    api_key = str(profile.get("api_key") or saved.get("api_key") or "")
    if not api_key:
        env_field = catalog.get("env_key")
        api_key = getattr(_seed_settings(), env_field, "") if env_field else ""
    if not base_url:
        raise ValueError("未配置 Base URL")
    return OpenAI(api_key=api_key or "sk-no-key-required", base_url=base_url, timeout=15)


def is_public_host(base_url: str) -> bool:
    """端点是否指向公开托管服务（这类服务无 Key 必然调用失败）。

    供 RAG Embedding 的降级判断共用，避免两处主机名单不一致。
    """
    host = (base_url or "").split("//", 1)[-1].split("/", 1)[0].lower()
    return any(host.endswith(h) for h in _PUBLIC_HOSTS)


def _needs_api_key(provider: str, base_url: str) -> bool:
    """公开托管服务必须带 Key；本地/自建端点允许免 Key。"""
    return is_public_host(base_url)


def list_remote_models(provider: str, profile: dict | None = None) -> dict:
    """通过 /v1/models 拉取服务端模型列表（失败时回退到目录里的建议列表）。"""
    catalog = PROVIDER_CATALOG.get(provider, {})
    try:
        client = _build_probe_client(provider, profile)
        models = sorted(m.id for m in client.models.list().data)
        return {"models": models, "source": "remote", "error": None}
    except Exception as exc:
        logger.info("拉取 %s 模型列表失败: %s", provider, exc)
        return {
            "models": list(catalog.get("models") or []),
            "source": "catalog",
            "error": str(exc),
        }


def check_connection(provider: str, profile: dict | None = None, model: str | None = None) -> dict:
    """连接测试：先探端点可达性，再用目标模型发一次极短对话。

    只做 models.list 无法发现「模型名写错」这类问题，因此追加一次最小 chat 调用。
    """
    profile = profile or {}
    catalog = PROVIDER_CATALOG.get(provider, {})
    cfg = load_config()
    saved = cfg["providers"].get(provider) or {}
    target_model = (
        model
        or profile.get("llm_model")
        or saved.get("llm_model")
        or catalog.get("default_model")
        or ""
    )

    try:
        client = _build_probe_client(provider, profile)
    except Exception as exc:
        return {"ok": False, "stage": "init", "detail": f"客户端初始化失败: {exc}"}

    base_url = str(client.base_url)
    if _needs_api_key(provider, base_url) and client.api_key in ("", "sk-no-key-required"):
        return {"ok": False, "stage": "auth", "detail": "该端点为公开托管服务，请先填写 API Key"}

    # 1) 端点可达性
    try:
        available = sorted(m.id for m in client.models.list().data)
    except Exception as exc:
        return {"ok": False, "stage": "reachable", "detail": f"无法连接 {base_url}：{exc}"}

    if not available:
        return {
            "ok": True, "stage": "reachable", "detail": f"端点可达（{base_url}），但未返回任何模型",
            "models": [],
        }

    if not target_model:
        return {
            "ok": True, "stage": "reachable",
            "detail": f"端点可达，共 {len(available)} 个模型；未指定模型名，无法验证具体模型",
            "models": available,
        }

    if target_model not in available:
        return {
            "ok": False, "stage": "model",
            "detail": f"模型 {target_model} 不在服务端列表中（可用 {len(available)} 个，如 {available[:5]}）",
            "models": available,
        }

    # 2) 最小对话验证模型确实可用
    try:
        client.chat.completions.create(
            model=target_model,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=1,
            temperature=0,
        )
    except Exception as exc:
        return {
            "ok": False, "stage": "chat",
            # 该 detail 会返回给前端：上游异常可能回显提交的 Key，先脱敏
            "detail": f"模型 {target_model} 调用失败：{mask_secrets(str(exc))}",
            "models": available,
        }

    return {
        "ok": True, "stage": "chat",
        "detail": f"连接正常，模型 {target_model} 可用（服务端共 {len(available)} 个模型）",
        "models": available,
    }


def catalog_public() -> list[dict]:
    """提供商目录（供前端渲染下拉与表单默认值）。"""
    seed = _seed_settings()
    items = []
    for key, meta in PROVIDER_CATALOG.items():
        env_field = meta.get("env_key")
        has_env_key = bool(getattr(seed, env_field, "")) if env_field else False
        items.append(
            {
                "key": key,
                "label": meta["label"],
                "default_base_url": meta.get("base_url", ""),
                "default_model": meta.get("default_model", ""),
                "default_embedding_model": meta.get("embedding_model", ""),
                "default_vision_model": meta.get("vision_model", ""),
                "models": list(meta.get("models") or []),
                "local": bool(meta.get("local")),
                "supports_vision": bool(meta.get("supports_vision")),
                "requires_api_key": key in PROVIDER_CATALOG and not meta.get("local"),
                "has_env_key": has_env_key,
            }
        )
    return items
