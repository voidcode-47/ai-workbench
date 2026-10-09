"""模型接入路由：提供商目录、模型列表、配置读写与连接测试。

权限：所有登录用户可读取「当前生效的模型」；修改配置、拉取模型列表与连接测试
仅限管理员 —— 这些操作会带出已保存的 API Key，或改写全局模型配置、影响所有用户。
"""
from __future__ import annotations

from fastapi import APIRouter, Depends

from api.schemas import (
    EffectiveModelInfo,
    ModelConfigRequest,
    ModelConfigResponse,
    ModelListResponse,
    ModelProbeRequest,
    ModelProviderInfo,
    ModelTestResponse,
    SavedProfileInfo,
)
from app import model_config
from app.dependencies import require_admin, require_user
from app.exceptions import AppError

router = APIRouter(dependencies=[Depends(require_user)])


def _snapshot() -> ModelConfigResponse:
    """当前配置快照：目录 + 实际生效值 + 已保存参数（均不含明文 Key）。"""
    effective = model_config.resolve_config()
    return ModelConfigResponse(
        catalog=[ModelProviderInfo(**item) for item in model_config.catalog_public()],
        effective=EffectiveModelInfo(**effective.public()),
        saved={
            provider: SavedProfileInfo(**profile)
            for provider, profile in model_config.saved_profiles_public().items()
        },
    )


def _validated(provider: str) -> None:
    if provider not in model_config.PROVIDER_CATALOG:
        raise AppError(f"未知提供商: {provider}，可选: {sorted(model_config.PROVIDER_CATALOG)}")


@router.get("/config", response_model=ModelConfigResponse, tags=["Models"])
def get_model_config():
    return _snapshot()


@router.post("/config", response_model=ModelConfigResponse, tags=["Models"])
def update_model_config(payload: ModelConfigRequest, _: dict = Depends(require_admin)):
    """保存接入参数并立即生效（无需重启）。"""
    _validated(payload.provider)
    # exclude_none：未提交的字段保持原值（api_key 传空串同样视为不修改）
    model_config.save_profile(payload.provider, payload.profile.model_dump(exclude_none=True))
    model_config.apply_runtime_config()
    return _snapshot()


@router.delete("/config", response_model=ModelConfigResponse, tags=["Models"])
def reset_model_config(_: dict = Depends(require_admin)):
    """清除界面配置，回到 .env 默认值。"""
    model_config.clear_config()
    model_config.apply_runtime_config()
    return _snapshot()


@router.post("/list", response_model=ModelListResponse, tags=["Models"])
def list_models(payload: ModelProbeRequest, _: dict = Depends(require_admin)):
    """从服务端 /v1/models 拉取可用模型列表（失败回退到目录建议列表）。"""
    _validated(payload.provider)
    result = model_config.list_remote_models(
        payload.provider, payload.profile.model_dump(exclude_none=True)
    )
    return ModelListResponse(**result)


@router.post("/test", response_model=ModelTestResponse, tags=["Models"])
def test_model(payload: ModelProbeRequest, _: dict = Depends(require_admin)):
    """连接测试：端点可达性 + 模型可用性（发一次极短对话）。"""
    _validated(payload.provider)
    result = model_config.check_connection(
        payload.provider, payload.profile.model_dump(exclude_none=True)
    )
    return ModelTestResponse(**result)
