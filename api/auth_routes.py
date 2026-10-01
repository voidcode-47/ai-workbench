"""认证与用户管理路由：登录、当前用户、改密、用户管理（管理员）。"""
from __future__ import annotations

from fastapi import APIRouter, Depends

from api.schemas import (
    ChangePasswordRequest,
    LoginRequest,
    LoginResponse,
    UserCreateRequest,
    UserInfo,
)
from app.auth import (
    authenticate,
    change_password,
    create_token,
    create_user,
    delete_user,
    kb_delete_meta,
    kb_get_meta,
    list_users,
)
from app.dependencies import get_rag_service, require_admin, require_user

router = APIRouter(tags=["Auth"])


@router.post("/auth/login", response_model=LoginResponse)
def login(payload: LoginRequest):
    user = authenticate(payload.username, payload.password)
    token = create_token(user["username"], user["role"])
    return LoginResponse(token=token, username=user["username"], role=user["role"])


@router.get("/auth/me", response_model=UserInfo)
def me(user: dict = Depends(require_user)):
    return UserInfo(username=user["username"], role=user["role"])


@router.post("/auth/change-password")
def change_my_password(payload: ChangePasswordRequest, user: dict = Depends(require_user)):
    change_password(user["username"], payload.old_password, payload.new_password)
    return {"ok": True}


@router.get("/users", response_model=list[UserInfo])
def list_all_users(_: dict = Depends(require_admin)):
    return [UserInfo(**u) for u in list_users()]


@router.post("/users", response_model=UserInfo)
def add_user(payload: UserCreateRequest, _: dict = Depends(require_admin)):
    user = create_user(payload.username, payload.password, payload.role)
    return UserInfo(**user)


@router.delete("/users/{username}", response_model=UserInfo)
def remove_user(username: str, _: dict = Depends(require_admin)):
    # 级联删除该员工名下的全部知识库（向量数据 + 元数据），随账号一并清除
    service = get_rag_service()
    for kb in service.list_knowledge_bases():
        kb_id = kb["kb_id"]
        meta = kb_get_meta(kb_id)
        owner = (meta or {}).get("owner") or "admin"
        if owner == username:
            service.delete_knowledge_base(kb_id)
            kb_delete_meta(kb_id)
    user = delete_user(username)
    return UserInfo(**user)
