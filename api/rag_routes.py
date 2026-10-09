"""RAG 路由：知识库管理、文档上传/删除/重建、检索增强问答（带权限控制）。

权限矩阵：
- admin 的系统库（owner=admin）：admin 可读写，普通员工只读（公共库）；
- 员工私有库（owner=员工本人）：仅创建者本人可见可读写，管理员及其他员工均不可见；
- 删除员工账号时，其名下知识库级联删除。
"""
from __future__ import annotations

import re
import uuid

from fastapi import APIRouter, Depends, File, Path, UploadFile

from api.schemas import (
    DocumentResponse,
    KBCreateRequest,
    KBResponse,
    QueryRequest,
    QueryResponse,
)
from app.auth import kb_access_level, kb_delete_meta, kb_get_meta, kb_list_meta, kb_set_meta
from app.config import get_settings
from app.dependencies import get_chat_chain, get_rag_service, require_user
from app.exceptions import FileParseError, ForbiddenError
from rag.document_loader import SUPPORTED_EXTENSIONS

router = APIRouter(dependencies=[Depends(require_user)])
_UNSAFE_NAME_RE = re.compile(r'[\\/:*?"<>|\s]+')
# kb_id 参与文件系统路径拼接，路径参数必须与 body 校验一致（防路径穿越）
_KB_ID_PARAM = Path(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")


def _safe_filename(filename: str) -> str:
    name = _UNSAFE_NAME_RE.sub("_", filename).strip("._")
    return name or uuid.uuid4().hex


def _kb_access(kb_id: str, user: dict) -> str:
    """返回知识库对当前用户的访问级别：owner（可读写）/ readonly / none（不可见）。

    规则集中在 app.auth.kb_access_level，与工作流节点的校验共用同一份。
    """
    return kb_access_level(kb_id, user)


def _require_write(kb_id: str, user: dict) -> None:
    if _kb_access(kb_id, user) != "owner":
        raise ForbiddenError("该知识库为只读或不可见，无权修改")


def _require_read(kb_id: str, user: dict) -> None:
    if _kb_access(kb_id, user) == "none":
        raise ForbiddenError("知识库不存在或无权访问")


@router.post("/knowledge-bases", response_model=KBResponse, tags=["RAG"])
def create_knowledge_base(payload: KBCreateRequest, user: dict = Depends(require_user)):
    service = get_rag_service()
    existing = kb_get_meta(payload.kb_id)
    known_ids = {item["kb_id"] for item in service.list_knowledge_bases()}

    if existing is not None or payload.kb_id in known_ids:
        # 已存在的知识库只允许属主"重复创建"（幂等）。
        # kb_set_meta 是 INSERT OR REPLACE，若不校验，任何登录用户都能靠
        # POST 一个已知 kb_id 把 owner 改写成自己，从而夺走并删除别人的库。
        existing_owner = (existing or {}).get("owner") or "admin"
        if existing_owner != user["username"]:
            raise ForbiddenError("该知识库已存在，且不属于当前用户")

    service.create_knowledge_base(payload.kb_id)
    kb_set_meta(payload.kb_id, payload.name, user["username"])
    return KBResponse(
        kb_id=payload.kb_id,
        name=payload.name,
        documents=0,
        chunks=0,
        owner=user["username"],
        role="owner",
    )


@router.get("/knowledge-bases", response_model=list[KBResponse], tags=["RAG"])
def list_knowledge_bases(user: dict = Depends(require_user)):
    service = get_rag_service()
    meta_map = {m["kb_id"]: m for m in kb_list_meta()}
    result: list[KBResponse] = []
    for item in service.list_knowledge_bases():
        kb_id = item["kb_id"]
        owner = (meta_map.get(kb_id) or {}).get("owner") or "admin"
        access = _kb_access(kb_id, user)
        if access == "none":
            continue
        result.append(
            KBResponse(
                kb_id=kb_id,
                name=(meta_map.get(kb_id) or {}).get("name"),
                documents=item["documents"],
                chunks=item["chunks"],
                owner=owner,
                role=access,
            )
        )
    return result


@router.delete("/knowledge-bases/{kb_id}", tags=["RAG"])
def delete_knowledge_base(kb_id: str = _KB_ID_PARAM, user: dict = Depends(require_user)):
    _require_write(kb_id, user)
    get_rag_service().delete_knowledge_base(kb_id)
    kb_delete_meta(kb_id)
    return {"deleted": kb_id}


@router.get("/knowledge-bases/{kb_id}/documents", response_model=list[DocumentResponse], tags=["RAG"])
def list_documents(kb_id: str = _KB_ID_PARAM, user: dict = Depends(require_user)):
    _require_read(kb_id, user)
    docs = get_rag_service().list_documents(kb_id)
    return [DocumentResponse(kb_id=kb_id, **doc) for doc in docs]


@router.post("/knowledge-bases/{kb_id}/documents", response_model=list[DocumentResponse], tags=["RAG"])
async def upload_documents(
    kb_id: str = _KB_ID_PARAM,
    files: list[UploadFile] = File(...),
    user: dict = Depends(require_user),
):
    _require_write(kb_id, user)
    settings = get_settings()
    service = get_rag_service()
    service.create_knowledge_base(kb_id)  # 幂等：不存在则创建

    upload_dir = settings.files_dir / kb_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    max_bytes = settings.max_upload_mb * 1024 * 1024

    results: list[DocumentResponse] = []
    for upload in files:
        filename = upload.filename or "unnamed"
        ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        if ext not in SUPPORTED_EXTENSIONS:
            raise FileParseError(f"不支持的文件格式: {filename}")
        if upload.size is not None and upload.size > max_bytes:
            raise FileParseError(f"文件 {filename} 超过 {settings.max_upload_mb}MB 限制")

        # 存储名带 doc_id 前缀：同名文档互不覆盖，删除任一文档不会误删共享源文件
        doc_id = uuid.uuid4().hex
        stored_path = upload_dir / f"{doc_id}__{_safe_filename(filename)}"

        # 流式落盘 + 硬上限，避免整文件先读入内存
        written = 0
        try:
            with stored_path.open("wb") as f:
                while chunk := await upload.read(1024 * 1024):
                    written += len(chunk)
                    if written > max_bytes:
                        raise FileParseError(f"文件 {filename} 超过 {settings.max_upload_mb}MB 限制")
                    f.write(chunk)
        except Exception:
            stored_path.unlink(missing_ok=True)
            raise
        if written == 0:
            stored_path.unlink(missing_ok=True)
            raise FileParseError(f"文件为空: {filename}")

        try:
            info = service.ingest_file(stored_path, kb_id, doc_id=doc_id, filename=filename)
        except Exception:
            # 解析/向量化失败时清掉已落盘的源文件：
            # 否则 data/files 会不断累积既不在文档列表里、也无法从界面删除的无主文件
            stored_path.unlink(missing_ok=True)
            raise
        results.append(
            DocumentResponse(
                doc_id=info["doc_id"],
                kb_id=kb_id,
                filename=filename,
                chunks=info["chunks"],
                created_at=info["created_at"],
            )
        )
    return results


@router.delete("/knowledge-bases/{kb_id}/documents/{doc_id}", tags=["RAG"])
def delete_document(
    kb_id: str = _KB_ID_PARAM,
    doc_id: str = Path(min_length=1, max_length=64),
    user: dict = Depends(require_user),
):
    _require_write(kb_id, user)
    get_rag_service().delete_document(kb_id, doc_id)
    return {"deleted": doc_id}


@router.post("/knowledge-bases/{kb_id}/reindex", tags=["RAG"])
def reindex(kb_id: str = _KB_ID_PARAM, user: dict = Depends(require_user)):
    _require_write(kb_id, user)
    return get_rag_service().reindex(kb_id)


@router.post("/query", response_model=QueryResponse, tags=["RAG"])
def query(payload: QueryRequest, user: dict = Depends(require_user)):
    _require_read(payload.kb_id, user)
    chain = get_chat_chain()
    history = [m.model_dump() for m in payload.history] if payload.history else None
    result = chain.answer(
        kb_id=payload.kb_id,
        question=payload.question,
        history=history,
        top_k=payload.top_k,
    )
    return QueryResponse(**result)
