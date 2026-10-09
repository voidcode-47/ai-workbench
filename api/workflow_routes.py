"""工作流路由：流程校验/执行、示例模板查询。"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from api.schemas import TemplateSummary, WorkflowRunRequest, WorkflowRunResponse
from app.dependencies import get_workflow_executor, require_user
from workflow.executor import get_template, list_templates

router = APIRouter(dependencies=[Depends(require_user)])


@router.post("/run", response_model=WorkflowRunResponse, tags=["Workflow"])
def run_workflow(payload: WorkflowRunRequest, user: dict = Depends(require_user)):
    executor = get_workflow_executor()
    # 先显式校验，让定义错误以 400 返回
    executor.validate(payload.definition)
    # 传入当前用户：retrieval 节点依赖它做知识库权限校验
    result = executor.run(payload.definition, payload.inputs, user=user)
    return WorkflowRunResponse(**result)


@router.get("/templates", response_model=list[TemplateSummary], tags=["Workflow"])
def templates():
    return [TemplateSummary(**item) for item in list_templates()]


@router.get("/templates/{key}", tags=["Workflow"])
def template_detail(key: str):
    try:
        return get_template(key)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"模板不存在: {key}")
