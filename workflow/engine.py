"""DAG 引擎：流程定义校验（含环检测）、拓扑排序、按激活路径执行。"""
from __future__ import annotations

import time
from collections import defaultdict, deque
from typing import Any, Callable

from app.exceptions import WorkflowExecutionError, WorkflowValidationError
from workflow.nodes import NODE_TYPES


class WorkflowEngine:
    def validate(self, definition: dict) -> dict:
        nodes = definition.get("nodes")
        edges = definition.get("edges", [])
        if not isinstance(nodes, list) or not nodes:
            raise WorkflowValidationError("工作流必须包含非空 nodes 列表")

        nodes_by_id: dict[str, dict] = {}
        for node in nodes:
            nid = node.get("id")
            ntype = node.get("type")
            if not nid:
                raise WorkflowValidationError("存在缺少 id 的节点")
            if nid in nodes_by_id:
                raise WorkflowValidationError(f"节点 id 重复: {nid}")
            if ntype not in NODE_TYPES:
                raise WorkflowValidationError(f"节点 {nid} 类型非法: {ntype}，支持 {sorted(NODE_TYPES)}")
            nodes_by_id[nid] = node

        adjacency: dict[str, list[dict]] = defaultdict(list)
        indegree: dict[str, int] = {nid: 0 for nid in nodes_by_id}
        for edge in edges:
            source, target = edge.get("source"), edge.get("target")
            if source not in nodes_by_id or target not in nodes_by_id:
                raise WorkflowValidationError(f"边引用了不存在的节点: {source} -> {target}")
            if source == target:
                raise WorkflowValidationError(f"节点 {source} 不允许自环")
            adjacency[source].append(edge)
            indegree[target] += 1

        # 入口节点必须在拓扑排序（会原地递减 indegree）之前计算
        entry_nodes = [nid for nid, deg in indegree.items() if deg == 0]
        topo_order = self._topo_sort(nodes_by_id, adjacency, indegree)
        output_nodes = [nid for nid, node in nodes_by_id.items() if node["type"] == "output"]
        if not output_nodes:
            raise WorkflowValidationError("工作流至少需要一个 output 节点")

        return {
            "nodes_by_id": nodes_by_id,
            "adjacency": dict(adjacency),
            "topo_order": topo_order,
            "entry_nodes": entry_nodes,
            "output_nodes": output_nodes,
        }

    def execute(
        self,
        parsed: dict,
        inputs: dict,
        runner: Callable[[dict, dict], tuple[Any, dict]],
        timeout: float | None = None,
        user: dict | None = None,
    ) -> dict:
        """按拓扑顺序执行；runner(node, context) -> (output, meta)，返回执行上下文与日志。

        timeout：整体执行超时（秒），在节点之间检查，防止慢节点串行拖垮服务。
        user：当前登录用户，放入上下文供需要鉴权的节点（如 retrieval）使用。
        """
        nodes_by_id = parsed["nodes_by_id"]
        adjacency = parsed["adjacency"]
        context: dict[str, dict] = {"__inputs__": inputs, "__user__": user}
        logs: list[dict] = []
        active: set[str] = set(parsed["entry_nodes"])
        workflow_start = time.time()

        for nid in parsed["topo_order"]:
            if timeout and time.time() - workflow_start > timeout:
                raise WorkflowExecutionError(f"工作流执行超时（>{timeout:g}s），已终止")
            node = nodes_by_id[nid]
            if nid not in active:
                logs.append({"node": nid, "type": node["type"], "status": "skipped"})
                continue

            started = time.time()
            if timeout:
                # 把整体截止时间放进上下文：节点内部（如 LLM 调用）据此设置
                # 自己的请求超时，避免单个慢节点远超整体预算
                context["__deadline__"] = workflow_start + timeout
            try:
                output, meta = runner(node, context)
            except Exception as exc:
                logs.append(
                    {
                        "node": nid,
                        "type": node["type"],
                        "status": "failed",
                        "error": str(exc),
                        "elapsed_ms": round((time.time() - started) * 1000),
                    }
                )
                raise

            elapsed = round((time.time() - started) * 1000)
            context[nid] = {"type": node["type"], "status": "success", "output": output, "meta": meta}
            logs.append(
                {"node": nid, "type": node["type"], "status": "success", "elapsed_ms": elapsed, "meta": meta}
            )

            # 条件分支：仅激活 label 匹配的边；普通节点：激活无 label 或 default 的边
            out_edges = adjacency.get(nid, [])
            if node["type"] == "condition":
                matched = [e for e in out_edges if e.get("label") == output]
                if not matched:
                    matched = [e for e in out_edges if e.get("label") in (None, "default")]
                targets = [e["target"] for e in matched]
            else:
                targets = [
                    e["target"] for e in out_edges if e.get("label") is None or e.get("label") == "default"
                ]
            active.update(targets)

        outputs: dict[str, Any] = {}
        for nid in parsed["output_nodes"]:
            if nid in context:
                node_output = context[nid].get("output")
                if isinstance(node_output, dict):
                    outputs.update(node_output)
                else:
                    outputs[nid] = node_output
        return {"outputs": outputs, "context": context, "logs": logs}

    @staticmethod
    def _topo_sort(nodes_by_id: dict, adjacency: dict, indegree: dict) -> list[str]:
        queue = deque(sorted(nid for nid, deg in indegree.items() if deg == 0))
        order: list[str] = []
        while queue:
            nid = queue.popleft()
            order.append(nid)
            for edge in adjacency.get(nid, []):
                target = edge["target"]
                indegree[target] -= 1
                if indegree[target] == 0:
                    queue.append(target)
        if len(order) != len(nodes_by_id):
            cyclic = sorted(set(nodes_by_id) - set(order))
            raise WorkflowValidationError(f"工作流存在环，无法构成 DAG，涉及节点: {cyclic}")
        return order
