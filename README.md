# AI Workbench

**版本 0.1.0**

一体化 AI 能力平台：**企业知识库 RAG + 工具增强 Agent + 多模态理解 + 可视化工作流编排**，基于 FastAPI 提供工程化服务，附带玻璃拟态前端与 RAG 评测脚本。

## 功能模块

| 模块 | 能力 | 关键技术 |
| --- | --- | --- |
| 知识库 RAG | PDF/Word/TXT/MD 上传解析、段落感知切分、向量索引、按知识库隔离、引用溯源问答 | pypdf、python-docx、Embedding、本地向量库、Prompt 模板 |
| 工具 Agent | 计算器、日期、联网搜索、只读 SQL、文件读取、Python 执行（默认禁用，需 `ENABLE_PYTHON_TOOL=true` 开启）；ReAct 循环、步数限制、失败保护 | LangChain Tools、Function Calling、LangGraph |
| 多模态 | 图片描述、视觉问答、OCR、表格图片转 Markdown | VLM 多模态接口、Base64、结构化 Prompt + 重试 |
| 工作流 | input/llm/retrieval/tool/condition/output 六类节点，JSON 定义 DAG，条件路由、逐节点日志 | 拓扑排序、节点执行引擎、上下文模板传递 |
| 工程化 | Pydantic 校验、统一异常、账号登录鉴权（HMAC token）、滑动窗口限流、日志、Docker 部署 | FastAPI、中间件、Docker Compose |

> Embedding 优先走远端 OpenAI 兼容接口：配置了密钥、或端点指向本地/自建服务（如 LM Studio）时均使用真实向量；仅当端点是公开托管服务（OpenAI / DeepSeek / 智谱 / 通义 / Kimi）且未配置密钥时，才降级为本地哈希向量（无语义理解能力，仅适合离线演示）。LLM / VLM 调用仍需密钥。
>
> 安全默认：登录采用 HMAC 签名 token，改密或删除账号后旧 token 立即失效；首次登录后请务必修改默认管理员密码；跨域默认仅同源（配置 `CORS_ORIGINS` 白名单后放开）；`run_python` 默认禁用。

## 快速开始

```bash
# 1. 安装依赖（建议 Python 3.11+）
pip install -r requirements.txt

# 2. 配置环境变量
cp .env.example .env   # Windows 可直接编辑 .env，填入 OPENAI_API_KEY / OPENAI_BASE_URL

# 3. 启动服务
uvicorn main:app --reload
```

- 前端页面：<http://127.0.0.1:8000/>
- OpenAPI 文档：<http://127.0.0.1:8000/docs>

Docker 方式：

```bash
docker compose up --build
```

## 主要 API（前缀 `/api/v1`，除 `/auth/login` 与 `/health` 外均需 `Authorization: Bearer <token>`）

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| POST | `/rag/knowledge-bases` | 创建知识库 |
| GET | `/rag/knowledge-bases` | 知识库列表 |
| POST | `/rag/knowledge-bases/{kb}/documents` | 上传文档（multipart，支持多文件） |
| DELETE | `/rag/knowledge-bases/{kb}/documents/{doc_id}` | 删除文档 |
| POST | `/rag/knowledge-bases/{kb}/reindex` | 重建索引 |
| POST | `/rag/query` | 检索增强问答（返回 answer + citations） |
| GET | `/agent/tools` | Agent 工具列表 |
| POST | `/agent/chat` | Agent 对话（返回思考/工具调用/观察步骤） |
| POST | `/multimodal/analyze` | 图片分析（task: describe/question/ocr/table） |
| POST | `/workflow/run` | 执行 JSON 工作流定义 |
| GET | `/workflow/templates` | 内置示例流程（文档问答/Agent 分析/图片理解/报告生成） |

### 工作流定义示例

```json
{
  "nodes": [
    {"id": "in", "type": "input"},
    {"id": "search", "type": "tool", "config": {"name": "web_search", "args": {"query": "{{inputs.topic}}"}}},
    {"id": "summary", "type": "llm", "config": {"prompt": "搜索结果：\n{{search}}\n\n总结要点：{{inputs.topic}}"}},
    {"id": "out", "type": "output", "config": {"fields": {"summary": "summary"}}}
  ],
  "edges": [
    {"source": "in", "target": "search"},
    {"source": "search", "target": "summary"},
    {"source": "summary", "target": "out"}
  ]
}
```

条件分支：`condition` 节点配置 `{"field": "node_id", "routes": {"a": "值1", "b": "值2"}, "default": "a"}`，出边通过 `label` 匹配分支。

## RAG 评测

准备用例文件 `cases.json`：

```json
[
  {"question": "年假有多少天？", "expected_doc_id": "上传后返回的 doc_id", "keywords": ["年假", "5 天"]}
]
```

运行：

```bash
python -m eval.eval_rag --kb demo --cases cases.json --top-k 4
```

输出 Hit Rate、MRR、关键词召回率、Context Precision 四项指标。

## 项目结构

```
ai-workbench/
├── main.py                 # FastAPI 入口（中间件/静态页/异常）
├── requirements.txt
├── .env / .env.example
├── Dockerfile / docker-compose.yml
├── app/                    # 配置、日志、异常、公共依赖
├── api/                    # 路由层 + Pydantic schemas
├── rag/                    # 文档加载、切分、嵌入、向量库、检索、问答链
├── agent/                  # 工具、状态、LangGraph 图、执行器
├── multimodal/             # 图片校验、VLM、结构化输出
├── workflow/               # 节点、DAG 引擎、执行器与模板
├── frontend/               # index.html / app.js / style.css
├── data/files/             # 上传原文
├── data/vector_db/         # 本地向量库（按 kb_id 隔离）
└── eval/eval_rag.py        # 检索效果评测
```

## 版权声明

本项目**未采用任何开源许可证**。版权所有 © 2026 本项目作者，保留一切权利（All Rights Reserved）。

在遵守 GitHub 服务条款的前提下，你可以在 GitHub 上查看本仓库并进行 Fork；但未经版权方书面许可，不得将本项目的全部或部分代码、文档及相关资源用于任何商业用途，不得再分发，也不得公开发布修改后的衍生版本。课程学习与个人研究中的参考引用，请注明出处。
