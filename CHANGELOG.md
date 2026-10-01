# 更新日志

本项目版本号遵循[语义化版本](https://semver.org/lang/zh-CN/)（SemVer）。

## [0.1.0] - 2026-10-01

首个公开发布版本。

### Added

- 企业知识库 RAG：PDF/Word/TXT/Markdown/CSV/Excel/PPT 解析、段落感知切分、本地向量索引、按知识库隔离与引用溯源问答
- 工具增强 Agent：计算器、日期、联网搜索、只读 SQL、文件读取等工具，ReAct 循环、步数限制与失败保护
- 多模态理解：图片描述、视觉问答、OCR、表格图片转 Markdown，含结构化输出重试
- 可视化工作流：input/llm/retrieval/tool/condition/output 六类节点 DAG 编排、条件分支与逐节点执行日志
- 工程化：FastAPI 服务、PBKDF2 用户鉴权与 HMAC token、滑动窗口限流、统一异常处理、Docker 部署
- 113 个单元测试，覆盖文本切分、向量库、工作流引擎、鉴权、文档加载等核心模块
