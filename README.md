# 🎯 应届生 AI 求职助手 (Career AI Platform)

## 项目简介

多智能体协作的 AI 求职系统，覆盖从职业探索到面试模拟的完整链路，专门面向应届生解决"从学校到企业"的求职痛点。通过 6 个专业 Agent（职业探索师/JD翻译官/简历包装师/岗位匹配师/面试官/学习规划师）协同工作，提供一站式求职辅导。

## 方向

**方向一：Agentic AI 原生开发**

基于 LangGraph 框架构建的多智能体系统，采用 ReAct 循环（Reasoning + Acting）和状态机模式，实现 Agent 间的自主协作与上下文传递。系统从零设计，充分利用 LLM 的推理能力，而非对传统软件进行简单的 Agent 化改造。

## 技术栈

- **AI IDE**: Trae CN
- **LLM**: DeepSeek API (推理型任务) + Qwen API (文本型任务)
- **AI 框架**: LangGraph (多 Agent 编排) + LangChain (LLM 集成)
- **后端框架**: FastAPI + SSE (Server-Sent Events)
- **状态管理**: Pydantic BaseModel
- **前端**: 原生 HTML/JS + marked.js (Markdown 渲染)
- **容器化**: Docker + Caddy (反向代理)
- **可观测性**: Langfuse (Tracing) + pytest (Benchmark)
- **MCP 协议**: 自定义 MCP Client + 7 个 MCP Server

## 目录结构

```
src/
├── backend/
│   ├── app/
│   │   ├── agents/              # 6 个 Agent 实现
│   │   │   ├── agent_0_exploration.py   # 职业探索 (Qwen)
│   │   │   ├── agent_1_jd_translator.py # 岗位分析 (DeepSeek)
│   │   │   ├── agent_2_resume_wrapper.py # 简历优化 (Qwen)
│   │   │   ├── agent_3_job_matcher.py   # 岗位匹配 (DeepSeek)
│   │   │   ├── agent_4_interviewer.py   # 面试模拟 (DeepSeek)
│   │   │   └── agent_5_planner.py       # 学习规划 (Qwen)
│   │   ├── core/                # 核心组件
│   │   │   ├── state.py         # Pydantic 状态模型 (OrchestratorState)
│   │   │   ├── base_agent.py    # Agent 基类 (ReAct 循环)
│   │   │   ├── mcp_client.py    # MCP 客户端 (HTTP + 降级)
│   │   │   ├── config.py        # 配置管理 (LLM/MCP/内存)
│   │   │   ├── memory.py        # 会话内存 (内存 + 磁盘)
│   │   │   ├── cache.py         # 语义缓存 (相似查询)
│   │   │   ├── router.py        # 语义路由 (余弦相似度)
│   │   │   └── knowledge_graph.py # 知识图谱 (技能/薄弱点)
│   │   ├── orchestrator/
│   │   │   └── graph.py         # LangGraph 编排器 (路由/执行/后处理)
│   │   ├── main.py              # FastAPI 入口 + SSE Generator
│   │   └── interview_service.py # 面试模式独立服务
│   ├── mcp-servers/             # MCP Server 实现
│   │   └── resume-parser-mcp/   # 简历解析 MCP Server
│   ├── tests/benchmark/         # Benchmark 测试 (18 个用例)
│   │   ├── test_agents.py
│   │   └── test_routing_and_graph.py
│   └── requirements.txt
├── frontend/
│   └── index.html               # 单页面应用 (SSE 客户端)
├── docker/                      # Docker 部署配置
│   ├── Dockerfile.backend
│   ├── Dockerfile.mcp
│   └── docker-compose.yml
├── tracing/                     # Langfuse 可观测性
│   └── langfuse-config.json
└── docs/                        # 文档
    ├── 实验报告_LangGraph多智能体求职AI平台.md
    ├── architecture-diagrams.md
    └── BUG_FIX_REPORT_20240618.md
```

**核心模块职责**：
- `agents/`：6 个专业 Agent，每个负责一个求职场景
- `core/`：状态管理、MCP 通信、配置、记忆、缓存、路由
- `orchestrator/`：LangGraph 状态图，负责意图识别、Agent 分发、自反思唤醒
- `main.py`：FastAPI 入口，SSE 流式输出，会话管理
- `mcp-servers/`：外部工具服务（简历解析、JD 搜索、校招日历等）

## 环境搭建

### 1. 依赖安装

```bash
cd backend
pip install -r requirements.txt
```

### 2. 环境变量配置（⚠️ 不硬编码 API Key）

```bash
# 复制环境变量模板
cp .env.example .env

# 编辑 .env 文件，填入你的 API Key
# 必须配置的变量：
# - DEEPSEEK_API_KEY: DeepSeek API 密钥
# - QWEN_API_KEY: Qwen API 密钥
# - LANGFUSE_PUBLIC_KEY: Langfuse 公钥（可选）
# - LANGFUSE_SECRET_KEY: Langfuse 私钥（可选）
```

### 3. 启动步骤

**方式一：本地开发**

```bash
# 终端 1：启动后端
cd backend
python -m uvicorn app.main:app --reload

# 终端 2：启动前端（静态文件服务）
cd frontend
python -m http.server 8080

# 访问 http://localhost:8080
```

**方式二：Docker 部署**

```bash
cd docker
docker-compose up --build
```

### 4. 运行测试

```bash
cd backend
python -m pytest tests/benchmark/ -v

# 预期输出：18 passed in 0.5s
```

## 项目状态

- [x] **Proposal** - 项目提案、技术选型、架构设计
- [x] **MVP** - 6 个 Agent 实现、SSE 流式输出、MCP 集成、18 个测试用例
- [ ] **Final** - 生产环境部署、性能优化、真实 LLM API 接入、更多 MCP Server

**当前版本**：v0.1.0  
**最后更新**：2024-06-21