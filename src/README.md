# 🎯 应届生 AI 求职助手 (Career AI Platform)

多智能体协作的 AI 求职系统，覆盖从职业探索到面试模拟的完整链路，专门面向应届生解决"从学校到企业"的求职痛点。

## 架构亮点

- **6 大专业 Agent**：职业探索师/JD翻译官/简历包装师/岗位匹配师/面试官/学习规划师
- **1 个协调者 Orchestrator**：基于 LangGraph 状态机的多步推理调度
- **MCP 协议扩展**：标准化的工具接口（简历解析/校招日历/技术题库等）
- **记忆机制**：短期（对话窗口）+ 长期（ChromaDB 向量存储）
- **可观测性**：Langfuse Tracing + Benchmark 测试 + Docker 沙箱评估

## 核心技术要素

| 要素 | 实现 | 状态 |
|:---|:---|:---:|
| SDD 规格驱动开发 | `.spec/` 目录下 7 份规格文档 + Pydantic 运行时校验 | ✅ |
| 工具使用 / Function Calling / MCP 协议 | MCP Server 集群 + HTTP 客户端 + 本地 fallback | ✅ |
| 记忆机制 | SessionMemory (短期) + ChromaDB VectorMemory (长期) | ✅ |
| 状态管理与多步骤推理 | LangGraph StateGraph + ReAct 循环 | ✅ |
| 多智能体协作 | 6 Agent + 1 Orchestrator 协同 | ✅ |
| 可观测性与评估 | Langfuse Tracing + 18 个 Benchmark 测试 | ✅ |

## 快速开始

### 1. 安装依赖

```bash
cd backend
pip install -r requirements.txt
```

### 2. 配置环境变量（可选）

```bash
cp .env.example .env
# 编辑 .env 填入你的 API Key
```

### 3. 启动后端服务

```bash
cd backend
python -m uvicorn app.main:app --reload
```

### 4. 启动前端

直接用浏览器打开 `frontend/index.html`，或者:

```bash
# 用 Python 启动简单 HTTP 服务
cd frontend
python -m http.server 8080
# 然后访问 http://localhost:8080
```

### 5. 运行测试

```bash
cd backend
python -m pytest tests/benchmark/ -v
```

## Agent 工作流

```
用户输入 → Orchestrator(意图识别) → 路由到对应 Agent
  → Agent 执行 ReAct 循环 (Think→Act→Observe→Repeat)
  → 调用 MCP 工具获取数据
  → 生成结构化输出
  → Orchestrator 更新状态 & 返回响应
```

### 6 个 Agent 职能

| ID | Agent | 职能 | 默认 LLM |
|:---|:---|:---|:---:|
| 0 | **Career Explorer** | 技术栈盘点、兴趣分析、输出《岗位可行性报告》 | Qwen |
| 1 | **JD Translator** | JD 解析、门槛分级、过滤条件提取、黑话翻译 | DeepSeek |
| 2 | **Resume Wrapper** | 学术转商业包装、STAR 重构、ATS 评分优化 | Qwen |
| 3 | **Job Matcher** | 校招日历匹配、DDL 提醒、岗位推荐 | DeepSeek |
| 4 | **Interviewer** | 八股文/项目深挖/HR 面模拟、答案评分 | DeepSeek |
| 5 | **Learning Planner** | 技能差距分析、8 周学习路线、开源项目推荐 | Qwen |

**MoE 路由策略**：
- **DeepSeek**（推理型）：Agent 1/3/4 — 需要逻辑推理、分类、对话的任务
- **Qwen**（文本型）：Agent 0/2/5 — 需要文本生成、润色、结构化输出的任务

## 项目结构

```
career-ai/
├── .spec/                    # SDD 规格文档
├── backend/
│   ├── app/
│   │   ├── main.py           # FastAPI 入口
│   │   ├── orchestrator/     # LangGraph 状态图
│   │   ├── agents/           # 6 个 Agent 实现
│   │   ├── core/             # 核心（State/Config/Memory/MCP Client）
│   │   ├── tools/            # 工具函数
│   │   └── mcp/              # MCP 客户端
│   ├── mcp-servers/          # MCP Server 实现
│   │   └── resume-parser-mcp/ # 简历解析 MCP Server
│   ├── tests/benchmark/      # Benchmark 测试
│   └── requirements.txt
├── frontend/                 # Web 交互界面
├── docker/                   # Docker 部署配置
├── tracing/                  # Langfuse 可观测性
└── README.md
```

## API 接口

### 核心接口

| 方法 | 路径 | 说明 |
|:---|:---|:---|
| POST | `/chat` | 同步对话接口（简单场景） |
| POST | `/chat/stream` | SSE 流式对话接口（推荐，打字机效果） |
| GET | `/health` | 健康检查 + 活跃会话数 |
| GET | `/sessions` | 获取会话列表 |
| GET | `/session/{id}` | 获取会话详情 |
| GET | `/session/{id}/messages` | 获取完整对话历史 |
| DELETE | `/session/{id}` | 删除会话 |
| GET | `/agents` | 列出所有可用 Agent |
| POST | `/upload-resume` | 上传并解析简历文件（PDF/DOCX/TXT） |

### 面试模式接口

| 方法 | 路径 | 说明 |
|:---|:---|:---|
| POST | `/interview/start` | 开始新面试（初始化 Phase A） |
| POST | `/interview/answer` | 提交面试回答（Phase B/C 循环） |

### SSE 事件协议（/chat/stream）

```
data: {"type": "status", "node": "Orchestrator", "message": "正在分析..."}
data: {"type": "status", "node": "Orchestrator", "message": "唤醒【Resume Wrapper】"}
data: {"type": "message", "chunk": "**"}
data: {"type": "message", "chunk": "优化"}
data: {"type": "message", "chunk": "建议"}
...
data: {"type": "done", "session_id": "abc123", "result_type": "resume_optimization"}
data: [DONE]
```

**事件类型**：
- `status`：状态更新（节点名称 + 消息）
- `message`：内容块（逐 token 推送）
- `done`：完成标识（包含 session_id 和结果类型）
- `[DONE]`：流结束标记

## Docker 部署

```bash
cd docker
docker-compose up --build
```

## 前端功能

- **流式打字机**：SSE 逐 token 推送，Markdown 实时渲染
- **对话历史侧边栏**：左侧列出所有历史会话，支持切换/删除
- **快捷入口**：5 个场景快捷按钮（职业探索/岗位分析/简历优化/面试模拟/学习规划）
- **文件上传**：PDF/DOCX/TXT 简历文件上传 + 内容预览
- **60s 超时兜底**：AbortController 防止请求永久挂起
- **HTTP 错误处理**：500/502 时显示错误提示 + 关闭 loading

## 测试

```bash
# 运行所有测试
cd backend
python -m pytest tests/benchmark/ -v

# 预期输出：18 passed in 0.5s
```

**测试覆盖**：
- 状态管理（OrchestratorState、UserProfile、Skill 枚举）
- Agent 输出格式（6 个 Agent 的返回结构）
- MCP 降级（本地 fallback、服务不可用）
- 内存管理（Session 创建/获取/更新）
- 意图路由（关键词匹配、语义路由）

## 贡献指南

欢迎提交 Issue 和 Pull Request！

## 许可证

MIT License
