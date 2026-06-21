# Career AI Platform - 系统架构详细说明

## 1. 整体架构概览

系统采用 8 层架构设计，从用户界面到基础设施逐层递进：

```
第1层：前端层 (Browser)
    ↓
第2层：接入层 (Caddy)
    ↓
第3层：API 层 (FastAPI)
    ↓
第4层：编排层 (LangGraph)
    ↓
第5层：Agent 层 (6 个专业 Agent)
    ↓
第6层：LLM 层 (DeepSeek + Qwen)
    ↓
第7层：MCP 工具层 (7 个 MCP Server)
    ↓
第8层：基础设施层 (State/Memory/Cache/Knowledge Graph)
```

---

## 2. 各层详细说明

### 第1层：前端层 (Browser)

**技术栈**：原生 HTML/JS/CSS + marked.js

**核心组件**：
- **Chat UI**：单页面应用，支持 Markdown 实时渲染
- **SSE Client**：基于 fetch 和 ReadableStream 实现 Server-Sent Events 客户端
- **File Upload**：支持 PDF/DOCX/TXT 格式简历文件上传
- **会话管理**：左侧边栏列出所有历史会话，支持切换/删除/新建
- **快捷入口**：5 个场景快捷按钮（职业探索/岗位分析/简历优化/面试模拟/学习规划）

**关键特性**：
- 流式打字机效果：SSE 逐 token 推送，Markdown 实时渲染
- 60 秒超时兜底：AbortController 防止请求永久挂起
- HTTP 错误处理：500/502 时显示错误提示

---

### 第2层：接入层 (Caddy)

**技术栈**：Caddy Web Server

**核心功能**：
- **反向代理**：将 API 请求转发到 FastAPI 后端
- **静态文件服务**：直接提供前端 HTML/CSS/JS 资源
- **SSL 终止**：生产环境支持 HTTPS
- **负载均衡**：多实例部署时分配请求

---

### 第3层：API 层 (FastAPI)

**技术栈**：FastAPI + SSE (Server-Sent Events)

**核心接口**：

| 方法 | 路径 | 说明 |
|:---|:---|:---|
| POST | `/chat` | 同步对话接口（简单场景） |
| POST | `/chat/stream` | SSE 流式对话接口（推荐，打字机效果） |
| GET | `/health` | 健康检查 + 活跃会话数 |
| GET | `/sessions` | 获取会话列表 |
| GET | `/session/{id}` | 获取会话详情 |
| DELETE | `/session/{id}` | 删除会话 |
| GET | `/agents` | 列出所有可用 Agent |
| POST | `/upload-resume` | 上传并解析简历文件 |
| POST | `/interview/start` | 开始新面试 |
| POST | `/interview/answer` | 提交面试回答 |

**SSE 事件协议**：
- `status`：状态更新（节点名称 + 消息）
- `message`：内容块（逐 token 推送）
- `done`：完成标识（包含 session_id 和结果类型）
- `[DONE]`：流结束标记

**事件流示例**：
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

---

### 第4层：编排层 (LangGraph)

**技术栈**：LangGraph StateGraph

**核心职责**：
1. **意图路由**：先关键词匹配，无匹配时走语义路由（余弦相似度计算）
2. **面试锁校验**：面试进行中时强制路由到面试 Agent，防止状态污染
3. **自反思机制**：检测面试评分低于 6 分时自动唤醒学习规划 Agent 生成学习计划
4. **状态重置**：每轮请求调用重置方法清空临时状态

**工作流程**：
```
用户输入
    ↓
[Orchestrator]
    ↓
[Route Request]
    ├─ 面试锁校验 → 强制面试 Agent
    ├─ 关键词匹配 → 对应 Agent
    └─ 语义路由 → 最匹配的 Agent
    ↓
[Run Agent]
    └─ 调用对应 Agent 的处理方法
    ↓
[Post-process]
    ├─ 自反思：检测面试评分 → 唤醒学习规划 Agent
    └─ 知识图谱：记录技能和薄弱点
```

---

### 第5层：Agent 层（6 大智能体）

**设计模式**：每个 Agent 实现 ReAct 循环（思考 → 行动 → 观察 → 重复）

| Agent | 角色 | 核心功能 | LLM 分配 | 触发场景 |
|:---|:---|:---|:---:|:---|
| Agent 0 | Career Explorer | 技术栈盘点、兴趣分析、输出岗位可行性报告 | Qwen | "不知道选什么方向" |
| Agent 1 | JD Translator | JD 解析、门槛分级、过滤条件提取、黑话翻译 | DeepSeek | "帮我看看这个 JD" |
| Agent 2 | Resume Wrapper | 学术转商业包装、STAR 重构、ATS 评分优化 | Qwen | "帮我优化简历" |
| Agent 3 | Job Matcher | 校招日历匹配、DDL 提醒、岗位推荐 | DeepSeek | "有哪些岗位适合我" |
| Agent 4 | Interviewer | 八股文/项目深挖/HR 面模拟、答案评分 | DeepSeek | "模拟技术面试" |
| Agent 5 | Learning Planner | 技能差距分析、8 周学习路线、开源项目推荐 | Qwen | "我该怎么学习" |

**Agent 协作示例**（面试 → 学习规划）：
```
用户: "模拟后端技术面试"
    ↓
[Agent 4: Interviewer]
    ↓
Phase A: 生成开场白 + 第一题
    ↓
用户回答第 1 题 → Phase B: 反馈 + 第 2 题
    ↓
用户回答第 2 题 → Phase B: 反馈 + 第 3 题
    ↓
用户回答第 3 题 → Phase C: 生成最终评估报告
    ↓
[Post-process] 检测到评分 < 6.0
    ↓
[Auto-wake] Agent 5: Learning Planner
    ↓
输出 8 周学习规划
```

---

### 第6层：LLM 层（MoE 路由）

**技术栈**：DeepSeek API + Qwen API

**MoE 路由策略**：
- **DeepSeek**（推理型）：处理需要逻辑推理、分类、对话的任务
  - Agent 1: JD 分析（分类、难度评估）
  - Agent 3: 岗位匹配（逻辑推理、排序）
  - Agent 4: 面试模拟（自由对话、评分）
  
- **Qwen**（文本型）：处理需要文本生成、润色、结构化输出的任务
  - Agent 0: 职业探索（文本生成、润色）
  - Agent 2: 简历优化（STAR 改写、术语注入）
  - Agent 5: 学习规划（结构化输出、计划生成）

**容错机制**：
- Circuit Breaker：单 Provider 连续失败 3 次自动切换
- 超时控制：LLM 调用超时 30 秒
- 降级策略：失败时返回预设的 fallback 响应

---

### 第7层：MCP 工具层

**技术栈**：HTTP Client + 7 个 MCP Server

**MCP Server 列表**：

| Server | 功能 | 使用 Agent |
|:---|:---|:---|
| Resume Parser MCP | PDF/DOCX 简历文本提取 | Agent 2 |
| JD Search MCP | 招聘信息检索和验证 | Agent 1 |
| Campus Calendar MCP | 校招时间线和截止日期管理 | Agent 3 |
| Open Source MCP | 开源项目推荐 | Agent 5 |
| Tech Quiz MCP | 技术题库（八股文） | Agent 4 |
| Industry Data MCP | 行业趋势和薪资数据 | Agent 0 |
| Interview Eval MCP | 面试评分和评估 | Agent 4 |

**调用方式**：
- 工具名格式：`<server_name>/<tool_name>`
- 示例：`resume_parser/parse`、`jd_search/fetch_jd`
- 自动解析为：`http://localhost:8001/tools/parse`

**降级策略**：
- MCP Server 不可用时，自动切换到本地 Mock 执行
- 每个 Server 都有对应的本地实现，确保开发/测试不受影响
- 超时设置：2 秒

---

### 第8层：基础设施层

**核心组件**：

#### 8.1 状态管理（OrchestratorState）
- 使用 Pydantic BaseModel 定义
- 包含 session_id、user_input、pipeline_complete 等字段
- 提供 reset_for_new_request() 方法，每轮请求重置临时状态
- 保留 user_profile 和 conversation_history 跨会话持久化

#### 8.2 跨 Agent 上下文（AgentContext）
- 作为跨 Agent 的"共享内存"
- 包含 current_agent_id、user_profile、各 Agent 的输出结果
- conversation_history 累积所有对话历史
- 单轮结果（如 resume_report）每轮重置

#### 8.3 会话持久化（SessionMemory）
- 内存 + JSON 文件双重存储
- 支持会话创建、获取、更新、删除
- 历史会话上限：57 个

#### 8.4 语义缓存（Semantic Cache）
- 基于嵌入向量的相似查询缓存
- 相似度阈值：0.95
- 目标命中率：> 30%
- 减少 LLM 调用成本

#### 8.5 知识图谱（Knowledge Graph）
- 记录用户技能和薄弱点
- 技能节点：名称、等级、练习次数、最后练习时间
- 薄弱点节点：名称、紧急程度、发现次数、首次发现时间
- 支持跨 Agent 推理和个性化推荐

---

## 3. 数据流设计

### 3.1 单 Agent 流程（如简历优化）

```
用户输入: "帮我优化简历"
    ↓
[Orchestrator] 路由到 Agent 2
    ↓
[Agent 2: Resume Wrapper]
    ├─ Think: 构建 Prompt
    ├─ Act: 调用 LLM (Qwen)
    └─ Observe: 解析 JSON 响应
    ↓
[Orchestrator] 格式化响应
    ↓
[SSE] 流式推送给前端
    ↓
[SessionMemory] 持久化对话历史
    ↓
[Knowledge Graph] 更新技能画像
```

### 3.2 多 Agent 协作流程（面试 → 学习规划）

```
用户: "模拟后端技术面试"
    ↓
[Agent 4: Interviewer]
    ↓
Phase A: 生成开场白 + 第一题
    ↓
用户回答 → Phase B: 反馈 + 新问题（最多 8 轮）
    ↓
Phase C: 生成最终评估报告
    ↓
[Post-process] 检测到评分 < 6.0
    ↓
[Auto-wake] Agent 5: Learning Planner
    ↓
输出 8 周学习规划
```

### 3.3 状态流转

**OrchestratorState 结构**：
- session_id: 会话唯一标识
- user_input: 用户输入
- pipeline_complete: 流程是否完成
- current_node: 当前节点
- agent_context: 跨 Agent 上下文

**AgentContext 结构**：
- current_agent_id: 当前 Agent ID
- user_profile: 长期用户画像（跨会话持久化）
- career_report/jd_report/resume_report: 单轮结果（每轮重置）
- interview_session: 面试会话（单轮）
- is_interviewing: 面试状态标志（单轮）
- conversation_history: 对话历史（累积）
- errors: 错误列表（单轮）

**关键设计**：
- user_profile 和 conversation_history 跨会话持久化
- 其他字段每轮重置，防止状态污染

---

## 4. 核心设计模式

### 4.1 ReAct 循环

每个 Agent 遵循 思考 → 行动 → 观察 → 重复 的循环：
1. **思考**：构建 Prompt，调用 LLM 推理
2. **行动**：通过 MCP Client 调用外部工具
3. **观察**：解析 LLM 响应，更新状态
4. **重复**：直到任务完成

### 4.2 状态机模式

面试模拟采用三阶段状态机：
- **Phase A**：初始化阶段，生成开场白和第一题
- **Phase B**：自由对话阶段，LLM 自主决策（追问/换题/结束）
- **Phase C**：结束阶段，生成评估报告，重置状态

### 4.3 降级策略

三级降级机制确保系统稳定性：
1. **MCP 降级**：MCP Server 不可用时，自动切换到本地 Mock 执行
2. **LLM Circuit Breaker**：单 Provider 连续失败 3 次自动切换
3. **SSE 超时兜底**：60 秒超时 + 异常捕获，确保流正常结束

### 4.4 上下文传递

- AgentContext 作为跨 Agent 的"共享内存"
- 每轮请求调用 reset_for_new_request() 清空临时状态
- 保留 user_profile 和 conversation_history 实现长期记忆

---

## 5. 性能优化

### 5.1 语义缓存
- 基于嵌入向量的相似查询缓存
- 相似度阈值：0.95
- 目标命中率：> 30%
- 减少 LLM 调用成本和延迟

### 5.2 MoE 路由
- 推理型任务（JD 分析、面试模拟）使用 DeepSeek
- 文本型任务（职业探索、简历优化）使用 Qwen
- 平衡成本和效果

### 5.3 流式输出
- 逐 token 推送，降低首 token 感知延迟
- CJK 分词优化：中文逐字、英文按词
- 流控：20ms/token，提供流畅的打字机效果

---

## 6. 可观测性

### 6.1 Langfuse Tracing
- 追踪 LLM 调用延迟
- 追踪 Token 消耗
- 追踪 Agent 执行时间
- 支持 A/B 测试和 Prompt 优化

### 6.2 Benchmark 测试
- 18 个单元测试用例
- 覆盖状态管理、Agent 输出格式、MCP 降级、内存管理、意图路由
- 测试执行时间：< 1 秒

### 6.3 日志记录
- Agent 执行日志：记录每个 Agent 的输入输出
- MCP 调用日志：记录工具调用和降级情况
- SSE 事件日志：记录 status/message/done 事件

---

## 7. 安全设计

### 7.1 API Key 管理
- 使用 .env 文件存储 API Key
- 永远不要将 .env 提交到 Git
- 生产环境建议使用 Vault 或环境变量
- 支持 OAuth2.0 和 JWT 认证

### 7.2 输入验证
- Pydantic 模型运行时校验
- 用户输入截断（限制长度）
- JSON 解析容错，失败时返回 fallback 数据

### 7.3 超时控制
- LLM 调用超时：30 秒
- MCP 调用超时：2 秒
- SSE 请求超时：60 秒（前端 AbortController）

---

## 8. 扩展性设计

### 8.1 新增 Agent
1. 继承 BaseAgent，实现 process 方法
2. 在 Orchestrator 的路由逻辑中添加匹配规则
3. 在配置文件中指定 LLM Provider

### 8.2 新增 MCP Server
1. 实现 MCP Server（提供 /tools/{tool_name} 接口）
2. 在配置文件中添加 Server URL
3. 在 MCP Client 的注册表中注册

### 8.3 新增 LLM Provider
1. 在配置文件中添加 Provider 配置
2. 实现 LLM 调用方法
3. 在 Agent 中指定使用新的 Provider

---

## 9. 部署架构

### 9.1 本地开发
```
前端 (http.server:8080)
    ↓
Caddy (:80)
    ↓
FastAPI (:8000)
    ↓
LangGraph + 6 Agent
    ↓
DeepSeek/Qwen API
```

### 9.2 Docker 生产
```
Internet
    ↓
Caddy (HTTPS)
    ↓
FastAPI (多实例)
    ↓
Redis (会话存储)
    ↓
PostgreSQL (持久化)
    ↓
Langfuse (可观测性)
```

---

## 10. 未来演进

### v0.2：微调 + 评估
- 收集用户反馈数据
- 微调小模型（面试评分、简历打分）
- 增加 Agent 行为评估指标

### v1.0：RAG + 知识增强
- 向量数据库存储领域知识
- 动态 Prompt 生成（基于用户画像）
- 多轮对话记忆优化

### v2.0：自主进化
- Agent 自我反思（Self-Reflection）
- 自动 Prompt 优化
- 跨用户知识迁移（联邦学习）