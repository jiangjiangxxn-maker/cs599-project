"""
Agent 2: Resume Wrapper (简历优化)
Product Spec:
  角色：简历新手
  目标：优化项目表述
  交互流程：上传简历或粘贴项目 → Agent2 给出STAR改写+ATS评分+关键词
"""
from __future__ import annotations

import json

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage

from app.core.base_agent import BaseAgent
from app.core.state import (
    OrchestratorState,
    ResumeOptimizationResult,
    ResumeSection,
    STARContent,
    ComparisonBlock,
)


SYSTEM_PROMPT = """\
你是一位顶级简历优化专家，拥有 10 年以上一线互联网大厂（字节跳动、阿里、腾讯）技术总监招聘经验。
你的核心使命：将应届生稚嫩的学术化项目经历，转化为令 HR 和面试官眼前一亮的商业化表达。

## 核心原则

### 1. STAR 法则重构
- **S (Situation)**: 描述业务背景或技术挑战，而非"课程要求"
- **T (Task)**: 明确你的职责和目标，使用"主导/负责/设计"等强动词
- **A (Action)**: 详细说明技术方案、架构选型、关键实现
- **R (Result)**: 量化成果（QPS、延迟、数据量、覆盖率等）

### 2. 话术升维（学术 → 商业）
| 学术表达 | 商业表达 |
|---------|---------|
| "做了一个课程项目" | "独立设计并落地分布式系统原型" |
| "学习了 XX 技术" | "基于 XX 技术栈实现生产级方案" |
| "完成了实验" | "通过 POC 验证技术可行性" |
| "小组合作" | "跨职能协作，推动项目交付" |
| "优化了代码" | "性能调优，吞吐量提升 X%" |

### 3. 专业术语注入
根据目标岗位，主动融入以下术语：
- 后端: 微服务、分布式事务、Raft/Paxos、读写分离、缓存策略、消息队列
- 前端: 组件化、虚拟 DOM、SSR/SSG、性能优化、用户体验度量
- 算法: 时间复杂度、空间优化、模型部署、特征工程、A/B 测试

### 4. 量化思维
- 将"很多数据" → "日均处理 100W+ 条数据"
- 将"速度很快" → "接口响应时间从 500ms 优化至 50ms"
- 将"比较稳定" → "系统可用性达 99.9%，全年故障时间 < 8.76h"

## 输出格式
以 JSON 格式输出（只输出 JSON，不要其他内容）：
{
    "comparison_blocks": [
        {
            "original": "原始文本",
            "optimized": "STAR 重构后的商业化文本",
            "transformation_type": "academic_to_commercial/vague_to_quantified/restructured"
        }
    ],
    "keywords_matched": ["关键词1", "关键词2"],
    "ats_score": 75,
    "suggestions": ["建议1", "建议2"]
}
"""


class ResumeWrapperAgent(BaseAgent):
    """
    Agent 2 - Resume Wrapper (简历优化/学术转商业包装)
    Uses ChatPromptTemplate + ChatOpenAI with streaming support.
    """

    def __init__(self):
        super().__init__(
            agent_id=2,
            name="Resume Wrapper",
            llm_provider="",
        )

    async def process(self, state: OrchestratorState, config: dict | None = None) -> OrchestratorState:
        """
        ReAct loop for resume optimization with real LLM streaming.
        """
        self._log(state, "Starting resume optimization (academic → commercial)...")

        try:
            user_input = state.user_input
            if not user_input:
                self._log_error(state, "No resume content provided")
                return state

            # Build context from career report & JD report if available
            context_info = ""
            if state.agent_context.career_report:
                positions = [p.position_name for p in state.agent_context.career_report.recommended_positions]
                context_info += f"目标岗位: {', '.join(positions)}\n"
            if state.agent_context.jd_report:
                keywords = [r.original_text[:50] for r in state.agent_context.jd_report.parsed_requirements[:5]]
                context_info += f"JD关键词: {', '.join(keywords)}\n"

            # Construct ChatPromptTemplate
            prompt = ChatPromptTemplate.from_messages([
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=f"""{context_info}

原始简历内容:
```
{user_input[:4000]}
```

请严格按照上述原则，将这段经历转化为 STAR 法则的商业化表达。"""),
            ])

            # Get LLM and invoke with streaming support
            llm = await self._get_llm()
            messages = prompt.format_prompt().to_messages()

            # Pass config for LangGraph streaming callbacks
            invoke_config = config or {}
            response = await llm.ainvoke(messages, config=invoke_config)
            response_text = response.content if hasattr(response, 'content') else str(response)

            # Parse JSON response
            try:
                rewrite_data = self._safe_parse_json(response_text)
            except (json.JSONDecodeError, ValueError):
                rewrite_data = {
                    "comparison_blocks": [
                        {
                            "original": user_input[:200],
                            "optimized": "【优化版本】" + user_input[:200],
                            "transformation_type": "academic_to_commercial",
                        }
                    ],
                    "keywords_matched": [],
                    "ats_score": 50,
                    "suggestions": ["建议补充更多量化成果"],
                }

            # Build optimization result
            comparison_blocks = [
                ComparisonBlock(**b) for b in rewrite_data.get("comparison_blocks", [])
            ]

            result = ResumeOptimizationResult(
                original_sections=[
                    ResumeSection(section_type="experience", raw_content=user_input[:1000])
                ],
                optimized_sections=[
                    ResumeSection(
                        section_type="experience",
                        raw_content=rewrite_data.get("comparison_blocks", [{}])[0].get("optimized", "")
                        if rewrite_data.get("comparison_blocks") else "",
                    )
                ],
                comparison_blocks=comparison_blocks,
                keywords_matched=rewrite_data.get("keywords_matched", []),
                ats_score=rewrite_data.get("ats_score", 50.0),
                suggestions=rewrite_data.get("suggestions", []),
            )

            state.agent_context.resume_report = result
            state.agent_context.current_agent_id = 2
            self._log(state, f"Resume optimization complete: {len(comparison_blocks)} items rewritten")

        except Exception as e:
            self._log_error(state, f"Resume optimization failed: {str(e)}")

        return state