"""
Agent 1: Job Analysis / JD Translator (岗位分析)
Product Spec:
  角色：海投者
  目标：精准解读JD
  交互流程：粘贴JD文本 → Agent1 翻译黑话+区分硬性/加分项+难度评估
"""
from __future__ import annotations

import json
from typing import Optional

from app.core.base_agent import BaseAgent
from app.core.state import (
    OrchestratorState,
    JDAnalysisReport,
    ParsedRequirement,
    GapItem,
)


class JDTranslatorAgent(BaseAgent):
    """
    Agent 1 - JD Translator (岗位分析/黑话翻译)
    LLM: DeepSeek (cost-effective for classification tasks)
    """

    def __init__(self):
        super().__init__(
            agent_id=1,
            name="JD Translator",
            llm_provider="",
        )

    async def process(self, state: OrchestratorState) -> OrchestratorState:
        """
        ReAct loop for JD analysis:
        1. Think: Parse JD text
        2. Act: Call jd-search-mcp to validate/fetch JD
        3. Observe: Confirm JD source
        4. Think: Classify each requirement
        5. Act: Call resume-parser-mcp to compare with user profile
        6. Observe: Get matching results
        7. Think: Generate gap analysis
        8. Final: Output JDAnalysisReport
        """
        self._log(state, "Starting JD analysis...")

        try:
            user_input = state.user_input

            # Step 1 & 2: Parse JD from user input or fetch via MCP
            jd_text = user_input
            if "http" in user_input.lower():
                self._log(state, "Detected URL, attempting to fetch JD...")
                jd_result = await self._act("jd_search/fetch_jd", {"url": user_input})
                jd_text = jd_result.get("data", {}).get("jd_text", user_input)

            # Step 3: Analyze JD via LLM
            self._log(state, "Analyzing JD requirements...")

            # Build user profile context if available
            profile_context = ""
            if state.agent_context.user_profile:
                skills = ", ".join([s.name for s in state.agent_context.user_profile.tech_stack])
                profile_context = f"用户技术栈: {skills}"

            analysis_prompt = f"""
            你是一位专业的 JD 翻译官。你的任务是将招聘 JD 中的专业术语和"黑话"翻译成
            应届生能理解的描述，并区分硬性过滤条件和加分项。

            JD 原文:
            ```
            {jd_text[:3000]}
            ```

            {profile_context}

            请完成以下任务:
            1. 解析每一条要求，判断是硬性过滤条件(hard_filter)、加分项(nice_to_have)、还是岗位职责(responsibility)
            2. 将晦涩的术语"翻译"成清晰的语言
            3. 对每项要求按重要性打分(1-5分)
            4. 如果提供了用户技术栈，判断用户是否满足条件
            5. 评估整体难度

            以JSON格式输出（只输出JSON，不要其他内容）:
            {{
                "parsed_requirements": [
                    {{
                        "original_text": "原始要求文本",
                        "translated_text": "翻译后的清晰描述",
                        "category": "hard_filter/nice_to_have/responsibility",
                        "importance": 5,
                        "is_user_qualified": true/false/null
                    }}
                ],
                "gap_analysis": [
                    {{
                        "skill": "技能名称",
                        "current_status": "用户当前状态",
                        "recommended_action": "建议行动",
                        "urgency": "immediate/short-term/long-term"
                    }}
                ],
                "overall_difficulty": "low/medium/high",
                "suggestion": "总体建议"
            }}
            """

            analysis_str = await self._think(analysis_prompt)
            try:
                analysis_data = self._safe_parse_json(analysis_str)
            except (json.JSONDecodeError, ValueError):
                raise

            # Build report
            requirements = [
                ParsedRequirement(**r) for r in analysis_data.get("parsed_requirements", [])
            ]
            gaps = [
                GapItem(**g) for g in analysis_data.get("gap_analysis", [])
            ]

            report = JDAnalysisReport(
                jd_raw_text=jd_text[:2000],
                parsed_requirements=requirements,
                gap_analysis=gaps,
                overall_difficulty=analysis_data.get("overall_difficulty", "medium"),
                suggestion=analysis_data.get("suggestion", ""),
            )

            state.agent_context.jd_report = report
            state.agent_context.current_agent_id = 1
            self._log(state, f"JD analysis complete: {len(requirements)} requirements classified")

        except Exception as e:
            self._log_error(state, f"JD analysis failed: {str(e)}")

        return state