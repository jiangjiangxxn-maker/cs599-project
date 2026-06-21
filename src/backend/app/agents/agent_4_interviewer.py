"""
Agent 4: Interviewer (面试模拟)
Product Spec:
  角色：候选人
  目标：模拟面试训练
  交互流程：选择岗位+面试类型 → Agent4 多轮对话+评分+优势/不足
"""
from __future__ import annotations

import json

from app.core.base_agent import BaseAgent
from app.core.state import (
    OrchestratorState,
    InterviewSimulationSession,
    QAItem,
)


class InterviewerAgent(BaseAgent):
    """
    Agent 4 - Interviewer (面试模拟)
    Free-form conversational interview:
    Phase A: Initialize → opening + first question
    Phase B: Free dialogue → LLM decides next action (follow-up / new question / end)
    Phase C: End → generate final evaluation report
    """

    MAX_TURNS = 8  # Safety limit to prevent infinite loops

    def __init__(self):
        super().__init__(
            agent_id=4,
            name="Interviewer",
            llm_provider="",
        )

    def _detect_mode(self, user_input: str) -> str:
        """Detect interview mode from input."""
        input_lower = user_input.lower()
        if any(w in input_lower for w in ["hr", "behavior", "behavioral", "缺点", "分歧"]):
            return "hr"
        elif any(w in input_lower for w in ["project", "项目", "项目深挖", "deep-dive"]):
            return "project_deep_dive"
        return "technical"

    def _detect_position(self, state: OrchestratorState) -> str:
        """Detect position type from state."""
        if state.agent_context.career_report:
            positions = state.agent_context.career_report.recommended_positions
            if positions:
                pt = positions[0].position_name.lower()
                if "算法" in pt or "algorithm" in pt:
                    return "algorithm"
                elif "前端" in pt or "frontend" in pt:
                    return "frontend"
                elif "测试" in pt or "test" in pt:
                    return "testing"
                return pt
        return "backend"

    async def _phase_a_initialize(self, state: OrchestratorState) -> OrchestratorState:
        """Phase A: Initialize interview — opening + first question only."""
        ctx = state.agent_context
        mode = self._detect_mode(state.user_input)
        position_type = self._detect_position(state)

        # Build resume context
        resume_context = ""
        if ctx.resume_report:
            for section in ctx.resume_report.optimized_sections:
                resume_context += section.raw_content[:500] + "\n"

        mode_descriptions = {
            "technical": "技术面试 - 考察计算机基础、算法、系统设计等八股文知识",
            "project_deep_dive": "项目深挖 - 针对简历项目的技术难点、排查思路、架构选择进行追问",
            "hr": "HR行为面试 - 考察沟通能力、团队协作、职业规划等软技能",
        }

        prompt = f"""
你是一位经验丰富的面试官，正在进行应届生校招{mode_descriptions.get(mode, '技术面试')}。
面试岗位: {position_type}
{"简历项目参考:\n" + resume_context[:2000] if resume_context else ""}

用户消息: {state.user_input[:1000]}

请执行：
1. 输出一段简短的开场白，说明面试开始
2. 抛出第一个面试问题

以JSON格式输出（只输出JSON，不要其他内容）：
{{
    "opening": "简短的开场白，说明面试开始",
    "first_question": "第一个面试问题",
    "question_type": "fundamental/algorithm/system_design/project/behavior"
}}
"""
        response_str = await self._think(prompt)
        try:
            data = self._safe_parse_json(response_str)
        except (json.JSONDecodeError, ValueError):
            data = {"opening": "面试开始，我们聊聊你的技术背景吧。", "first_question": "请先做个自我介绍", "question_type": "behavior"}

        opening = data.get("opening", "面试开始，我们聊聊你的技术背景吧。")
        first_question = data.get("first_question", "请先做个自我介绍")
        question_type = data.get("question_type", "behavior")

        # Create first QA item
        qa = QAItem(
            question=first_question,
            question_type=question_type,
            model_answer="",
        )

        session = InterviewSimulationSession(
            user_id=state.session_id,
            position_type=position_type,
            mode=mode,
            qa_history=[qa],
            overall_score=0.0,
            strengths=[],
            weaknesses=[],
            improvement_plan=[],
        )
        ctx.interview_session = session

        # Mark interview as active
        ctx.is_interviewing = True
        # Track turn count for safety limit
        ctx.interview_total = self.MAX_TURNS
        ctx.interview_current = 1
        ctx.interview_questions = []

        # Save to conversation history
        ctx.conversation_history.append({
            "role": "assistant",
            "content": f"{opening}\n\n**{first_question}**",
            "agent_id": 4,
            "agent": "Interviewer",
        })

        self._log(state, f"Interview started: mode={mode}, position={position_type}")
        return state

    async def _phase_b_dialogue(self, state: OrchestratorState) -> OrchestratorState:
        """
        Phase B: Free-form dialogue.
        LLM receives full conversation context and decides:
        - Give feedback + ask follow-up question
        - Move to a new topic/question
        - End the interview with final report
        """
        ctx = state.agent_context
        session = ctx.interview_session
        if not session:
            return await self._phase_a_initialize(state)

        user_answer = state.user_input[:2000]
        turn_count = ctx.interview_current
        max_turns = self.MAX_TURNS

        # Build full conversation context
        mode_label = {"technical": "技术", "project_deep_dive": "项目深挖", "hr": "HR行为"}.get(session.mode, "技术")
        
        # Add user's current answer to history BEFORE building the prompt
        # This ensures the LLM sees the latest user response
        ctx.conversation_history.append({
            "role": "user",
            "content": user_answer,
            "agent_id": 0,
        })
        
        recent_history = "\n".join(
            f"{'面试官' if m.get('role') == 'assistant' else '考生'}：{m.get('content', '')[:300]}"
            for m in ctx.conversation_history[-8:]
        )

        # Check if we've reached max turns
        force_end = turn_count >= max_turns

        prompt = f"""
你是{mode_label}面试官。正在面试岗位 {session.position_type}。

当前是第 {turn_count} 轮对话（最多 {max_turns} 轮）。

历史对话：
{recent_history}

用户刚才的回答：{user_answer}

请根据面试情况执行以下操作之一：

1. 如果用户回答需要追问或深挖 → 给出简短反馈 + 追问一个问题
2. 如果当前话题已充分讨论 → 给出反馈 + 换一个新问题
3. 如果已经充分评估了用户的能力 → 给出反馈 + 输出最终评估报告
{"4. 注意：这已经是最后一轮对话，请输出最终评估报告" if force_end else ""}

以JSON格式输出（只输出JSON，不要其他内容）：
{{
    "action": "follow_up" 或 "new_question" 或 "end",
    "feedback": "对刚才回答的简短点评（1-2句话）",
    "score": 7.5,
    "next_question": "如果action不是end，这里填写下一个问题",
    "question_type": "fundamental/algorithm/system_design/project/behavior",
    "final_report": {{
        "overall_score": 7.0,
        "strengths": ["优势1", "优势2"],
        "weaknesses": ["不足1", "不足2"],
        "improvement_plan": ["建议1", "建议2"]
    }}
}}
"""
        response_str = await self._think(prompt)

        try:
            data = self._safe_parse_json(response_str)
        except (json.JSONDecodeError, ValueError):
            if force_end:
                data = {"action": "end", "feedback": "面试结束。", "score": 6.0, "final_report": {"overall_score": 6.0, "strengths": ["有基本了解"], "weaknesses": ["深度不够"], "improvement_plan": ["加强系统学习"]}}
            else:
                data = {"action": "new_question", "feedback": "好的。", "score": 6.0, "next_question": "请继续谈谈你的理解。"}

        action = data.get("action", "new_question")
        feedback_text = data.get("feedback", "")
        score = data.get("score", 5.0)

        # Update the current QA item with user answer and score
        if session.qa_history:
            current_qa = session.qa_history[-1]
            current_qa.user_answer = user_answer
            current_qa.score = score
            current_qa.feedback = feedback_text

        if action == "end" or force_end:
            # Phase C: End interview
            report = data.get("final_report", {})
            session.overall_score = report.get("overall_score", score)
            session.strengths = report.get("strengths", [])
            session.weaknesses = report.get("weaknesses", [])
            session.improvement_plan = report.get("improvement_plan", [])

            # Build final message
            final_msg = f"**面试结束！**\n\n反馈：{feedback_text}\n\n"
            final_msg += f"**总分：{session.overall_score:.1f}/10**\n\n"
            if session.strengths:
                final_msg += "**优势：**\n" + "\n".join(f"- {s}" for s in session.strengths) + "\n\n"
            if session.weaknesses:
                final_msg += "**薄弱点：**\n" + "\n".join(f"- {w}" for w in session.weaknesses) + "\n\n"
            if session.improvement_plan:
                final_msg += "**改进建议：**\n" + "\n".join(f"- {p}" for p in session.improvement_plan)

            # Reset interview state
            ctx.is_interviewing = False
            ctx.interview_total = 0
            ctx.interview_current = 0
            ctx.interview_questions = []

            ctx.conversation_history.append({
                "role": "assistant",
                "content": final_msg,
                "agent_id": 4,
                "agent": "Interviewer",
            })

            self._log(state, f"Interview completed. Score: {session.overall_score}/10, Turns: {turn_count}")

        else:
            # Continue interview: follow-up or new question
            ctx.interview_current += 1
            next_question = data.get("next_question", "请继续谈谈你的理解。")
            question_type = data.get("question_type", "fundamental")

            # Add new QA item
            new_qa = QAItem(
                question=next_question,
                question_type=question_type,
                model_answer="",
            )
            session.qa_history.append(new_qa)

            action_label = "追问" if action == "follow_up" else "新问题"
            next_msg = f"**反馈：** {feedback_text}\n\n**{next_question}**"

            ctx.conversation_history.append({
                "role": "assistant",
                "content": next_msg,
                "agent_id": 4,
                "agent": "Interviewer",
            })

            self._log(state, f"Interview progress: turn {ctx.interview_current}, action={action_label}")

        return state

    async def process(self, state: OrchestratorState) -> OrchestratorState:
        """Main entry: determine which phase we're in."""
        ctx = state.agent_context

        try:
            if not ctx.is_interviewing:
                # Phase A: Start new interview
                state = await self._phase_a_initialize(state)
            else:
                # Phase B: Free-form dialogue
                state = await self._phase_b_dialogue(state)

            state.agent_context.current_agent_id = 4

        except Exception as e:
            self._log_error(state, f"Interview failed: {str(e)}")
            # Safety reset
            ctx.is_interviewing = False

        return state