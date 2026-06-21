"""
Orchestrator for Career AI Platform.
Uses LangGraph StateGraph to manage the multi-agent pipeline.
Routes user requests to the appropriate agent and manages state flow.
"""
from __future__ import annotations

import json
from typing import Any, Optional, Literal

from app.core.state import OrchestratorState
from app.core.base_agent import BaseAgent
from app.core.memory import memory_manager
from app.core.cache import semantic_cache
from app.core.router import semantic_router
from app.core.knowledge_graph import knowledge_graph
from app.core.tracing import tracing_manager

# Import all agents
from app.agents.agent_0_exploration import CareerExplorerAgent
from app.agents.agent_1_jd_translator import JDTranslatorAgent
from app.agents.agent_2_resume_wrapper import ResumeWrapperAgent
from app.agents.agent_3_job_matcher import JobMatcherAgent
from app.agents.agent_4_interviewer import InterviewerAgent
from app.agents.agent_5_planner import LearningPlannerAgent


class CareerOrchestrator:
    """
    Orchestrator for the career AI multi-agent pipeline.
    Manages agent routing, state transitions, and the LangGraph workflow.
    """

    def __init__(self):
        self.agents: dict[int, BaseAgent] = {
            0: CareerExplorerAgent(),
            1: JDTranslatorAgent(),
            2: ResumeWrapperAgent(),
            3: JobMatcherAgent(),
            4: InterviewerAgent(),
            5: LearningPlannerAgent(),
        }

        # LangGraph graph (lazy init)
        self._graph = None

    async def route_request(self, state: OrchestratorState) -> OrchestratorState:
        """
        Route user request to the appropriate agent based on intent.
        Uses keyword matching first, then semantic router as fallback.
        If interview is in progress, lock routing to Agent 4.
        """
        ctx = state.agent_context

        # Safety check: if state flags indicate interview but session is None, reset
        if ctx.is_interviewing and not ctx.interview_session:
            print("[Orchestrator] Warning: is_interviewing=True but interview_session=None. Resetting interview state.")
            ctx.is_interviewing = False
            ctx.interview_total = 0
            ctx.interview_current = 0
            ctx.interview_questions = []

        # Interview lock: if interviewing, force route to Agent 4
        if ctx.is_interviewing:
            user_lower = state.user_input.lower()
            end_keywords = ["结束", "不面了", "停", "stop", "exit", "quit", "完了", "答完了", "没有问题了"]
            if any(kw in user_lower for kw in end_keywords):
                # Force end interview
                ctx.interview_current = ctx.interview_total
                print(f"[Orchestrator] User requested interview end.")
            ctx.current_agent_id = 4
            return state

        user_input = state.user_input.lower()

        # Intent detection keywords
        intents = {
            0: ["探索", "方向", "迷茫", "适合", "职业", "规划", "career", "方向选择", "不知道"],
            1: ["jd", "岗位分析", "招聘", "职位描述", "需求", "黑话", "job", "requirement"],
            2: ["简历", "resume", "优化", "包装", "改写", "经历", "项目描述", "cv"],
            3: ["匹配", "投递", "校招", "岗位", "内推", "秋招", "春招", "招聘"],
            4: ["面试", "interview", "八股", "模拟", "hr", "技术面", "算法题"],
            5: ["学习", "learning", "规划", "技能", "提升", "课程", "路线", "project"],
        }

        # Score each intent
        scores = {}
        for agent_id, keywords in intents.items():
            scores[agent_id] = sum(1 for kw in keywords if kw in user_input)

        best_agent_id = max(scores, key=scores.get)

        # If keyword matching is ambiguous (score 0), try semantic routing
        if scores[best_agent_id] == 0:
            print(f"[Orchestrator] No keyword match, trying semantic router...")
            semantic_result = await semantic_router.route(state.user_input)
            if semantic_result >= 0:
                best_agent_id = semantic_result
                print(f"[Orchestrator] Semantic route: Agent {best_agent_id}")
            else:
                best_agent_id = 0  # Default fallback

        ctx.current_agent_id = best_agent_id
        return state

    async def run_agent(self, agent_id: int, state: OrchestratorState) -> OrchestratorState:
        """Run a specific agent and return updated state."""
        agent = self.agents.get(agent_id)
        if not agent:
            state.agent_context.errors.append(f"Unknown agent ID: {agent_id}")
            return state

        return await agent.process(state)

    async def run_full_pipeline(self, state: OrchestratorState) -> OrchestratorState:
        """
        Run the full career pipeline sequentially.
        Each agent builds upon the output of the previous one.
        """
        pipeline_order = [0, 1, 2, 3, 4, 5]

        for agent_id in pipeline_order:
            if state.pipeline_complete:
                break

            agent = self.agents.get(agent_id)
            if not agent:
                continue

            print(f"[Orchestrator] Running Agent {agent_id}: {agent.name}...")
            state = await agent.process(state)

            # Check for errors that should stop the pipeline
            critical_errors = [e for e in state.agent_context.errors if "failed" in e.lower()]
            if len(critical_errors) > 3:
                print(f"[Orchestrator] Too many errors, stopping pipeline.")
                break

        state.pipeline_complete = True
        return state

    async def _post_process(self, state: OrchestratorState) -> OrchestratorState:
        """
        P2: Self-reflection and cross-agent wake.
        P3: Knowledge graph injection.
        After an agent completes, check if downstream agents should be triggered.
        SKIP if interview is in progress (handled by interviewer agent itself).
        """
        ctx = state.agent_context

        # Skip cross-agent wake during interview — interviewer manages its own flow
        if ctx.is_interviewing:
            return state

        # P3: Inject knowledge graph context for cross-agent reasoning
        user_id = state.session_id
        if ctx.current_agent_id == 5 and ctx.learning_plan:
            weaknesses = knowledge_graph.get_weaknesses(user_id)
            if weaknesses:
                ctx.errors.append(f"[知识图谱] 已注入薄弱领域: {', '.join(weaknesses[:3])}")

        # 1. Agent 4 (Interview) → auto-wake Agent 5 (Learning Plan) only on completion
        if ctx.current_agent_id == 4 and ctx.interview_session and not ctx.is_interviewing:
            session = ctx.interview_session

            # P3: Record weaknesses in knowledge graph
            if session.weaknesses:
                for weakness in session.weaknesses[:3]:
                    knowledge_graph.add_weakness(user_id, weakness, source="interview")

            # Record skills
            if session.strengths:
                for strength in session.strengths[:3]:
                    knowledge_graph.add_skill(user_id, strength, level="strong", source="interview")

            if session.overall_score < 6.0 and session.weaknesses:
                print(f"[Self-Reflection] Interview score {session.overall_score}/10 < 6.0, auto-waking Agent 5 (Learning Planner)")
                weak_areas = "、".join(session.weaknesses[:3])
                original_input = state.user_input
                state.user_input = f"针对以下面试薄弱点制定学习计划：{weak_areas}"
                state = await self.agents[5].process(state)
                state.user_input = original_input
                ctx.current_agent_id = 4
                ctx.errors.append(f"[自反思] 面试评分偏低({session.overall_score}/10)，已自动生成学习规划")

        # 2. Agent 2 (Resume) → self-reflection on ATS score
        if ctx.current_agent_id == 2 and ctx.resume_report:
            report = ctx.resume_report
            if report.ats_score < 60:
                ctx.errors.append(f"[自反思] ATS评分偏低({report.ats_score}/100)，建议补充量化成果和关键词")

        return state

    async def process(self, state: OrchestratorState) -> OrchestratorState:
        """
        Main entry point: determine if single agent or full pipeline.
        Includes semantic cache check and Langfuse tracing.
        """
        # Langfuse trace: root span for this request
        with tracing_manager.trace(
            "orchestrator.process",
            session_id=state.session_id,
            metadata={"user_input_length": len(state.user_input)},
        ) as trace_ctx:
            # Store session in memory
            memory_manager.update_session(state.session_id, state)

            # P0: Semantic Cache - check if we have a cached result
            cached_result = await semantic_cache.get(state.user_input)
            if cached_result:
                print(f"[Orchestrator] Cache hit for query: {state.user_input[:50]}...")
                ctx = state.agent_context
                if cached_result.get("type") == "career_feasibility_report":
                    ctx.current_agent_id = 0
                elif cached_result.get("type") == "jd_analysis_report":
                    ctx.current_agent_id = 1
                elif cached_result.get("type") == "resume_optimization":
                    ctx.current_agent_id = 2
                elif cached_result.get("type") == "job_matching":
                    ctx.current_agent_id = 3
                elif cached_result.get("type") == "interview_simulation":
                    ctx.current_agent_id = 4
                elif cached_result.get("type") == "learning_plan":
                    ctx.current_agent_id = 5
                state.pipeline_complete = True
                memory_manager.update_session(state.session_id, state)
                return state

            # Route to appropriate agent(s)
            state = await self.route_request(state)
            agent_id = state.agent_context.current_agent_id

            # Check if user wants full pipeline
            if any(kw in state.user_input.lower() for kw in ["完整", "全流程", "all", "pipeline", "全套"]):
                state = await self.run_full_pipeline(state)
            else:
                state = await self.run_agent(agent_id, state)

            # P2: Self-reflection + cross-agent wake
            state = await self._post_process(state)

            # P0: Cache the result for future similar queries
            response_data = self.format_response(state)
            if response_data.get("result"):
                await semantic_cache.set(state.user_input, response_data["result"])

            # Update memory
            memory_manager.update_session(state.session_id, state)

            return state

    def format_response(self, state: OrchestratorState) -> dict:
        """Format the state into a user-friendly response."""
        response = {
            "session_id": state.session_id,
            "agent_id": state.agent_context.current_agent_id,
            "conversation_history": state.agent_context.conversation_history[-5:],  # Last 5 messages
            "pipeline_complete": state.pipeline_complete,
        }

        # Include the most relevant result based on which agent was active
        ctx = state.agent_context
        if ctx.career_report and ctx.current_agent_id == 0:
            report = ctx.career_report
            response["result"] = {
                "type": "career_feasibility_report",
                "positions": [
                    {
                        "name": p.position_name,
                        "match_score": p.match_score,
                        "daily_tasks": p.daily_tasks[:3],
                        "skill_barriers": p.skill_barriers[:3],
                    }
                    for p in report.recommended_positions
                ],
                "skill_gaps": [
                    {
                        "skill": g.skill_name,
                        "from": g.current_level,
                        "to": g.required_level,
                    }
                    for g in report.skill_gaps
                ],
                "confidence_score": report.confidence_score,
            }

        elif ctx.jd_report and ctx.current_agent_id == 1:
            report = ctx.jd_report
            response["result"] = {
                "type": "jd_analysis_report",
                "overall_difficulty": report.overall_difficulty,
                "requirements": [
                    {
                        "original": r.original_text[:100],
                        "translated": r.translated_text[:100],
                        "category": r.category,
                        "importance": r.importance,
                    }
                    for r in report.parsed_requirements
                ],
                "gaps": [
                    {
                        "skill": g.skill,
                        "urgency": g.urgency,
                        "action": g.recommended_action,
                    }
                    for g in report.gap_analysis
                ],
                "suggestion": report.suggestion,
            }

        elif ctx.resume_report and ctx.current_agent_id == 2:
            report = ctx.resume_report
            response["result"] = {
                "type": "resume_optimization",
                "comparisons": [
                    {
                        "original": c.original[:150],
                        "optimized": c.optimized[:150],
                        "type": c.transformation_type,
                    }
                    for c in report.comparison_blocks
                ],
                "keywords_matched": report.keywords_matched,
                "ats_score": report.ats_score,
                "suggestions": report.suggestions,
            }

        elif ctx.job_matching and ctx.current_agent_id == 3:
            result = ctx.job_matching
            response["result"] = {
                "type": "job_matching",
                "matched_positions": [
                    {
                        "company": p.company_name,
                        "position": p.position_name,
                        "match_score": p.match_score,
                        "deadline": str(p.deadline),
                        "days_left": p.days_remaining,
                    }
                    for p in result.matched_positions
                ],
                "reminders": [
                    {
                        "company": r.company,
                        "urgency": r.urgency,
                        "message": r.reminder_message,
                    }
                    for r in result.application_reminders
                ],
                "strategy": result.overall_strategy,
            }

        elif ctx.interview_session and ctx.current_agent_id == 4:
            session = ctx.interview_session
            response["result"] = {
                "type": "interview_simulation",
                "mode": session.mode,
                "is_interviewing": ctx.is_interviewing,
                "qa": [
                    {
                        "question": q.question[:100],
                        "user_answer": q.user_answer[:200] if q.user_answer else "",
                        "score": q.score,
                        "feedback": q.feedback[:100],
                        "follow_up": q.follow_up_question,
                    }
                    for q in session.qa_history
                ],
                "overall_score": session.overall_score,
                "strengths": session.strengths,
                "weaknesses": session.weaknesses,
                "improvement_plan": session.improvement_plan,
            }

        elif ctx.learning_plan and ctx.current_agent_id == 5:
            plan = ctx.learning_plan
            response["result"] = {
                "type": "learning_plan",
                "target_position": plan.target_position,
                "duration_weeks": plan.estimated_duration_weeks,
                "weekly_plan": [
                    {
                        "week": w.week_number,
                        "topic": w.topic,
                        "hours": w.estimated_hours,
                        "objectives": w.learning_objectives,
                    }
                    for w in plan.weekly_plan
                ],
                "projects": [
                    {
                        "name": p.project_name,
                        "difficulty": p.difficulty,
                        "tech_stack": p.tech_stack,
                        "resume_value": p.resume_value,
                    }
                    for p in plan.recommended_projects
                ],
            }

        # Include any errors
        if ctx.errors:
            response["errors"] = ctx.errors[-3:]  # Last 3 errors

        return response


# Global singleton
orchestrator = CareerOrchestrator()