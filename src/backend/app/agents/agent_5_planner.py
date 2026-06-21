"""
Agent 5: Learning Planner (学习规划)
Product Spec:
  角色：学习者
  目标：弥补短板
  交互流程：面试薄弱点 → Agent5 生成周计划+项目推荐+里程碑
"""
from __future__ import annotations

import json
from datetime import datetime

from app.core.base_agent import BaseAgent
from app.core.state import (
    OrchestratorState,
    LearningPlan,
    WeeklyTask,
    RecommendedProject,
    Milestone,
)


class LearningPlannerAgent(BaseAgent):
    """
    Agent 5 - Learning Planner (学习规划)
    LLM: Claude (best at creating structured, actionable plans)
    """

    def __init__(self):
        super().__init__(
            agent_id=5,
            name="Learning Planner",
            llm_provider="",
        )

    async def process(self, state: OrchestratorState) -> OrchestratorState:
        """
        ReAct loop for learning planning:
        1. Think: Analyze skill gaps & target requirements
        2. Act: Call open-source-mcp → find practice projects
        3. Observe: Get project recommendations
        4. Think: Design weekly learning roadmap
        5. Act: Call industry-data-mcp → verify relevance
        6. Observe: Confirm tech stack demand
        7. Think: Generate milestones
        8. Final: Output LearningPlan
        """
        self._log(state, "Starting learning planning...")

        try:
            # Step 1: Gather context from previous agents
            target_position = "backend"
            skill_gaps = []
            user_skills = []

            if state.agent_context.career_report:
                positions = state.agent_context.career_report.recommended_positions
                if positions:
                    target_position = positions[0].position_name
                skill_gaps = [
                    f"{g.skill_name}: {g.current_level} → {g.required_level}"
                    for g in state.agent_context.career_report.skill_gaps
                ]
            elif state.agent_context.jd_report:
                target_position = "target_position"
                skill_gaps = [
                    f"{g.skill}: {g.urgency}"
                    for g in state.agent_context.jd_report.gap_analysis
                ]

            if state.agent_context.user_profile:
                user_skills = [s.name for s in state.agent_context.user_profile.tech_stack]

            self._log(state, f"Planning learning path for: {target_position}")

            # Step 2-3: Get open source project recommendations
            project_result = await self._act("open_source/recommend_projects", {
                "target_position": target_position,
                "current_skills": user_skills,
                "skill_gaps": skill_gaps,
            })
            project_data = project_result.get("data", {})

            # Determine if this is backend or algorithm focused
            is_backend = any(w in target_position.lower() for w in ["后端", "back", "java", "spring", "go", "服务"])
            is_algorithm = any(w in target_position.lower() for w in ["算法", "algorithm", "ml", "ai", "data", "数据"])

            # Step 4-7: Generate learning plan via LLM
            skill_gap_str = "\n".join(skill_gaps) if skill_gaps else "待从探索阶段获取"
            user_skills_str = ", ".join(user_skills) if user_skills else "基础编程能力"

            plan_prompt = f"""
            你是一位资深的学习规划师，专门帮助应届生填补"从学校到企业的最后一公里"技能鸿沟。

            目标岗位: {target_position}
            用户现有技能: {user_skills_str}
            技能差距: {skill_gap_str}
            {"这是后端开发方向" if is_backend else ""}
            {"这是算法/AI方向" if is_algorithm else ""}

            请设计一份8周的学习计划，要求:
            1. 每周聚焦一个实战主题，直接切入企业级技术栈
            2. 如果是后端方向，路线包含: Spring框架/Redis/消息队列/微服务等
            3. 如果是算法方向，路线包含: 高级数据结构/DP优化/ML系统设计/Kaggle等
            4. 每周包含具体学习目标、练习任务和预计学时
            5. 推荐2-3个适合写进简历的开源练手项目

            以JSON格式输出（只输出JSON，不要其他内容）:
            {{
                "weekly_plan": [
                    {{
                        "week_number": 1,
                        "topic": "主题名称",
                        "learning_objectives": ["目标1", "目标2"],
                        "resources": ["资源链接或书籍"],
                        "practice_exercises": ["练习任务1", "练习任务2"],
                        "estimated_hours": 10
                    }}
                ],
                "recommended_projects": [
                    {{
                        "project_name": "项目名称",
                        "description": "项目描述",
                        "tech_stack": ["技术1", "技术2"],
                        "difficulty": "beginner/intermediate/advanced",
                        "github_url": "",
                        "resume_value": "这个项目对简历的价值说明"
                    }}
                ],
                "milestone_checkpoints": [
                    {{
                        "week": 4,
                        "checkpoint_name": "阶段节点名称",
                        "validation_criteria": "如何验证学习成果",
                        "status": "pending"
                    }}
                ],
                "estimated_duration_weeks": 8,
                "current_level": "用户当前水平描述",
                "summary": "总体学习路线总结"
            }}
            """

            plan_str = await self._think(plan_prompt)
            try:
                plan_data = self._safe_parse_json(plan_str)
            except (json.JSONDecodeError, ValueError):
                raise

            # Build learning plan
            weekly_tasks = [
                WeeklyTask(**w) for w in plan_data.get("weekly_plan", [])
            ]
            projects = [
                RecommendedProject(**p) for p in plan_data.get("recommended_projects", [])
            ]
            milestones = [
                Milestone(**m) for m in plan_data.get("milestone_checkpoints", [])
            ]

            plan = LearningPlan(
                user_id=state.session_id,
                target_position=target_position,
                current_level=plan_data.get("current_level", "入门"),
                estimated_duration_weeks=plan_data.get("estimated_duration_weeks", 8),
                weekly_plan=weekly_tasks,
                recommended_projects=projects,
                milestone_checkpoints=milestones,
            )

            state.agent_context.learning_plan = plan
            state.agent_context.current_agent_id = 5
            self._log(state, f"Learning plan complete: {len(weekly_tasks)} weeks, {len(projects)} projects")

        except Exception as e:
            self._log_error(state, f"Learning planning failed: {str(e)}")

        return state