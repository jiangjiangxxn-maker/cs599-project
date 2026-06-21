"""
Agent 0: Career Exploration (职业探索)
====================================
Product Spec:
  角色：应届生
  目标：探索适合职业方向
  交互流程：输入专业/技能/兴趣 → Agent0 输出岗位列表+匹配度+技能差距
"""
from __future__ import annotations

import json
import random
from typing import Any

from app.core.base_agent import BaseAgent
from app.core.state import (
    OrchestratorState,
    CareerFeasibilityReport,
    PositionAnalysis,
    SkillGap,
)


class CareerExplorerAgent(BaseAgent):
    """
    角色：应届生
    目标：探索适合职业方向
    交互流程：输入专业/技能/兴趣 → 输出岗位列表+匹配度+技能差距

    职责：
    - 分析用户的教育背景和技术栈
    - 推荐匹配的职业方向（岗位名称、匹配度、日常工作内容）
    - 识别技能差距（当前水平 → 目标水平）
    - 输出置信度评分
    """

    def __init__(self):
        super().__init__(
            agent_id=0,
            name="CareerExplorer",
            llm_provider="qwen",
        )
        self._mock_positions = [
            PositionAnalysis(
                position_name="后端开发工程师",
                match_score=0.85,
                daily_tasks=["设计REST API", "数据库表结构设计", "业务逻辑实现"],
                skill_barriers=["分布式系统设计", "高并发处理", "微服务架构"],
                growth_potential="3-5年到高级工程师",
                average_salary_range=(200000, 400000),
            ),
            PositionAnalysis(
                position_name="大数据工程师",
                match_score=0.72,
                daily_tasks=["数据ETL开发", "实时计算任务", "数据仓库建设"],
                skill_barriers=["Spark/Flink框架", "数据建模", "实时计算"],
                growth_potential="2-4年到高级工程师",
                average_salary_range=(250000, 450000),
            ),
            PositionAnalysis(
                position_name="全栈开发工程师",
                match_score=0.68,
                daily_tasks=["前后端功能开发", "接口联调", "系统部署维护"],
                skill_barriers=["前端框架(Vue/React)", "DevOps实践", "全链路排查"],
                growth_potential="2-3年可独立带项目",
                average_salary_range=(180000, 350000),
            ),
        ]

    async def process(self, state: OrchestratorState) -> OrchestratorState:
        """执行职业探索分析"""
        # 构建 prompt 调用 LLM
        # ...
        # 返回推荐结果
        report = CareerFeasibilityReport(
            user_id=state.session_id,
            recommended_positions=self._mock_positions,
            skill_gaps=[
                SkillGap(
                    skill_name="Redis",
                    current_level="beginner",
                    required_level="intermediate",
                ),
                SkillGap(
                    skill_name="消息队列",
                    current_level="beginner",
                    required_level="intermediate",
                ),
            ],
            confidence_score=0.8,
        )
        state.agent_context.career_report = report
        self._log(state, f"已完成职业探索，推荐 {len(report.recommended_positions)} 个岗位")
        return state