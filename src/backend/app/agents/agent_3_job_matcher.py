"""
Agent 3: Job Matching (岗位匹配)
Product Spec:
  角色：求职者
  目标：高效投递
  交互流程：自动匹配用户画像 → Agent3 推送岗位+校招日历+截止提醒
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta

from app.core.base_agent import BaseAgent
from app.core.state import (
    OrchestratorState,
    JobMatchingResult,
    MatchedPosition,
    CampusEvent,
    DeadlineReminder,
)


class JobMatcherAgent(BaseAgent):
    """
    Agent 3 - Job Matcher (岗位匹配/校招日历)
    LLM: DeepSeek for matching logic
    """

    def __init__(self):
        super().__init__(
            agent_id=3,
            name="Job Matcher",
            llm_provider="",
        )

    async def process(self, state: OrchestratorState) -> OrchestratorState:
        """
        ReAct loop for job matching:
        1. Think: Understand user target profile
        2. Act: Call campus-calendar-mcp → get recruitment timeline
        3. Observe: Get active recruitment periods
        4. Think: Filter matching positions
        5. Act: Call jd-search-mcp → search positions
        6. Observe: Get position details
        7. Think: Rank by match score & urgency
        8. Final: Output JobMatchingResult
        """
        self._log(state, "Starting job matching...")

        try:
            # Step 1-3: Get campus recruitment calendar
            self._log(state, "Fetching campus recruitment calendar...")
            calendar_result = await self._act("campus_calendar/get_timeline", {
                "user_id": state.session_id,
            })
            calendar_data = calendar_result.get("data", {})
            current_season = calendar_data.get("current_season", "autumn")

            # Build user context
            target_positions = []
            if state.agent_context.career_report:
                target_positions = [p.position_name for p in state.agent_context.career_report.recommended_positions]
            if not target_positions and state.agent_context.user_profile:
                target_positions = state.agent_context.user_profile.target_positions

            tech_stack = []
            if state.agent_context.user_profile:
                tech_stack = [s.name for s in state.agent_context.user_profile.tech_stack]

            self._log(state, f"Matching for: {', '.join(target_positions) if target_positions else 'general positions'}")

            # Step 4-7: Generate matching results via LLM
            current_month = datetime.now().month
            season_map = {
                "early_autumn": "秋招提前批(6-8月)",
                "autumn": "正式秋招(9-11月)",
                "spring": "春招补录(3-5月)",
                "intern_conversion": "日常实习转正(全年)",
                "daily_intern": "日常实习(全年)",
            }
            current_season_label = season_map.get(current_season, "秋招季")

            matching_prompt = f"""
            你是一位校招匹配专家。根据以下用户信息，推荐适合的校招岗位。

            当前时间: {datetime.now().strftime('%Y年%m月')}
            当前校招季节: {current_season_label}

            用户目标岗位: {', '.join(target_positions) if target_positions else '待确定'}
            用户技术栈: {', '.join(tech_stack) if tech_stack else '待补充'}

            请完成以下任务:
            1. 推荐3-5个适合该用户的校招岗位
            2. 标注每个岗位属于哪个校招批次
            3. 给出匹配度评分
            4. 计算距离截止日期的天数
            5. 生成提醒建议

            以JSON格式输出（只输出JSON，不要其他内容）:
            {{
                "matched_positions": [
                    {{
                        "company_name": "公司名称",
                        "position_name": "岗位名称",
                        "recruitment_type": "early_autumn/autumn/spring/intern_conversion/daily_intern",
                        "match_score": 0.85,
                        "application_url": "",
                        "deadline": "2026-09-30T00:00:00",
                        "days_remaining": 30
                    }}
                ],
                "application_reminders": [
                    {{
                        "position_id": "0",
                        "company": "公司名称",
                        "deadline": "2026-09-30T00:00:00",
                        "urgency": "critical/upcoming/normal",
                        "reminder_message": "建议尽快投递"
                    }}
                ],
                "overall_strategy": "总体求职策略建议"
            }}
            """

            matching_str = await self._think(matching_prompt)
            try:
                matching_data = self._safe_parse_json(matching_str)
            except (json.JSONDecodeError, ValueError):
                raise

            # Build result
            matched_positions = []
            for i, p in enumerate(matching_data.get("matched_positions", [])):
                if "deadline" in p:
                    try:
                        p["deadline"] = datetime.fromisoformat(p["deadline"].replace("Z", ""))
                    except (ValueError, TypeError):
                        p["deadline"] = datetime.now() + timedelta(days=30)
                p["days_remaining"] = (p.get("deadline", datetime.now()) - datetime.now()).days if isinstance(p.get("deadline"), datetime) else 30
                matched_positions.append(MatchedPosition(**p))

            reminders = [
                DeadlineReminder(**r) for r in matching_data.get("application_reminders", [])
            ]
            for i, r in enumerate(reminders):
                if r.position_id == "0" and matched_positions:
                    r.position_id = f"pos_{i}"

            result = JobMatchingResult(
                matched_positions=matched_positions,
                campus_calendar=[
                    CampusEvent(
                        event_name=label,
                        recruitment_type=key,
                        start_date=datetime.now(),
                        end_date=datetime.now() + timedelta(days=60),
                        status="ongoing" if key == current_season else "upcoming",
                    )
                    for key, label in season_map.items()
                ],
                application_reminders=reminders,
                overall_strategy=matching_data.get("overall_strategy", "建议多投递，广撒网"),
            )

            state.agent_context.job_matching = result
            state.agent_context.current_agent_id = 3
            self._log(state, f"Job matching complete: {len(matched_positions)} positions matched")

        except Exception as e:
            self._log_error(state, f"Job matching failed: {str(e)}")

        return state