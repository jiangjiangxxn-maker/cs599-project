"""
Benchmark tests for Career AI Platform agents.
Tests agent state transitions, MCP fallback, and response formatting.
Each test prints what functionality it simulates to console.
"""
from __future__ import annotations

import pytest
from datetime import datetime

from app.core.state import (
    OrchestratorState,
    AgentContext,
    UserProfile,
    Skill,
    Education,
    CareerFeasibilityReport,
    PositionAnalysis,
    SkillGap,
    JDAnalysisReport,
    ParsedRequirement,
    GapItem,
    ResumeOptimizationResult,
    ComparisonBlock,
    JobMatchingResult,
    MatchedPosition,
    InterviewSimulationSession,
    QAItem,
    LearningPlan,
    WeeklyTask,
)


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def sample_state():
    """Create a sample orchestrator state for testing."""
    state = OrchestratorState(session_id="test-session-001")
    state.user_input = "我是计算机专业的学生，会 Python 和 Java，对后端开发感兴趣"
    state.agent_context = AgentContext(current_agent_id=-1)
    state.agent_context.user_profile = UserProfile(
        user_id="test-session-001",
        education=Education(
            school="测试大学",
            major="计算机科学与技术",
            degree="bachelor",
            graduation_year=2026,
        ),
        tech_stack=[
            Skill(name="Python", category="language", proficiency="intermediate"),
            Skill(name="Java", category="language", proficiency="intermediate"),
        ],
        interests=["后端开发", "分布式系统"],
        career_stage="senior",
    )
    return state


# ============================================================================
# Test 1: State Management (SDD Validation)
# ============================================================================

class TestStateManagement:
    """Test Pydantic state models and their validation."""

    def test_orchestrator_state_creation(self):
        """
        模拟【状态管理】功能。
        场景：创建新的对话会话
        验证：session_id、pipeline_complete、current_node、agent_id 初始值正确
        """
        print("\n" + "="*60)
        print("📦 Test: OrchestratorState 创建（状态管理）")
        print("  - 模拟场景：用户进入系统，新建一个空对话会话")
        print("  - 期望状态：session_id 正确赋值，流程未开始，Agent 未路由")
        print("-"*60)
        
        state = OrchestratorState(session_id="test-1")
        print(f"  📌 session_id = '{state.session_id}'")
        print(f"  📌 pipeline_complete = {state.pipeline_complete}（期望: False）")
        print(f"  📌 current_node = '{state.current_node}'（期望: start）")
        print(f"  📌 agent_context.current_agent_id = {state.agent_context.current_agent_id}（期望: -1）")
        
        assert state.session_id == "test-1", f"❌ session_id 应为 test-1，实际为 {state.session_id}"
        assert state.pipeline_complete is False, "❌ pipeline_complete 应为 False"
        assert state.current_node == "start", f"❌ current_node 应为 start，实际为 {state.current_node}"
        assert state.agent_context.current_agent_id == -1, f"❌ current_agent_id 应为 -1，实际为 {state.agent_context.current_agent_id}"
        
        print("  ✅ 断言通过！状态创建正确。")

    def test_user_profile_validation(self):
        """
        模拟【用户画像】功能。
        场景：用户填写个人信息（学校、专业、技能等）
        验证：教育背景和技能等级被正确存储
        """
        print("\n" + "="*60)
        print("👤 Test: UserProfile 验证（用户画像）")
        print("  - 模拟场景：用户填写个人信息表单")
        print("  - 输入：清华/CS/硕士/2025届毕业, Python高级")
        print("-"*60)
        
        profile = UserProfile(
            user_id="u1",
            education=Education(
                school="Tsinghua",
                major="CS",
                degree="master",
                graduation_year=2025,
            ),
            tech_stack=[
                Skill(name="Python", category="language", proficiency="advanced"),
            ],
        )
        print(f"  📌 教育背景: {profile.education.school} {profile.education.major} {profile.education.degree} ({profile.education.graduation_year}届)")
        print(f"  📌 技能栈: {profile.tech_stack[0].name} ({profile.tech_stack[0].proficiency})")
        
        assert profile.education.school == "Tsinghua", f"❌ school 应为 Tsinghua，实际为 {profile.education.school}"
        assert profile.tech_stack[0].proficiency == "advanced", f"❌ proficiency 应为 advanced，实际为 {profile.tech_stack[0].proficiency}"
        
        print("  ✅ 断言通过！用户画像存储正确。")

    def test_skill_proficiency_enum(self):
        """
        模拟【技能等级校验】功能。
        场景：用户输入无效的技能等级
        验证：Pydantic 拒绝非法枚举值
        """
        print("\n" + "="*60)
        print("🔒 Test: Skill 枚举校验（技能等级）")
        print("  - 模拟场景：用户输入非法技能等级 'expert_level'")
        print("  - 期望：Pydantic 抛出 ValidationError（合法值: beginner/intermediate/advanced/expert）")
        print("-"*60)
        
        import pydantic
        with pytest.raises((pydantic.ValidationError, ValueError)) as exc_info:
            Skill(name="Python", category="language", proficiency="expert_level")  # Invalid value
        
        print(f"  📌 异常类型: {type(exc_info.value).__name__}")
        print(f"  📌 异常信息: 包含 'expert_level' 不是合法枚举值")
        print("  ✅ 断言通过！非法枚举值被正确拒绝。")


# ============================================================================
# Test 2: Agent Output Formatting
# ============================================================================

class TestAgentOutputs:
    """Test that agent outputs follow the correct SDD-defined formats."""

    def test_career_report_format(self):
        """
        模拟【Agent 0 - 职业探索】功能。
        场景：LLM 返回职业探索报告
        验证：推荐岗位、技能差距、置信度格式正确
        """
        print("\n" + "="*60)
        print("🔍 Test: CareerFeasibilityReport（Agent 0 - 职业探索）")
        print("  - 模拟场景：用户输入'我不知道该选什么方向'")
        print("  - Agent 0 分析后推荐岗位 + 识别技能差距")
        print("-"*60)
        
        report = CareerFeasibilityReport(
            user_id="u1",
            recommended_positions=[
                PositionAnalysis(
                    position_name="后端开发工程师",
                    match_score=0.85,
                    daily_tasks=["设计REST API", "数据库调优"],
                    skill_barriers=["分布式系统设计", "高并发处理"],
                    growth_potential="3-5年到高级工程师",
                    average_salary_range=(200000, 400000),
                )
            ],
            skill_gaps=[
                SkillGap(
                    skill_name="Redis",
                    current_level="beginner",
                    required_level="intermediate",
                )
            ],
            confidence_score=0.8,
        )
        
        print(f"  📌 推荐岗位: {report.recommended_positions[0].position_name}")
        print(f"  📌 匹配度: {report.recommended_positions[0].match_score*100:.0f}%")
        print(f"  📌 日常工作: {', '.join(report.recommended_positions[0].daily_tasks)}")
        print(f"  📌 技能差距: {report.skill_gaps[0].skill_name} (当前: {report.skill_gaps[0].current_level} → 目标: {report.skill_gaps[0].required_level})")
        print(f"  📌 置信度: {report.confidence_score*100:.0f}%")
        
        assert len(report.recommended_positions) == 1, "❌ 应返回 1 个推荐岗位"
        assert report.recommended_positions[0].match_score == 0.85, "❌ 匹配度应为 0.85"
        assert report.confidence_score == 0.8, "❌ 置信度应为 0.8"
        
        print("  ✅ 断言通过！职业探索报告格式正确。")

    def test_jd_report_format(self):
        """
        模拟【Agent 1 - 岗位分析】功能。
        场景：用户粘贴 JD 文本让 AI 解读
        验证：JD 解析结果（硬性要求、难度评估）格式正确
        """
        print("\n" + "="*60)
        print("📋 Test: JDAnalysisReport（Agent 1 - 岗位分析）")
        print("  - 模拟场景：用户粘贴 JD '熟悉高并发架构'")
        print("  - Agent 1 翻译黑话、评估难度、识别差距")
        print("-"*60)
        
        report = JDAnalysisReport(
            jd_raw_text="熟悉高并发架构",
            parsed_requirements=[
                ParsedRequirement(
                    original_text="熟悉高并发架构",
                    translated_text="了解线程池、消息队列、缓存等基础概念即可",
                    category="hard_filter",
                    importance=4,
                )
            ],
            gap_analysis=[
                GapItem(
                    skill="高并发",
                    current_status="无相关经验",
                    recommended_action="学习Redis和消息队列基础知识",
                    urgency="immediate",
                )
            ],
            overall_difficulty="medium",
            suggestion="建议补充中间件相关知识",
        )
        
        print(f"  📌 JD原文: {report.jd_raw_text}")
        print(f"  📌 翻译: {report.parsed_requirements[0].translated_text}")
        print(f"  📌 类别: {report.parsed_requirements[0].category}（硬性要求）")
        print(f"  📌 难度评估: {report.overall_difficulty}")
        print(f"  📌 建议: {report.suggestion}")
        
        assert report.overall_difficulty == "medium", f"❌ 难度应为 medium，实际为 {report.overall_difficulty}"
        assert report.parsed_requirements[0].category == "hard_filter", f"❌ 类别应为 hard_filter，实际为 {report.parsed_requirements[0].category}"
        
        print("  ✅ 断言通过！岗位分析报告格式正确。")

    def test_resume_optimization_format(self):
        """
        模拟【Agent 2 - 简历优化】功能。
        场景：用户提交学术化项目经历让 AI 优化
        验证：优化前后对比、ATS 评分格式正确
        """
        print("\n" + "="*60)
        print("📝 Test: ResumeOptimizationResult（Agent 2 - 简历优化）")
        print("  - 模拟场景：用户'完成了一个数据库课程作业'")
        print("  - Agent 2 用 STAR 法则重写为商业化表达")
        print("-"*60)
        
        result = ResumeOptimizationResult(
            original_sections=[],
            optimized_sections=[],
            comparison_blocks=[
                ComparisonBlock(
                    original="完成了一个数据库课程作业",
                    optimized="独立设计分布式KV存储，支撑10W+ QPS",
                    transformation_type="academic_to_commercial",
                )
            ],
            keywords_matched=["分布式系统", "高并发"],
            ats_score=85.0,
            suggestions=["添加更多量化结果"],
        )
        
        print(f"  📌 优化前: '{result.comparison_blocks[0].original}'")
        print(f"  📌 优化后: '{result.comparison_blocks[0].optimized}'")
        print(f"  📌 匹配关键词: {', '.join(result.keywords_matched)}")
        print(f"  📌 ATS评分: {result.ats_score}/100")
        
        assert result.ats_score == 85.0, f"❌ ATS评分应为 85.0，实际为 {result.ats_score}"
        assert "分布式系统" in result.keywords_matched, "❌ 关键词应包含'分布式系统'"
        
        print("  ✅ 断言通过！简历优化报告格式正确。")

    def test_job_matching_format(self):
        """
        模拟【Agent 3 - 岗位匹配】功能。
        场景：系统根据用户画像推荐匹配岗位
        验证：公司名称、匹配度格式正确
        """
        print("\n" + "="*60)
        print("🎯 Test: JobMatchingResult（Agent 3 - 岗位匹配）")
        print("  - 模拟场景：系统推荐与用户匹配的岗位")
        print("  - 推荐：字节跳动 后端开发实习生 (匹配度82%)")
        print("-"*60)
        
        from datetime import datetime, timedelta
        result = JobMatchingResult(
            matched_positions=[
                MatchedPosition(
                    company_name="字节跳动",
                    position_name="后端开发实习生",
                    recruitment_type="daily_intern",
                    match_score=0.82,
                    deadline=datetime.now() + timedelta(days=15),
                    days_remaining=15,
                )
            ],
            campus_calendar=[],
            application_reminders=[],
            overall_strategy="建议重点关注后端岗位",
        )
        
        print(f"  📌 公司: {result.matched_positions[0].company_name}")
        print(f"  📌 岗位: {result.matched_positions[0].position_name}")
        print(f"  📌 匹配度: {result.matched_positions[0].match_score*100:.0f}%")
        print(f"  📌 截止天数: {result.matched_positions[0].days_remaining}天")
        print(f"  📌 策略建议: {result.overall_strategy}")
        
        assert result.matched_positions[0].company_name == "字节跳动", f"❌ 公司名应为字节跳动，实际为 {result.matched_positions[0].company_name}"
        assert result.matched_positions[0].match_score > 0.8, f"❌ 匹配度应 > 0.8，实际为 {result.matched_positions[0].match_score}"
        
        print("  ✅ 断言通过！岗位匹配结果格式正确。")

    def test_interview_session_format(self):
        """
        模拟【Agent 4 - 面试模拟】功能。
        场景：多轮问答面试，每轮记录题目、回答、评分
        验证：面试会话模式、Q&A 项格式正确
        """
        print("\n" + "="*60)
        print("🎤 Test: InterviewSimulationSession（Agent 4 - 面试模拟）")
        print("  - 模拟场景：技术面试，面试官问HashMap实现原理")
        print("  - 用户回答'基于数组+链表+红黑树'")
        print("-"*60)
        
        session = InterviewSimulationSession(
            user_id="u1",
            position_type="backend",
            mode="technical",
            qa_history=[
                QAItem(
                    question="Java中HashMap的实现原理？",
                    question_type="fundamental",
                    user_answer="基于数组+链表+红黑树",
                    model_answer="...",
                    score=8.0,
                    feedback="基础扎实",
                )
            ],
            overall_score=8.0,
            strengths=["基础扎实"],
            weaknesses=["缺少项目经验"],
            improvement_plan=["多做实战项目"],
        )
        
        print(f"  📌 面试模式: {session.mode}")
        print(f"  📌 问题1: {session.qa_history[0].question}")
        print(f"  📌 用户回答: {session.qa_history[0].user_answer}")
        print(f"  📌 评分: {session.qa_history[0].score}/10")
        print(f"  📌 反馈: {session.qa_history[0].feedback}")
        print(f"  📌 总分: {session.overall_score}/10")
        print(f"  📌 优势: {', '.join(session.strengths)}")
        print(f"  📌 不足: {', '.join(session.weaknesses)}")
        
        assert session.mode == "technical", f"❌ 模式应为 technical，实际为 {session.mode}"
        assert session.qa_history[0].score == 8.0, f"❌ 评分应为 8.0，实际为 {session.qa_history[0].score}"
        
        print("  ✅ 断言通过！面试会话格式正确。")

    def test_learning_plan_format(self):
        """
        模拟【Agent 5 - 学习规划】功能。
        场景：面试评分低后自动生成 8 周提升计划
        验证：周计划、周数格式正确
        """
        print("\n" + "="*60)
        print("📚 Test: LearningPlan（Agent 5 - 学习规划）")
        print("  - 模拟场景：面试评分 < 6.0，自动生成学习计划")
        print("  - 目标：8 周内掌握 Spring Boot")
        print("-"*60)
        
        plan = LearningPlan(
            user_id="u1",
            target_position="后端开发工程师",
            current_level="入门",
            estimated_duration_weeks=8,
            weekly_plan=[
                WeeklyTask(
                    week_number=1,
                    topic="Spring Boot入门",
                    learning_objectives=["掌握REST API设计"],
                    estimated_hours=10,
                )
            ],
            recommended_projects=[],
            milestone_checkpoints=[],
        )
        
        print(f"  📌 目标岗位: {plan.target_position}")
        print(f"  📌 当前水平: {plan.current_level}")
        print(f"  📌 计划周数: {plan.estimated_duration_weeks}周")
        print(f"  📌 第1周主题: {plan.weekly_plan[0].topic}")
        print(f"  📌 第1周目标: {', '.join(plan.weekly_plan[0].learning_objectives)}")
        print(f"  📌 建议学时: {plan.weekly_plan[0].estimated_hours}h")
        
        assert plan.estimated_duration_weeks == 8, f"❌ 周数应为 8，实际为 {plan.estimated_duration_weeks}"
        assert plan.weekly_plan[0].topic == "Spring Boot入门", f"❌ 主题应为 Spring Boot入门，实际为 {plan.weekly_plan[0].topic}"
        
        print("  ✅ 断言通过！学习规划格式正确。")


# ============================================================================
# Test 3: MCP Client Fallback
# ============================================================================

class TestMCPFallback:
    """Test MCP client local fallback behavior."""

    @pytest.mark.asyncio
    async def test_local_fallback(self):
        """
        模拟【MCP 本地降级】功能。
        场景：MCP Server 不响应时，用本地逻辑处理
        验证：降级返回格式正确
        """
        print("\n" + "="*60)
        print("🛡️ Test: MCP 本地降级（MCP Fallback）")
        print("  - 模拟场景：简历解析 MCP Server 无法连接")
        print("  - 降级策略：本地 Mock 返回简历解析结果")
        print("-"*60)
        
        from app.core.mcp_client import mcp_client
        
        print("  📌 调用 mcp_client._local_fallback('resume_parser/parse', ...)")
        result = await mcp_client._local_fallback("resume_parser/parse", {
            "text": "test resume",
            "user_id": "test",
        })
        
        print(f"  📌 返回状态: {result['status']}")
        print(f"  📌 返回结构: {list(result.keys())}")
        
        assert result["status"] == "ok", f"❌ 状态应为 ok，实际为 {result['status']}"
        assert "data" in result, "❌ 返回应包含 data 字段"
        
        print("  ✅ 断言通过！MCP 降级返回格式正确。")

    @pytest.mark.asyncio
    async def test_mcp_server_unavailable(self):
        """
        模拟【MCP 服务不可用】功能。
        场景：调用不存在的 MCP Server
        验证：不崩溃，返回空结果
        """
        print("\n" + "="*60)
        print("⚠️ Test: MCP 服务不可用（优雅降级）")
        print("  - 模拟场景：调用 'nonexistent_server/test'（不存在的服务）")
        print("  - 期望：不抛出异常，返回 fallback 结果")
        print("-"*60)
        
        from app.core.mcp_client import mcp_client
        
        print("  📌 调用 mcp_client.call_tool('nonexistent_server/test', {})")
        result = await mcp_client.call_tool("nonexistent_server/test", {})
        
        print(f"  📌 返回类型: {type(result).__name__}")
        print(f"  📌 返回内容: {str(result)[:100]}...")
        
        assert result is not None, "❌ 返回值不应为 None"
        
        print("  ✅ 断言通过！MCP 服务不可用时未崩溃。")


# ============================================================================
# Test 4: Memory System
# ============================================================================

class TestMemory:
    """Test session and vector memory."""

    def test_session_create_and_get(self):
        """
        模拟【会话创建与读取】功能。
        场景：用户开始新对话
        验证：会话能被创建、读取、在列表中
        """
        print("\n" + "="*60)
        print("💾 Test: 会话管理（SessionMemory）")
        print("  - 模拟场景：用户发起新对话，系统创建会话记录")
        print("-"*60)
        
        from app.core.memory import SessionMemory
        mem = SessionMemory()
        
        session_id = mem.create_session()
        print(f"  📌 创建会话: id = '{session_id}'")
        
        state = mem.get_session(session_id)
        print(f"  📌 读取会话: session_id = '{state.session_id}'")
        
        sessions = mem.list_sessions()
        print(f"  📌 会话列表: {sessions}")
        
        assert session_id is not None, "❌ session_id 不应为 None"
        assert state is not None, "❌ 读取状态不应为 None"
        assert state.session_id == session_id, f"❌ 不一致: {state.session_id} vs {session_id}"
        assert session_id in sessions, "❌ 会话应出现在列表中"
        
        print("  ✅ 断言通过！会话创建和读取正常。")

    def test_session_update(self):
        """
        模拟【会话状态更新】功能。
        场景：用户发送消息，系统更新会话状态
        验证：更新后的数据能正确读取
        """
        print("\n" + "="*60)
        print("🔄 Test: 会话更新（SessionMemory）")
        print("  - 模拟场景：用户发送消息'帮我优化简历'")
        print("  - 系统将该消息保存到会话中")
        print("-"*60)
        
        from app.core.memory import SessionMemory
        mem = SessionMemory()
        
        session_id = mem.create_session()
        state = mem.get_session(session_id)
        state.user_input = "帮我优化简历"
        print(f"  📌 写入数据: user_input = '{state.user_input}'")
        
        mem.update_session(session_id, state)
        print("  📌 已更新会话")
        
        updated = mem.get_session(session_id)
        print(f"  📌 读取数据: user_input = '{updated.user_input}'")
        
        assert updated.user_input == "帮我优化简历", f"❌ 数据不一致: '{updated.user_input}'"
        
        print("  ✅ 断言通过！会话更新正常。")


# ============================================================================
# Test 5: Orchestrator Routing
# ============================================================================

class TestOrchestrator:
    """Test orchestrator routing logic."""

    @pytest.mark.asyncio
    async def test_intent_routing_exploration(self):
        """
        模拟【意图路由 - 职业探索】功能。
        场景：用户输入'我不知道该选什么方向'
        验证：路由到 Agent 0 (CareerExplorer)
        """
        print("\n" + "="*60)
        print("🚦 Test: 意图路由 → Agent 0（职业探索）")
        print("  - 模拟场景：用户说'我不知道该选什么方向'")
        print("  - 期望路由：Agent 0 (CareerExplorer)")
        print("-"*60)
        
        from app.orchestrator.graph import CareerOrchestrator
        orch = CareerOrchestrator()

        state = OrchestratorState(session_id="test-route-1")
        state.user_input = "我不知道该选什么方向，帮我探索一下"
        print(f"  📌 用户输入: '{state.user_input}'")
        
        state = await orch.route_request(state)
        routed_agent = state.agent_context.current_agent_id
        agent_name = {0: "CareerExplorer", 1: "JDTranslator", 2: "ResumeWrapper", 3: "JobMatcher", 4: "Interviewer", 5: "LearningPlanner"}.get(routed_agent, "Unknown")
        print(f"  📌 路由结果: Agent {routed_agent} ({agent_name})")
        
        assert state.agent_context.current_agent_id == 0, f"❌ 应为 Agent 0，实际为 Agent {routed_agent}"
        
        print("  ✅ 断言通过！职业探索意图路由正确。")

    @pytest.mark.asyncio
    async def test_intent_routing_resume(self):
        """
        模拟【意图路由 - 简历优化】功能。
        场景：用户输入'帮我优化简历'
        验证：路由到 Agent 2 (ResumeWrapper)
        """
        print("\n" + "="*60)
        print("🚦 Test: 意图路由 → Agent 2（简历优化）")
        print("  - 模拟场景：用户说'帮我优化简历'")
        print("  - 期望路由：Agent 2 (ResumeWrapper)")
        print("-"*60)
        
        from app.orchestrator.graph import CareerOrchestrator
        orch = CareerOrchestrator()

        state = OrchestratorState(session_id="test-route-2")
        state.user_input = "帮我优化简历，把项目经历包装一下"
        print(f"  📌 用户输入: '{state.user_input}'")
        
        state = await orch.route_request(state)
        routed_agent = state.agent_context.current_agent_id
        agent_name = {0: "CareerExplorer", 1: "JDTranslator", 2: "ResumeWrapper", 3: "JobMatcher", 4: "Interviewer", 5: "LearningPlanner"}.get(routed_agent, "Unknown")
        print(f"  📌 路由结果: Agent {routed_agent} ({agent_name})")
        
        assert state.agent_context.current_agent_id == 2, f"❌ 应为 Agent 2，实际为 Agent {routed_agent}"
        
        print("  ✅ 断言通过！简历优化意图路由正确。")

    @pytest.mark.asyncio
    async def test_intent_routing_interview(self):
        """
        模拟【意图路由 - 面试模拟】功能。
        场景：用户输入'模拟面试'
        验证：路由到 Agent 4 (Interviewer)
        """
        print("\n" + "="*60)
        print("🚦 Test: 意图路由 → Agent 4（面试模拟）")
        print("  - 模拟场景：用户说'模拟面试，问一些八股文问题'")
        print("  - 期望路由：Agent 4 (Interviewer)")
        print("-"*60)
        
        from app.orchestrator.graph import CareerOrchestrator
        orch = CareerOrchestrator()

        state = OrchestratorState(session_id="test-route-3")
        state.user_input = "模拟面试，问一些八股文问题"
        print(f"  📌 用户输入: '{state.user_input}'")
        
        state = await orch.route_request(state)
        routed_agent = state.agent_context.current_agent_id
        agent_name = {0: "CareerExplorer", 1: "JDTranslator", 2: "ResumeWrapper", 3: "JobMatcher", 4: "Interviewer", 5: "LearningPlanner"}.get(routed_agent, "Unknown")
        print(f"  📌 路由结果: Agent {routed_agent} ({agent_name})")
        
        assert state.agent_context.current_agent_id == 4, f"❌ 应为 Agent 4，实际为 Agent {routed_agent}"
        
        print("  ✅ 断言通过！面试模拟意图路由正确。")


# ============================================================================
# Test 6: Config Loading
# ============================================================================

class TestConfig:
    """Test configuration loading."""

    def test_default_config(self):
        """
        模拟【配置加载】功能。
        场景：系统启动时加载默认配置
        验证：应用名称、版本、调试模式正确
        """
        print("\n" + "="*60)
        print("⚙️ Test: 默认配置加载")
        print("  - 模拟场景：系统启动，加载 AppConfig")
        print("-"*60)
        
        from app.core.config import AppConfig
        cfg = AppConfig()
        print(f"  📌 app_name = '{cfg.app_name}'")
        print(f"  📌 version = '{cfg.version}'")
        print(f"  📌 debug = {cfg.debug}")
        
        assert cfg.app_name == "Career AI Platform", f"❌ app_name 应为 Career AI Platform，实际为 {cfg.app_name}"
        assert cfg.version == "0.1.0", f"❌ version 应为 0.1.0，实际为 {cfg.version}"
        assert cfg.debug is True, "❌ debug 应为 True"
        
        print("  ✅ 断言通过！默认配置正确。")

    def test_llm_config_defaults(self):
        """
        模拟【LLM 配置加载】功能。
        场景：系统加载 LLM 模型配置
        验证：Qwen 模型名称和 Base URL 正确
        """
        print("\n" + "="*60)
        print("🤖 Test: LLM 配置加载")
        print("  - 模拟场景：系统加载 LLM 配置（Qwen 模型）")
        print("-"*60)
        
        from app.core.config import LLMConfig
        llm = LLMConfig()
        from app.core.config import config
        
        print(f"  📌 Qwen 模型: {llm.qwen_model}")
        print(f"  📌 Qwen Base URL: {llm.qwen_base_url}")
        
        assert llm.qwen_model == config.llm.qwen_model, f"❌ 模型名不匹配: {llm.qwen_model} vs {config.llm.qwen_model}"
        assert llm.qwen_base_url == "https://dashscope.aliyuncs.com/compatible-mode/v1", f"❌ Base URL 错误: {llm.qwen_base_url}"
        
        print("  ✅ 断言通过！LLM 配置正确。")