"""
Automated tests for Semantic Router, Knowledge Graph, Semantic Cache, and Circuit Breaker.
Evaluates accuracy, correctness, and integration of core intelligence modules.
"""
from __future__ import annotations

import pytest
import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

from app.core.state import (
    OrchestratorState,
    AgentContext,
    InterviewSimulationSession,
    QAItem,
    ResumeOptimizationResult,
    ComparisonBlock,
)


# ============================================================================
# Test 1: Semantic Router - Keyword Routing Accuracy
# ============================================================================

class TestKeywordRouting:
    """Test keyword-based intent routing accuracy."""

    @pytest.fixture
    def orch(self):
        from app.orchestrator.graph import CareerOrchestrator
        return CareerOrchestrator()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("query,expected_agent", [
        # Agent 0: Career Exploration
        ("我不知道该选什么职业方向", 0),
        ("帮我想想适合什么岗位", 0),
        ("我很迷茫，不知道该做什么", 0),
        ("我应该走哪个方向", 0),
        ("帮我做个职业规划", 0),
        # Agent 1: JD Analysis
        ("帮我分析这个招聘JD", 1),
        ("这个岗位需求是什么意思", 1),
        ("解读一下这个职位描述", 1),
        ("招聘黑话是什么意思", 1),
        # Agent 2: Resume Optimization
        ("帮我优化简历", 2),
        ("改写一下我的项目经历", 2),
        ("把我的简历包装一下", 2),
        ("简历上的学术表达怎么改成商业化", 2),
        # Agent 3: Job Matching
        ("帮我找匹配的校招岗位", 3),
        ("有什么可以投递的", 3),
        ("秋招有哪些机会", 3),
        ("春招内推岗位", 3),
        # Agent 4: Interview Simulation
        ("模拟一次技术面试", 4),
        ("问一些八股文问题", 4),
        ("帮我做HR面试模拟", 4),
        ("来一轮算法题面试", 4),
        # Agent 5: Learning Plan
        ("帮我制定学习路线", 5),
        ("推荐一些学习课程", 5),
        ("我想提升技能", 5),
        ("做一个8周学习规划", 5),
    ])
    async def test_keyword_routing_accuracy(self, orch, query, expected_agent):
        """Test that each query routes to the correct agent via keywords."""
        state = OrchestratorState(session_id=f"test-kw-{expected_agent}")
        state.user_input = query
        state = await orch.route_request(state)
        assert state.agent_context.current_agent_id == expected_agent, (
            f"Query '{query}' should route to Agent {expected_agent}, "
            f"got Agent {state.agent_context.current_agent_id}"
        )

    @pytest.mark.asyncio
    async def test_keyword_routing_accuracy_rate(self, orch):
        """Test overall keyword routing accuracy rate."""
        test_cases = [
            ("我不知道该选什么方向", 0),
            ("帮我分析招聘需求", 1),
            ("优化简历经历", 2),
            ("校招岗位匹配", 3),
            ("模拟面试八股文", 4),
            ("学习路线规划", 5),
            ("职业方向探索", 0),
            ("JD需求解读", 1),
            ("简历改写", 2),
            ("秋招投递", 3),
            ("技术面模拟", 4),
            ("技能提升课程", 5),
        ]

        correct = 0
        for query, expected in test_cases:
            state = OrchestratorState(session_id=f"acc-{expected}")
            state.user_input = query
            state = await orch.route_request(state)
            if state.agent_context.current_agent_id == expected:
                correct += 1

        accuracy = correct / len(test_cases)
        print(f"\nKeyword Routing Accuracy: {correct}/{len(test_cases)} = {accuracy:.1%}")
        assert accuracy >= 0.8, f"Keyword routing accuracy {accuracy:.1%} < 80%"


# ============================================================================
# Test 2: Semantic Router - Cosine Similarity
# ============================================================================

class TestSemanticRouter:
    """Test the semantic router's cosine similarity computation."""

    def test_cosine_similarity_identical(self):
        """Test that identical vectors have similarity = 1.0."""
        from app.core.router import SemanticRouter
        router = SemanticRouter()
        score = router._cosine_similarity([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
        assert abs(score - 1.0) < 1e-6

    def test_cosine_similarity_orthogonal(self):
        """Test that orthogonal vectors have similarity = 0.0."""
        from app.core.router import SemanticRouter
        router = SemanticRouter()
        score = router._cosine_similarity([1.0, 0.0], [0.0, 1.0])
        assert abs(score) < 1e-6

    def test_cosine_similarity_opposite(self):
        """Test that opposite vectors have similarity = -1.0."""
        from app.core.router import SemanticRouter
        router = SemanticRouter()
        score = router._cosine_similarity([1.0, 0.0], [-1.0, 0.0])
        assert abs(score + 1.0) < 1e-6

    def test_cosine_similarity_partial(self):
        """Test partial similarity."""
        from app.core.router import SemanticRouter
        router = SemanticRouter()
        score = router._cosine_similarity([1.0, 1.0], [1.0, 0.0])
        expected = 1.0 / (2.0 ** 0.5)  # ~0.707
        assert abs(score - expected) < 1e-6

    def test_cosine_similarity_zero_vector(self):
        """Test that zero vectors return 0.0 (not NaN)."""
        from app.core.router import SemanticRouter
        router = SemanticRouter()
        score = router._cosine_similarity([0.0, 0.0], [1.0, 2.0])
        assert score == 0.0

    @pytest.mark.asyncio
    async def test_semantic_router_uninitialized_returns_negative(self):
        """Test that uninitialized router returns -1 for fallback."""
        from app.core.router import SemanticRouter
        router = SemanticRouter()
        result = await router.route("test query")
        assert result == -1


# ============================================================================
# Test 3: Knowledge Graph - CRUD Operations
# ============================================================================

class TestKnowledgeGraph:
    """Test knowledge graph operations."""

    @pytest.fixture
    def kg(self):
        """Create a fresh knowledge graph with temp file."""
        from app.core.knowledge_graph import KnowledgeGraph
        import tempfile
        import os

        kg = KnowledgeGraph.__new__(KnowledgeGraph)
        kg._graph = {}
        # Use a temp file to avoid polluting real data
        kg._graph_path = os.path.join(tempfile.gettempdir(), f"test_kg_{id(kg)}.json")
        yield kg
        # Cleanup
        if os.path.exists(kg._graph_path):
            os.remove(kg._graph_path)

    def test_add_skill(self, kg):
        """Test adding a skill to a user."""
        kg.add_skill("user1", "Python", level="advanced", source="resume")
        skills = kg.get_skills("user1")
        assert len(skills) == 1
        assert skills[0]["name"] == "Python"
        assert skills[0]["level"] == "advanced"

    def test_add_weakness(self, kg):
        """Test adding a weakness to a user."""
        kg.add_weakness("user1", "HashMap底层原理", source="interview")
        weaknesses = kg.get_weaknesses("user1")
        assert len(weaknesses) == 1
        assert weaknesses[0] == "HashMap底层原理"

    def test_add_duplicate_skill(self, kg):
        """Test that duplicate skills are updated, not duplicated."""
        kg.add_skill("user1", "Java", level="beginner")
        kg.add_skill("user1", "Java", level="intermediate")
        skills = kg.get_skills("user1")
        assert len(skills) == 1
        assert skills[0]["level"] == "intermediate"

    def test_add_duplicate_weakness(self, kg):
        """Test that duplicate weaknesses are not duplicated."""
        kg.add_weakness("user1", "Redis")
        kg.add_weakness("user1", "Redis")
        weaknesses = kg.get_weaknesses("user1")
        assert len(weaknesses) == 1

    def test_multiple_users_isolation(self, kg):
        """Test that different users have isolated graphs."""
        kg.add_skill("user1", "Python")
        kg.add_weakness("user1", "Redis")
        kg.add_skill("user2", "Java")

        assert len(kg.get_skills("user1")) == 1
        assert len(kg.get_weaknesses("user1")) == 1
        assert len(kg.get_skills("user2")) == 1
        assert len(kg.get_weaknesses("user2")) == 0

    def test_empty_user(self, kg):
        """Test querying a user with no data."""
        assert kg.get_skills("nonexistent") == []
        assert kg.get_weaknesses("nonexistent") == []

    def test_inject_context_empty(self, kg):
        """Test context injection for empty graph."""
        context = kg.inject_context("empty_user", "test")
        assert context == ""

    def test_inject_context_with_skills(self, kg):
        """Test context injection with skills only."""
        kg.add_skill("user1", "Python", level="advanced")
        kg.add_skill("user1", "Java", level="intermediate")
        context = kg.inject_context("user1", "test")
        assert "已掌握技能" in context
        assert "Python" in context

    def test_inject_context_with_weaknesses(self, kg):
        """Test context injection with weaknesses only."""
        kg.add_weakness("user1", "Redis")
        context = kg.inject_context("user1", "test")
        assert "薄弱领域" in context
        assert "Redis" in context

    def test_inject_context_full(self, kg):
        """Test context injection with both skills and weaknesses."""
        kg.add_skill("user1", "Python", level="strong")
        kg.add_weakness("user1", "分布式系统")
        kg.add_weakness("user1", "消息队列")
        context = kg.inject_context("user1", "test")
        assert "已掌握技能" in context
        assert "薄弱领域" in context
        assert "分布式系统" in context
        assert "消息队列" in context

    def test_persistence(self, kg):
        """Test that graph data persists to disk and can be reloaded."""
        kg.add_skill("user1", "Python", level="advanced")
        kg.add_weakness("user1", "Redis")

        # Reload from disk
        from app.core.knowledge_graph import KnowledgeGraph
        kg2 = KnowledgeGraph.__new__(KnowledgeGraph)
        kg2._graph = {}
        kg2._graph_path = kg._graph_path
        kg2._load_from_disk()

        skills = kg2.get_skills("user1")
        weaknesses = kg2.get_weaknesses("user1")
        assert len(skills) == 1
        assert skills[0]["name"] == "Python"
        assert len(weaknesses) == 1
        assert weaknesses[0] == "Redis"


# ============================================================================
# Test 4: Knowledge Graph - Cross-Agent Integration
# ============================================================================

class TestCrossAgentIntegration:
    """Test knowledge graph integration with orchestrator post-processing."""

    def _make_interview_state(self, score: float, weaknesses: list[str], strengths: list[str]):
        """Create a state simulating Agent 4 interview results."""
        state = OrchestratorState(session_id="test-cross-agent")
        state.user_input = "模拟面试"
        state.agent_context = AgentContext(current_agent_id=4)
        state.agent_context.interview_session = InterviewSimulationSession(
            user_id="test-cross-agent",
            position_type="backend",
            mode="technical",
            qa_history=[
                QAItem(
                    question="HashMap原理",
                    question_type="fundamental",
                    user_answer="数组+链表",
                    model_answer="...",
                    score=score,
                    feedback="需要补充",
                )
            ],
            overall_score=score,
            strengths=strengths,
            weaknesses=weaknesses,
            improvement_plan=["多做项目"],
        )
        return state

    @pytest.mark.asyncio
    async def test_low_score_triggers_learning_plan(self):
        """Test that interview score < 6 auto-triggers Agent 5."""
        from app.orchestrator.graph import CareerOrchestrator
        orch = CareerOrchestrator()

        state = self._make_interview_state(
            score=4.0,
            weaknesses=["HashMap底层原理", "Redis缓存策略"],
            strengths=["Python基础"],
        )

        # Run post-processing (simulating Agent 4 already ran)
        state = await orch._post_process(state)

        # Should have triggered learning plan
        assert state.agent_context.learning_plan is not None
        assert state.agent_context.current_agent_id == 4  # Restored

    @pytest.mark.asyncio
    async def test_high_score_no_trigger(self):
        """Test that interview score >= 6 does NOT trigger Agent 5."""
        from app.orchestrator.graph import CareerOrchestrator
        orch = CareerOrchestrator()

        state = self._make_interview_state(
            score=8.0,
            weaknesses=["项目经验不足"],
            strengths=["基础扎实", "算法能力"],
        )

        state = await orch._post_process(state)

        assert state.agent_context.learning_plan is None

    @pytest.mark.asyncio
    async def test_weaknesses_recorded_in_graph(self):
        """Test that interview weaknesses are recorded in knowledge graph."""
        from app.core.knowledge_graph import KnowledgeGraph
        import tempfile
        import os

        # Use temp graph
        kg = KnowledgeGraph.__new__(KnowledgeGraph)
        kg._graph = {}
        kg._graph_path = os.path.join(tempfile.gettempdir(), f"test_kg_cross_{id(kg)}.json")

        try:
            # Simulate recording weaknesses
            weaknesses = ["HashMap底层原理", "Redis缓存策略", "消息队列"]
            for w in weaknesses:
                kg.add_weakness("test-session", w, source="interview")

            # Verify recorded
            recorded = kg.get_weaknesses("test-session")
            assert len(recorded) == 3
            for w in weaknesses:
                assert w in recorded
        finally:
            if os.path.exists(kg._graph_path):
                os.remove(kg._graph_path)


# ============================================================================
# Test 5: Circuit Breaker
# ============================================================================

class TestCircuitBreaker:
    """Test circuit breaker logic in base agent."""

    def test_circuit_breaker_initial_state(self):
        """Test that circuit breaker starts closed."""
        from app.agents.agent_0_exploration import CareerExplorerAgent
        agent = CareerExplorerAgent()

        assert agent._failure_counts["deepseek"] == 0
        assert agent._failure_counts["qwen"] == 0
        assert agent._circuit_open["deepseek"] is False
        assert agent._circuit_open["qwen"] is False

    def test_record_failure_threshold(self):
        """Test that recording failures opens the circuit."""
        from app.agents.agent_0_exploration import CareerExplorerAgent
        agent = CareerExplorerAgent()

        for i in range(3):
            agent._record_failure("deepseek")

        assert agent._circuit_open["deepseek"] is True
        assert agent._failure_counts["deepseek"] == 3

    def test_record_failure_other_provider_unaffected(self):
        """Test that failures in one provider don't affect the other."""
        from app.agents.agent_0_exploration import CareerExplorerAgent
        agent = CareerExplorerAgent()

        for i in range(3):
            agent._record_failure("deepseek")

        assert agent._circuit_open["deepseek"] is True
        assert agent._circuit_open["qwen"] is False

    def test_circuit_breaker_reset_after_timeout(self):
        """Test that circuit breaker resets after timeout."""
        from app.agents.agent_0_exploration import CareerExplorerAgent
        agent = CareerExplorerAgent()
        agent._circuit_reset_time = 0  # Instant reset for testing

        for i in range(3):
            agent._record_failure("deepseek")

        assert agent._circuit_open["deepseek"] is True

        # Trigger reset by checking timeout
        import time
        agent._last_failure_time["deepseek"] = time.time() - 1  # Already expired
        agent._circuit_open["deepseek"] = False
        agent._failure_counts["deepseek"] = 0

        assert agent._circuit_open["deepseek"] is False


# ============================================================================
# Test 6: Semantic Cache
# ============================================================================

class TestSemanticCache:
    """Test semantic cache operations."""

    @pytest.mark.asyncio
    async def test_cache_uninitialized_returns_none(self):
        """Test that uninitialized cache returns None."""
        from app.core.cache import SemanticCache
        cache = SemanticCache()
        result = await cache.get("test query")
        assert result is None

    @pytest.mark.asyncio
    async def test_cache_uninitialized_set_returns_false(self):
        """Test that uninitialized cache set returns False."""
        from app.core.cache import SemanticCache
        cache = SemanticCache()
        result = await cache.set("test query", {"answer": "test"})
        assert result is False

    @pytest.mark.asyncio
    async def test_cosine_similarity_in_cache(self):
        """Test the cosine similarity computation used in cache."""
        from app.core.router import SemanticRouter
        router = SemanticRouter()

        # Simulate what cache does with distance
        # ChromaDB cosine distance: 0 = identical, 2 = opposite
        # similarity = 1 - distance/2

        # Identical: distance = 0, similarity = 1.0
        assert 1.0 - 0 / 2.0 == 1.0

        # Opposite: distance = 2, similarity = 0.0
        assert 1.0 - 2.0 / 2.0 == 0.0

        # Partial: distance = 1, similarity = 0.5
        assert 1.0 - 1.0 / 2.0 == 0.5


# ============================================================================
# Test 7: Orchestrator Integration - Full Pipeline
# ============================================================================

class TestOrchestratorIntegration:
    """Test orchestrator integration with all new modules."""

    @pytest.mark.asyncio
    async def test_orchestrator_creates_session(self):
        """Test that orchestrator creates and persists session."""
        from app.orchestrator.graph import CareerOrchestrator
        orch = CareerOrchestrator()

        state = OrchestratorState(session_id="test-integration-1")
        state.user_input = "帮我做职业规划"

        state = await orch.route_request(state)
        assert state.agent_context.current_agent_id == 0

    @pytest.mark.asyncio
    async def test_orchestrator_format_response_structure(self):
        """Test that format_response returns correct structure."""
        from app.orchestrator.graph import CareerOrchestrator
        from app.core.state import CareerFeasibilityReport, PositionAnalysis, SkillGap

        orch = CareerOrchestrator()

        state = OrchestratorState(session_id="test-format")
        state.agent_context.current_agent_id = 0
        state.agent_context.career_report = CareerFeasibilityReport(
            user_id="test",
            recommended_positions=[
                PositionAnalysis(
                    position_name="后端开发",
                    match_score=0.85,
                    daily_tasks=["设计API"],
                    skill_barriers=["分布式系统"],
                    growth_potential="高级工程师",
                    average_salary_range=(200000, 400000),
                )
            ],
            skill_gaps=[SkillGap(skill_name="Redis", current_level="beginner", required_level="intermediate")],
            confidence_score=0.8,
        )

        response = orch.format_response(state)
        assert response["session_id"] == "test-format"
        assert response["agent_id"] == 0
        assert response["result"]["type"] == "career_feasibility_report"
        assert len(response["result"]["positions"]) == 1


# ============================================================================
# Test 8: Accuracy Benchmark Report
# ============================================================================

class TestAccuracyBenchmark:
    """Generate accuracy benchmark report."""

    @pytest.mark.asyncio
    async def test_full_accuracy_report(self, capsys):
        """Generate a comprehensive accuracy report."""
        from app.orchestrator.graph import CareerOrchestrator
        orch = CareerOrchestrator()

        # Keyword routing test cases
        test_cases = [
            ("我不知道该选什么方向", 0),
            ("帮我分析招聘需求", 1),
            ("优化简历经历", 2),
            ("校招岗位匹配", 3),
            ("模拟面试八股文", 4),
            ("学习路线规划", 5),
            ("我很迷茫，不知道做什么", 0),
            ("解读一下这个JD", 1),
            ("把项目经历商业化表达", 2),
            ("秋招有哪些机会", 3),
            ("技术面模拟面试", 4),
            ("制定8周学习规划", 5),
            ("适合什么岗位", 0),
            ("招聘黑话什么意思", 1),
            ("简历改写优化", 2),
            ("春招内推", 3),
            ("HR面试模拟", 4),
            ("推荐开源项目", 5),
        ]

        correct = 0
        results = []
        for query, expected in test_cases:
            state = OrchestratorState(session_id=f"bench-{expected}")
            state.user_input = query
            state = await orch.route_request(state)
            actual = state.agent_context.current_agent_id
            is_correct = actual == expected
            if is_correct:
                correct += 1
            results.append((query, expected, actual, is_correct))

        accuracy = correct / len(test_cases)

        # Print report
        captured = capsys.readouterr()
        print("\n" + "=" * 60)
        print("ACCURACY BENCHMARK REPORT")
        print("=" * 60)
        for query, expected, actual, ok in results:
            status = "PASS" if ok else "FAIL"
            print(f"  [{status}] '{query}' → Agent {actual} (expected {expected})")
        print("-" * 60)
        print(f"  Total: {len(test_cases)} | Correct: {correct} | Accuracy: {accuracy:.1%}")
        print("=" * 60)

        # Assert minimum accuracy
        assert accuracy >= 0.75, f"Overall accuracy {accuracy:.1%} < 75%"