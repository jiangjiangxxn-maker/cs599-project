"""
Base Agent class for Career AI Platform.
Provides common LLM integration, ReAct loop, and tool calling infrastructure.
All specialized agents inherit from this base.

LLM Provider: Qwen only (通义千问 via DashScope OpenAI-compatible API)
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any, Optional

from app.core.config import config
from app.core.state import OrchestratorState


class BaseAgent(ABC):
    """
    Abstract base agent implementing the ReAct pattern:
    Think -> Act (call tools) -> Observe -> Repeat -> Final
    """

    def __init__(self, agent_id: int, name: str, llm_provider: str = ""):
        self.agent_id = agent_id
        self.name = name
        self.llm_provider = llm_provider or config.default_agent_llm
        self._llm = None
        # MoE routing: map agent to preferred provider
        # reasoning_agents use DeepSeek, text_agents use Qwen
        self._provider_config = self._resolve_provider()
        # Circuit breaker: track failures per provider
        self._failure_counts: dict[str, int] = {"deepseek": 0, "qwen": 0}
        self._circuit_open: dict[str, bool] = {"deepseek": False, "qwen": False}
        self._circuit_threshold = 3  # Open circuit after 3 consecutive failures
        self._circuit_reset_time = 60  # Reset after 60 seconds
        self._last_failure_time: dict[str, float] = {"deepseek": 0, "qwen": 0}

    @abstractmethod
    async def process(self, state: OrchestratorState) -> OrchestratorState:
        """
        Main entry point for agent processing.
        Implements the ReAct loop internally.
        Must return updated OrchestratorState.
        """
        pass

    def _resolve_provider(self) -> dict:
        """Resolve which LLM provider to use based on agent type (MoE routing)."""
        # Reasoning-heavy agents → DeepSeek
        reasoning_agents = {
            1,  # JD Translator - complex requirement parsing
            3,  # Job Matcher - matching logic
            4,  # Interviewer - technical question generation
        }
        # Text polish agents → Qwen
        text_agents = {
            0,  # Career Explorer - report generation
            2,  # Resume Wrapper - text rewriting
            5,  # Learning Planner - plan generation
        }

        if self.agent_id in reasoning_agents and config.llm.deepseek_api_key:
            return {
                "provider": "deepseek",
                "api_key": config.llm.deepseek_api_key,
                "model": config.llm.deepseek_model,
                "base_url": config.llm.deepseek_base_url,
                "temperature": 0.3,
            }
        elif self.agent_id in text_agents and config.llm.qwen_api_key:
            return {
                "provider": "qwen",
                "api_key": config.llm.qwen_api_key,
                "model": config.llm.qwen_model,
                "base_url": config.llm.qwen_base_url,
                "temperature": 0.7,
            }
        else:
            # Fallback: use whichever provider is available
            if config.llm.deepseek_api_key:
                return {
                    "provider": "deepseek",
                    "api_key": config.llm.deepseek_api_key,
                    "model": config.llm.deepseek_model,
                    "base_url": config.llm.deepseek_base_url,
                    "temperature": 0.3,
                }
            return {
                "provider": "qwen",
                "api_key": config.llm.qwen_api_key,
                "model": config.llm.qwen_model,
                "base_url": config.llm.qwen_base_url,
                "temperature": 0.7,
            }

    async def _get_llm(self):
        """Lazy-initialize LLM with MoE-style provider routing and circuit breaker."""
        if self._llm is not None:
            return self._llm

        prov = self._provider_config
        provider = prov.get("provider", "qwen")

        # Check circuit breaker
        if self._circuit_open.get(provider, False):
            import time
            if time.time() - self._last_failure_time.get(provider, 0) < self._circuit_reset_time:
                print(f"[{self.name}] Circuit open for {provider}, trying fallback...")
                # Try the other provider
                fallback = "qwen" if provider == "deepseek" else "deepseek"
                if fallback == "qwen" and config.llm.qwen_api_key:
                    prov = {
                        "provider": "qwen",
                        "api_key": config.llm.qwen_api_key,
                        "model": config.llm.qwen_model,
                        "base_url": config.llm.qwen_base_url,
                        "temperature": 0.7,
                    }
                elif fallback == "deepseek" and config.llm.deepseek_api_key:
                    prov = {
                        "provider": "deepseek",
                        "api_key": config.llm.deepseek_api_key,
                        "model": config.llm.deepseek_model,
                        "base_url": config.llm.deepseek_base_url,
                        "temperature": 0.3,
                    }
                else:
                    print(f"[{self.name}] No fallback available. Using mock mode.")
                    return self._get_mock_llm()
            else:
                # Reset circuit breaker
                self._circuit_open[provider] = False
                self._failure_counts[provider] = 0

        if not prov.get("api_key"):
            print(f"[{self.name}] No API key for {prov['provider']}. Using mock mode.")
            return self._get_mock_llm()

        try:
            from langchain_openai import ChatOpenAI

            self._llm = ChatOpenAI(
                model=prov["model"],
                api_key=prov["api_key"],
                base_url=prov["base_url"],
                temperature=prov["temperature"],
            )
            print(f"[{self.name}] Using LLM: {prov['provider']}/{prov['model']}")
            return self._llm
        except Exception as e:
            print(f"[{self.name}] LLM init failed ({e}). Falling back to mock mode.")
            return self._get_mock_llm()

    def _record_failure(self, provider: str):
        """Record a failure for circuit breaker tracking."""
        import time
        self._failure_counts[provider] = self._failure_counts.get(provider, 0) + 1
        self._last_failure_time[provider] = time.time()
        if self._failure_counts[provider] >= self._circuit_threshold:
            self._circuit_open[provider] = True
            print(f"[{self.name}] Circuit breaker OPEN for {provider} after {self._circuit_threshold} failures")

    def _get_api_key(self) -> str:
        """Get the Qwen API key."""
        return config.llm.qwen_api_key

    def _get_mock_llm(self):
        """
        Return a mock LLM that returns structured responses without API calls.
        Uses duck-typed object instead of BaseLanguageModel inheritance to avoid
        abstract method requirements in newer langchain-core versions.
        """
        import asyncio
        import re

        class MockResponse:
            """Simulates an LLM response with .content attribute."""
            def __init__(self, content: str):
                self.content = content

        _captured_agent_id = self.agent_id

        class MockLLM:
            """Lightweight mock LLM with duck-typed interface."""

            def __init__(self):
                self._model = "mock-llm"
                self.agent_id = _captured_agent_id

            async def ainvoke(self, prompt, **kwargs):
                """Async invoke - returns mock response."""
                return await self._mock_response(prompt)

            def invoke(self, prompt, **kwargs):
                """Sync invoke - returns mock response."""
                return asyncio.run(self._mock_response(prompt))

            def bind_tools(self, tools, **kwargs):
                """bind_tools is a no-op for mock."""
                return self

            async def _mock_response(self, prompt):
                """
                Return a single valid JSON string with all possible fields that any Agent might need.
                Uses the agent_id to return role-appropriate mock data.
                """
                import json
                import re

                # Determine which agent field data to return based on self.agent_id
                _agent_id = self.agent_id

                base_mock = {
                    "message": "Mock response: No API key configured. Set QWEN_API_KEY in .env for real results.",
                }

                # Agent 0: Career Exploration
                if _agent_id == 0:
                    mock_data = {
                        **base_mock,
                        "recommended_positions": [
                            {"position_name": "后端开发工程师", "match_score": 0.85, "daily_tasks": ["设计 RESTful API", "数据库设计与优化", "编写单元测试"], "skill_barriers": ["分布式系统设计", "高并发处理", "微服务架构"], "growth_potential": "3-5年成长为高级工程师", "average_salary_range": [200000, 400000]}
                        ],
                        "skill_gaps": [{"skill_name": "Spring Boot", "current_level": "beginner", "required_level": "intermediate"}],
                        "confidence_score": 0.80,
                        "summary": "建议重点准备后端开发方向",
                    }
                elif _agent_id == 1:
                    mock_data = {**base_mock, "parsed_requirements": [{"original_text": "熟练掌握Java及Spring框架", "translated_text": "需要会Java基础和Spring Boot使用经验", "category": "hard_filter", "importance": 4}], "gap_analysis": [{"skill": "分布式系统", "current_status": "理论基础", "recommended_action": "通过实战项目积累经验", "urgency": "short-term"}], "overall_difficulty": "medium", "suggestion": "建议补充微服务和中间件相关知识"}
                elif _agent_id == 2:
                    mock_data = {**base_mock, "comparison_blocks": [{"original": "完成了一个数据库课程作业", "optimized": "独立设计并部署分布式KV存储集群，基于Raft协议实现强一致性，支撑10W+ QPS读写请求", "transformation_type": "academic_to_commercial"}], "keywords_matched": ["分布式系统", "高并发", "Raft协议"], "ats_score": 78.0, "suggestions": ["添加更多量化成果数据", "补充开源项目经验"]}
                elif _agent_id == 3:
                    mock_data = {**base_mock, "matched_positions": [{"company_name": "字节跳动", "position_name": "后端开发实习生", "recruitment_type": "daily_intern", "match_score": 0.82, "deadline": "2026-09-30T00:00:00", "days_remaining": 30}, {"company_name": "腾讯", "position_name": "后台开发实习生", "recruitment_type": "daily_intern", "match_score": 0.78, "deadline": "2026-10-15T00:00:00", "days_remaining": 45}], "application_reminders": [{"company": "字节跳动", "deadline": "2026-09-30T00:00:00", "urgency": "upcoming", "reminder_message": "距离截止还有30天"}], "overall_strategy": "建议优先投递大厂后端实习岗位"}
                elif _agent_id == 4:
                    # Check prompt content to determine which phase
                    prompt_lower = prompt.lower() if isinstance(prompt, str) else ""
                    is_phase_a = "opening" in prompt_lower or "questions" in prompt_lower
                    if is_phase_a:
                        # Phase A: Initialize — return opening + 3 questions
                        mock_data = {**base_mock, "opening": "本次后端开发模拟面试共计 3 个问题，请准备好后开始。问题 1/3：请解释 Java 中 HashMap 的实现原理。", "questions": [{"question": "请解释 Java 中 HashMap 的实现原理", "question_type": "fundamental", "model_answer": "HashMap基于数组+链表+红黑树实现。"}, {"question": "ConcurrentHashMap 如何保证线程安全？", "question_type": "fundamental", "model_answer": "ConcurrentHashMap采用分段锁和CAS操作。"}, {"question": "Redis 和 Memcached 有什么区别？", "question_type": "system_design", "model_answer": "Redis支持丰富数据类型和持久化。"}]}
                    else:
                        # Phase B/C: Return feedback + next question or final report
                        is_last = "final_report" in prompt_lower or "最后一题" in prompt_lower
                        if is_last:
                            mock_data = {**base_mock, "feedback": "回答完毕。整体表现不错。", "score": 7.0, "final_report": {"overall_score": 7.0, "strengths": ["基础知识扎实", "学习能力强"], "weaknesses": ["项目经验不足"], "improvement_plan": ["参与开源项目"]}}
                        else:
                            mock_data = {**base_mock, "feedback": "回答思路清晰。", "score": 7.5, "next_question": "问题 2/3：ConcurrentHashMap 如何保证线程安全？"}
                elif _agent_id == 5:
                    mock_data = {**base_mock, "weekly_plan": [{"week_number": 1, "topic": "Spring Boot核心原理与REST API设计", "learning_objectives": ["掌握依赖注入原理"], "estimated_hours": 10}, {"week_number": 2, "topic": "数据库设计与MyBatis/JPA整合", "learning_objectives": ["掌握MySQL索引优化"], "estimated_hours": 12}], "recommended_projects": [{"project_name": "Mini-Spring Boot 电商系统", "description": "构建完整电商后端", "tech_stack": ["Spring Boot", "MySQL", "Redis"], "difficulty": "intermediate", "resume_value": "展示全栈后端开发能力"}], "estimated_duration_weeks": 8, "current_level": "入门", "target_position": "后端开发工程师"}
                else:
                    mock_data = base_mock
                return MockResponse(content=json.dumps(mock_data, ensure_ascii=False))

        self._llm = MockLLM()
        return self._llm

    async def _think(self, prompt: str, tools: Optional[list[dict]] = None) -> str:
        """
        ReAct Think step: send prompt to LLM and get response.
        Can include tool definitions for function calling.
        Returns a clean string, stripping markdown code fences if present.
        """
        llm = await self._get_llm()

        if tools:
            llm_with_tools = llm.bind_tools(tools)
            response = await llm_with_tools.ainvoke(prompt)
        else:
            response = await llm.ainvoke(prompt)

        raw = response.content if hasattr(response, 'content') else str(response)

        # Strip markdown JSON code fences if present
        import re
        json_fence = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', raw, re.DOTALL)
        if json_fence:
            return json_fence.group(1).strip()

        return raw.strip()

    def _safe_parse_json(self, text: str) -> dict:
        """
        Safely parse JSON from LLM response text.
        Handles multiple JSON objects, trailing text, and markdown fences.
        Returns parsed dict or raises ValueError if no valid JSON found.
        """
        import re

        text = text.strip()

        # Try direct parse first
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try to find and parse a single JSON object
        # Match from the first { to its matching }
        brace_depth = 0
        json_start = -1
        for i, ch in enumerate(text):
            if ch == '{':
                if json_start == -1:
                    json_start = i
                brace_depth += 1
            elif ch == '}':
                brace_depth -= 1
                if brace_depth == 0 and json_start != -1:
                    candidate = text[json_start:i + 1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        json_start = -1

        # Try with markdown fences stripped
        json_fence = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
        if json_fence:
            try:
                return json.loads(json_fence.group(1))
            except json.JSONDecodeError:
                pass

        raise ValueError(f"Could not parse JSON from: {text[:200]}...")

    async def _act(self, tool_name: str, arguments: dict) -> Any:
        """
        ReAct Act step: execute a tool call.
        Delegates to the MCP client for external tool execution.
        """
        from app.core.mcp_client import mcp_client
        return await mcp_client.call_tool(tool_name, arguments)

    async def _observe(self, result: Any) -> str:
        """
        ReAct Observe step: format tool result back into text for LLM.
        """
        if isinstance(result, str):
            return result
        try:
            return json.dumps(result, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            return str(result)

    def _log(self, state: OrchestratorState, message: str):
        """Add a log entry to the conversation history."""
        state.agent_context.conversation_history.append({
            "agent": self.name,
            "agent_id": self.agent_id,
            "message": message,
        })

    def _log_error(self, state: OrchestratorState, error: str):
        """Add an error to the state."""
        state.agent_context.errors.append(f"[{self.name}] {error}")
        self._log(state, f"Error: {error}")