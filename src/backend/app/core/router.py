"""
Semantic Router for Career AI Platform.
Uses embedding-based similarity to route user queries to the appropriate agent.
Falls back to keyword matching if embedding is unavailable.
"""
from __future__ import annotations

import json
from typing import Optional

from app.core.config import config


# Agent capability descriptions for semantic matching
AGENT_CAPABILITIES = {
    0: "职业方向探索，分析专业背景、技术栈和兴趣，推荐适合的职业方向，岗位可行性分析",
    1: "岗位分析，解读招聘JD中的专业术语，区分硬性要求和加分项，技能差距分析",
    2: "简历优化，把学术化项目经历转化为企业认可的商业化表达，ATS评分，STAR框架改写",
    3: "岗位匹配，校招岗位匹配，投递时间线提醒，秋招春招日历",
    4: "面试模拟，技术面试、项目深挖、HR行为面试，评分反馈，八股文",
    5: "学习规划，制定8周技能提升路线图，推荐开源项目，学习路线设计",
}


class SemanticRouter:
    """
    Semantic router using embedding similarity.
    Routes user queries to the appropriate agent based on semantic meaning.
    """

    def __init__(self):
        self._embeddings = None
        self._initialized = False

    async def initialize(self):
        """Initialize the embedding model."""
        if self._initialized:
            return
        try:
            from langchain_openai import OpenAIEmbeddings
            self._embeddings = OpenAIEmbeddings(
                model="text-embedding-v3",
                api_key=config.llm.qwen_api_key,
                base_url=config.llm.qwen_base_url,
            )
            self._initialized = True
            print("[SemanticRouter] Initialized with Qwen embeddings")
        except Exception as e:
            print(f"[SemanticRouter] Failed to initialize: {e}")

    async def route(self, query: str) -> int:
        """
        Route a query to the best matching agent.
        Returns agent_id (0-5).
        """
        if not self._initialized or self._embeddings is None:
            return -1  # Signal to use keyword fallback

        try:
            # Embed the query and all agent capabilities
            query_embedding = await self._embeddings.aembed_query(query)
            capability_embeddings = await self._embeddings.aembed_documents(
                list(AGENT_CAPABILITIES.values())
            )

            # Compute cosine similarity
            best_agent_id = 0
            best_score = -1.0
            for agent_id, cap_embedding in zip(AGENT_CAPABILITIES.keys(), capability_embeddings):
                score = self._cosine_similarity(query_embedding, cap_embedding)
                if score > best_score:
                    best_score = score
                    best_agent_id = agent_id

            print(f"[SemanticRouter] Best match: Agent {best_agent_id} (score={best_score:.3f})")
            return best_agent_id

        except Exception as e:
            print(f"[SemanticRouter] Routing failed: {e}")
            return -1

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)


# Global singleton
semantic_router = SemanticRouter()