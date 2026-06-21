"""
Central configuration for the Career AI Platform.
Supports environment variable overrides for production deployment.
Hybrid Model Routing: DeepSeek for reasoning, Qwen for text polish.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

from dotenv import load_dotenv

# Load .env file from backend/ directory
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), ".env"))


@dataclass
class LLMProviderConfig:
    """Configuration for a single LLM provider"""
    api_key: str = ""
    model: str = ""
    base_url: str = ""
    temperature: float = 0.7
    max_tokens: int = 4096


@dataclass
class LLMConfig:
    """
    Multi-provider LLM configuration (MoE-style routing).
    - DeepSeek: reasoning-heavy tasks (JD analysis, interview simulation)
    - Qwen: text polish tasks (resume optimization, learning planning)
    - Default: configurable fallback
    """
    # DeepSeek (reasoning-heavy)
    deepseek_api_key: str = os.getenv("DEEPSEEK_API_KEY", "")
    deepseek_model: str = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
    deepseek_base_url: str = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

    # Qwen (text polish)
    qwen_api_key: str = os.getenv("QWEN_API_KEY", "")
    qwen_model: str = os.getenv("QWEN_MODEL", "qwen-plus")
    qwen_base_url: str = os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")


@dataclass
class MemoryConfig:
    """Memory/Vector DB configuration"""
    vector_db_type: str = os.getenv("VECTOR_DB_TYPE", "chromadb")
    chromadb_path: str = os.getenv("CHROMADB_PATH", "./data/chromadb")
    collection_name: str = os.getenv("CHROMADB_COLLECTION", "career_memories")


@dataclass
class TracingConfig:
    """Observability / Langfuse configuration"""
    langfuse_public_key: str = os.getenv("LANGFUSE_PUBLIC_KEY", "")
    langfuse_secret_key: str = os.getenv("LANGFUSE_SECRET_KEY", "")
    langfuse_host: str = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
    enable_tracing: bool = os.getenv("ENABLE_TRACING", "false").lower() == "true"


@dataclass
class MCPConfig:
    """MCP Server connection configurations"""
    resume_parser_url: str = os.getenv("RESUME_PARSER_MCP_URL", "http://localhost:8001")
    campus_calendar_url: str = os.getenv("CAMPUS_CALENDAR_MCP_URL", "http://localhost:8002")
    tech_quiz_url: str = os.getenv("TECH_QUIZ_MCP_URL", "http://localhost:8003")
    industry_data_url: str = os.getenv("INDUSTRY_DATA_MCP_URL", "http://localhost:8004")
    open_source_url: str = os.getenv("OPEN_SOURCE_MCP_URL", "http://localhost:8005")
    interview_eval_url: str = os.getenv("INTERVIEW_EVAL_MCP_URL", "http://localhost:8006")
    jd_search_url: str = os.getenv("JD_SEARCH_MCP_URL", "http://localhost:8007")


@dataclass
class AppConfig:
    """Top-level application configuration"""
    app_name: str = "Career AI Platform"
    version: str = "0.1.0"
    debug: bool = os.getenv("DEBUG", "true").lower() == "true"
    secret_key: str = os.getenv("SECRET_KEY", "dev-secret-key-change-in-production")

    # Sub-configs
    llm: LLMConfig = field(default_factory=LLMConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    tracing: TracingConfig = field(default_factory=TracingConfig)
    mcp: MCPConfig = field(default_factory=MCPConfig)

    # Server
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "8000"))

    # Agent defaults - always qwen
    default_agent_llm: str = os.getenv("DEFAULT_AGENT_LLM", "qwen")


# Global singleton
config = AppConfig()