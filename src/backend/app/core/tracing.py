"""
Langfuse tracing integration for Career AI Platform.
Provides trace context management and decorators for observability.
"""
from __future__ import annotations

import time
import json
import uuid
from typing import Optional, Callable, Any
from functools import wraps

from app.core.config import config


class TracingManager:
    """
    Manages Langfuse trace contexts.
    Falls back to no-op if Langfuse is not configured.
    """

    def __init__(self):
        self._langfuse = None
        self._initialized = False
        self._traces: dict[str, dict] = {}  # {trace_id: trace_data}
        self._spans: dict[str, dict] = {}   # {span_id: span_data}

    async def initialize(self):
        """Initialize Langfuse client."""
        if self._initialized:
            return
        if not config.tracing.enable_tracing:
            print("[Tracing] Tracing disabled (set ENABLE_TRACING=true to enable)")
            self._initialized = True
            return

        pk = config.tracing.langfuse_public_key
        sk = config.tracing.langfuse_secret_key
        if not pk or not sk:
            print("[Tracing] Langfuse keys not configured. Tracing disabled.")
            self._initialized = True
            return

        try:
            from langfuse import Langfuse
            self._langfuse = Langfuse(
                public_key=pk,
                secret_key=sk,
                host=config.tracing.langfuse_host,
                release="0.1.0",
            )
            self._initialized = True
            print(f"[Tracing] Langfuse initialized (host: {config.tracing.langfuse_host})")
        except ImportError:
            print("[Tracing] langfuse package not installed. Install with: pip install langfuse")
            self._initialized = True
        except Exception as e:
            print(f"[Tracing] Failed to initialize Langfuse: {e}")
            self._initialized = True

    def _noop_generator(self, name: str, **kwargs):
        """Return a no-op trace context manager when Langfuse is unavailable."""
        class NoopSpan:
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def end(self): pass
            def update(self, **kwargs): pass
            def set_input(self, v): pass
            def set_output(self, v): pass
            def generation(self, **kwargs): return NoopSpan()
        return NoopSpan()

    def trace(self, name: str, session_id: str = "", metadata: dict = None):
        """
        Create a trace or span context.
        Returns a context manager.
        """
        if not self._initialized or self._langfuse is None:
            return self._noop_generator(name)

        try:
            trace_id = str(uuid.uuid4())
            self._traces[trace_id] = {
                "name": name,
                "session_id": session_id,
                "start_time": time.time(),
                "metadata": metadata or {},
            }

            trace = self._langfuse.trace(
                id=trace_id,
                name=name,
                session_id=session_id,
                metadata=metadata or {},
            )

            class TraceSpan:
                def __init__(self, langfuse_trace, trace_id):
                    self._trace = langfuse_trace
                    self._trace_id = trace_id
                    self._span = None

                def __enter__(self):
                    return self

                def __exit__(self, exc_type, exc_val, exc_tb):
                    if exc_val:
                        if self._span:
                            self._span.end(output={"error": str(exc_val)})
                        else:
                            self._trace.update(output={"error": str(exc_val)})
                    else:
                        if self._span:
                            self._span.end()
                    del self

                def span(self, span_name: str, **kwargs):
                    span = self._trace.span(name=span_name, **kwargs)
                    self._span = span
                    return span

            return TraceSpan(trace, trace_id)

        except Exception as e:
            print(f"[Tracing] Failed to create trace: {e}")
            return self._noop_generator(name)

    async def record_llm_call(
        self,
        trace_id: str,
        provider: str,
        model: str,
        prompt: str,
        response: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        duration_ms: float = 0,
        error: Optional[str] = None,
    ):
        """Record an LLM generation in the trace."""
        if not self._initialized or self._langfuse is None:
            return

        try:
            generation = self._langfuse.generation(
                name=f"llm_{provider}",
                model=f"{provider}/{model}",
                model_parameters={"temperature": 0.7},
                input=prompt[:2000],
                output=response[:2000] if response else "",
                usage={"input": input_tokens, "output": output_tokens, "unit": "TOKENS"},
                metadata={
                    "provider": provider,
                    "duration_ms": duration_ms,
                    "error": error,
                },
            )
        except Exception as e:
            print(f"[Tracing] Failed to record LLM call: {e}")

    def get_trace_stats(self) -> dict:
        """Get in-memory trace statistics for health endpoint."""
        active = len(self._traces)
        total_spans = sum(
            1 for t in self._traces.values()
            if t.get("end_time", 0) == 0
        )
        return {
            "active_traces": active,
            "total_spans": total_spans,
            "enabled": self._initialized and self._langfuse is not None,
        }


# Global singleton
tracing_manager = TracingManager()