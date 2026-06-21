"""
MCP Client for interacting with MCP Servers via HTTP.
Implements the Function Calling / MCP Protocol integration layer.
"""
from __future__ import annotations

import json
from typing import Any, Optional

import httpx

from app.core.config import config


class MCPClient:
    """
    HTTP client for MCP servers.
    Each MCP server exposes tools that agents can call via this client.
    """

    def __init__(self):
        self._server_registry: dict[str, str] = {
            "resume_parser": config.mcp.resume_parser_url,
            "campus_calendar": config.mcp.campus_calendar_url,
            "tech_quiz": config.mcp.tech_quiz_url,
            "industry_data": config.mcp.industry_data_url,
            "open_source": config.mcp.open_source_url,
            "interview_eval": config.mcp.interview_eval_url,
            "jd_search": config.mcp.jd_search_url,
        }
        self._http_client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(timeout=2.0)
        return self._http_client

    def _resolve_server(self, tool_name: str) -> tuple[str, str]:
        """
        Resolve a tool name to (server_url, tool_path).
        Format: <server_name>/<tool_name>
        Example: "resume_parser/parse" → ("http://localhost:8001", "parse")
        """
        parts = tool_name.split("/", 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid tool name format: {tool_name}. Expected '<server>/<tool>'")

        server_name, tool_path = parts
        server_url = self._server_registry.get(server_name)
        if not server_url:
            raise ValueError(f"Unknown MCP server: {server_name}. Available: {list(self._server_registry.keys())}")

        return server_url, tool_path

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        """
        Call an MCP tool via HTTP.
        The tool_name should be in format '<server_name>/<tool_name>'.
        Falls back to local execution if MCP server is unavailable.
        """
        client = await self._get_client()

        try:
            server_url, tool_path = self._resolve_server(tool_name)
            url = f"{server_url}/tools/{tool_path}"

            response = await client.post(url, json=arguments)
            response.raise_for_status()
            result = response.json()

            return result.get("data", result)

        except (httpx.ConnectError, httpx.TimeoutException):
            # Fallback to local execution if MCP server is unavailable
            return await self._local_fallback(tool_name, arguments)

        except Exception as e:
            # Try local fallback on error
            try:
                return await self._local_fallback(tool_name, arguments)
            except Exception:
                raise e

    async def _local_fallback(self, tool_name: str, arguments: dict) -> dict:
        """
        Local fallback execution when MCP server is not running.
        Returns mock/simulated results for development.
        """
        print(f"[MCP Client] Local fallback for '{tool_name}' with args: {arguments}")

        server_name = tool_name.split("/")[0]

        if server_name == "resume_parser":
            return {"status": "ok", "data": {"sections": [], "parsed": True}}
        elif server_name == "jd_search":
            return {"status": "ok", "data": {"jd_text": arguments.get("text", ""), "source": "local_fallback"}}
        elif server_name == "campus_calendar":
            return {"status": "ok", "data": {"events": [], "current_season": "autumn"}}
        elif server_name == "tech_quiz":
            return {"status": "ok", "data": {"questions": [], "position_type": arguments.get("position_type", "backend")}}
        elif server_name == "industry_data":
            return {"status": "ok", "data": {"trends": [], "salaries": {}}}
        elif server_name == "open_source":
            return {"status": "ok", "data": {"projects": []}}
        elif server_name == "interview_eval":
            return {"status": "ok", "data": {"score": 7.0, "feedback": "Local evaluation (fallback)"}}
        else:
            return {"status": "ok", "data": {"message": f"Local fallback for {tool_name}"}}

    async def list_available_tools(self) -> list[dict]:
        """List all available tools from all registered MCP servers."""
        tools = []
        for server_name, server_url in self._server_registry.items():
            try:
                client = await self._get_client()
                response = await client.get(f"{server_url}/tools")
                if response.status_code == 200:
                    server_tools = response.json().get("tools", [])
                    for tool in server_tools:
                        tool["server"] = server_name
                        tool["full_name"] = f"{server_name}/{tool.get('name', 'unknown')}"
                    tools.extend(server_tools)
            except Exception:
                # Server not available, add fallback tool entry
                tools.append({
                    "server": server_name,
                    "name": f"{server_name}/local_fallback",
                    "description": f"Local fallback for {server_name}",
                    "full_name": f"{server_name}/local_fallback",
                })
        return tools


# Global singleton
mcp_client = MCPClient()