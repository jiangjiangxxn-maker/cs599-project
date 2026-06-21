"""
Knowledge Graph for Career AI Platform.
Tracks user skills, weaknesses, and relationships as a graph.
Enables cross-agent context injection.
"""
from __future__ import annotations

import json
import os
from typing import Optional
from datetime import datetime


class KnowledgeGraph:
    """
    User skill knowledge graph using JSON persistence.
    Tracks: (user) -[has_skill]-> (skill) -[lacks]-> (topic)
    Enables Agent 5 to automatically address weaknesses from Agent 4.
    """

    def __init__(self):
        self._graph_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "data", "knowledge_graph.json"
        )
        self._graph: dict[str, dict] = {}  # {user_id: {nodes: [...], edges: [...]}}
        self._load_from_disk()

    def _load_from_disk(self):
        """Load graph from JSON file."""
        try:
            if os.path.exists(self._graph_path):
                with open(self._graph_path, "r", encoding="utf-8") as f:
                    self._graph = json.load(f)
                print(f"[KnowledgeGraph] Loaded {len(self._graph)} user graphs")
        except Exception as e:
            print(f"[KnowledgeGraph] Failed to load: {e}")

    def _save_to_disk(self):
        """Save graph to JSON file."""
        try:
            os.makedirs(os.path.dirname(self._graph_path), exist_ok=True)
            with open(self._graph_path, "w", encoding="utf-8") as f:
                json.dump(self._graph, f, ensure_ascii=False, default=str)
        except Exception as e:
            print(f"[KnowledgeGraph] Failed to save: {e}")

    def add_skill(self, user_id: str, skill: str, level: str = "unknown", source: str = "observation"):
        """Add a skill node for a user."""
        if user_id not in self._graph:
            self._graph[user_id] = {"nodes": [], "edges": []}

        # Check if skill already exists
        for node in self._graph[user_id]["nodes"]:
            if node["id"] == f"skill:{skill}":
                node["level"] = level
                node["updated_at"] = datetime.now().isoformat()
                self._save_to_disk()
                return

        self._graph[user_id]["nodes"].append({
            "id": f"skill:{skill}",
            "type": "skill",
            "name": skill,
            "level": level,
            "source": source,
            "created_at": datetime.now().isoformat(),
        })
        self._save_to_disk()

    def add_weakness(self, user_id: str, topic: str, source: str = "interview"):
        """Add a weakness/learning need for a user."""
        if user_id not in self._graph:
            self._graph[user_id] = {"nodes": [], "edges": []}

        # Check if weakness already exists
        for node in self._graph[user_id]["nodes"]:
            if node["id"] == f"weakness:{topic}":
                node["updated_at"] = datetime.now().isoformat()
                self._save_to_disk()
                return

        self._graph[user_id]["nodes"].append({
            "id": f"weakness:{topic}",
            "type": "weakness",
            "name": topic,
            "source": source,
            "created_at": datetime.now().isoformat(),
        })

        # Add edge: user -[lacks]-> weakness
        edge = {
            "source": f"user:{user_id}",
            "target": f"weakness:{topic}",
            "relation": "lacks",
        }
        if edge not in self._graph[user_id]["edges"]:
            self._graph[user_id]["edges"].append(edge)

        self._save_to_disk()

    def get_weaknesses(self, user_id: str) -> list[str]:
        """Get all weaknesses for a user."""
        if user_id not in self._graph:
            return []
        return [
            node["name"]
            for node in self._graph[user_id]["nodes"]
            if node["type"] == "weakness"
        ]

    def get_skills(self, user_id: str) -> list[dict]:
        """Get all skills for a user."""
        if user_id not in self._graph:
            return []
        return [
            {"name": node["name"], "level": node.get("level", "unknown")}
            for node in self._graph[user_id]["nodes"]
            if node["type"] == "skill"
        ]

    def inject_context(self, user_id: str, query: str) -> str:
        """
        Inject relevant context from the knowledge graph into the query.
        Returns a context string to prepend to the agent's prompt.
        """
        weaknesses = self.get_weaknesses(user_id)
        skills = self.get_skills(user_id)

        if not weaknesses and not skills:
            return ""

        context_parts = ["[用户画像]"]

        if skills:
            skill_list = ", ".join([f"{s['name']}({s['level']})" for s in skills[:5]])
            context_parts.append(f"已掌握技能: {skill_list}")

        if weaknesses:
            weakness_list = ", ".join(weaknesses[:5])
            context_parts.append(f"薄弱领域: {weakness_list}")

        return "\n".join(context_parts)


# Global singleton
knowledge_graph = KnowledgeGraph()