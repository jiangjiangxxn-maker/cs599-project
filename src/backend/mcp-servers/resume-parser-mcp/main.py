"""
Resume Parser MCP Server
Parses PDF/DOCX/text resumes into structured JSON format.
Exposes tools for the Career AI Platform agents.
"""
from __future__ import annotations

import json
import os
import re
from typing import Any

from mcp.server import FastMCP, Server
from mcp.server.models import InitializeResult
from mcp.types import Tool, TextContent

# Create MCP server
mcp = FastMCP(
    "resume-parser",
    description="Parse and structure resume content from PDF, DOCX, or plain text",
)


def parse_sections(text: str) -> list[dict]:
    """Parse resume text into structured sections."""
    sections = []
    section_patterns = {
        "education": r"(教育|教育背景|学[历校]|Education|educational background)",
        "experience": r"(工作经历|实习经历|工作[经]验|Experience|work experience)",
        "project": r"(项目[经]?验|项目经历|Projects?|project experience)",
        "skill": r"(技能|技术栈|专业技能|Skills?|technical skills)",
        "summary": r"(个人简介|个人总结|自我评价|Summary|Profile|About me)",
    }

    lines = text.split("\n")
    current_section = "summary"
    current_content = []

    for line in lines:
        line_stripped = line.strip()
        if not line_stripped:
            continue

        matched_section = None
        for section_name, pattern in section_patterns.items():
            if re.search(pattern, line_stripped, re.IGNORECASE):
                matched_section = section_name
                break

        if matched_section:
            if current_content:
                sections.append({
                    "section_type": current_section,
                    "raw_content": "\n".join(current_content).strip(),
                })
                current_content = []
            current_section = matched_section
        else:
            current_content.append(line_stripped)

    # Add last section
    if current_content:
        sections.append({
            "section_type": current_section,
            "raw_content": "\n".join(current_content).strip(),
        })

    return sections


def extract_skills(text: str) -> list[str]:
    """Extract technical skills from resume text."""
    skill_keywords = [
        "Python", "Java", "JavaScript", "TypeScript", "Go", "Rust", "C\\+\\+", "C#",
        "React", "Vue", "Angular", "Spring", "Spring Boot", "Django", "Flask",
        "MySQL", "PostgreSQL", "Redis", "MongoDB", "Elasticsearch",
        "Docker", "Kubernetes", "AWS", "GCP", "Azure",
        "Git", "Linux", "Nginx", "Kafka", "RabbitMQ",
        "TensorFlow", "PyTorch", "Scikit-learn", "Pandas",
        "GraphQL", "REST", "gRPC", "WebSocket",
    ]

    found_skills = []
    text_lower = text.lower()
    for skill in skill_keywords:
        if skill.lower() in text_lower:
            found_skills.append(skill)

    return found_skills


@mcp.tool()
async def parse_resume(text: str, user_id: str = "") -> dict:
    """
    Parse resume text into structured JSON format.
    
    Args:
        text: Raw resume text content (up to 10000 chars)
        user_id: Optional user identifier
    
    Returns:
        Structured resume data with sections and extracted skills
    """
    if not text or not text.strip():
        return {"status": "error", "error": "Empty resume text provided"}

    # Truncate if too long
    text = text[:10000]

    # Parse sections
    sections = parse_sections(text)

    # Extract skills
    skills = extract_skills(text)

    # Basic ATS scoring
    ats_score = min(100.0, len(skills) * 5 + len(sections) * 10)
    if len(text) < 200:
        ats_score = max(ats_score, 30.0)

    result = {
        "status": "ok",
        "data": {
            "sections": sections,
            "skills": skills,
            "ats_score": ats_score,
            "word_count": len(text.split()),
            "has_email": bool(re.search(r'[\w\.-]+@[\w\.-]+\.\w+', text)),
            "has_phone": bool(re.search(r'1[3-9]\d{9}', text)),
        }
    }

    return result


@mcp.tool()
async def validate_ats(resume_text: str) -> dict:
    """
    Validate resume ATS (Applicant Tracking System) compatibility.
    
    Args:
        resume_text: Raw resume text to validate
    
    Returns:
        ATS compatibility score and suggestions
    """
    if not resume_text:
        return {"status": "error", "error": "Empty resume text"}

    sections = parse_sections(resume_text)
    skills = extract_skills(resume_text)
    word_count = len(resume_text.split())

    # Scoring factors
    score = 50.0  # Base score
    suggestions = []

    # Check word count
    if word_count < 300:
        score -= 10
        suggestions.append("简历内容过少，建议扩充到500字以上")
    elif word_count > 2000:
        score -= 5
        suggestions.append("简历过长，建议精简到1000字以内")

    # Check section coverage
    section_types = {s["section_type"] for s in sections}
    required_sections = ["education", "experience", "skill", "project"]
    missing_sections = [s for s in required_sections if s not in section_types]
    if missing_sections:
        score -= len(missing_sections) * 8
        for s in missing_sections:
            suggestions.append(f"缺少 '{s}' 板块")

    # Check quantification
    quantification_patterns = [r'\d+%', r'\d+x', r'\d+倍', r'\d+万', r'\d+人']
    has_quantification = any(re.search(p, resume_text) for p in quantification_patterns)
    if not has_quantification:
        score -= 10
        suggestions.append("缺少量化成果数据，建议加入具体数字指标")

    # Check contact info
    has_email = bool(re.search(r'[\w\.-]+@[\w\.-]+\.\w+', resume_text))
    has_phone = bool(re.search(r'1[3-9]\d{9}', resume_text))
    if not has_email:
        suggestions.append("缺少邮箱联系方式")
    if not has_phone:
        suggestions.append("缺少手机号码")

    # Final score
    score = max(0.0, min(100.0, score))
    if len(skills) > 5:
        score += 10
    if len(skills) > 10:
        score += 5

    return {
        "status": "ok",
        "data": {
            "ats_score": min(100.0, score),
            "word_count": word_count,
            "sections_found": list(section_types),
            "skills_found": skills,
            "suggestions": suggestions[:5],
        }
    }


@mcp.tool()
async def extract_keywords(resume_text: str, target_industry: str = "") -> dict:
    """
    Extract keywords from resume for matching against job descriptions.
    
    Args:
        resume_text: Raw resume text
        target_industry: Optional target industry to filter keywords
    
    Returns:
        Extracted keywords categorized by type
    """
    skills = extract_skills(resume_text)

    # Extract experience level keywords
    experience_keywords = []
    exp_patterns = [
        (r'(\d+)\s*年', "工作经验年数"),
        (r'(实习|intern)', "实习经历"),
        (r'(项目|project)', "项目经验"),
        (r'(开源|open.?source)', "开源贡献"),
    ]
    for pattern, label in exp_patterns:
        if re.search(pattern, resume_text, re.IGNORECASE):
            experience_keywords.append(label)

    # Extract education keywords
    edu_keywords = []
    edu_patterns = [
        (r'(本科|学士|Bachelor)', "本科学历"),
        (r'(硕士|研究生|Master)', "硕士学历"),
        (r'(博士|PhD|Ph\.D)', "博士学历"),
        (r'(985|211|双一流)', "重点院校"),
        (r'(海外|留学|交换)', "海外经历"),
    ]
    for pattern, label in edu_patterns:
        if re.search(pattern, resume_text, re.IGNORECASE):
            edu_keywords.append(label)

    return {
        "status": "ok",
        "data": {
            "technical_skills": skills,
            "experience_indicators": experience_keywords,
            "education_indicators": edu_keywords,
            "total_keywords": len(skills) + len(experience_keywords) + len(edu_keywords),
        }
    }


if __name__ == "__main__":
    port = int(os.getenv("MCP_SERVER_PORT", "8001"))
    print(f"[Resume Parser MCP] Starting on port {port}...")
    mcp.run(transport="sse", port=port)