"""
SDD Runtime State Definitions
All state models use Pydantic for runtime validation per SDD spec requirements.
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional, Literal
from pydantic import BaseModel, Field


# ============================================================================
# User Profile Models
# ============================================================================

class Education(BaseModel):
    """User education background"""
    school: str
    major: str
    degree: Literal["bachelor", "master", "phd", "other"]
    graduation_year: int
    university_tier: Optional[Literal["T1", "T2", "T3", "other"]] = None


class Skill(BaseModel):
    """A single skill with proficiency level"""
    name: str
    category: Literal["language", "framework", "tool", "domain_knowledge", "soft_skill"]
    proficiency: Literal["beginner", "intermediate", "advanced", "expert"]
    years_of_experience: int = 0


class Experience(BaseModel):
    """Project or internship experience"""
    title: str
    type: Literal["academic_project", "internship", "personal_project", "open_source", "competition"]
    organization: Optional[str] = None
    description: str
    tech_stack: list[str]
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    highlights: list[str] = []
    quantified_results: list[str] = []


class UserProfile(BaseModel):
    """Complete user profile - SDD runtime validated"""
    user_id: str
    education: Optional[Education] = None
    tech_stack: list[Skill] = []
    interests: list[str] = []
    experience: list[Experience] = []
    target_positions: list[str] = []
    career_stage: Literal["freshman", "sophomore", "junior", "senior", "grad", "intern"] = "senior"
    github_url: Optional[str] = None
    linkedin_url: Optional[str] = None


# ============================================================================
# Agent 0 - Career Exploration Models
# ============================================================================

class PositionAnalysis(BaseModel):
    """Analysis of a single position"""
    position_name: str
    match_score: float = Field(ge=0.0, le=1.0)
    daily_tasks: list[str]
    skill_barriers: list[str]
    growth_potential: str
    average_salary_range: tuple[int, int] = (0, 0)


class SkillGap(BaseModel):
    """Identified skill gap"""
    skill_name: str
    current_level: Literal["beginner", "intermediate", "advanced"]
    required_level: Literal["beginner", "intermediate", "advanced"]
    suggested_resources: list[str] = []


class CareerFeasibilityReport(BaseModel):
    """Agent 0 output"""
    user_id: str
    recommended_positions: list[PositionAnalysis]
    skill_gaps: list[SkillGap]
    confidence_score: float = Field(ge=0.0, le=1.0)
    exploration_timestamp: datetime = Field(default_factory=datetime.now)


# ============================================================================
# Agent 1 - Job Analysis Models
# ============================================================================

class ParsedRequirement(BaseModel):
    """A single requirement from a JD"""
    original_text: str
    translated_text: str
    category: Literal["hard_filter", "nice_to_have", "responsibility"]
    importance: int = Field(ge=1, le=5)
    is_user_qualified: Optional[bool] = None


class GapItem(BaseModel):
    """Skill gap item with recommended action"""
    skill: str
    current_status: str
    recommended_action: str
    urgency: Literal["immediate", "short-term", "long-term"]


class JDAnalysisReport(BaseModel):
    """Agent 1 output"""
    jd_raw_text: str
    parsed_requirements: list[ParsedRequirement]
    gap_analysis: list[GapItem]
    overall_difficulty: Literal["low", "medium", "high"]
    suggestion: str


# ============================================================================
# Agent 2 - Resume Optimization Models
# ============================================================================

class STARContent(BaseModel):
    """STAR-structured experience"""
    situation: str
    task: str
    action: str
    result: str


class ResumeSection(BaseModel):
    """A section from a resume"""
    section_type: Literal["education", "experience", "project", "skill", "summary"]
    raw_content: str
    star_structured: Optional[STARContent] = None


class ComparisonBlock(BaseModel):
    """Before/after comparison for a single item"""
    original: str
    optimized: str
    transformation_type: str  # e.g., "academic_to_commercial", "vague_to_quantified"


class ResumeOptimizationResult(BaseModel):
    """Agent 2 output"""
    original_sections: list[ResumeSection]
    optimized_sections: list[ResumeSection]
    comparison_blocks: list[ComparisonBlock]
    keywords_matched: list[str]
    ats_score: float = Field(ge=0.0, le=100.0)
    suggestions: list[str]


# ============================================================================
# Agent 3 - Job Matching Models
# ============================================================================

class CampusEvent(BaseModel):
    """Campus recruitment event"""
    event_name: str
    recruitment_type: Literal["early_autumn", "autumn", "spring", "intern_conversion", "daily_intern"]
    start_date: datetime
    end_date: datetime
    status: Literal["upcoming", "ongoing", "ended"]


class DeadlineReminder(BaseModel):
    """Application deadline reminder"""
    position_id: str
    company: str
    deadline: datetime
    urgency: Literal["critical", "upcoming", "normal"]
    reminder_message: str


class MatchedPosition(BaseModel):
    """A matched position"""
    company_name: str
    position_name: str
    recruitment_type: Literal["early_autumn", "autumn", "spring", "intern_conversion", "daily_intern"]
    match_score: float = Field(ge=0.0, le=1.0)
    application_url: str = ""
    deadline: datetime
    days_remaining: int


class JobMatchingResult(BaseModel):
    """Agent 3 output"""
    matched_positions: list[MatchedPosition]
    campus_calendar: list[CampusEvent]
    application_reminders: list[DeadlineReminder]
    overall_strategy: str


# ============================================================================
# Agent 4 - Interview Simulation Models
# ============================================================================

class QAItem(BaseModel):
    """Single Q&A item in interview"""
    question: str
    question_type: Literal["fundamental", "algorithm", "system_design", "project", "behavior"]
    user_answer: str = ""
    model_answer: str = ""
    score: float = Field(ge=0.0, le=10.0, default=0.0)
    feedback: str = ""
    follow_up_question: Optional[str] = None


class InterviewEvaluation(BaseModel):
    """Overall interview evaluation"""
    technical_depth: float = Field(ge=0.0, le=10.0)
    communication_clarity: float = Field(ge=0.0, le=10.0)
    problem_solving: float = Field(ge=0.0, le=10.0)
    overall: float = Field(ge=0.0, le=10.0)
    suggestions: list[str]


class InterviewSimulationSession(BaseModel):
    """Agent 4 output"""
    user_id: str
    position_type: str
    mode: Literal["technical", "project_deep_dive", "hr"]
    qa_history: list[QAItem]
    overall_score: float = Field(ge=0.0, le=10.0)
    strengths: list[str]
    weaknesses: list[str]
    improvement_plan: list[str]


# ============================================================================
# Agent 5 - Learning Planning Models
# ============================================================================

class WeeklyTask(BaseModel):
    """Weekly learning task"""
    week_number: int
    topic: str
    learning_objectives: list[str]
    resources: list[str] = []
    practice_exercises: list[str] = []
    estimated_hours: int


class RecommendedProject(BaseModel):
    """Project recommendation"""
    project_name: str
    description: str
    tech_stack: list[str]
    difficulty: Literal["beginner", "intermediate", "advanced"]
    github_url: Optional[str] = None
    resume_value: str


class Milestone(BaseModel):
    """Learning milestone"""
    week: int
    checkpoint_name: str
    validation_criteria: str
    status: Literal["pending", "in_progress", "completed"] = "pending"


class LearningPlan(BaseModel):
    """Agent 5 output"""
    user_id: str
    target_position: str
    current_level: str
    estimated_duration_weeks: int
    weekly_plan: list[WeeklyTask]
    recommended_projects: list[RecommendedProject]
    milestone_checkpoints: list[Milestone]


# ============================================================================
# Orchestrator State (LangGraph)
# ============================================================================

class AgentContext(BaseModel):
    """Context passed between agents"""
    current_agent_id: int
    user_profile: Optional[UserProfile] = None
    career_report: Optional[CareerFeasibilityReport] = None
    jd_report: Optional[JDAnalysisReport] = None
    resume_report: Optional[ResumeOptimizationResult] = None
    job_matching: Optional[JobMatchingResult] = None
    interview_session: Optional[InterviewSimulationSession] = None
    learning_plan: Optional[LearningPlan] = None
    conversation_history: list[dict] = []
    errors: list[str] = []

    # Interview loop control
    is_interviewing: bool = False
    interview_total: int = 0
    interview_current: int = 0
    interview_questions: list[str] = []

    def reset_for_new_request(self) -> None:
        """
        Reset all per-request state fields to prevent leakage between turns.
        Preserves user_profile (long-term user data).
        """
        self.current_agent_id = -1
        self.is_interviewing = False
        self.interview_total = 0
        self.interview_current = 0
        self.interview_questions = []
        self.errors = []
        # Clear agent outputs (results from previous turns)
        self.career_report = None
        self.jd_report = None
        self.resume_report = None
        self.job_matching = None
        self.interview_session = None
        self.learning_plan = None


class OrchestratorState(BaseModel):
    """Full orchestrator state for LangGraph"""
    session_id: str
    user_input: str = ""
    agent_context: AgentContext = Field(default_factory=lambda: AgentContext(current_agent_id=-1))
    pipeline_complete: bool = False
    current_node: str = "start"

    def reset_for_new_request(self) -> None:
        """Reset state for a new user request within the same session."""
        self.pipeline_complete = False
        self.current_node = "start"
        self.agent_context.reset_for_new_request()