"""Pydantic schemas: the public API contract (models.py -> api.py) and the
internal catalog record shape (produced by scraper.py, consumed by retriever.py).
"""
from typing import Literal, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Public API contract. Field names/shape here are dictated by the assignment
# spec and must not change -- the automated evaluator depends on them exactly.
# ---------------------------------------------------------------------------

class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(default_factory=list)


class RecommendationItem(BaseModel):
    name: str
    url: str
    test_type: str = ""


class ChatResponse(BaseModel):
    reply: str
    recommendations: list[RecommendationItem] = Field(default_factory=list)
    end_of_conversation: bool = False


class HealthResponse(BaseModel):
    status: str = "ok"


# ---------------------------------------------------------------------------
# Internal catalog record. One per SHL Individual Test Solution.
# ---------------------------------------------------------------------------

TEST_TYPE_LABELS = {
    "A": "Ability & Aptitude",
    "B": "Biodata & Situational Judgement",
    "C": "Competencies",
    "D": "Development & 360",
    "E": "Assessment Exercises",
    "K": "Knowledge & Skills",
    "P": "Personality & Behavior",
    "S": "Simulations",
}


class Assessment(BaseModel):
    id: str
    name: str
    url: str
    description: str = ""
    test_type: str = ""
    test_type_label: str = ""
    duration_minutes: Optional[int] = None
    remote_testing: bool = False
    adaptive_irt: bool = False
    job_levels: list[str] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)
    languages: list[str] = Field(default_factory=list)
    source: str = ""
    current_redirect_target: str = ""  # where `url` currently 301s to; informational only
