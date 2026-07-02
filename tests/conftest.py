import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agent import Agent
from app.retriever import Retriever

_EXTRACTION_DEFAULTS = {
    "in_scope": True,
    "prompt_injection": False,
    "intent": "recommend",
    "comparison_targets": [],
    "role_title": None,
    "skills": [],
    "seniority": None,
    "add_test_types": [],
    "remove_test_types": [],
    "max_duration_minutes": None,
    "remote_required": None,
    "num_requested": None,
    "no_preference_signaled": False,
}


class StubLLMClient:
    """Deterministic stand-in for agent.LLMClient -- lets tests drive exact
    extraction output per call without hitting a real API."""

    def __init__(self, extractions=None, compare_answer: str | None = None):
        self._extractions = list(extractions or [])
        self.compare_answer = compare_answer
        self.compare_calls: list[tuple[str, list[str]]] = []
        self.extract_calls = 0

    def extract(self, history):
        self.extract_calls += 1
        if self._extractions:
            override = self._extractions.pop(0)
        else:
            override = {}
        merged = dict(_EXTRACTION_DEFAULTS)
        merged.update(override)
        return merged

    def compare(self, question, snippets):
        self.compare_calls.append((question, snippets))
        return self.compare_answer


@pytest.fixture(scope="session")
def shared_retriever():
    return Retriever()


def make_agent(retriever, extractions=None, compare_answer=None):
    return Agent(retriever=retriever, llm=StubLLMClient(extractions=extractions, compare_answer=compare_answer))
