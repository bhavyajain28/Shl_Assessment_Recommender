"""The conversational agent.

Design (see approach.md for full rationale): the LLM is used for exactly two
things -- (1) structured extraction of facts/intent from the conversation,
and (2) grounded comparison synthesis. Every branch decision (ask vs.
retrieve vs. answer vs. refuse) is explicit Python control flow over those
extracted facts, not left to an LLM's free-form judgement. This keeps the
agent debuggable, unit-testable without a live API key, and resistant to
prompt injection: injected text can influence what the *extraction* call
reports, but it can never make the agent skip a refusal, since scope/
injection is also checked with regexes before the LLM is even called, and
the LLM's own verdict is combined with, not substituted for, that check.

The whole module is stateless: handle_chat() re-derives every constraint
from the full message history on every call. That single property is what
makes refinement ("Actually, add personality tests") "just work" -- the
next call re-extracts the updated constraint set and re-retrieves, rather
than needing an explicit "edit the previous shortlist" code path.
"""
from __future__ import annotations

import json
import logging
import re

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from app import prompts
from app.config import get_settings
from app.models import ChatMessage, ChatResponse, RecommendationItem
from app.retriever import Retriever

logger = logging.getLogger(__name__)

# --- Guardrails: fast, LLM-independent first pass. Injection detection in
# particular must not depend on the LLM, since injected text is exactly what
# would be trying to manipulate that LLM call. ---
_INJECTION_PATTERNS = [
    r"ignore (all |any |the )?(previous|prior|above) instructions",
    r"disregard (all |any |the )?(previous|prior|above)",
    r"you are now",
    r"act as (?!.*(recruiter|hiring manager))",  # "act as a recruiter" is fine; "act as DAN" is not
    r"reveal (your |the )?(system prompt|instructions)",
    r"what (is|are) your (system prompt|instructions)",
    r"pretend (you|to be)",
    r"developer mode",
    r"jailbreak",
    r"\bDAN\b",
    r"forget (everything|all) (you|above)",
]
_INJECTION_RE = re.compile("|".join(_INJECTION_PATTERNS), re.IGNORECASE)

_OFFTOPIC_PATTERNS = [
    r"legal advice", r"can i fire", r"how do i fire", r"sue (my|the) (employer|employee|company)",
    r"weather", r"stock price", r"who (won|wins)", r"write me a (poem|song|essay)",
    r"general hiring advice", r"how (should|do) i interview",
]
_OFFTOPIC_RE = re.compile("|".join(_OFFTOPIC_PATTERNS), re.IGNORECASE)

_VALID_TEST_TYPES = {"A", "B", "C", "D", "E", "K", "P", "S"}


def _regex_guardrail(latest_user_text: str) -> tuple[bool, bool]:
    """Returns (is_injection, is_offtopic) from fast regex checks alone."""
    return bool(_INJECTION_RE.search(latest_user_text)), bool(_OFFTOPIC_RE.search(latest_user_text))


class LLMClient:
    def __init__(self) -> None:
        settings = get_settings()
        self._settings = settings
        self._client = OpenAI(
            api_key=settings.llm_api_key or "unset",
            base_url=settings.resolved_base_url,
            timeout=settings.llm_timeout_seconds,
        )

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=0.5, max=3), reraise=True)
    def _complete(self, messages: list[dict], json_mode: bool) -> str:
        kwargs = {}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        response = self._client.chat.completions.create(
            model=self._settings.resolved_model,
            messages=messages,
            temperature=self._settings.llm_temperature,
            max_tokens=600,
            **kwargs,
        )
        return response.choices[0].message.content or ""

    def extract(self, history: list[dict]) -> dict | None:
        try:
            raw = self._complete(prompts.build_extraction_messages(history), json_mode=True)
            return json.loads(raw)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Extraction LLM call failed, falling back to heuristics: %s", exc)
            return None

    def compare(self, question: str, snippets: list[str]) -> str | None:
        try:
            return self._complete(prompts.build_comparison_messages(question, snippets), json_mode=False).strip()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Comparison LLM call failed, falling back to templated diff: %s", exc)
            return None


# ---------------------------------------------------------------------------
# Rule-based fallback extractor. Used only if the LLM call errors out, so a
# provider outage degrades the agent to "less smart" rather than "broken".
# ---------------------------------------------------------------------------

_SENIORITY_HINTS = [
    "intern", "entry", "junior", "graduate", "mid-level", "mid level", "senior",
    "manager", "lead", "director", "executive", "principal", "associate",
]
_NO_PREFERENCE_RE = re.compile(r"no preference|don'?t know|not sure|any (is fine|works)|doesn'?t matter", re.IGNORECASE)
_REMOTE_RE = re.compile(r"remote", re.IGNORECASE)
_DURATION_RE = re.compile(r"(\d+)\s*(?:minutes|mins|min)\b", re.IGNORECASE)
_ADD_PERSONALITY_RE = re.compile(r"(add|include).{0,20}personality", re.IGNORECASE)
_ADD_COGNITIVE_RE = re.compile(r"(add|include).{0,20}(cognitive|aptitude|ability)", re.IGNORECASE)


def rule_based_extract(history: list[dict]) -> dict:
    all_text = " ".join(m["content"] for m in history if m["role"] == "user")
    latest = history[-1]["content"] if history else ""
    seniority = next((hint for hint in _SENIORITY_HINTS if hint in all_text.lower()), None)
    duration_match = _DURATION_RE.search(all_text)
    return {
        "in_scope": True,
        "prompt_injection": False,
        "intent": "compare" if re.search(r"difference between|compare", latest, re.IGNORECASE) else "recommend",
        "comparison_targets": [],
        "role_title": None,
        "skills": [],
        "seniority": seniority,
        "add_test_types": (["P"] if _ADD_PERSONALITY_RE.search(all_text) else [])
        + (["A"] if _ADD_COGNITIVE_RE.search(all_text) else []),
        "remove_test_types": [],
        "max_duration_minutes": int(duration_match.group(1)) if duration_match else None,
        "remote_required": True if _REMOTE_RE.search(all_text) else None,
        "num_requested": None,
        "no_preference_signaled": bool(_NO_PREFERENCE_RE.search(latest)),
    }


def _sanitize_extraction(raw: dict | None, fallback: dict) -> dict:
    if not raw:
        return fallback
    out = dict(fallback)
    for key in fallback:
        if key in raw and raw[key] is not None:
            out[key] = raw[key]
    out["add_test_types"] = [t for t in (out.get("add_test_types") or []) if t in _VALID_TEST_TYPES]
    out["remove_test_types"] = [t for t in (out.get("remove_test_types") or []) if t in _VALID_TEST_TYPES]
    out["skills"] = [s for s in (out.get("skills") or []) if isinstance(s, str)][:10]
    out["comparison_targets"] = [c for c in (out.get("comparison_targets") or []) if isinstance(c, str)][:3]
    return out


# ---------------------------------------------------------------------------
# History introspection. Relies on our own reply templates being fixed
# strings, so we can reliably tell what kind of turn we last took without
# any separate state store.
# ---------------------------------------------------------------------------

def _prior_assistant_texts(history: list[dict]) -> list[str]:
    return [m["content"] for m in history if m["role"] == "assistant"]


def _already_recommended(history: list[dict]) -> bool:
    return any(t.startswith("Here are") or t.startswith("Updated the shortlist") for t in _prior_assistant_texts(history))


def _clarify_count(history: list[dict]) -> int:
    canned = set(prompts.CLARIFY_QUESTIONS.values())
    return sum(1 for t in _prior_assistant_texts(history) if t in canned)


class Agent:
    def __init__(self, retriever: Retriever | None = None, llm: LLMClient | None = None):
        self.retriever = retriever or Retriever()
        self.llm = llm or LLMClient()

    def handle_chat(self, messages: list[ChatMessage]) -> ChatResponse:
        history = [{"role": m.role, "content": m.content} for m in messages if m.content and m.content.strip()]
        if not history or history[-1]["role"] != "user":
            return ChatResponse(reply="I'm ready when you are -- tell me about the role you're hiring for.")

        latest_user_text = history[-1]["content"]
        regex_injection, regex_offtopic = _regex_guardrail(latest_user_text)

        fallback = rule_based_extract(history)
        try:
            extracted_raw = self.llm.extract(history)
        except Exception as exc:  # noqa: BLE001
            logger.warning("llm.extract raised unexpectedly, using rule-based fallback: %s", exc)
            extracted_raw = None
        facts = _sanitize_extraction(extracted_raw, fallback)

        if regex_injection or facts.get("prompt_injection"):
            return ChatResponse(reply=prompts.INJECTION_REFUSAL, recommendations=[], end_of_conversation=False)
        if regex_offtopic or not facts.get("in_scope", True):
            return ChatResponse(reply=prompts.SCOPE_REFUSAL, recommendations=[], end_of_conversation=False)

        if facts.get("intent") == "compare" and len(facts.get("comparison_targets", [])) >= 2:
            return self._handle_compare(latest_user_text, facts["comparison_targets"])

        return self._handle_recommend(history, facts)

    # ------------------------------------------------------------------

    def _handle_compare(self, question: str, targets: list[str]) -> ChatResponse:
        found, missing = [], []
        for name in targets[:2]:
            match = self.retriever.get_by_name(name)
            (found if match else missing).append(match or name)
        if len(found) < 2:
            return ChatResponse(reply=prompts.compare_not_found_reply(missing), recommendations=[], end_of_conversation=False)

        snippets = [f"{a.name}: {a.description}" for a in found]
        answer = self.llm.compare(question, snippets)
        if not answer:
            a, b = found
            answer = (
                f"{a.name} is a {a.test_type_label or 'general'} assessment (~{a.duration_minutes or 'unspecified'} min, "
                f"remote testing {'supported' if a.remote_testing else 'not supported'}). "
                f"{b.name} is a {b.test_type_label or 'general'} assessment (~{b.duration_minutes or 'unspecified'} min, "
                f"remote testing {'supported' if b.remote_testing else 'not supported'}). "
                f"Main difference on record: {a.test_type_label or 'n/a'} vs {b.test_type_label or 'n/a'}."
            )
        return ChatResponse(reply=answer, recommendations=[], end_of_conversation=False)

    def _handle_recommend(self, history: list[dict], facts: dict) -> ChatResponse:
        role_title = facts.get("role_title")
        skills = facts.get("skills") or []
        seniority = facts.get("seniority")
        no_pref = bool(facts.get("no_preference_signaled"))
        already_recommended = _already_recommended(history)
        clarify_count = _clarify_count(history)

        role_or_skills_known = bool(role_title) or bool(skills)

        if not role_or_skills_known and not no_pref and clarify_count < 2:
            return ChatResponse(reply=prompts.clarify_reply("role_or_skills"), recommendations=[], end_of_conversation=False)
        if (
            role_or_skills_known
            and not already_recommended
            and not seniority
            and not no_pref
            and clarify_count < 2
        ):
            return ChatResponse(reply=prompts.clarify_reply("seniority"), recommendations=[], end_of_conversation=False)

        query = " ".join(filter(None, [role_title, " ".join(skills), seniority])).strip() or history[-1]["content"]
        settings = get_settings()
        candidates = self.retriever.search(query, k=settings.top_k_retrieval)
        assessments = [a for a, _ in candidates]

        remove_types = set(facts.get("remove_test_types") or [])
        if remove_types:
            assessments = [a for a in assessments if a.test_type not in remove_types]

        max_duration = facts.get("max_duration_minutes")
        remote_required = facts.get("remote_required")
        strict = self.retriever.apply_filters(assessments, max_duration_minutes=max_duration, remote_required=remote_required)
        pool = strict if strict else assessments  # relax soft constraints rather than dead-ending

        if not pool:
            return ChatResponse(reply=prompts.no_match_reply(), recommendations=[], end_of_conversation=False)

        target_n = facts.get("num_requested") or 5
        target_n = max(settings.min_recommendations, min(target_n, settings.max_recommendations))

        add_types = set(facts.get("add_test_types") or [])
        if add_types:
            # The requested type may not appear in the top-k semantic/lexical
            # candidates at all (e.g. "add personality tests" for a Java
            # query) -- pull matching items from the whole catalog, subject
            # to the same hard filters, so refinement can actually surface them.
            pool_ids = {a.id for a in pool}
            type_matches = [a for a in self.retriever.get_all() if a.test_type in add_types and a.id not in pool_ids]
            type_matches = self.retriever.apply_filters(type_matches, max_duration_minutes=max_duration, remote_required=remote_required)
            pool = pool + type_matches
        shortlist = self._select_shortlist(pool, add_types, target_n)

        recommendations = [RecommendationItem(name=a.name, url=a.url, test_type=a.test_type) for a in shortlist]
        reply = prompts.recommend_reply(len(recommendations), role_title, refined=already_recommended)
        return ChatResponse(reply=reply, recommendations=recommendations, end_of_conversation=True)

    @staticmethod
    def _select_shortlist(pool: list, add_types: set[str], target_n: int) -> list:
        if not add_types:
            return pool[:target_n]
        boosted = [a for a in pool if a.test_type in add_types]
        rest = [a for a in pool if a.test_type not in add_types]
        quota = max(1, target_n // 3)
        shortlist = boosted[:quota] + rest[: target_n - min(quota, len(boosted))]
        return shortlist[:target_n] if shortlist else pool[:target_n]
