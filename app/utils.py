"""Small stateless helpers shared across scraper/retriever/agent."""
import logging
import re
import sys

_DURATION_RE = re.compile(r"(\d+)")

# Coarse keyword -> skill tag map used to enrich catalog descriptions/retrieval
# text for items whose only scraped signal is a product name (e.g. "Java 8 (New)").
_SKILL_KEYWORDS = {
    "java": ["Java"], "python": ["Python"], ".net": [".NET"], "c#": ["C#"],
    "c++": ["C++"], "javascript": ["JavaScript"], "sql": ["SQL"],
    "html": ["HTML/CSS"], "css": ["HTML/CSS"], "angular": ["Angular"],
    "react": ["React"], "node": ["Node.js"], "aws": ["Cloud/AWS"],
    "azure": ["Cloud/Azure"], "docker": ["DevOps"], "kubernetes": ["DevOps"],
    "hadoop": ["Big Data"], "spark": ["Big Data"], "data science": ["Data Science"],
    "machine learning": ["Machine Learning"], "sales": ["Sales"],
    "customer service": ["Customer Service"], "call center": ["Contact Center"],
    "accounts": ["Finance/Accounting"], "financial": ["Finance/Accounting"],
    "administrative": ["Administrative"], "clerical": ["Administrative"],
    "leadership": ["Leadership"], "manager": ["Management/Leadership"],
    "supervisor": ["Management/Leadership"], "verbal": ["Verbal Reasoning"],
    "numerical": ["Numerical Reasoning"], "mechanical": ["Mechanical Reasoning"],
    "personality": ["Personality"], "opq": ["Personality"],
    "situational judgement": ["Situational Judgement"],
    "sjt": ["Situational Judgement"], "motivation": ["Motivation"],
    "coding": ["Software Development"], "developer": ["Software Development"],
    "testing": ["QA/Testing"], "sap": ["SAP/ERP"], "excel": ["MS Office"],
    "word": ["MS Office"], "typing": ["Data Entry/Typing"], "data entry": ["Data Entry/Typing"],
}

_LEVEL_KEYWORDS = {
    "graduate": "Graduate", "entry": "Entry-Level", "manager": "Manager",
    "supervisor": "Supervisor", "executive": "Executive", "director": "Director",
    "senior": "Senior Professional", "professional": "Professional",
    "front line": "Front Line", "frontline": "Front Line",
}


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )


def parse_duration_minutes(raw: str) -> int | None:
    """Extract an integer minute count from strings like
    'Approximate Completion Time in minutes = 30' or '30.0'. Returns None for
    non-numeric values such as 'Untimed', 'Variable', 'TBC', 'N/A', ''."""
    if not raw:
        return None
    match = _DURATION_RE.search(raw)
    if not match:
        return None
    return int(match.group(1))


def test_type_letter_from_label(label: str) -> str:
    mapping = {
        "ability & aptitude": "A",
        "biodata & situational judgement": "B",
        "competencies": "C",
        "development & 360": "D",
        "assessment exercises": "E",
        "knowledge & skills": "K",
        "personality & behavior": "P",
        "simulations": "S",
    }
    return mapping.get(label.strip().lower(), "")


def infer_skills(name: str) -> list[str]:
    name_lower = name.lower()
    found: list[str] = []
    for kw, tags in _SKILL_KEYWORDS.items():
        if kw in name_lower:
            for t in tags:
                if t not in found:
                    found.append(t)
    return found


def infer_job_levels(name: str) -> list[str]:
    name_lower = name.lower()
    levels = [label for kw, label in _LEVEL_KEYWORDS.items() if kw in name_lower]
    return levels or ["General/Mid-Professional"]


def slugify(text: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9]+", "-", text.strip().lower())
    return re.sub(r"-+", "-", text).strip("-")
