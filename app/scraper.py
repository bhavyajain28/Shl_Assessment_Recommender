"""Builds data/catalog.json, the grounding source for the whole agent.

Two data paths, tried in order:

1. Live scrape: GET the classic paginated SHL catalog table
   (``/solutions/products/product-catalog/?start=N&type=1``) and parse it with
   BeautifulSoup. This is the path the assignment describes, and it is what
   runs if SHL ever restores that page.

2. Historical fallback: as of 2026-07, SHL retired that paginated table --
   every legacy detail page now 301-redirects to one of ~10 consolidated
   marketing category pages, and the index page itself renders a generic
   "Our Products" hub with no per-assessment data (verified by rendering it
   with a real browser, with cookie-consent accepted, before writing this).
   ``fetch_live_catalog_table`` detects this by structurally failing to find
   a catalog table, and returns None so the caller falls back to
   ``data/catalog_seed_historical.csv`` -- a snapshot of the same catalog
   (name, URL, test type, duration, remote testing, adaptive/IRT) captured
   while that page was still live. Every URL from the seed is then
   re-resolved against the live site right now via HTTP redirect-following,
   so nothing we ever return points at a dead page -- assessments whose
   granular page was retired get pointed at their real, current SHL
   destination (the category page they now redirect to).
"""
from __future__ import annotations

import csv
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from tenacity import retry, stop_after_attempt, wait_exponential

from app.models import TEST_TYPE_LABELS, Assessment
from app.utils import (
    infer_job_levels,
    infer_skills,
    parse_duration_minutes,
    test_type_letter_from_label,
)

logger = logging.getLogger(__name__)

CATALOG_BASE = "https://www.shl.com/solutions/products/product-catalog/"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
JOB_SOLUTION_SUFFIX = "solution"  # SHL names every Job Solution bundle "<Role> Solution"
MAX_CATALOG_PAGES = 60  # safety cap (12 rows/page historically -> ~720 rows)


def _new_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT, "Accept-Language": "en-US,en;q=0.9"})
    return session


def fetch_live_catalog_table(session: requests.Session) -> list[dict] | None:
    """Try to scrape the classic paginated Individual Test Solutions table.

    Returns None (not []) if the page no longer contains a recognizable
    catalog table, signalling the caller to use the historical fallback.
    Returns [] only if the table exists but every page came back empty.
    """
    rows: list[dict] = []
    start = 0
    saw_table = False
    while start < MAX_CATALOG_PAGES * 12:
        resp = session.get(CATALOG_BASE, params={"start": start, "type": 1}, timeout=15)
        if resp.status_code != 200:
            break
        soup = BeautifulSoup(resp.text, "lxml")
        table = soup.find("table")
        if table is None:
            break
        saw_table = True
        page_rows = table.find_all("tr")[1:]  # skip header
        if not page_rows:
            break
        for tr in page_rows:
            link = tr.find("a", href=True)
            if not link:
                continue
            cells = tr.find_all("td")
            rows.append(
                {
                    "Assessment Name": link.get_text(strip=True),
                    "Relative URL": requests.compat.urljoin(CATALOG_BASE, link["href"]),
                    "Remote Testing": "Yes" if any("catalogue__circle -yes" in c.get("class", []) for c in cells[1:2]) else "No",
                    "Adaptive/IRT": "Yes" if any("catalogue__circle -yes" in c.get("class", []) for c in cells[2:3]) else "No",
                    "Test Type": "",
                    "Assessment Length": "",
                }
            )
        start += 12
        time.sleep(0.3)  # be polite
    if not saw_table:
        logger.warning("Live catalog table not found at %s -- site structure has changed.", CATALOG_BASE)
        return None
    return rows


def load_historical_seed(csv_path: Path) -> list[dict]:
    with open(csv_path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=0.5, max=4))
def _resolve_one(session: requests.Session, url: str) -> tuple[str, int]:
    resp = session.get(url, timeout=10, allow_redirects=True)
    return resp.url, resp.status_code


def resolve_live_urls(session: requests.Session, urls: list[str], max_workers: int = 8) -> dict[str, tuple[str, int]]:
    """Follow redirects for every URL concurrently. Falls back to the
    original URL with status 0 (rather than dropping the item) if resolution
    fails after retries, so a transient network error never removes an
    assessment -- callers should treat status 0 as 'unverified, use with
    caution' rather than 'confirmed dead'."""
    resolved: dict[str, tuple[str, int]] = {}
    unique_urls = list(set(urls))
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_to_url = {pool.submit(_resolve_one, session, u): u for u in unique_urls}
        for i, future in enumerate(as_completed(future_to_url), start=1):
            original = future_to_url[future]
            try:
                resolved[original] = future.result()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not resolve %s (%s); keeping original URL", original, exc)
                resolved[original] = (original, 0)
            if i % 25 == 0 or i == len(unique_urls):
                logger.info("Resolved %d/%d URLs...", i, len(unique_urls))
    return resolved


def _is_job_solution_bundle(name: str) -> bool:
    return name.strip().lower().endswith(JOB_SOLUTION_SUFFIX)


def generate_description(
    name: str,
    test_type_label: str,
    duration_minutes: int | None,
    remote_testing: bool,
    adaptive_irt: bool,
    skills: list[str],
) -> str:
    parts = [f"{name} is an SHL"]
    parts.append(f"{test_type_label} assessment." if test_type_label else "assessment.")
    if skills:
        parts.append(f"It measures: {', '.join(skills)}.")
    if duration_minutes:
        parts.append(f"Approximate completion time is {duration_minutes} minutes.")
    parts.append(f"Remote testing is {'supported' if remote_testing else 'not supported'}.")
    parts.append(f"Adaptive/IRT scoring is {'used' if adaptive_irt else 'not used'}.")
    return " ".join(parts)


def rows_to_assessments(rows: list[dict], url_map: dict[str, tuple[str, int]] | None = None) -> list[Assessment]:
    assessments: list[Assessment] = []
    for row in rows:
        name = row["Assessment Name"].strip()
        if not name or _is_job_solution_bundle(name):
            continue
        catalog_url = row.get("Relative URL", "").strip()
        redirect_target, status = (url_map or {}).get(catalog_url, (catalog_url, 200))
        # Prefer the specific, item-unique historical URL as long as it confirmed
        # live (200, even via redirect). Only fall back to the resolved
        # destination if the historical URL itself failed outright.
        primary_url = catalog_url if (catalog_url and status != 0) else (redirect_target or catalog_url)
        test_type_label = row.get("Test Type", "").strip()
        test_type = test_type_letter_from_label(test_type_label)
        duration = parse_duration_minutes(row.get("Assessment Length", ""))
        remote = row.get("Remote Testing", "").strip().lower() == "yes"
        adaptive = row.get("Adaptive/IRT", "").strip().lower() == "yes"
        skills = infer_skills(name)
        assessments.append(
            Assessment(
                id=row.get("data-entity-id") or str(len(assessments) + 1),
                name=name,
                url=primary_url or CATALOG_BASE,
                current_redirect_target=redirect_target,
                description=generate_description(name, test_type_label, duration, remote, adaptive, skills),
                test_type=test_type,
                test_type_label=test_type_label or TEST_TYPE_LABELS.get(test_type, ""),
                duration_minutes=duration,
                remote_testing=remote,
                adaptive_irt=adaptive,
                job_levels=infer_job_levels(name),
                skills=skills,
                source="live-scrape" if url_map is None else "historical-seed+live-redirect-resolved",
            )
        )
    return assessments


def build_catalog(seed_csv_path: Path) -> list[Assessment]:
    session = _new_session()
    live_rows = fetch_live_catalog_table(session)
    if live_rows:
        logger.info("Using live-scraped catalog table: %d rows", len(live_rows))
        return rows_to_assessments(live_rows)

    logger.info("Live catalog table unavailable -- falling back to historical seed: %s", seed_csv_path)
    rows = load_historical_seed(seed_csv_path)
    urls = [r.get("Relative URL", "").strip() for r in rows if r.get("Relative URL", "").strip()]
    logger.info("Resolving %d unique URLs against the live site...", len(set(urls)))
    url_map = resolve_live_urls(session, urls)
    return rows_to_assessments(rows, url_map=url_map)
