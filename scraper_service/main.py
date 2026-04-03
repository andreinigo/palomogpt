"""Scraper micro-service for Sofascore formations.

Deployed on Railway with Playwright + Chromium pre-installed (see Dockerfile).

Strategy:
1. Open headless browser, navigate to Sofascore team page
2. Use page.evaluate(fetch) to call Sofascore internal API (has session context)
3. Close browser, return structured formation data

The browser's fetch shares the same anti-bot tokens/cookies that Sofascore sets
via JavaScript, which is why httpx/requests with extracted cookies doesn't work.
"""

from __future__ import annotations

import os
import re
import time as _time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from difflib import SequenceMatcher
from typing import Optional

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="Sofascore Scraper Service")

_API_KEY = os.getenv("SCRAPER_API_KEY", "")
_EXECUTOR = ThreadPoolExecutor(max_workers=1)

BROWSER_ARGS = [
    "--no-sandbox",
    "--disable-gpu",
    "--disable-dev-shm-usage",
    "--disable-software-rasterizer",
]

# Hard timeout for the entire crawl operation (seconds).
CRAWL_TIMEOUT = 90


# ── request / response models ────────────────────────────────────────────

class CrawlRequest(BaseModel):
    team_name: str = Field(..., min_length=1, max_length=200)
    limit: int = Field(default=10, ge=1, le=30)
    team_url: Optional[str] = Field(
        default=None,
        description="Direct Sofascore team URL (bypasses search)",
    )


class FormationEntry(BaseModel):
    index: int
    match_url: str
    match_date: Optional[str] = None
    home_team: str
    away_team: str
    target_team: str
    opponent: str
    target_side: str
    formation: Optional[str] = None
    players: list[str]
    image_base64: Optional[str] = None


class CrawlResponse(BaseModel):
    team_name: str
    count: int
    formations: list[FormationEntry]


# ── helpers ───────────────────────────────────────────────────────────────

def _check_auth(authorization: str | None) -> None:
    if not _API_KEY:
        return
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
    if authorization.split(" ", 1)[1] != _API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")


# ── endpoints ─────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/crawl", response_model=CrawlResponse)
def crawl(req: CrawlRequest, authorization: str | None = Header(default=None)):
    _check_auth(authorization)
    print(f"[crawl] START team={req.team_name!r} limit={req.limit} team_url={req.team_url!r}", flush=True)

    # Run the actual crawl in a thread with a hard timeout so we never block
    # the uvicorn worker indefinitely.
    future = _EXECUTOR.submit(_do_crawl, req.team_name, req.limit, req.team_url)
    try:
        entries = future.result(timeout=CRAWL_TIMEOUT)
    except FuturesTimeout:
        future.cancel()
        print(f"[crawl] HARD TIMEOUT after {CRAWL_TIMEOUT}s", flush=True)
        raise HTTPException(status_code=504, detail=f"Crawl timed out after {CRAWL_TIMEOUT}s")
    except HTTPException:
        raise
    except Exception as exc:
        print(f"[crawl] ERROR: {exc}", flush=True)
        raise HTTPException(status_code=500, detail=str(exc))

    return CrawlResponse(
        team_name=req.team_name,
        count=len(entries),
        formations=entries,
    )


def _do_crawl(
    team_name: str,
    limit: int,
    team_url: str | None,
) -> list[FormationEntry]:
    """Run the full crawl pipeline inside Playwright."""
    from playwright.sync_api import sync_playwright
    from sofascore_formations_crawler import (
        build_context, _new_stealth_page, resolve_team_url, dismiss_overlays,
    )

    t0 = _time.time()

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=BROWSER_ARGS)
        context = build_context(browser)
        page = _new_stealth_page(context)
        page.set_default_timeout(10000)
        page.set_default_navigation_timeout(15000)

        try:
            # Resolve team URL
            if team_url:
                team_page_url = team_url
            else:
                team_page_url = resolve_team_url(page, team_query=team_name)
            print(f"[crawl] [{_time.time()-t0:.1f}s] team URL: {team_page_url}", flush=True)

            # Extract team_id from URL
            team_id_match = re.search(r"/(\d+)$", team_page_url.rstrip("/"))
            if not team_id_match:
                raise RuntimeError(f"Cannot extract team_id from URL: {team_page_url}")
            team_id = team_id_match.group(1)

            # Navigate to team page (establishes session cookies/tokens for API)
            page.goto(team_page_url, wait_until="domcontentloaded")
            dismiss_overlays(page)
            page.wait_for_timeout(2000)
            print(f"[crawl] [{_time.time()-t0:.1f}s] team page loaded", flush=True)

            # Fetch events via internal API
            events = _fetch_events(page, team_id, limit, t0)
            print(f"[crawl] [{_time.time()-t0:.1f}s] found {len(events)} finished events", flush=True)

            # Fetch lineups for each event
            entries = _fetch_lineups(page, events, team_name, limit, t0)
            print(f"[crawl] [{_time.time()-t0:.1f}s] DONE count={len(entries)}", flush=True)

            return entries
        finally:
            page.close()
            context.close()
            browser.close()


def _fetch_events(page, team_id: str, limit: int, t0: float) -> list[dict]:
    """Fetch recent finished events via Sofascore API from browser context."""
    all_events: list[dict] = []

    for page_num in range(3):
        try:
            data = page.evaluate(
                """async ([teamId, pageNum]) => {
                    try {
                        const resp = await fetch(`/api/v1/team/${teamId}/events/last/${pageNum}`);
                        if (!resp.ok) return {error: resp.status};
                        return await resp.json();
                    } catch(e) { return {error: e.message}; }
                }""",
                [team_id, page_num],
            )
        except Exception as exc:
            print(f"[api] [{_time.time()-t0:.1f}s] events page {page_num} failed: {exc}", flush=True)
            break

        if not data:
            print(f"[api] [{_time.time()-t0:.1f}s] events page {page_num}: null response", flush=True)
            break
        if "error" in data:
            print(f"[api] [{_time.time()-t0:.1f}s] events page {page_num}: error={data['error']}", flush=True)
            break
        if "events" not in data:
            break

        page_events = data["events"]
        if not page_events:
            break

        all_events.extend(page_events)
        print(f"[api] [{_time.time()-t0:.1f}s] events page {page_num}: {len(page_events)} events", flush=True)

        finished = [e for e in all_events if e.get("status", {}).get("type") == "finished"]
        if len(finished) >= limit * 2:
            break

    # Return only finished events
    return [e for e in all_events if e.get("status", {}).get("type") == "finished"]


def _fetch_lineups(
    page,
    events: list[dict],
    team_name: str,
    limit: int,
    t0: float,
) -> list[FormationEntry]:
    """Fetch lineup data for each event via API."""
    entries: list[FormationEntry] = []
    consecutive_failures = 0

    for event in events:
        if len(entries) >= limit:
            break
        if consecutive_failures >= 5:
            print(f"[api] [{_time.time()-t0:.1f}s] too many failures, stopping", flush=True)
            break

        event_id = event.get("id")
        home = event.get("homeTeam", {}).get("name", "")
        away = event.get("awayTeam", {}).get("name", "")

        try:
            lineups = page.evaluate(
                """async (eventId) => {
                    try {
                        const resp = await fetch(`/api/v1/event/${eventId}/lineups`);
                        if (!resp.ok) return {error: resp.status};
                        return await resp.json();
                    } catch(e) { return {error: e.message}; }
                }""",
                event_id,
            )
        except Exception as exc:
            print(f"[api] [{_time.time()-t0:.1f}s] lineups failed for {event_id}: {exc}", flush=True)
            consecutive_failures += 1
            continue

        if not lineups:
            print(f"[api] [{_time.time()-t0:.1f}s] lineups null for {event_id}", flush=True)
            consecutive_failures += 1
            continue
        if "error" in lineups:
            print(f"[api] [{_time.time()-t0:.1f}s] lineups error for {event_id}: {lineups['error']}", flush=True)
            consecutive_failures += 1
            continue

        entry = _parse_lineup(lineups, event, team_name, len(entries) + 1)
        if entry:
            entries.append(entry)
            consecutive_failures = 0
            print(f"[api] [{_time.time()-t0:.1f}s] ✓ {entry.formation} vs {entry.opponent} ({entry.match_date})", flush=True)
        else:
            consecutive_failures += 1

    return entries


def _parse_lineup(
    lineups: dict,
    event: dict,
    team_name: str,
    index: int,
) -> FormationEntry | None:
    """Parse lineup API response into a FormationEntry."""
    home = event.get("homeTeam", {}).get("name", "")
    away = event.get("awayTeam", {}).get("name", "")

    home_lineup = lineups.get("home", {})
    away_lineup = lineups.get("away", {})

    # Match target team
    home_ratio = SequenceMatcher(None, team_name.lower(), home.lower()).ratio()
    away_ratio = SequenceMatcher(None, team_name.lower(), away.lower()).ratio()
    if home_ratio >= away_ratio:
        target_side, target_team, opponent = "home", home, away
        target_lineup = home_lineup
    else:
        target_side, target_team, opponent = "away", away, home
        target_lineup = away_lineup

    formation = target_lineup.get("formation")
    if not formation:
        return None

    # Extract starters
    player_names = []
    for p in target_lineup.get("players", []):
        player = p.get("player", {})
        shirt = p.get("shirtNumber", "")
        name = player.get("name", "")
        if name and not p.get("substitute", False):
            player_names.append(f"{shirt} {name}")

    # Match date
    start_ts = event.get("startTimestamp")
    match_date = None
    if start_ts:
        from datetime import datetime, timezone
        match_date = datetime.fromtimestamp(start_ts, tz=timezone.utc).strftime("%Y-%m-%d")

    # Match URL
    slug = event.get("slug", "")
    custom_id = event.get("customId", "")
    match_url = f"https://www.sofascore.com/football/match/{slug}/{custom_id}" if slug else ""

    return FormationEntry(
        index=index,
        match_url=match_url,
        match_date=match_date,
        home_team=home,
        away_team=away,
        target_team=target_team,
        opponent=opponent,
        target_side="left" if target_side == "home" else "right",
        formation=formation,
        players=player_names,
        image_base64=None,
    )


# ── main ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port, timeout_keep_alive=300)
