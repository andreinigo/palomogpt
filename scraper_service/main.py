"""Scraper micro-service for Sofascore formations.

Deployed on Railway with Playwright + Chromium pre-installed (see Dockerfile).

Strategy:
1. Open headless browser -> navigate to Sofascore team page -> extract session cookies
2. Close browser immediately (free memory)
3. Use Python httpx with extracted cookies to call Sofascore internal API
4. Return structured formation data
"""

from __future__ import annotations

import os
import time as _time
from difflib import SequenceMatcher
from typing import Optional

import httpx
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="Sofascore Scraper Service")

_API_KEY = os.getenv("SCRAPER_API_KEY", "")

BROWSER_ARGS = [
    "--no-sandbox",
    "--disable-gpu",
    "--disable-dev-shm-usage",
    "--disable-software-rasterizer",
]


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


def _get_sofascore_session(team_page_url: str) -> tuple[str, str]:
    """Open browser, navigate to team page, extract cookies + user-agent, close browser."""
    from playwright.sync_api import sync_playwright
    from sofascore_formations_crawler import build_context, _new_stealth_page, dismiss_overlays

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=BROWSER_ARGS)
        context = build_context(browser)
        page = _new_stealth_page(context)
        page.set_default_timeout(12000)
        page.set_default_navigation_timeout(20000)

        page.goto(team_page_url, wait_until="domcontentloaded")
        dismiss_overlays(page)
        page.wait_for_timeout(2000)

        cookies = context.cookies("https://www.sofascore.com")
        cookie_header = "; ".join(f"{c['name']}={c['value']}" for c in cookies)
        user_agent = page.evaluate("() => navigator.userAgent")

        page.close()
        context.close()
        browser.close()

    return cookie_header, user_agent


def _resolve_team_url_browser(team_name: str) -> str:
    """Use browser to resolve team name to Sofascore team URL."""
    from playwright.sync_api import sync_playwright
    from sofascore_formations_crawler import build_context, _new_stealth_page, resolve_team_url

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=BROWSER_ARGS)
        context = build_context(browser)
        page = _new_stealth_page(context)
        page.set_default_timeout(12000)
        page.set_default_navigation_timeout(20000)

        url = resolve_team_url(page, team_query=team_name)

        page.close()
        context.close()
        browser.close()

    return url


# ── endpoints ─────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/crawl", response_model=CrawlResponse)
def crawl(req: CrawlRequest, authorization: str | None = Header(default=None)):
    _check_auth(authorization)
    t0 = _time.time()
    MAX_SECONDS = 120
    print(f"[crawl] START team={req.team_name!r} limit={req.limit} team_url={req.team_url!r}", flush=True)

    import re

    try:
        # Step 1: Resolve team URL if not provided
        if req.team_url:
            team_page_url = req.team_url
        else:
            team_page_url = _resolve_team_url_browser(req.team_name)
        print(f"[crawl] [{_time.time()-t0:.1f}s] team URL: {team_page_url}", flush=True)

        # Extract team_id from URL
        team_id_match = re.search(r"/(\d+)$", team_page_url.rstrip("/"))
        if not team_id_match:
            raise HTTPException(status_code=400, detail=f"Cannot extract team_id from URL: {team_page_url}")
        team_id = team_id_match.group(1)

        # Step 2: Get session cookies from browser
        print(f"[crawl] [{_time.time()-t0:.1f}s] getting session cookies...", flush=True)
        cookie_header, user_agent = _get_sofascore_session(team_page_url)
        print(f"[crawl] [{_time.time()-t0:.1f}s] got cookies ({len(cookie_header)} chars)", flush=True)

        # Step 3: Use API to get formations
        entries = _fetch_formations_via_api(
            team_id=team_id,
            team_name=req.team_name,
            limit=req.limit,
            cookie_header=cookie_header,
            user_agent=user_agent,
            t0=t0,
            max_seconds=MAX_SECONDS,
        )

        print(f"[crawl] [{_time.time()-t0:.1f}s] DONE count={len(entries)}", flush=True)
        return CrawlResponse(
            team_name=req.team_name,
            count=len(entries),
            formations=entries,
        )
    except HTTPException:
        raise
    except Exception as exc:
        elapsed = _time.time() - t0
        print(f"[crawl] [{elapsed:.1f}s] ERROR: {exc}", flush=True)
        raise HTTPException(status_code=500, detail=str(exc))


def _fetch_formations_via_api(
    team_id: str,
    team_name: str,
    limit: int,
    cookie_header: str,
    user_agent: str,
    t0: float,
    max_seconds: float,
) -> list[FormationEntry]:
    """Fetch formations using Sofascore API with browser-extracted cookies."""

    headers = {
        "Cookie": cookie_header,
        "User-Agent": user_agent,
        "Referer": "https://www.sofascore.com/",
        "Accept": "application/json",
    }

    # Collect finished events (paginate if needed)
    all_events: list[dict] = []
    for page_num in range(3):
        if _time.time() - t0 > max_seconds:
            break
        try:
            resp = httpx.get(
                f"https://www.sofascore.com/api/v1/team/{team_id}/events/last/{page_num}",
                headers=headers,
                timeout=15.0,
            )
            if resp.status_code != 200:
                print(f"[api] [{_time.time()-t0:.1f}s] events page {page_num}: HTTP {resp.status_code}", flush=True)
                break
            data = resp.json()
        except Exception as exc:
            print(f"[api] [{_time.time()-t0:.1f}s] events page {page_num} failed: {exc}", flush=True)
            break

        page_events = data.get("events", [])
        if not page_events:
            break
        all_events.extend(page_events)
        print(f"[api] [{_time.time()-t0:.1f}s] events page {page_num}: {len(page_events)} events (total: {len(all_events)})", flush=True)

        finished_count = sum(1 for e in all_events if e.get("status", {}).get("type") == "finished")
        if finished_count >= limit * 2:
            break

    if not all_events:
        raise RuntimeError("Sofascore API returned no events (cookies may have expired)")

    # Fetch lineups for each finished event
    entries: list[FormationEntry] = []
    consecutive_failures = 0

    for event in all_events:
        if len(entries) >= limit:
            break
        if _time.time() - t0 > max_seconds:
            print(f"[api] [{_time.time()-t0:.1f}s] time limit reached", flush=True)
            break
        if consecutive_failures >= 5:
            print(f"[api] [{_time.time()-t0:.1f}s] too many failures, stopping", flush=True)
            break

        event_id = event.get("id")
        home = event.get("homeTeam", {}).get("name", "")
        away = event.get("awayTeam", {}).get("name", "")
        status_type = event.get("status", {}).get("type", "")

        if status_type != "finished":
            continue

        try:
            resp = httpx.get(
                f"https://www.sofascore.com/api/v1/event/{event_id}/lineups",
                headers=headers,
                timeout=10.0,
            )
            if resp.status_code != 200:
                consecutive_failures += 1
                continue
            lineups_data = resp.json()
        except Exception as exc:
            print(f"[api] [{_time.time()-t0:.1f}s] lineups failed for event {event_id}: {exc}", flush=True)
            consecutive_failures += 1
            continue

        if not lineups_data:
            consecutive_failures += 1
            continue

        home_lineup = lineups_data.get("home", {})
        away_lineup = lineups_data.get("away", {})

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
            consecutive_failures += 1
            continue

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

        entries.append(FormationEntry(
            index=len(entries) + 1,
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
        ))
        consecutive_failures = 0
        print(f"[api] [{_time.time()-t0:.1f}s] ✓ {formation} vs {opponent} ({match_date})", flush=True)

    return entries


# ── main ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port, timeout_keep_alive=300)
