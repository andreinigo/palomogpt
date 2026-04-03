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
CRAWL_TIMEOUT = 150


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
    debug_log: list[str] = Field(default_factory=list, description="Diagnostic log entries")


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
        entries, debug_log = future.result(timeout=CRAWL_TIMEOUT)
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
        debug_log=debug_log,
    )


def _do_crawl(
    team_name: str,
    limit: int,
    team_url: str | None,
) -> tuple[list[FormationEntry], list[str]]:
    """Run the full crawl pipeline inside Playwright.

    Strategy (April 2026):
    - Sofascore internal API now returns 403 for page.evaluate(fetch) from
      datacenter IPs.  However the SSR-rendered team page already contains
      match links in the DOM, and navigating to each match page + clicking
      the Lineups tab causes the *page's own React code* to fetch lineups
      (which passes anti-bot checks).
    - We intercept those lineup API responses via page.on("response").
    - Fallback: if interception doesn't fire, try page.evaluate(fetch).
    """
    from playwright.sync_api import sync_playwright
    from sofascore_formations_crawler import (
        build_context, _new_stealth_page, resolve_team_url, dismiss_overlays,
    )

    t0 = _time.time()
    log: list[str] = []

    def _log(msg: str):
        line = f"[{_time.time()-t0:.1f}s] {msg}"
        log.append(line)
        print(f"[crawl] {line}", flush=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=BROWSER_ARGS)
        context = build_context(browser)
        page = _new_stealth_page(context)
        page.set_default_timeout(12000)
        page.set_default_navigation_timeout(20000)

        try:
            # Resolve team URL
            if team_url:
                team_page_url = team_url
            else:
                team_page_url = resolve_team_url(page, team_query=team_name)
            _log(f"team URL: {team_page_url}")

            # Navigate to team page
            page.goto(team_page_url, wait_until="domcontentloaded")
            dismiss_overlays(page)
            page.wait_for_timeout(3000)
            _log(f"team page loaded: {page.url}")

            # Extract match links from DOM (SSR-rendered)
            match_links = _extract_match_links(page, limit, _log)
            _log(f"found {len(match_links)} match links in DOM")

            if not match_links:
                return [], log

            # Try API approach first (works locally, may fail on Railway)
            team_id_match = re.search(r"/(\d+)$", page.url.rstrip("/"))
            team_id = team_id_match.group(1) if team_id_match else None

            if team_id:
                api_events = _try_api_events(page, team_id, limit, _log)
                if api_events:
                    _log(f"API approach: {len(api_events)} events")
                    entries = _try_api_lineups(page, api_events, team_name, limit, _log)
                    if entries:
                        _log(f"API approach SUCCESS: {len(entries)} formations")
                        return entries, log
                    _log("API lineups failed, falling back to DOM scraping")
                else:
                    _log("API events failed, falling back to DOM scraping")

            # Fallback: navigate to each match page and scrape lineups from DOM
            entries = _scrape_match_lineups(page, match_links, team_name, limit, _log)
            _log(f"DONE count={len(entries)}")

            return entries, log
        except Exception as exc:
            _log(f"EXCEPTION: {exc}")
            return [], log
        finally:
            page.close()
            context.close()
            browser.close()


def _extract_match_links(page, limit: int, _log) -> list[dict]:
    """Extract match links and basic info from the team page DOM."""
    data = page.evaluate("""(limit) => {
        const links = document.querySelectorAll('a[href*="/football/match/"]');
        const seen = new Set();
        const results = [];
        for (const a of links) {
            const href = a.getAttribute('href');
            if (!href || seen.has(href)) continue;
            seen.add(href);
            // Extract event ID from href hash: #id:14083607
            const idMatch = href.match(/#id:(\\d+)/);
            if (!idMatch) continue;
            results.push({
                href: href,
                event_id: parseInt(idMatch[1]),
                text: a.innerText.substring(0, 100).replace(/\\n/g, ' | '),
            });
            if (results.length >= limit * 3) break;
        }
        return results;
    }""", limit)
    return data or []


def _try_api_events(page, team_id: str, limit: int, _log) -> list[dict]:
    """Try the direct API approach for events (fast, works locally)."""
    try:
        data = page.evaluate(
            """async (teamId) => {
                try {
                    const resp = await fetch('/api/v1/team/' + teamId + '/events/last/0');
                    if (!resp.ok) return {error: resp.status};
                    return await resp.json();
                } catch(e) { return {error: e.message}; }
            }""",
            team_id,
        )
        if not data or "error" in data:
            _log(f"API events: {data.get('error') if data else 'null'}")
            return []
        events = data.get("events", [])
        return [e for e in events if e.get("status", {}).get("type") == "finished"]
    except Exception as exc:
        _log(f"API events exception: {exc}")
        return []


def _try_api_lineups(page, events: list[dict], team_name: str, limit: int, _log) -> list[FormationEntry]:
    """Try fetching lineups via direct API (fast, works locally)."""
    entries: list[FormationEntry] = []
    failures = 0
    for event in events:
        if len(entries) >= limit:
            break
        if failures >= 3:
            break
        event_id = event.get("id")
        try:
            lineups = page.evaluate(
                """async (eventId) => {
                    try {
                        const resp = await fetch('/api/v1/event/' + eventId + '/lineups');
                        if (!resp.ok) return {error: resp.status};
                        return await resp.json();
                    } catch(e) { return {error: e.message}; }
                }""",
                event_id,
            )
            if not lineups or "error" in lineups:
                failures += 1
                continue
            entry = _parse_lineup(lineups, event, team_name, len(entries) + 1)
            if entry:
                entries.append(entry)
                failures = 0
            else:
                failures += 1
        except Exception:
            failures += 1
    return entries


def _scrape_match_lineups(
    page, match_links: list[dict], team_name: str, limit: int, _log,
) -> list[FormationEntry]:
    """Navigate to each match page, click Lineups tab, and scrape data from DOM."""
    import threading

    entries: list[FormationEntry] = []
    consecutive_failures = 0

    for ml in match_links:
        if len(entries) >= limit:
            break
        if consecutive_failures >= 5:
            _log("too many DOM scraping failures, stopping")
            break

        event_id = ml["event_id"]
        match_url = f"https://www.sofascore.com{ml['href']}"
        _log(f"navigating to match {event_id}...")

        try:
            # Set up response interception for lineups
            lineup_data = {}
            lock = threading.Lock()

            def on_response(resp):
                if f"/event/{event_id}/lineups" in resp.url and resp.status == 200:
                    try:
                        with lock:
                            lineup_data["data"] = resp.json()
                    except Exception:
                        pass

            page.on("response", on_response)

            # Navigate to match page
            page.goto(match_url, wait_until="domcontentloaded")
            page.wait_for_timeout(2000)

            # Click Lineups tab
            try:
                tab = page.locator("text=Lineups").first
                if tab.is_visible(timeout=3000):
                    tab.click()
                    page.wait_for_timeout(3000)
            except Exception:
                pass

            # Remove listener
            page.remove_listener("response", on_response)

            # Try intercepted data first
            lineups = lineup_data.get("data")

            # If interception didn't get data, try direct API fetch
            if not lineups:
                try:
                    lineups = page.evaluate(
                        """async (eventId) => {
                            try {
                                const resp = await fetch('/api/v1/event/' + eventId + '/lineups');
                                if (!resp.ok) return null;
                                return await resp.json();
                            } catch(e) { return null; }
                        }""",
                        event_id,
                    )
                except Exception:
                    pass

            # If still no API data, try scraping formation from DOM
            if not lineups:
                lineups = _scrape_lineup_from_dom(page, _log)

            if not lineups:
                _log(f"no lineup data for match {event_id}")
                consecutive_failures += 1
                continue

            # Get event info from page
            event_info = page.evaluate("""() => {
                const nd = window.__NEXT_DATA__;
                if (!nd || !nd.props || !nd.props.pageProps || !nd.props.pageProps.initialProps) return null;
                const ev = nd.props.pageProps.initialProps.event;
                if (!ev) return null;
                return {
                    id: ev.id,
                    slug: ev.slug || '',
                    customId: ev.customId || '',
                    startTimestamp: ev.startTimestamp,
                    homeTeam: ev.homeTeam ? {name: ev.homeTeam.name} : {name: ''},
                    awayTeam: ev.awayTeam ? {name: ev.awayTeam.name} : {name: ''},
                    status: ev.status || {}
                };
            }""")

            if not event_info:
                # Construct minimal event info from match link
                event_info = {
                    "id": event_id,
                    "slug": "",
                    "customId": "",
                    "startTimestamp": None,
                    "homeTeam": {"name": ""},
                    "awayTeam": {"name": ""},
                    "status": {},
                }

            entry = _parse_lineup(lineups, event_info, team_name, len(entries) + 1)
            if entry:
                entries.append(entry)
                consecutive_failures = 0
                _log(f"OK {entry.formation} vs {entry.opponent} ({entry.match_date})")
            else:
                _log(f"no formation parsed for match {event_id}")
                consecutive_failures += 1

        except Exception as exc:
            _log(f"match {event_id} exception: {exc}")
            consecutive_failures += 1
            page.remove_listener("response", on_response) if "on_response" in dir() else None

    return entries


def _scrape_lineup_from_dom(page, _log) -> dict | None:
    """Last resort: scrape lineup data directly from rendered DOM."""
    try:
        data = page.evaluate("""() => {
            // Get formations from page text
            const text = document.body.innerText;
            const formationPattern = /\\b(\\d-\\d+-\\d+(?:-\\d+)?)\\b/g;
            const formations = [...new Set(text.match(formationPattern) || [])];
            if (formations.length < 2) return null;

            // Try to get player names from lineup elements
            const lineupEls = document.querySelectorAll('[data-testid*="lineup"], [class*="lineupPlayer"]');
            
            // Build a basic lineup structure
            return {
                home: {formation: formations[0], players: []},
                away: {formation: formations[1], players: []},
                _source: 'dom_scrape'
            };
        }""")
        return data
    except Exception:
        return None


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
