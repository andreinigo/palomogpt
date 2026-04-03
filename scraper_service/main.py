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

# ── Known Sofascore team slug→ID mapping ──────────────────────────────────
# Used to resolve team names without relying on Google/Sofascore search
# (which fails from datacenter IPs).  slug → numeric ID.
_TEAM_IDS: dict[str, int] = {
    # La Liga
    "barcelona": 2817, "real-madrid": 2829, "villarreal": 2819,
    "atletico-madrid": 2836, "real-betis": 2816, "celta-vigo": 2821,
    "real-sociedad": 2824, "athletic-club": 2825, "sevilla": 2833,
    "valencia": 2828, "getafe": 2859, "osasuna": 2820, "mallorca": 2826,
    "girona-fc": 24264, "rayo-vallecano": 2818, "deportivo-alaves": 2885,
    "espanyol": 2814, "levante-ud": 2849, "elche": 2846, "real-oviedo": 2851,
    # Premier League
    "arsenal": 42, "liverpool": 44, "manchester-city": 17,
    "manchester-united": 35, "chelsea": 38, "tottenham-hotspur": 33,
    "newcastle-united": 39, "aston-villa": 40, "west-ham-united": 37,
    "brighton-and-hove-albion": 30, "wolverhampton": 3, "bournemouth": 60,
    "fulham": 43, "crystal-palace": 7, "brentford": 50, "everton": 48,
    "nottingham-forest": 14,
    # Bundesliga
    "fc-bayern-munchen": 2672, "borussia-dortmund": 2673,
    "bayer-04-leverkusen": 2681, "rb-leipzig": 36360,
    "eintracht-frankfurt": 2674, "vfb-stuttgart": 2677,
    "sc-freiburg": 2538, "vfl-wolfsburg": 2524,
    "tsg-hoffenheim": 2569, "fc-augsburg": 2600,
    "1-fsv-mainz-05": 2556, "sv-werder-bremen": 2534,
    "borussia-mgladbach": 2527, "fc-st-pauli": 2526,
    "1-fc-union-berlin": 2547, "1-fc-heidenheim": 5885,
    # Serie A
    "inter": 2697, "milan": 2692, "juventus": 2687, "napoli": 2714,
    "atalanta": 2686, "roma": 2702, "lazio": 2699, "fiorentina": 2693,
    "bologna": 2685, "torino": 2696, "udinese": 2695,
    # Ligue 1
    "paris-saint-germain": 1644, "olympique-de-marseille": 1641,
    "as-monaco": 1653, "lille": 1643, "olympique-lyonnais": 1649,
    "nice": 1661, "stade-rennais": 1658, "rc-lens": 1648,
    "stade-brestois": 1715,
    # Portugal
    "benfica": 3006, "fc-porto": 3002, "sporting": 3001,
    "sporting-braga": 2999,
    # Netherlands
    "ajax": 2953, "psv-eindhoven": 2952, "feyenoord": 2959,
    # National teams
    "spain": 4687, "germany": 4711, "france": 4481, "england": 4713,
    "brazil": 4230, "argentina": 4819, "italy": 4707, "portugal": 4704,
    "netherlands": 4708, "belgium": 4717, "colombia": 4753,
    "uruguay": 4724, "mexico": 4535, "usa": 4837, "croatia": 4715,
    "japan": 4387, "south-korea": 4342, "poland": 4703,
    "switzerland": 4699, "denmark": 4476, "austria": 4697,
    "sweden": 4694, "chile": 4761, "ecuador": 4757, "peru": 4765,
    "costa-rica": 4775, "honduras": 4793, "panama": 4773,
    "paraguay": 4769, "venezuela": 4755, "bolivia": 4759,
}

# Friendly name → slug mapping for fuzzy lookup
_NAME_TO_SLUG: dict[str, str] = {}
for _slug in _TEAM_IDS:
    # "real-madrid" → "real madrid", "fc-bayern-munchen" → "fc bayern munchen"
    _NAME_TO_SLUG[_slug.replace("-", " ")] = _slug


def _lookup_team_url(team_name: str) -> str | None:
    """Resolve a team name to a Sofascore URL using the known mapping."""
    name = team_name.lower().strip()
    # Strip women's team suffix
    name = re.sub(r"\s*\(equipo femenino\)\s*$", "", name)

    # 1. Try exact slug match  (e.g. "real-madrid")
    if name.replace(" ", "-") in _TEAM_IDS:
        slug = name.replace(" ", "-")
        return f"https://www.sofascore.com/team/football/{slug}/{_TEAM_IDS[slug]}"

    # 2. Try exact friendly-name match
    if name in _NAME_TO_SLUG:
        slug = _NAME_TO_SLUG[name]
        return f"https://www.sofascore.com/team/football/{slug}/{_TEAM_IDS[slug]}"

    # 3. Common aliases
    _ALIASES: dict[str, str] = {
        "atletico": "atletico-madrid", "atlético": "atletico-madrid",
        "atlético madrid": "atletico-madrid", "atletico de madrid": "atletico-madrid",
        "real": "real-madrid", "barça": "barcelona", "barca": "barcelona",
        "betis": "real-betis", "sociedad": "real-sociedad",
        "athletic": "athletic-club", "athletic bilbao": "athletic-club",
        "man city": "manchester-city", "man united": "manchester-united",
        "man utd": "manchester-united", "spurs": "tottenham-hotspur",
        "tottenham": "tottenham-hotspur", "wolves": "wolverhampton",
        "bayern": "fc-bayern-munchen", "bayern munich": "fc-bayern-munchen",
        "bayern munchen": "fc-bayern-munchen", "bayern münchen": "fc-bayern-munchen",
        "dortmund": "borussia-dortmund", "leverkusen": "bayer-04-leverkusen",
        "leipzig": "rb-leipzig", "frankfurt": "eintracht-frankfurt",
        "stuttgart": "vfb-stuttgart", "freiburg": "sc-freiburg",
        "wolfsburg": "vfl-wolfsburg", "hoffenheim": "tsg-hoffenheim",
        "mainz": "1-fsv-mainz-05", "werder": "sv-werder-bremen",
        "werder bremen": "sv-werder-bremen",
        "gladbach": "borussia-mgladbach", "monchengladbach": "borussia-mgladbach",
        "psg": "paris-saint-germain", "paris": "paris-saint-germain",
        "marseille": "olympique-de-marseille", "lyon": "olympique-lyonnais",
        "monaco": "as-monaco", "lens": "rc-lens", "rennes": "stade-rennais",
        "brest": "stade-brestois",
        "psv": "psv-eindhoven",
        "porto": "fc-porto", "braga": "sporting-braga",
        "ac milan": "milan",
        "españa": "spain", "alemania": "germany", "francia": "france",
        "inglaterra": "england", "brasil": "brazil",
        "holanda": "netherlands", "paises bajos": "netherlands",
        "belgica": "belgium", "bélgica": "belgium",
        "estados unidos": "usa", "eeuu": "usa",
        "corea del sur": "south-korea", "corea": "south-korea",
        "japon": "japan", "japón": "japan",
        "suiza": "switzerland", "dinamarca": "denmark",
        "suecia": "sweden", "croacia": "croatia",
    }
    if name in _ALIASES:
        slug = _ALIASES[name]
        return f"https://www.sofascore.com/team/football/{slug}/{_TEAM_IDS[slug]}"

    # 4. Fuzzy match (threshold 0.75)
    best_score, best_slug = 0.0, None
    for friendly, slug in _NAME_TO_SLUG.items():
        score = SequenceMatcher(None, name, friendly).ratio()
        if score > best_score:
            best_score = score
            best_slug = slug
    if best_score >= 0.75 and best_slug:
        return f"https://www.sofascore.com/team/football/{best_slug}/{_TEAM_IDS[best_slug]}"

    return None

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
# Must allow time for up to 2 attempts with browser relaunch.
CRAWL_TIMEOUT = 250


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
    1. Resolve team URL via known mapping → Google → Sofascore search.
    2. Try fast API path (page.evaluate(fetch)) — works when anti-bot allows.
    3. Fallback: navigate to each match page, click Lineups tab, intercept
       the lineup API response from the page's own React code.
    4. If first attempt returns 0 results, retry once with fresh browser.
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

    # Resolve team URL upfront (before browser, uses mapping)
    if not team_url:
        mapped = _lookup_team_url(team_name)
        if mapped:
            team_url = mapped
            _log(f"team URL (mapped): {team_url}")

    max_attempts = 2
    for attempt in range(1, max_attempts + 1):
        if attempt > 1:
            _log(f"RETRY attempt {attempt}/{max_attempts} with fresh browser")
            import random
            _time.sleep(random.uniform(2, 5))

        entries = _do_crawl_attempt(
            team_name, limit, team_url, attempt, _log,
        )
        if entries:
            _log(f"DONE count={len(entries)}")
            return entries, log

    _log(f"DONE count=0 after {max_attempts} attempts")
    return [], log


def _do_crawl_attempt(
    team_name: str,
    limit: int,
    team_url: str | None,
    attempt: int,
    _log,
) -> list[FormationEntry]:
    """Single crawl attempt with its own browser instance."""
    from playwright.sync_api import sync_playwright
    from sofascore_formations_crawler import (
        build_context, _new_stealth_page, resolve_team_url, dismiss_overlays,
    )

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=BROWSER_ARGS)
        context = build_context(browser)
        page = _new_stealth_page(context)
        page.set_default_timeout(12000)
        page.set_default_navigation_timeout(20000)

        try:
            # Warmup: visit sofascore.com first to establish cookies/tokens
            try:
                page.goto("https://www.sofascore.com/", wait_until="domcontentloaded")
                page.wait_for_timeout(2000)
                dismiss_overlays(page)
            except Exception:
                pass  # non-critical

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
            match_links = _extract_match_links(page, team_name, limit, _log)
            _log(f"found {len(match_links)} match links in DOM")

            if not match_links:
                return []

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
                        return entries
                    _log("API lineups failed, falling back to DOM scraping")
                else:
                    _log("API events failed, falling back to DOM scraping")

            # Fallback: navigate to each match page and scrape lineups from DOM
            entries = _scrape_match_lineups(page, match_links, team_name, limit, _log)
            return entries
        except Exception as exc:
            _log(f"EXCEPTION: {exc}")
            return []
        finally:
            page.close()
            context.close()
            browser.close()


def _extract_match_links(page, team_name: str, limit: int, _log) -> list[dict]:
    """Extract match links and basic info from the team page DOM + __NEXT_DATA__."""
    # 1. Try DOM-based extraction
    data = page.evaluate("""([limit, teamSlug]) => {
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
            // Filter: match URL should contain team slug
            if (teamSlug && !href.toLowerCase().includes(teamSlug.toLowerCase())) continue;
            results.push({
                href: href,
                event_id: parseInt(idMatch[1]),
                text: a.innerText.substring(0, 100).replace(/\\n/g, ' | '),
            });
            if (results.length >= limit * 3) break;
        }
        return results;
    }""", [limit, _team_slug(page.url)])
    dom_links = data or []

    if len(dom_links) >= limit:
        return dom_links

    # 2. Supplement with __NEXT_DATA__ events (team page may have them in SSR)
    nd_events = page.evaluate("""() => {
        try {
            const nd = window.__NEXT_DATA__;
            if (!nd) return [];
            const ip = nd.props?.pageProps?.initialProps;
            if (!ip) return [];
            // Check for events in various possible locations
            const events = ip.lastEvents || ip.events || ip.recentEvents || [];
            return events.filter(e => e.status?.type === 'finished').map(e => ({
                href: '/football/match/' + (e.slug || '') + '/' + (e.customId || '') + '#id:' + e.id,
                event_id: e.id,
                text: (e.homeTeam?.name || '') + ' vs ' + (e.awayTeam?.name || ''),
            })).slice(0, 30);
        } catch(e) { return []; }
    }""")

    if nd_events:
        seen_ids = {ml["event_id"] for ml in dom_links}
        for ev in nd_events:
            if ev["event_id"] not in seen_ids:
                dom_links.append(ev)
                seen_ids.add(ev["event_id"])
        _log(f"__NEXT_DATA__ added {len(nd_events)} extra events")

    return dom_links


def _team_slug(url: str) -> str:
    """Extract team slug from Sofascore URL, e.g. 'real-madrid' from .../real-madrid/2829."""
    m = re.search(r'/([\w-]+)/\d+$', url.rstrip('/'))
    return m.group(1) if m else ''


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
        if consecutive_failures >= 3:
            _log("too many DOM scraping failures, stopping")
            break

        event_id = ml["event_id"]
        href = ml['href']
        match_url = href if href.startswith('http') else f"https://www.sofascore.com{href}"
        _log(f"navigating to match {event_id}: {match_url[:80]}...")

        on_response = None
        try:
            # Set up response interception for lineups
            lineup_data = {}
            lock = threading.Lock()
            eid = event_id  # capture for closure

            def on_response(resp, _eid=eid, _ld=lineup_data, _lk=lock):
                if f"/event/{_eid}/lineups" in resp.url and resp.status == 200:
                    try:
                        with _lk:
                            _ld["data"] = resp.json()
                    except Exception:
                        pass

            page.on("response", on_response)

            # Navigate to match page
            try:
                page.goto(match_url, wait_until="domcontentloaded")
            except Exception as nav_exc:
                _log(f"navigation failed for {event_id}: {str(nav_exc)[:100]}")
                page.remove_listener("response", on_response)
                consecutive_failures += 1
                # Wait a bit and try to recover page state
                page.wait_for_timeout(1000)
                continue

            page.wait_for_timeout(2000)

            # Click Lineups tab — try multiple selectors
            lineups_clicked = False
            try:
                for selector in [
                    "text=Lineups",
                    "text=Alineaciones",
                    "[data-tabid='lineups']",
                    "button:has-text('Lineups')",
                ]:
                    try:
                        tab = page.locator(selector).first
                        if tab.is_visible(timeout=2000):
                            tab.click()
                            lineups_clicked = True
                            break
                    except Exception:
                        continue
            except Exception:
                pass

            if lineups_clicked:
                page.wait_for_timeout(5000)  # longer wait for lineup data to load
            else:
                page.wait_for_timeout(2000)

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
            _log(f"match {event_id} exception: {str(exc)[:120]}")
            consecutive_failures += 1
            if on_response:
                try:
                    page.remove_listener("response", on_response)
                except Exception:
                    pass

    return entries


def _scrape_lineup_from_dom(page, _log) -> dict | None:
    """Last resort: scrape lineup data directly from rendered DOM."""
    try:
        data = page.evaluate("""() => {
            // Get formations from page text
            const text = document.body.innerText;
            const formationPattern = /\\b(\\d-\\d+-\\d+(?:-\\d+)?)\\b/g;
            const formations = [...new Set(text.match(formationPattern) || [])];
            if (formations.length < 1) return null;

            // Try to extract player names from various possible selectors
            const playerSelectors = [
                '[data-testid*="lineup"] [data-testid*="player"]',
                '[class*="lineupPlayer"]',
                '[class*="lineup"] [class*="name"]',
                '[class*="Lineup"] span',
                '[class*="formation"] [class*="player"]',
            ];
            let homePlayers = [];
            let awayPlayers = [];
            for (const sel of playerSelectors) {
                const els = document.querySelectorAll(sel);
                if (els.length > 0) {
                    const names = [...els].map(el => el.textContent.trim()).filter(Boolean);
                    // Split roughly in half for home/away
                    const mid = Math.floor(names.length / 2);
                    homePlayers = names.slice(0, mid);
                    awayPlayers = names.slice(mid);
                    break;
                }
            }

            return {
                home: {formation: formations[0], players: homePlayers.map(n => ({player: {name: n}}))},
                away: {formation: formations.length > 1 ? formations[1] : formations[0], players: awayPlayers.map(n => ({player: {name: n}}))},
                _source: 'dom_scrape'
            };
        }""")
        if data:
            _log(f"DOM scrape found formations: {data.get('home',{}).get('formation')} / {data.get('away',{}).get('formation')}")
        return data
    except Exception as exc:
        _log(f"DOM scrape exception: {str(exc)[:80]}")
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
