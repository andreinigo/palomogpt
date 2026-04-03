"""Scraper micro-service — wraps sofascore_formations_crawler behind a FastAPI endpoint.

Deployed on Railway with Playwright + Chromium pre-installed (see Dockerfile).
The Streamlit app calls POST /crawl to get formation data without needing
a local browser.
"""

from __future__ import annotations

import base64
import os
import shutil
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from sofascore_formations_crawler import crawl_team_lineups  # noqa: E402

app = FastAPI(title="Sofascore Scraper Service")

# Optional bearer token for minimal auth (set SCRAPER_API_KEY env var).
_API_KEY = os.getenv("SCRAPER_API_KEY", "")


# ── request / response models ────────────────────────────────────────────

class CrawlRequest(BaseModel):
    team_name: str = Field(..., min_length=1, max_length=200)
    limit: int = Field(default=10, ge=1, le=30)
    team_url: Optional[str] = Field(default=None, description="Direct Sofascore team URL — bypasses search")


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
        return  # no key configured → open access
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
    if authorization.split(" ", 1)[1] != _API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")


# ── endpoints ─────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/debug")
def debug(team_url: str = "https://www.sofascore.com/team/football/real-madrid/2829"):
    """Debug: navigate to team page, find matches, try processing first match."""
    from playwright.sync_api import sync_playwright
    import time as _time
    log: list[str] = []
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            ctx = browser.new_context(
                user_agent="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
                viewport={"width": 1600, "height": 2200},
                color_scheme="dark",
            )
            page = ctx.new_page()
            try:
                from playwright_stealth import stealth_sync
                stealth_sync(page)
            except ImportError:
                pass
            page.set_default_timeout(12000)
            page.set_default_navigation_timeout(20000)

            t0 = _time.time()
            page.goto(team_url, wait_until="domcontentloaded", timeout=20000)
            page.wait_for_timeout(2000)
            log.append(f"[{_time.time()-t0:.1f}s] team page loaded: {page.title()}")

            # Try clicking Matches/Results tabs
            from sofascore_formations_crawler import click_tab_if_present, dismiss_overlays
            dismiss_overlays(page)
            click_tab_if_present(page, "Matches")
            click_tab_if_present(page, "Results")
            log.append(f"[{_time.time()-t0:.1f}s] clicked tabs")

            match_links = page.locator("a[href*='/football/match/']")
            hrefs = match_links.evaluate_all(
                "els => els.map(e => e.href || e.getAttribute('href')).filter(Boolean)"
            )
            log.append(f"[{_time.time()-t0:.1f}s] found {len(hrefs)} match hrefs")

            if hrefs:
                match_url = hrefs[0].split("?")[0].split("#")[0]
                if not match_url.startswith("http"):
                    match_url = "https://www.sofascore.com" + match_url
                log.append(f"[{_time.time()-t0:.1f}s] navigating to match: {match_url}")
                page.goto(match_url, wait_until="domcontentloaded", timeout=20000)
                page.wait_for_timeout(2000)
                log.append(f"[{_time.time()-t0:.1f}s] match page title: {page.title()}")

                # Try lineup tab
                from sofascore_formations_crawler import ensure_lineups_tab
                found_tab = ensure_lineups_tab(page)
                log.append(f"[{_time.time()-t0:.1f}s] lineup tab found: {found_tab}")

                if found_tab:
                    page.wait_for_timeout(2000)
                    card_loc = page.locator("div:has(.FootballTerrainHalf__root--side_left):has(.FootballTerrainHalf__root--side_right)").first
                    try:
                        card_loc.wait_for(timeout=12000)
                        log.append(f"[{_time.time()-t0:.1f}s] lineup card found!")
                        body_snippet = card_loc.inner_text(timeout=5000)[:200]
                        log.append(f"card text: {body_snippet}")
                    except Exception as e:
                        log.append(f"[{_time.time()-t0:.1f}s] lineup card NOT found: {e}")

            ctx.close()
            browser.close()
            log.append(f"[{_time.time()-t0:.1f}s] done")
        return {"log": log}
    except Exception as exc:
        log.append(f"ERROR: {exc}")
        return {"log": log}


@app.post("/crawl", response_model=CrawlResponse)
def crawl(req: CrawlRequest, authorization: str | None = Header(default=None)):
    """Crawl Sofascore for recent formations using a hybrid approach:
    1. Load team page in browser to get session cookies
    2. Extract cookies and use Python httpx to call Sofascore API (proper timeouts)
    3. Fall back to full page scraping only if API fails
    """
    import time as _time
    _check_auth(authorization)
    t0 = _time.time()
    MAX_SECONDS = 240
    print(f"[crawl] START team={req.team_name!r} limit={req.limit} team_url={req.team_url!r}", flush=True)

    from playwright.sync_api import sync_playwright
    from sofascore_formations_crawler import (
        build_context, _new_stealth_page, resolve_team_url,
        dismiss_overlays, click_tab_if_present,
    )
    import re

    tmp_dir = Path(tempfile.mkdtemp(prefix="scraper_"))
    tmp_dir.mkdir(parents=True, exist_ok=True)

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(
                headless=True,
                args=[
                    "--no-sandbox",
                    "--disable-gpu",
                    "--disable-dev-shm-usage",
                    "--disable-software-rasterizer",
                ],
            )
            context = build_context(browser)
            page = _new_stealth_page(context)
            page.set_default_timeout(12000)
            page.set_default_navigation_timeout(20000)

            # Resolve team URL and extract team_id
            team_page_url = req.team_url or resolve_team_url(page, req.team_name)
            print(f"[crawl] [{_time.time()-t0:.1f}s] team URL: {team_page_url}", flush=True)

            team_id_match = re.search(r"/(\d+)$", team_page_url.rstrip("/"))
            team_id = team_id_match.group(1) if team_id_match else None

            # Navigate to team page to establish session + get cookies
            page.goto(team_page_url, wait_until="domcontentloaded")
            dismiss_overlays(page)
            page.wait_for_timeout(2000)

            # Extract cookies from browser context for Python-side API calls
            cookies = context.cookies("https://www.sofascore.com")
            cookie_header = "; ".join(f"{c['name']}={c['value']}" for c in cookies)
            user_agent = page.evaluate("() => navigator.userAgent")
            print(f"[crawl] [{_time.time()-t0:.1f}s] extracted {len(cookies)} cookies", flush=True)

            # Close browser early — we don't need it for API calls
            page.close()
            context.close()
            browser.close()

        # Now use Python httpx for API calls (proper timeouts, no browser needed)
        entries: list[FormationEntry] = []
        api_success = False

        if team_id and cookie_header:
            print(f"[crawl] [{_time.time()-t0:.1f}s] trying API approach with team_id={team_id}", flush=True)
            try:
                entries = _crawl_via_api(
                    team_id, req.team_name, req.limit,
                    cookie_header, user_agent,
                    t0, MAX_SECONDS,
                )
                api_success = len(entries) > 0
                print(f"[crawl] [{_time.time()-t0:.1f}s] API returned {len(entries)} formations", flush=True)
            except Exception as exc:
                print(f"[crawl] [{_time.time()-t0:.1f}s] API approach failed: {exc}", flush=True)

        # Fall back to page scraping if API didn't work
        if not api_success:
            print(f"[crawl] [{_time.time()-t0:.1f}s] falling back to page scraping", flush=True)
            with sync_playwright() as p:
                browser = p.chromium.launch(
                    headless=True,
                    args=["--no-sandbox", "--disable-gpu", "--disable-dev-shm-usage", "--disable-software-rasterizer"],
                )
                context = build_context(browser)
                page = _new_stealth_page(context)
                page.set_default_timeout(12000)
                page.set_default_navigation_timeout(20000)
                page.goto(team_page_url, wait_until="domcontentloaded")
                dismiss_overlays(page)
                page.wait_for_timeout(1500)

                entries = _crawl_via_pages(
                    page, context, req.team_name, req.limit,
                    tmp_dir, t0, MAX_SECONDS,
                )
                page.close()
                context.close()
                browser.close()

        print(f"[crawl] [{_time.time()-t0:.1f}s] DONE count={len(entries)}", flush=True)
        return CrawlResponse(
            team_name=req.team_name,
            count=len(entries),
            formations=entries,
        )
    except Exception as exc:
        print(f"[crawl] [{_time.time()-t0:.1f}s] ERROR: {exc}", flush=True)
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _crawl_via_api(
    team_id: str,
    team_name: str,
    limit: int,
    cookie_header: str,
    user_agent: str,
    t0: float,
    max_seconds: float,
) -> list[FormationEntry]:
    """Use Sofascore API with cookies extracted from browser session."""
    import time as _time
    import httpx
    from difflib import SequenceMatcher

    headers = {
        "Cookie": cookie_header,
        "User-Agent": user_agent,
        "Referer": "https://www.sofascore.com/",
        "Accept": "application/json",
    }

    # Fetch events (paginate if needed)
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
        raise RuntimeError("API returned no events")

    entries: list[FormationEntry] = []
    consecutive_failures = 0

    for event in all_events:
        if len(entries) >= limit:
            break
        if _time.time() - t0 > max_seconds:
            break
        if consecutive_failures >= 5:
            print(f"[api] [{_time.time()-t0:.1f}s] too many consecutive failures, stopping", flush=True)
            break

        event_id = event.get("id")
        home = event.get("homeTeam", {}).get("name", "")
        away = event.get("awayTeam", {}).get("name", "")
        status_type = event.get("status", {}).get("type", "")

        if status_type != "finished":
            continue

        # Get lineups via API with httpx timeout
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

        # Extract formation and players
        home_lineup = lineups_data.get("home", {})
        away_lineup = lineups_data.get("away", {})

        # Determine which side is our target team
        home_ratio = SequenceMatcher(None, team_name.lower(), home.lower()).ratio()
        away_ratio = SequenceMatcher(None, team_name.lower(), away.lower()).ratio()
        if home_ratio >= away_ratio:
            target_side = "home"
            target_team = home
            opponent = away
            target_lineup = home_lineup
        else:
            target_side = "away"
            target_team = away
            opponent = home
            target_lineup = away_lineup

        formation = target_lineup.get("formation")
        if not formation:
            consecutive_failures += 1
            continue

        # Extract players (starters)
        players_raw = target_lineup.get("players", [])
        player_names = []
        for p in players_raw:
            player = p.get("player", {})
            shirt = p.get("shirtNumber", "")
            name = player.get("name", "")
            if name and p.get("substitute", False) is False:
                player_names.append(f"{shirt} {name}")

        # Extract match date
        start_ts = event.get("startTimestamp")
        match_date = None
        if start_ts:
            from datetime import datetime, timezone
            match_date = datetime.fromtimestamp(start_ts, tz=timezone.utc).strftime("%Y-%m-%d")

        # Build match URL
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
            image_base64=None,  # API approach skips screenshots
        ))
        consecutive_failures = 0
        print(f"[api] [{_time.time()-t0:.1f}s] ✓ {formation} vs {opponent} ({match_date})", flush=True)

    return entries


def _crawl_via_pages(
    page: "Page",
    context: "BrowserContext",
    team_name: str,
    limit: int,
    tmp_dir: Path,
    t0: float,
    max_seconds: float,
) -> list[FormationEntry]:
    """Fall back to full page-based scraping when API doesn't work."""
    import time as _time
    import re
    from urllib.parse import urljoin
    from sofascore_formations_crawler import (
        _new_stealth_page, click_tab_if_present, process_match,
    )

    # Collect match URLs from current page
    click_tab_if_present(page, "Matches")
    click_tab_if_present(page, "Results")
    page.wait_for_timeout(1500)

    hrefs_raw = page.locator("a[href*='/football/match/']").evaluate_all(
        "els => els.map(e => e.href || e.getAttribute('href')).filter(Boolean)"
    )
    seen: set[str] = set()
    match_urls: list[str] = []
    for href in hrefs_raw:
        full = urljoin("https://www.sofascore.com", href.split("?")[0].split("#")[0])
        if re.search(r"/football/match/", full) and full not in seen:
            seen.add(full)
            match_urls.append(full)

    print(f"[pages] [{_time.time()-t0:.1f}s] found {len(match_urls)} match URLs", flush=True)

    entries: list[FormationEntry] = []
    consecutive_failures = 0
    for idx, match_url in enumerate(match_urls):
        if len(entries) >= limit:
            break
        if _time.time() - t0 > max_seconds:
            print(f"[pages] [{_time.time()-t0:.1f}s] time limit reached", flush=True)
            break
        if consecutive_failures >= 5:
            print(f"[pages] [{_time.time()-t0:.1f}s] too many consecutive failures, stopping", flush=True)
            break

        print(f"[pages] [{_time.time()-t0:.1f}s] match {idx+1}: {match_url}", flush=True)
        match_page = _new_stealth_page(context)
        match_page.set_default_timeout(12000)
        match_page.set_default_navigation_timeout(20000)
        try:
            result = process_match(
                page=match_page,
                match_url=match_url,
                team_query=team_name,
                output_dir=tmp_dir,
                index=len(entries) + 1,
            )
            if result is None:
                consecutive_failures += 1
                continue

            d = asdict(result)
            img_path = Path(d.get("image_path", ""))
            img_b64: str | None = None
            if img_path.exists():
                img_b64 = base64.b64encode(img_path.read_bytes()).decode()

            entries.append(FormationEntry(
                index=d["index"],
                match_url=d["match_url"],
                match_date=d.get("match_date"),
                home_team=d.get("home_team", ""),
                away_team=d.get("away_team", ""),
                target_team=d.get("target_team", ""),
                opponent=d.get("opponent", ""),
                target_side=d.get("target_side", ""),
                formation=d.get("formation"),
                players=d.get("players", []),
                image_base64=img_b64,
            ))
            consecutive_failures = 0
            print(f"[pages] [{_time.time()-t0:.1f}s] ✓ {result.formation} vs {result.opponent}", flush=True)
        except Exception as exc:
            consecutive_failures += 1
            print(f"[pages] [{_time.time()-t0:.1f}s] ✗ skip: {exc}", flush=True)
        finally:
            match_page.close()

    return entries


# ── main ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port, timeout_keep_alive=300)
