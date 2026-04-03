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
    import time as _time
    import signal
    _check_auth(authorization)
    t0 = _time.time()
    MAX_SECONDS = 240  # 4 minute hard cap
    print(f"[crawl] START team={req.team_name!r} limit={req.limit} team_url={req.team_url!r}", flush=True)

    from playwright.sync_api import sync_playwright
    from sofascore_formations_crawler import (
        build_context, _new_stealth_page, resolve_team_url,
        dismiss_overlays, wait_short, click_tab_if_present,
        ensure_lineups_tab, process_match,
    )
    from urllib.parse import urljoin
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
                    "--single-process",
                ],
            )
            context = build_context(browser)
            page = _new_stealth_page(context)
            page.set_default_timeout(15000)
            page.set_default_navigation_timeout(25000)

            # Resolve team URL
            team_page_url = req.team_url or resolve_team_url(page, req.team_name)
            print(f"[crawl] [{_time.time()-t0:.1f}s] team URL: {team_page_url}", flush=True)

            page.goto(team_page_url, wait_until="domcontentloaded")
            dismiss_overlays(page)
            page.wait_for_timeout(2000)

            # Collect match URLs directly (simplified — no scrolling loop)
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

            print(f"[crawl] [{_time.time()-t0:.1f}s] found {len(match_urls)} match URLs", flush=True)

            # Process matches
            entries: list[FormationEntry] = []
            for idx, match_url in enumerate(match_urls):
                if len(entries) >= req.limit:
                    break
                if _time.time() - t0 > MAX_SECONDS:
                    print(f"[crawl] [{_time.time()-t0:.1f}s] time limit reached, returning {len(entries)} results", flush=True)
                    break
                print(f"[crawl] [{_time.time()-t0:.1f}s] match {idx+1}: {match_url}", flush=True)
                try:
                    result = process_match(
                        page=page,
                        match_url=match_url,
                        team_query=req.team_name,
                        output_dir=tmp_dir,
                        index=len(entries) + 1,
                    )
                    if result is None:
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
                    print(f"[crawl] [{_time.time()-t0:.1f}s] ✓ {result.formation} vs {result.opponent}", flush=True)
                except Exception as exc:
                    print(f"[crawl] [{_time.time()-t0:.1f}s] ✗ skip: {exc}", flush=True)
                    continue

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


# ── main ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port, timeout_keep_alive=300)
