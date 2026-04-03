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
def debug():
    """Hit Sofascore and return what the browser sees (title + snippet)."""
    from playwright.sync_api import sync_playwright
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            ctx = browser.new_context(
                user_agent="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
                viewport={"width": 1600, "height": 900},
            )
            page = ctx.new_page()
            try:
                from playwright_stealth import stealth_sync
                stealth_sync(page)
            except ImportError:
                pass
            page.goto("https://www.sofascore.com", wait_until="domcontentloaded", timeout=20000)
            page.wait_for_timeout(3000)
            title = page.title()
            body = page.locator("body").inner_text(timeout=5000)[:500]
            ctx.close()
            browser.close()
        return {"title": title, "body_snippet": body}
    except Exception as exc:
        return {"error": str(exc)}


@app.post("/crawl", response_model=CrawlResponse)
def crawl(req: CrawlRequest, authorization: str | None = Header(default=None)):
    _check_auth(authorization)

    tmp_dir = Path(tempfile.mkdtemp(prefix="scraper_"))
    try:
        lineups = crawl_team_lineups(
            team_query=req.team_name,
            limit=req.limit,
            output_dir=tmp_dir,
            team_url=req.team_url,
            headless=True,
        )

        entries: list[FormationEntry] = []
        for lu in lineups:
            d = asdict(lu)
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

        return CrawlResponse(
            team_name=req.team_name,
            count=len(entries),
            formations=entries,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ── main ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
