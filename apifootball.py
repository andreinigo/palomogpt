"""API-Football.com client — formations, squads, fixtures.

Pro plan (7 500 req/day, all seasons).  Called directly from the
Streamlit app; no Railway scraper service needed.
"""
from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any, Optional

import requests

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_BASE = "https://v3.football.api-sports.io"

# Position mapping: API-Football → app convention
_POS_MAP = {
    "Goalkeeper": "GK",
    "Defender": "DEF",
    "Midfielder": "MID",
    "Attacker": "FWD",
}

# In-memory cache: lowered team name → (team_id, canonical_name)
_team_cache: dict[str, tuple[int, str]] = {}


def _headers() -> dict[str, str]:
    import streamlit as _st
    key = _st.secrets.get("APIFOOTBALL_KEY", "")
    if not key:
        raise RuntimeError("APIFOOTBALL_KEY not set in Streamlit secrets")
    return {"x-apisports-key": key}


def _current_season() -> int:
    """Return the season year for the current football season.

    API-Football uses the *start* year of a season, so the 2025/26 season
    is ``2025``.  We assume Aug 1 as the cutover.
    """
    from config import CURRENT_DATE
    from datetime import datetime

    try:
        dt = datetime.strptime(CURRENT_DATE, "%d de %B de %Y")
    except ValueError:
        dt = datetime.now()
    return dt.year if dt.month >= 8 else dt.year - 1


# ---------------------------------------------------------------------------
# Team resolution
# ---------------------------------------------------------------------------

def resolve_team(name: str) -> tuple[int, str] | None:
    """Search API-Football for *name* and return ``(team_id, canonical_name)``
    or ``None``.  Results are cached in-process."""
    clean = re.sub(r"\s*\(equipo femenino\)\s*$", "", name.strip(), flags=re.I)
    key = clean.lower()
    if key in _team_cache:
        return _team_cache[key]

    resp = requests.get(
        f"{_BASE}/teams",
        headers=_headers(),
        params={"search": clean},
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()

    if data.get("errors"):
        print(f"[api-football] team search errors: {data['errors']}")
        return None

    teams = data.get("response", [])
    if not teams:
        print(f"[api-football] no team found for '{clean}'")
        return None

    best_id, best_score, best_name = None, 0.0, ""
    for t in teams:
        tm = t["team"]
        score = SequenceMatcher(None, key, tm["name"].lower()).ratio()
        if score > best_score:
            best_score = score
            best_id = tm["id"]
            best_name = tm["name"]

    if best_id is None:
        return None

    result = (best_id, best_name)
    _team_cache[key] = result
    print(f"[api-football] resolved '{clean}' → {best_name} (id={best_id}, score={best_score:.2f})")
    return result


# ---------------------------------------------------------------------------
# Squad / roster
# ---------------------------------------------------------------------------

def get_squad(team_id: int) -> list[dict[str, Any]]:
    """Return the current squad via ``/players/squads``.

    Returns ``[{name, position, number, photo}]`` with positions mapped to
    ``GK / DEF / MID / FWD``.
    """
    resp = requests.get(
        f"{_BASE}/players/squads",
        headers=_headers(),
        params={"team": team_id},
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()

    entries = data.get("response", [])
    if not entries:
        return []

    players: list[dict[str, Any]] = []
    for squad_block in entries:
        for p in squad_block.get("players", []):
            pos_raw = p.get("position", "")
            players.append({
                "name": p.get("name", ""),
                "position": _POS_MAP.get(pos_raw, pos_raw),
                "number": p.get("number"),
                "photo": p.get("photo", ""),
            })

    # Sort: GK → DEF → MID → FWD
    order = {"GK": 0, "DEF": 1, "MID": 2, "FWD": 3}
    players.sort(key=lambda x: order.get(x["position"], 9))
    return players


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def get_fixtures(
    team_id: int,
    limit: int = 30,
    *,
    seasons: tuple[int, ...] | None = None,
) -> list[dict]:
    """Return the most recent finished fixtures for a team.

    Tries the current season first, then falls back to older seasons.
    Returns raw API-Football fixture dicts sorted by date descending.
    """
    if seasons is None:
        cur = _current_season()
        seasons = tuple(range(cur, cur - 3, -1))

    for season in seasons:
        resp = requests.get(
            f"{_BASE}/fixtures",
            headers=_headers(),
            params={"team": team_id, "season": season, "status": "FT"},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        if data.get("errors"):
            continue
        fixtures = data.get("response", [])
        if fixtures:
            fixtures.sort(
                key=lambda f: f["fixture"].get("date", ""), reverse=True,
            )
            print(f"[api-football] {len(fixtures)} finished fixtures (season={season})")
            return fixtures[:limit]

    print("[api-football] no fixtures found")
    return []


def get_fixture_results(
    team_id: int,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Return a compact list of recent results for grounding LLM prompts.

    Each dict: ``{date, home, away, score, tournament}``.
    """
    fixtures = get_fixtures(team_id, limit=limit)
    results: list[dict[str, Any]] = []
    for fx in fixtures:
        info = fx["fixture"]
        teams = fx["teams"]
        goals = fx.get("goals", {})
        league = fx.get("league", {})
        results.append({
            "date": (info.get("date") or "")[:10],
            "home": teams["home"]["name"],
            "away": teams["away"]["name"],
            "score": f"{goals.get('home', '?')}-{goals.get('away', '?')}",
            "tournament": league.get("name", ""),
        })
    return results


# ---------------------------------------------------------------------------
# Formations / lineups
# ---------------------------------------------------------------------------

def _get_lineup(
    fixture_id: int,
    fixture_data: dict,
    team_id: int,
    index: int,
) -> dict | None:
    """Fetch lineup for a single fixture and return a UI-compatible dict."""
    resp = requests.get(
        f"{_BASE}/fixtures/lineups",
        headers=_headers(),
        params={"fixture": fixture_id},
        timeout=15,
    )
    resp.raise_for_status()
    lineups = resp.json().get("response", [])
    if not lineups:
        return None

    target = None
    for lu in lineups:
        if lu["team"]["id"] == team_id:
            target = lu
            break
    if not target or not target.get("formation"):
        return None

    fx = fixture_data["fixture"]
    teams = fixture_data["teams"]
    home_name = teams["home"]["name"]
    away_name = teams["away"]["name"]
    is_home = teams["home"]["id"] == team_id

    starters = target.get("startXI", [])
    player_names = [
        f"{p['player'].get('number', '')} {p['player']['name']}"
        for p in starters
        if p.get("player", {}).get("name")
    ]

    return {
        "index": index,
        "match_url": "",
        "match_date": (fx.get("date") or "")[:10] or None,
        "home_team": home_name,
        "away_team": away_name,
        "target_team": home_name if is_home else away_name,
        "opponent": away_name if is_home else home_name,
        "target_side": "left" if is_home else "right",
        "formation": target["formation"],
        "players": player_names,
        "image_bytes": None,
    }


def get_formations(team_name: str, limit: int = 10) -> list[dict]:
    """Fetch recent formations for *team_name*.

    Resolves the team, fetches finished fixtures, then grabs lineups.
    Returns a list of dicts compatible with ``_render_formations``.
    """
    resolved = resolve_team(team_name)
    if not resolved:
        return []
    team_id, canonical = resolved

    fixtures = get_fixtures(team_id, limit=limit + 5)
    if not fixtures:
        return []

    entries: list[dict] = []
    failures = 0
    for fx in fixtures:
        if len(entries) >= limit:
            break
        if failures >= 3:
            print("[api-football] too many lineup failures, stopping")
            break
        fixture_id = fx["fixture"]["id"]
        entry = _get_lineup(fixture_id, fx, team_id, len(entries) + 1)
        if entry:
            entries.append(entry)
            failures = 0
        else:
            failures += 1

    print(f"[api-football] {len(entries)} formations for {canonical}")
    return entries
