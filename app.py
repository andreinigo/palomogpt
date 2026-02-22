#!/usr/bin/env python3
"""
Palomo Fact Agent (POC) — "facts que no alucinan" para jugador/equipo/estadio.

- Usa soccerdata para traer stats estructuradas (FBref + opcional Understat).
- Genera facts tipo “wow” (percentiles, rachas simples, splits casa/fuera, rankings).
- Cada fact trae: claim, metric, value, scope, method, provenance, confidence.
- Si no hay evidencia suficiente -> NO lo dice (lo descarta o lo marca como "insufficient_data").

NOTAS IMPORTANTES (realidad 2026):
- FBref ha cambiado disponibilidad de “advanced stats” en varias ocasiones; este POC está diseñado
  para degradar con gracia: si no encuentra una métrica, la omite y lo reporta.
- Para xG robusto, suele ser mejor Understat (si soccerdata lo soporta para tu liga/temporada).
"""

from __future__ import annotations

import os
import json
import math
import re
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

# --------- Dependencies ----------
# pip install soccerdata pandas requests beautifulsoup4 lxml
import soccerdata as sd

import requests
from bs4 import BeautifulSoup

# Timeout (seconds) for soccerdata network calls that may hang
SOCCERDATA_TIMEOUT = int(os.environ.get("SOCCERDATA_TIMEOUT", "60"))

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None


if load_dotenv is not None:
    load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")


# -----------------------------
# Configuration
# -----------------------------
DEFAULT_LEAGUE = "ENG-Premier League"   # Cambia a "ESP-La Liga", "ITA-Serie A", etc.
DEFAULT_SEASON = datetime.now().year    # dynamic: always current year
DATA_DIR = os.environ.get("SOCCERDATA_DIR", os.path.expanduser("~/soccerdata"))
REQUEST_TIMEOUT = 20
DEFAULT_AGENT_MODEL = os.environ.get("OPENAI_MODEL", "gpt-5.2")
OPENAI_API_BASE = os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")


# -----------------------------
# Data model
# -----------------------------
@dataclass
class Fact:
    claim: str
    metric: str
    value: Any
    scope: str
    method: str
    provenance: Dict[str, Any]  # source + table + columns + filters + timestamps
    confidence: str             # "HIGH" | "MED" (no usamos LOW por defecto)
    wow_score: int              # 0..100


# -----------------------------
# Utility helpers
# -----------------------------
def _now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def safe_div(a: float, b: float) -> Optional[float]:
    if b == 0 or pd.isna(b):
        return None
    return a / b


def to_float(x: Any) -> Optional[float]:
    try:
        if pd.isna(x):
            return None
        return float(x)
    except Exception:
        return None


def clamp_int(x: float, lo: int = 0, hi: int = 100) -> int:
    return int(max(lo, min(hi, round(x))))


def normalize_name(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip()).lower()


def percentile_rank(series: pd.Series, value: float, higher_is_better: bool = True) -> Optional[float]:
    """
    Returns percentile in [0,1] where 1.0 means best.
    """
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty or value is None or pd.isna(value):
        return None
    if higher_is_better:
        return (s <= value).mean()
    else:
        return (s >= value).mean()


# -----------------------------
# Fetch layer (soccerdata + web)
# -----------------------------
class DataClient:
    """
    Thin wrapper around soccerdata + a minimal web fetcher for stadium facts.
    """

    def __init__(self, league: str, season: int, data_dir: str = DATA_DIR, provider: str = "auto"):
        self.league = league
        self.season = season
        self.provider = provider
        data_root = Path(data_dir)
        self.fbref = None
        self.understat = None

        if provider in ("auto", "fbref"):
            try:
                self.fbref = sd.FBref(leagues=league, seasons=season, data_dir=data_root / "data" / "FBref")
            except Exception:
                self.fbref = None

        # Optional: Understat (xG etc). Some leagues/seasons supported.
        # If it fails, we just skip xG-based facts.
        if provider in ("auto", "understat"):
            try:
                self.understat = sd.Understat(leagues=league, seasons=season, data_dir=data_root / "data" / "Understat")
            except Exception:
                self.understat = None

    def read_schedule(self) -> pd.DataFrame:
        if self.fbref is not None:
            return self.fbref.read_schedule()
        if self.understat is not None:
            return self.understat.read_schedule()
        raise RuntimeError("No provider available for schedule.")

    def read_team_season_stats(self, stat_type: str) -> pd.DataFrame:
        if self.fbref is None:
            raise RuntimeError("FBref provider disabled/unavailable.")
        return self.fbref.read_team_season_stats(stat_type=stat_type)

    def read_player_season_stats(self, stat_type: str) -> pd.DataFrame:
        if self.fbref is None:
            raise RuntimeError("FBref provider disabled/unavailable.")
        return self.fbref.read_player_season_stats(stat_type=stat_type)

    def _run_with_timeout(self, fn, timeout: int = SOCCERDATA_TIMEOUT):
        """Run a callable in a thread with a timeout to prevent hangs."""
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(fn)
            return future.result(timeout=timeout)

    def read_understat_player_season(self) -> Optional[pd.DataFrame]:
        if not self.understat:
            return None
        # Understat schema differs; try a safe call. If it breaks, skip.
        try:
            return self._run_with_timeout(self.understat.read_player_season_stats)
        except FuturesTimeoutError:
            print(f"[DataClient] Timeout fetching Understat player season stats after {SOCCERDATA_TIMEOUT}s")
            return None
        except Exception as e:
            print(f"[DataClient] Error fetching Understat player stats: {type(e).__name__}: {e}")
            return None

    def read_understat_team_matches(self) -> Optional[pd.DataFrame]:
        if not self.understat:
            return None
        try:
            return self._run_with_timeout(self.understat.read_team_match_stats)
        except FuturesTimeoutError:
            print(f"[DataClient] Timeout fetching Understat team match stats after {SOCCERDATA_TIMEOUT}s")
            return None
        except Exception as e:
            print(f"[DataClient] Error fetching Understat team stats: {type(e).__name__}: {e}")
            return None

    def fetch_stadium_facts_wikipedia(self, stadium_name: str) -> Dict[str, Any]:
        """
        Minimal stadium fact fetch: Wikipedia page parse for infobox.
        This is *not* perfect, but good enough for MVP.

        Returns dict with: capacity, opened, record_attendance (if found), sources[]
        """
        # Simple heuristic: search URL directly by replacing spaces
        # If you want “no-guessing”, swap this for a Wikipedia search API call.
        title = stadium_name.replace(" ", "_")
        url = f"https://en.wikipedia.org/wiki/{title}"

        r = requests.get(url, timeout=REQUEST_TIMEOUT, headers={"User-Agent": "PalomoFactAgent/0.1"})
        if r.status_code != 200:
            return {"stadium": stadium_name, "error": f"http_{r.status_code}", "sources": [url]}

        soup = BeautifulSoup(r.text, "lxml")
        infobox = soup.select_one("table.infobox")

        out = {"stadium": stadium_name, "capacity": None, "opened": None, "record_attendance": None, "sources": [url], "fetched_at": _now_iso()}

        if not infobox:
            return out

        def pick_value(label: str) -> Optional[str]:
            row = infobox.find("th", string=re.compile(rf"^{re.escape(label)}$", re.I))
            if not row:
                return None
            td = row.find_parent("tr").find("td")
            if not td:
                return None
            return re.sub(r"\[\d+\]", "", td.get_text(" ", strip=True))

        out["capacity"] = pick_value("Capacity")
        out["opened"] = pick_value("Opened")
        out["record_attendance"] = pick_value("Record attendance")
        return out


# -----------------------------
# Fact generation (no hallucination)
# -----------------------------
class FactForge:
    def __init__(self, client: DataClient, narrator_instructions: str = ""):
        self.client = client
        self.narrator_instructions = narrator_instructions

    # ---------
    # Player facts
    # ---------
    def generate_player_facts(self, player_name: str) -> Tuple[List[Fact], List[str]]:
        """
        Returns (facts, notes). Notes are “what we couldn't do”.
        """
        notes: List[str] = []
        facts: List[Fact] = []

        if self.client.provider == "understat":
            return self._generate_player_facts_understat(player_name)

        # Pull multiple stat tables — degrade gracefully if some fail.
        tables: Dict[str, pd.DataFrame] = {}
        for stat_type in ["standard", "shooting", "passing", "gca", "defense", "possession", "playing_time", "misc"]:
            try:
                tables[stat_type] = self.client.read_player_season_stats(stat_type=stat_type)
            except Exception as e:
                notes.append(f"FBref player table failed: stat_type={stat_type} err={type(e).__name__}")

        if not tables:
            notes.append("No player season tables available; cannot generate player facts.")
            return facts, notes

        # Find player row by fuzzy name match across available tables
        player_key = normalize_name(player_name)

        def find_player_row(df: pd.DataFrame) -> Optional[pd.Series]:
            # Typical FBref columns include: 'player' (or 'player_name'), 'team', 'nation', etc.
            name_col = None
            for c in df.columns:
                if normalize_name(c) in ["player", "player_name", "name"]:
                    name_col = c
                    break
            if not name_col:
                return None

            # Fuzzy: exact normalized match first, then contains
            df2 = df.copy()
            df2["_nm"] = df2[name_col].astype(str).map(normalize_name)
            exact = df2[df2["_nm"] == player_key]
            if len(exact) == 1:
                return exact.iloc[0]

            contains = df2[df2["_nm"].str.contains(re.escape(player_key), na=False)]
            if len(contains) == 1:
                return contains.iloc[0]

            # If multiple matches, refuse to guess
            if len(exact) > 1 or len(contains) > 1:
                return None
            return None

        # Choose a “base” table to identify player and team
        base_df = tables.get("standard") or next(iter(tables.values()))
        row = find_player_row(base_df)
        if row is None:
            notes.append("Player name ambiguous or not found in base table; refusing to guess.")
            return facts, notes

        # Identify columns robustly
        def get_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
            norm_map = {normalize_name(c): c for c in df.columns}
            for cand in candidates:
                c = norm_map.get(normalize_name(cand))
                if c:
                    return c
            return None

        player_display = row.get(get_col(base_df, ["player", "player_name", "name"]), player_name)
        team = row.get(get_col(base_df, ["team", "squad"]), None)
        minutes = to_float(row.get(get_col(base_df, ["minutes", "min"]), None))
        matches = to_float(row.get(get_col(base_df, ["games", "matches", "mp"]), None))

        scope = f"{self.client.league} {self.client.season}"
        fetched_at = _now_iso()

        # 1) “Evergreen” fact: minutes share / durability
        if minutes is not None:
            wow = 45 if minutes >= 2000 else 25
            facts.append(Fact(
                claim=f"{player_display} ya suma {int(minutes)} minutos en liga esta temporada.",
                metric="minutes",
                value=int(minutes),
                scope=scope,
                method="From FBref player_season_stats(stat_type=standard).",
                provenance={"source": "FBref via soccerdata", "stat_type": "standard", "filters": {"player": str(player_display)}, "fetched_at": fetched_at},
                confidence="HIGH",
                wow_score=wow,
            ))

        # 2) Per-90 facts from standard: goals/assists if available
        g_col = get_col(base_df, ["goals", "gls"])
        a_col = get_col(base_df, ["assists", "ast"])
        if minutes and minutes > 0:
            if g_col and to_float(row.get(g_col)) is not None:
                gls = to_float(row.get(g_col))
                g90 = safe_div(gls, minutes) * 90 if gls is not None else None
                if g90 is not None:
                    # wow if high rate
                    wow = clamp_int(30 + 120 * max(0, g90 - 0.25))  # simple heuristic
                    facts.append(Fact(
                        claim=f"Producción de gol: {player_display} promedia {g90:.2f} goles por 90 minutos.",
                        metric="goals_per90",
                        value=round(g90, 3),
                        scope=scope,
                        method=f"goles/minutos*90 using columns {g_col} and minutes.",
                        provenance={"source": "FBref via soccerdata", "stat_type": "standard", "columns": [g_col, "minutes"], "fetched_at": fetched_at},
                        confidence="HIGH",
                        wow_score=wow,
                    ))
            if a_col and to_float(row.get(a_col)) is not None:
                ast = to_float(row.get(a_col))
                a90 = safe_div(ast, minutes) * 90 if ast is not None else None
                if a90 is not None:
                    wow = clamp_int(25 + 120 * max(0, a90 - 0.20))
                    facts.append(Fact(
                        claim=f"Creador: {player_display} promedia {a90:.2f} asistencias por 90.",
                        metric="assists_per90",
                        value=round(a90, 3),
                        scope=scope,
                        method=f"asistencias/minutos*90 using columns {a_col} and minutes.",
                        provenance={"source": "FBref via soccerdata", "stat_type": "standard", "columns": [a_col, "minutes"], "fetched_at": fetched_at},
                        confidence="HIGH",
                        wow_score=wow,
                    ))

        # 3) Percentile-based “wow”: passing (key passes / progressive passes)
        passing_df = tables.get("passing")
        if passing_df is not None:
            prow = find_player_row(passing_df)
            if prow is None:
                notes.append("Passing table found, but player not uniquely matched in passing.")
            else:
                kp_col = get_col(passing_df, ["key_passes", "kp"])
                prog_col = get_col(passing_df, ["progressive_passes", "prog"])
                if kp_col:
                    kp = to_float(prow.get(kp_col))
                    if kp is not None:
                        pct = percentile_rank(passing_df[kp_col], kp, higher_is_better=True)
                        if pct is not None:
                            wow = clamp_int(40 + 60 * pct)
                            facts.append(Fact(
                                claim=f"En creatividad pura, {player_display} está en el percentil {int(pct*100)} de la liga en pases clave.",
                                metric="key_passes_percentile",
                                value=int(pct * 100),
                                scope=scope,
                                method=f"percentile_rank over {kp_col} in FBref passing table.",
                                provenance={"source": "FBref via soccerdata", "stat_type": "passing", "columns": [kp_col], "fetched_at": fetched_at},
                                confidence="HIGH",
                                wow_score=wow,
                            ))
                if prog_col:
                    prog = to_float(prow.get(prog_col))
                    if prog is not None:
                        pct = percentile_rank(passing_df[prog_col], prog, higher_is_better=True)
                        if pct is not None:
                            wow = clamp_int(38 + 62 * pct)
                            facts.append(Fact(
                                claim=f"Rompe líneas: {player_display} está en el percentil {int(pct*100)} en pases progresivos.",
                                metric="progressive_passes_percentile",
                                value=int(pct * 100),
                                scope=scope,
                                method=f"percentile_rank over {prog_col} in FBref passing table.",
                                provenance={"source": "FBref via soccerdata", "stat_type": "passing", "columns": [prog_col], "fetched_at": fetched_at},
                                confidence="HIGH",
                                wow_score=wow,
                            ))

        # 4) Defensive “wow”: tackles+interceptions per 90 (if available)
        defense_df = tables.get("defense")
        if defense_df is not None and minutes and minutes > 0:
            drow = find_player_row(defense_df)
            if drow is None:
                notes.append("Defense table found, but player not uniquely matched in defense.")
            else:
                tkl_col = get_col(defense_df, ["tackles", "tkl"])
                int_col = get_col(defense_df, ["interceptions", "int"])
                tkl = to_float(drow.get(tkl_col)) if tkl_col else None
                intr = to_float(drow.get(int_col)) if int_col else None
                if tkl is not None or intr is not None:
                    total = (tkl or 0.0) + (intr or 0.0)
                    per90 = safe_div(total, minutes) * 90 if minutes else None
                    if per90 is not None:
                        wow = clamp_int(25 + 60 * max(0, min(1, per90 / 4.0)))  # ~4 per90 is strong
                        facts.append(Fact(
                            claim=f"Trabajo sucio: {player_display} genera {per90:.2f} (tackles+intercepciones) por 90.",
                            metric="tackles_plus_interceptions_per90",
                            value=round(per90, 3),
                            scope=scope,
                            method=f"(tackles+interceptions)/minutes*90 using {tkl_col},{int_col}.",
                            provenance={"source": "FBref via soccerdata", "stat_type": "defense", "columns": [c for c in [tkl_col, int_col] if c], "fetched_at": fetched_at},
                            confidence="HIGH",
                            wow_score=wow,
                        ))

        # Deduplicate similar facts by metric
        facts = self._dedupe_facts(facts)

        # Sort by wow_score descending
        facts.sort(key=lambda f: f.wow_score, reverse=True)
        return facts, notes

    def _generate_player_facts_understat(self, player_name: str) -> Tuple[List[Fact], List[str]]:
        """
        Collect all raw stats + computed per-90s / percentiles / ranks from Understat,
        then delegate to OpenAI to produce position-aware, non-trivial insights.
        Falls back to a deterministic generator if no API key is available.
        """
        notes: List[str] = []

        df = self.client.read_understat_player_season()
        if df is None or df.empty:
            notes.append("No Understat player season table available; cannot generate player facts.")
            return [], notes

        df = df.reset_index()
        player_key = normalize_name(player_name)

        def get_col(candidates: List[str]) -> Optional[str]:
            norm_map = {normalize_name(c): c for c in df.columns}
            for cand in candidates:
                c = norm_map.get(normalize_name(cand))
                if c:
                    return c
            return None

        name_col = get_col(["player", "player_name", "name"])
        if not name_col:
            notes.append("Understat player table has no player name column.")
            return [], notes

        df2 = df.copy()
        df2["_nm"] = df2[name_col].astype(str).map(normalize_name)
        exact = df2[df2["_nm"] == player_key]
        contains = df2[df2["_nm"].str.contains(re.escape(player_key), na=False)]

        row = None
        if len(exact) == 1:
            row = exact.iloc[0]
        elif len(contains) == 1:
            row = contains.iloc[0]
        else:
            notes.append("Player name ambiguous or not found in Understat table; refusing to guess.")
            return [], notes

        scope = f"{self.client.league} {self.client.season}"
        fetched_at = _now_iso()

        # --- Extract all available raw stats ---
        col_map = {
            "minutes": get_col(["minutes", "min"]),
            "goals": get_col(["goals", "gls"]),
            "assists": get_col(["assists", "ast"]),
            "xg": get_col(["xg"]),
            "xa": get_col(["xa"]),
            "key_passes": get_col(["key_passes", "kp"]),
            "shots": get_col(["shots", "sh"]),
            "np_goals": get_col(["np_goals", "npg", "non_penalty_goals"]),
            "np_xg": get_col(["np_xg", "npxg", "non_penalty_xg"]),
            "matches": get_col(["matches", "games", "appearances", "mp"]),
            "yellow_cards": get_col(["yellow_cards", "yellow", "yel"]),
            "red_cards": get_col(["red_cards", "red"]),
            "xg_chain": get_col(["xg_chain"]),
            "xg_buildup": get_col(["xg_buildup"]),
            "position": get_col(["position"]),
            "team": get_col(["team", "squad"]),
        }

        raw: Dict[str, Any] = {}
        for stat_name, col_name in col_map.items():
            if col_name is not None:
                val = row.get(col_name)
                if stat_name in ("position", "team"):
                    raw[stat_name] = str(val) if val is not None else None
                else:
                    raw[stat_name] = to_float(val)

        player_display = str(row.get(name_col, player_name))
        raw["player"] = player_display
        minutes = raw.get("minutes") or 0

        # --- Compute derived stats ---
        derived: Dict[str, Any] = {}
        if minutes > 0:
            for stat in ["goals", "assists", "xg", "xa", "shots", "key_passes",
                         "np_goals", "np_xg", "xg_chain", "xg_buildup"]:
                v = raw.get(stat)
                if v is not None:
                    derived[f"{stat}_per90"] = round(safe_div(v, minutes) * 90, 3)

            goals = raw.get("goals")
            assists = raw.get("assists")
            if goals is not None and assists is not None:
                derived["goal_involvements"] = int(goals + assists)
                derived["goal_involvements_per90"] = round(safe_div(goals + assists, minutes) * 90, 3)

            xg = raw.get("xg")
            xa = raw.get("xa")
            if xg is not None and xa is not None:
                derived["xg_xa_per90"] = round(safe_div(xg + xa, minutes) * 90, 3)

            if goals is not None and xg is not None:
                derived["goals_minus_xg"] = round(goals - xg, 2)

            npg = raw.get("np_goals")
            npxg = raw.get("np_xg")
            if npg is not None and npxg is not None:
                derived["npg_minus_npxg"] = round(npg - npxg, 2)

            shots = raw.get("shots")
            if goals is not None and shots and shots > 0:
                derived["shot_conversion_pct"] = round(safe_div(goals, shots) * 100, 1)

            if goals and goals > 0:
                derived["minutes_per_goal"] = round(safe_div(minutes, goals), 0)

        # --- Percentile ranks (among players with >=400 min) ---
        min_col = col_map.get("minutes")
        percentiles: Dict[str, int] = {}
        if min_col:
            qualified = df[pd.to_numeric(df[min_col], errors="coerce") >= 400].copy()
            n_qualified = len(qualified)
            derived["league_qualified_players"] = n_qualified

            for stat, real_col in col_map.items():
                if real_col and stat not in ("position", "team", "minutes") and raw.get(stat) is not None:
                    pct = percentile_rank(
                        qualified[real_col] if real_col in qualified.columns else pd.Series(),
                        raw[stat], higher_is_better=True,
                    )
                    if pct is not None:
                        percentiles[stat] = int(pct * 100)

        # --- League rank (top N) ---
        ranks: Dict[str, str] = {}
        if min_col:
            qualified = df[pd.to_numeric(df[min_col], errors="coerce") >= 400].copy()
            pool_size = len(qualified)
            for stat, real_col in col_map.items():
                if real_col and stat not in ("position", "team", "minutes") and raw.get(stat) is not None:
                    vals = pd.to_numeric(qualified[real_col], errors="coerce").dropna()
                    if not vals.empty:
                        rank = int((vals > raw[stat]).sum()) + 1
                        if rank <= 20:
                            ranks[stat] = f"#{rank}/{pool_size}"

        # --- Build context for OpenAI ---
        context = {
            "player": player_display,
            "position": raw.get("position", "unknown"),
            "team": raw.get("team"),
            "league": self.client.league,
            "season": self.client.season,
            "raw_stats": {k: v for k, v in raw.items() if k not in ("player", "position", "team") and v is not None},
            "derived_stats": derived,
            "percentiles_among_qualified": percentiles,
            "league_ranks_top20": ranks,
        }

        # --- Try OpenAI generation, fall back to deterministic ---
        api_key = os.environ.get("OPENAI_API_KEY")
        model = os.environ.get("OPENAI_MODEL", DEFAULT_AGENT_MODEL)

        if api_key:
            facts, gen_notes = _openai_generate_player_insights(
                context, scope, fetched_at, api_key, model,
                narrator_instructions=self.narrator_instructions,
            )
            notes.extend(gen_notes)
            if facts:
                return facts, notes
            notes.append("OpenAI insight generation returned no facts; falling back to deterministic.")

        # Deterministic fallback
        facts = self._deterministic_player_facts(raw, derived, percentiles, ranks, player_display, scope, fetched_at)
        return facts, notes

    def _deterministic_player_facts(
        self, raw: Dict, derived: Dict, percentiles: Dict, ranks: Dict,
        player_display: str, scope: str, fetched_at: str,
    ) -> List[Fact]:
        """Simple deterministic facts as fallback when OpenAI is unavailable."""
        facts: List[Fact] = []
        minutes = raw.get("minutes") or 0
        goals = raw.get("goals")
        assists = raw.get("assists")
        xg = raw.get("xg")
        position = raw.get("position", "")

        # G+A
        gi = derived.get("goal_involvements")
        gi90 = derived.get("goal_involvements_per90")
        if gi is not None and gi90 is not None:
            facts.append(Fact(
                claim=f"{player_display} suma {int(gi)} goles+asistencias ({gi90}/90).",
                metric="goal_involvement_per90", value=gi90, scope=scope,
                method="(goals+assists)/minutes*90 from Understat.",
                provenance={"source": "Understat via soccerdata", "fetched_at": fetched_at},
                confidence="HIGH", wow_score=50,
            ))

        # Goals-xG
        g_xg = derived.get("goals_minus_xg")
        if g_xg is not None and goals is not None and xg is not None:
            facts.append(Fact(
                claim=f"{player_display}: {g_xg:+.1f} goles vs xG ({int(goals)}G vs {xg:.1f}xG).",
                metric="goals_minus_xg", value=g_xg, scope=scope,
                method="goals - xG from Understat.",
                provenance={"source": "Understat via soccerdata", "fetched_at": fetched_at},
                confidence="HIGH", wow_score=55,
            ))

        # Shot conversion
        conv = derived.get("shot_conversion_pct")
        if conv is not None:
            facts.append(Fact(
                claim=f"{player_display} convierte {conv:.0f}% de sus disparos.",
                metric="shot_conversion", value=conv, scope=scope,
                method="goals/shots from Understat.",
                provenance={"source": "Understat via soccerdata", "fetched_at": fetched_at},
                confidence="HIGH", wow_score=45,
            ))

        # xG+xA per 90
        xgxa = derived.get("xg_xa_per90")
        if xgxa is not None:
            facts.append(Fact(
                claim=f"{player_display} genera {xgxa} (xG+xA)/90.",
                metric="xg_xa_per90", value=xgxa, scope=scope,
                method="(xG+xA)/minutes*90 from Understat.",
                provenance={"source": "Understat via soccerdata", "fetched_at": fetched_at},
                confidence="HIGH", wow_score=50,
            ))

        # Key passes percentile
        kp_pct = percentiles.get("key_passes")
        if kp_pct is not None:
            facts.append(Fact(
                claim=f"{player_display} está en el percentil {kp_pct} en pases clave.",
                metric="key_passes_percentile", value=kp_pct, scope=scope,
                method="percentile_rank in Understat player table.",
                provenance={"source": "Understat via soccerdata", "fetched_at": fetched_at},
                confidence="HIGH", wow_score=40 + int(kp_pct * 0.4),
            ))

        facts.sort(key=lambda f: f.wow_score, reverse=True)
        return facts

    # ---------
    # Team facts
    # ---------
    def generate_team_facts(self, team_name: str) -> Tuple[List[Fact], List[str]]:
        notes: List[str] = []
        facts: List[Fact] = []

        if self.client.provider == "understat":
            return self._generate_team_facts_understat(team_name)

        tables: Dict[str, pd.DataFrame] = {}
        for stat_type in ["standard", "shooting", "passing", "gca", "defense", "possession", "playing_time", "misc", "keeper"]:
            try:
                tables[stat_type] = self.client.read_team_season_stats(stat_type=stat_type)
            except Exception as e:
                notes.append(f"FBref team table failed: stat_type={stat_type} err={type(e).__name__}")

        if not tables:
            notes.append("No team season tables available; cannot generate team facts.")
            return facts, notes

        base_df = tables.get("standard") or next(iter(tables.values()))
        team_key = normalize_name(team_name)

        def find_team_row(df: pd.DataFrame) -> Optional[pd.Series]:
            name_col = None
            for c in df.columns:
                if normalize_name(c) in ["team", "squad", "club"]:
                    name_col = c
                    break
            if not name_col:
                return None
            df2 = df.copy()
            df2["_nm"] = df2[name_col].astype(str).map(normalize_name)
            exact = df2[df2["_nm"] == team_key]
            if len(exact) == 1:
                return exact.iloc[0]
            contains = df2[df2["_nm"].str.contains(re.escape(team_key), na=False)]
            if len(contains) == 1:
                return contains.iloc[0]
            return None

        def get_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
            norm_map = {normalize_name(c): c for c in df.columns}
            for cand in candidates:
                c = norm_map.get(normalize_name(cand))
                if c:
                    return c
            return None

        row = find_team_row(base_df)
        if row is None:
            notes.append("Team name ambiguous or not found; refusing to guess.")
            return facts, notes

        scope = f"{self.client.league} {self.client.season}"
        fetched_at = _now_iso()

        # Example: Goals For / Goals Against and goal difference
        gf_col = get_col(base_df, ["goals_for", "gf", "goals"])
        ga_col = get_col(base_df, ["goals_against", "ga"])
        mp_col = get_col(base_df, ["matches", "mp", "games"])

        gf = to_float(row.get(gf_col)) if gf_col else None
        ga = to_float(row.get(ga_col)) if ga_col else None
        mp = to_float(row.get(mp_col)) if mp_col else None

        if gf is not None and ga is not None:
            gd = gf - ga
            wow = clamp_int(40 + 3 * max(0, gd))
            facts.append(Fact(
                claim=f"{team_name} tiene diferencial de gol de {int(gd)} en liga esta temporada.",
                metric="goal_difference",
                value=int(gd),
                scope=scope,
                method=f"GF-GA using columns {gf_col} and {ga_col}.",
                provenance={"source": "FBref via soccerdata", "stat_type": "standard", "columns": [gf_col, ga_col], "fetched_at": fetched_at},
                confidence="HIGH",
                wow_score=wow,
            ))

        if gf is not None and mp:
            gf_per_game = safe_div(gf, mp)
            if gf_per_game is not None:
                wow = clamp_int(25 + 60 * max(0, min(1, gf_per_game / 2.2)))
                facts.append(Fact(
                    claim=f"Promedian {gf_per_game:.2f} goles por partido en liga.",
                    metric="goals_per_game",
                    value=round(gf_per_game, 3),
                    scope=scope,
                    method=f"GF/MP using columns {gf_col} and {mp_col}.",
                    provenance={"source": "FBref via soccerdata", "stat_type": "standard", "columns": [gf_col, mp_col], "fetched_at": fetched_at},
                    confidence="HIGH",
                    wow_score=wow,
                ))

        # Percentile example: possession (if available)
        poss_df = tables.get("possession")
        if poss_df is not None:
            trow = find_team_row(poss_df)
            if trow is not None:
                poss_col = get_col(poss_df, ["possession", "poss", "possession%"])
                if poss_col:
                    poss = to_float(trow.get(poss_col))
                    if poss is not None:
                        pct = percentile_rank(poss_df[poss_col], poss, higher_is_better=True)
                        if pct is not None:
                            wow = clamp_int(35 + 65 * pct)
                            facts.append(Fact(
                                claim=f"Con balón: {team_name} está en el percentil {int(pct*100)} de la liga en posesión.",
                                metric="possession_percentile",
                                value=int(pct * 100),
                                scope=scope,
                                method=f"percentile_rank over {poss_col} in FBref possession table.",
                                provenance={"source": "FBref via soccerdata", "stat_type": "possession", "columns": [poss_col], "fetched_at": fetched_at},
                                confidence="HIGH",
                                wow_score=wow,
                            ))

        facts = self._dedupe_facts(facts)
        facts.sort(key=lambda f: f.wow_score, reverse=True)
        return facts, notes

    def _generate_team_facts_understat(self, team_name: str) -> Tuple[List[Fact], List[str]]:
        """
        Collect all raw team match stats from Understat, aggregate them,
        then delegate to OpenAI to produce interesting insights.
        Falls back to deterministic generator if no API key is available.
        """
        notes: List[str] = []

        df = self.client.read_understat_team_matches()
        if df is None or df.empty:
            notes.append("No Understat team match table available; cannot generate team facts.")
            return [], notes

        team_key = normalize_name(team_name)
        home_team_col = "home_team" if "home_team" in df.columns else None
        away_team_col = "away_team" if "away_team" in df.columns else None
        if not home_team_col or not away_team_col:
            notes.append("Understat team match table missing team columns.")
            return [], notes

        home_mask = df[home_team_col].astype(str).map(normalize_name) == team_key
        away_mask = df[away_team_col].astype(str).map(normalize_name) == team_key
        team_matches = df[home_mask | away_mask].copy()
        if team_matches.empty:
            notes.append("Team not found in Understat match table.")
            return [], notes

        fetched_at = _now_iso()
        scope = f"{self.client.league} {self.client.season}"

        # --- Aggregate match-level stats ---
        gf = 0.0; ga = 0.0; xg_for = 0.0; xg_against = 0.0
        npxg_for = 0.0; npxg_against = 0.0
        points = 0.0; exp_points = 0.0
        wins = 0; draws = 0; losses = 0; clean_sheets = 0
        home_pts = 0.0; away_pts = 0.0; home_mp = 0; away_mp = 0
        home_gf = 0.0; away_gf = 0.0; home_ga = 0.0; away_ga = 0.0
        ppda_sum = 0.0; deep_sum = 0.0

        for _, r in team_matches.iterrows():
            is_home = normalize_name(str(r.get(home_team_col, ""))) == team_key
            pfx = "home" if is_home else "away"
            opp = "away" if is_home else "home"

            gf_match = to_float(r.get(f"{pfx}_goals")) or 0.0
            ga_match = to_float(r.get(f"{opp}_goals")) or 0.0
            gf += gf_match; ga += ga_match
            xg_for += to_float(r.get(f"{pfx}_xg")) or 0.0
            xg_against += to_float(r.get(f"{opp}_xg")) or 0.0
            npxg_for += to_float(r.get(f"{pfx}_np_xg")) or 0.0
            npxg_against += to_float(r.get(f"{opp}_np_xg")) or 0.0
            pts = to_float(r.get(f"{pfx}_points")) or 0.0
            points += pts
            exp_points += to_float(r.get(f"{pfx}_expected_points")) or 0.0
            ppda_sum += to_float(r.get(f"{pfx}_ppda")) or 0.0
            deep_sum += to_float(r.get(f"{pfx}_deep_completions")) or 0.0

            if is_home:
                home_pts += pts; home_mp += 1; home_gf += gf_match; home_ga += ga_match
            else:
                away_pts += pts; away_mp += 1; away_gf += gf_match; away_ga += ga_match

            if ga_match == 0:
                clean_sheets += 1
            if gf_match > ga_match:
                wins += 1
            elif gf_match == ga_match:
                draws += 1
            else:
                losses += 1

        mp = float(len(team_matches))
        gd = gf - ga; xgd = xg_for - xg_against

        raw: Dict[str, Any] = {
            "team": team_name, "matches_played": int(mp),
            "wins": wins, "draws": draws, "losses": losses,
            "goals_for": gf, "goals_against": ga, "goal_difference": int(gd),
            "xg_for": round(xg_for, 2), "xg_against": round(xg_against, 2),
            "npxg_for": round(npxg_for, 2), "npxg_against": round(npxg_against, 2),
            "points": points,
            "expected_points": round(exp_points, 2),
            "clean_sheets": clean_sheets,
        }

        derived: Dict[str, Any] = {}
        if mp > 0:
            derived["win_rate_pct"] = round(wins / mp * 100, 1)
            derived["clean_sheet_rate_pct"] = round(clean_sheets / mp * 100, 1)
            derived["goals_per_game"] = round(gf / mp, 2)
            derived["goals_conceded_per_game"] = round(ga / mp, 2)
            derived["xg_per_game"] = round(xg_for / mp, 2)
            derived["xga_per_game"] = round(xg_against / mp, 2)
            derived["xg_diff_per_game"] = round(xgd / mp, 2)
            derived["points_per_game"] = round(points / mp, 2)
            derived["expected_points_per_game"] = round(exp_points / mp, 2)
            derived["points_vs_expected"] = round(points - exp_points, 1)
            derived["goals_vs_xg"] = round(gf - xg_for, 1)
            derived["ga_vs_xga"] = round(ga - xg_against, 1)
            derived["avg_ppda"] = round(ppda_sum / mp, 2)
            derived["avg_deep_completions"] = round(deep_sum / mp, 2)

        if home_mp >= 3 and away_mp >= 3:
            derived["home_ppg"] = round(home_pts / home_mp, 2)
            derived["away_ppg"] = round(away_pts / away_mp, 2)
            derived["home_goals_per_game"] = round(home_gf / home_mp, 2)
            derived["away_goals_per_game"] = round(away_gf / away_mp, 2)

        context = {
            "team": team_name,
            "league": self.client.league,
            "season": self.client.season,
            "raw_stats": raw,
            "derived_stats": derived,
        }

        # --- Try OpenAI generation, fall back to deterministic ---
        api_key = os.environ.get("OPENAI_API_KEY")
        model = os.environ.get("OPENAI_MODEL", DEFAULT_AGENT_MODEL)

        if api_key:
            facts, gen_notes = _openai_generate_team_insights(
                context, scope, fetched_at, api_key, model,
                narrator_instructions=self.narrator_instructions,
            )
            notes.extend(gen_notes)
            if facts:
                return facts, notes
            notes.append("OpenAI team insight generation returned no facts; falling back to deterministic.")

        # Deterministic fallback
        facts = self._deterministic_team_facts(raw, derived, team_name, scope, fetched_at)
        return facts, notes

    def _deterministic_team_facts(
        self, raw: Dict, derived: Dict, team_name: str, scope: str, fetched_at: str,
    ) -> List[Fact]:
        """Simple deterministic team facts as fallback when OpenAI is unavailable."""
        facts: List[Fact] = []
        prov = {"source": "Understat via soccerdata", "fetched_at": fetched_at}

        w, d, l = raw.get("wins", 0), raw.get("draws", 0), raw.get("losses", 0)
        wr = derived.get("win_rate_pct")
        if wr is not None:
            facts.append(Fact(
                claim=f"{team_name}: {w}W-{d}D-{l}L ({wr:.0f}% victorias).",
                metric="win_rate", value=wr, scope=scope,
                method="Aggregate W/D/L from Understat.", provenance=prov,
                confidence="HIGH", wow_score=55,
            ))

        g_xg = derived.get("goals_vs_xg")
        if g_xg is not None:
            facts.append(Fact(
                claim=f"{team_name}: {g_xg:+.1f} goles vs xG esperado.",
                metric="team_goals_minus_xg", value=g_xg, scope=scope,
                method="GF - xGF from Understat.", provenance=prov,
                confidence="HIGH", wow_score=50,
            ))

        ppg = derived.get("points_per_game")
        if ppg is not None:
            facts.append(Fact(
                claim=f"{team_name} suma {ppg} pts/partido.",
                metric="points_per_game", value=ppg, scope=scope,
                method="points/MP from Understat.", provenance=prov,
                confidence="HIGH", wow_score=45,
            ))

        cs_pct = derived.get("clean_sheet_rate_pct")
        if cs_pct is not None:
            facts.append(Fact(
                claim=f"Portería a cero en {cs_pct:.0f}% de los partidos.",
                metric="clean_sheet_rate", value=cs_pct, scope=scope,
                method="Clean sheets / MP from Understat.", provenance=prov,
                confidence="HIGH", wow_score=45,
            ))

        facts.sort(key=lambda f: f.wow_score, reverse=True)
        return facts

    # ---------
    # Stadium facts
    # ---------
    def generate_stadium_facts(self, stadium_name: str) -> Tuple[List[Fact], List[str]]:
        notes: List[str] = []
        facts: List[Fact] = []

        info = self.client.fetch_stadium_facts_wikipedia(stadium_name)
        if "error" in info:
            notes.append(f"Stadium fetch error: {info['error']}")
            return facts, notes

        scope = f"{stadium_name}"
        fetched_at = info.get("fetched_at", _now_iso())
        sources = info.get("sources", [])

        # Capacity
        if info.get("capacity"):
            facts.append(Fact(
                claim=f"Este estadio tiene capacidad aproximada de {info['capacity']}.",
                metric="stadium_capacity",
                value=info["capacity"],
                scope=scope,
                method="Parsed Wikipedia infobox field 'Capacity'.",
                provenance={"source": "Wikipedia infobox", "url": sources[0] if sources else None, "fetched_at": fetched_at},
                confidence="MED",  # Wikipedia -> MED; puedes subir a HIGH si cruzas con fuente oficial.
                wow_score=55,
            ))
        else:
            notes.append("Stadium capacity not found in infobox.")

        # Opened
        if info.get("opened"):
            facts.append(Fact(
                claim=f"Inaugurado: {info['opened']}.",
                metric="stadium_opened",
                value=info["opened"],
                scope=scope,
                method="Parsed Wikipedia infobox field 'Opened'.",
                provenance={"source": "Wikipedia infobox", "url": sources[0] if sources else None, "fetched_at": fetched_at},
                confidence="MED",
                wow_score=45,
            ))

        # Record attendance
        if info.get("record_attendance"):
            facts.append(Fact(
                claim=f"Récord de asistencia: {info['record_attendance']}.",
                metric="record_attendance",
                value=info["record_attendance"],
                scope=scope,
                method="Parsed Wikipedia infobox field 'Record attendance'.",
                provenance={"source": "Wikipedia infobox", "url": sources[0] if sources else None, "fetched_at": fetched_at},
                confidence="MED",
                wow_score=60,
            ))

        facts = self._dedupe_facts(facts)
        facts.sort(key=lambda f: f.wow_score, reverse=True)
        return facts, notes

    # ---------
    # Internal: dedupe
    # ---------
    def _dedupe_facts(self, facts: List[Fact]) -> List[Fact]:
        best: Dict[str, Fact] = {}
        for f in facts:
            if f.metric not in best or f.wow_score > best[f.metric].wow_score:
                best[f.metric] = f
        return list(best.values())


# -----------------------------
# OpenAI-powered insight generation
# -----------------------------
_INSIGHT_SYSTEM_PROMPT = """You are FactForge, a football analytics engine. You receive verified raw stats, derived metrics, percentiles and league ranks for a player or team. Your job is to produce 8-12 genuinely interesting, non-trivial insights.

RULES:
1. ONLY use the numbers provided in the context. NEVER invent data.
2. Assign each insight a wow_score (0-100) based on how surprising, unusual or narratively interesting it is. Be a harsh judge — generic facts get low scores.
3. Adapt to the player's POSITION:
   - Forwards/Strikers (F/S): scoring efficiency, clinical finishing, movement, goal threat
   - Midfielders (M): creativity, progressive passing, goal involvement, buildup contribution
   - Defenders (D): defensive actions, clean sheets, progressive carrying, aerial ability
   - Goalkeepers (GK): save rate, clean sheets, distribution, xG prevented
   - Multi-position: highlight what makes them versatile
4. Insights do NOT have to be purely numeric. You can craft narrative observations like:
   - "Despite playing deeper this season, X still contributes more xA than most forwards"
   - "X's xG chain involvement suggests they're the heartbeat of their team's attacks"
   - Comparative context ("only 3 players in the league do X better")
5. NEVER include trivial facts like total minutes played, number of matches, or obvious position info.
6. Avoid repeating the same angle twice (e.g., don't show goals/90 AND minutes-per-goal — pick the more interesting one).
7. Each insight needs a unique short metric key (snake_case).
8. Respond in the same language as the player/team name context (default: Spanish).

OUTPUT: Return ONLY a JSON object:
{
  "insights": [
    {
      "claim": "Human-readable insight text",
      "metric": "unique_metric_key",
      "value": <number or string>,
      "wow_score": <0-100>,
      "confidence": "HIGH" or "MED"
    }
  ]
}"""


def _call_openai_for_insights(
    context: Dict[str, Any],
    system_prompt: str,
    api_key: str,
    model: str,
) -> Tuple[Optional[List[Dict]], List[str]]:
    """
    Sends context to OpenAI and parses the structured insight response.
    Returns (list_of_insight_dicts | None, notes).
    """
    notes: List[str] = []
    try:
        r = requests.post(
            f"{OPENAI_API_BASE.rstrip('/')}/responses",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "input": [
                    {
                        "role": "system",
                        "content": [{"type": "input_text", "text": system_prompt}],
                    },
                    {
                        "role": "user",
                        "content": [{"type": "input_text", "text": json.dumps(context, ensure_ascii=False)}],
                    },
                ],
            },
            timeout=45,
        )
    except Exception as e:
        notes.append(f"OpenAI insight request failed: {type(e).__name__}")
        return None, notes

    if r.status_code >= 400:
        notes.append(f"OpenAI insight HTTP error {r.status_code}")
        return None, notes

    try:
        payload = r.json()
    except Exception:
        notes.append("OpenAI response was not valid JSON.")
        return None, notes

    # Extract text from response
    text = payload.get("output_text") or ""
    if not text and isinstance(payload.get("output"), list):
        chunks = []
        for item in payload["output"]:
            for content in item.get("content", []):
                if isinstance(content.get("text"), str):
                    chunks.append(content["text"])
        text = "\n".join(chunks)

    text = text.strip()
    if not text:
        notes.append("OpenAI returned empty response.")
        return None, notes

    # Parse JSON (strip markdown fences if present)
    cleaned = text
    if cleaned.startswith("```"):
        first_nl = cleaned.find("\n")
        if first_nl != -1:
            cleaned = cleaned[first_nl + 1:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

    parsed = None
    try:
        parsed = json.loads(cleaned)
    except Exception:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end > start:
            try:
                parsed = json.loads(cleaned[start:end + 1])
            except Exception:
                pass

    if not isinstance(parsed, dict) or "insights" not in parsed:
        notes.append("OpenAI response did not contain valid insights JSON.")
        return None, notes

    insights = parsed.get("insights", [])
    if not isinstance(insights, list) or not insights:
        notes.append("OpenAI returned empty insights list.")
        return None, notes

    notes.append(f"OpenAI generated {len(insights)} insights using model {model}.")
    return insights, notes


def _insights_to_facts(
    insights: List[Dict], scope: str, fetched_at: str, source_label: str,
) -> List[Fact]:
    """Convert raw insight dicts from OpenAI into Fact dataclass instances."""
    facts: List[Fact] = []
    seen_metrics: set = set()

    for item in insights:
        if not isinstance(item, dict):
            continue
        claim = item.get("claim", "")
        metric = item.get("metric", "")
        if not claim or not metric or metric in seen_metrics:
            continue
        seen_metrics.add(metric)

        wow = item.get("wow_score", 50)
        if not isinstance(wow, (int, float)):
            wow = 50
        wow = clamp_int(wow)

        confidence = str(item.get("confidence", "HIGH")).upper()
        if confidence not in ("HIGH", "MED"):
            confidence = "MED"

        value = item.get("value", "—")

        facts.append(Fact(
            claim=str(claim),
            metric=str(metric),
            value=value,
            scope=scope,
            method=f"AI-generated insight from verified {source_label} data.",
            provenance={"source": source_label, "method": "openai_insight_generation", "fetched_at": fetched_at},
            confidence=confidence,
            wow_score=wow,
        ))

    facts.sort(key=lambda f: f.wow_score, reverse=True)
    return facts


def _openai_generate_player_insights(
    context: Dict[str, Any], scope: str, fetched_at: str,
    api_key: str, model: str,
    narrator_instructions: str = "",
) -> Tuple[List[Fact], List[str]]:
    """Generate player insights via OpenAI."""
    prompt = _INSIGHT_SYSTEM_PROMPT
    if narrator_instructions:
        prompt += "\n\n--- CONFIGURACIÓN DEL NARRADOR ---\n" + narrator_instructions
    insights, notes = _call_openai_for_insights(context, prompt, api_key, model)
    if not insights:
        return [], notes
    facts = _insights_to_facts(insights, scope, fetched_at, "Understat via soccerdata")
    return facts, notes


def _openai_generate_team_insights(
    context: Dict[str, Any], scope: str, fetched_at: str,
    api_key: str, model: str,
    narrator_instructions: str = "",
) -> Tuple[List[Fact], List[str]]:
    """Generate team insights via OpenAI."""
    team_system = _INSIGHT_SYSTEM_PROMPT.replace(
        "for a player or team",
        "for a football team"
    ).replace(
        "Adapt to the player's POSITION:",
        "Focus on what makes this team unique:"
    ).replace(
        "- Forwards/Strikers (F/S): scoring efficiency, clinical finishing, movement, goal threat\n"
        "   - Midfielders (M): creativity, progressive passing, goal involvement, buildup contribution\n"
        "   - Defenders (D): defensive actions, clean sheets, progressive carrying, aerial ability\n"
        "   - Goalkeepers (GK): save rate, clean sheets, distribution, xG prevented\n"
        "   - Multi-position: highlight what makes them versatile",
        "- Attack: scoring patterns, clinical finishing, chance creation, xG trends\n"
        "   - Defense: solidity, clean sheets, xGA performance, pressing intensity (PPDA)\n"
        "   - Style: home vs away splits, deep completions, expected vs actual points\n"
        "   - Narrative: form runs, overperformance, tactical identity, point pace"
    )
    if narrator_instructions:
        team_system += "\n\n--- CONFIGURACIÓN DEL NARRADOR ---\n" + narrator_instructions
    insights, notes = _call_openai_for_insights(context, team_system, api_key, model)
    if not insights:
        return [], notes
    facts = _insights_to_facts(insights, scope, fetched_at, "Understat via soccerdata")
    return facts, notes


# -----------------------------
# OpenAI agentic curation
# -----------------------------
class OpenAIFactAgent:
    def __init__(self, model: str, api_key: Optional[str] = None, api_base: str = OPENAI_API_BASE, narrator_instructions: str = ""):
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.api_base = api_base.rstrip("/")
        self.narrator_instructions = narrator_instructions

    def available(self) -> bool:
        return bool(self.api_key)

    def curate_facts(self, facts: List[Fact], max_items: int = 12) -> Tuple[List[Fact], List[str], bool]:
        notes: List[str] = []
        if not facts:
            notes.append("No facts available for agent curation.")
            return facts, notes, False

        if not self.available():
            notes.append("OPENAI_API_KEY missing; skipping agent curation.")
            return facts, notes, False

        fact_dicts = [asdict(f) for f in facts]
        prompt_payload = {
            "task": "Select and refine the best football facts for narration.",
            "rules": [
                "Use only the provided facts; do not invent numbers, metrics, or provenance.",
                "Keep confidence and provenance unchanged.",
                "You may rewrite claim text for clarity, but keep meaning faithful.",
                f"Return at most {max_items} items.",
                "Prefer higher wow_score when quality is similar.",
            ],
            "facts": fact_dicts,
            "output_format": {
                "selected": [
                    {
                        "metric": "existing metric string",
                        "claim": "rewritten claim string",
                        "wow_score": "integer 0..100"
                    }
                ]
            }
        }

        try:
            _curation_system = (
                "You are a fact curation assistant. Return ONLY valid JSON matching the requested output_format."
                + (f"\n\n--- CONFIGURACIÓN DEL NARRADOR ---\n{self.narrator_instructions}" if self.narrator_instructions else "")
            )
            r = requests.post(
                f"{self.api_base}/responses",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "input": [
                        {
                            "role": "system",
                            "content": [
                                {
                                    "type": "input_text",
                                    "text": _curation_system,
                                }
                            ],
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "input_text",
                                    "text": json.dumps(prompt_payload, ensure_ascii=False),
                                }
                            ],
                        },
                    ],
                    "max_output_tokens": 1200,
                },
                timeout=REQUEST_TIMEOUT,
            )
        except Exception as e:
            notes.append(f"Agent request failed: {type(e).__name__}")
            return facts, notes, False

        if r.status_code >= 400:
            details = r.text[:300].replace("\n", " ")
            notes.append(f"Agent HTTP error {r.status_code}: {details}")
            return facts, notes, False

        try:
            response_json = r.json()
        except Exception:
            notes.append("Agent response was not valid JSON.")
            return facts, notes, False

        text = self._extract_response_text(response_json)
        if not text:
            notes.append("Agent returned empty text output.")
            return facts, notes, False

        parsed = self._parse_json_object(text)
        if not parsed or not isinstance(parsed, dict):
            notes.append("Agent output did not contain a valid JSON object.")
            return facts, notes, False

        selected = parsed.get("selected", [])
        if not isinstance(selected, list) or not selected:
            notes.append("Agent output had no selected facts; keeping deterministic ranking.")
            return facts, notes, False

        by_metric: Dict[str, Fact] = {f.metric: f for f in facts}
        curated: List[Fact] = []

        for item in selected:
            if not isinstance(item, dict):
                continue
            metric = item.get("metric")
            if not isinstance(metric, str) or metric not in by_metric:
                continue

            base = by_metric[metric]
            claim = item.get("claim")
            wow_score = item.get("wow_score")

            new_claim = claim if isinstance(claim, str) and claim.strip() else base.claim
            new_wow = clamp_int(wow_score) if isinstance(wow_score, (int, float)) else base.wow_score

            curated.append(Fact(
                claim=new_claim,
                metric=base.metric,
                value=base.value,
                scope=base.scope,
                method=base.method,
                provenance=base.provenance,
                confidence=base.confidence,
                wow_score=new_wow,
            ))

        if not curated:
            notes.append("Agent output did not reference existing metrics; keeping deterministic ranking.")
            return facts, notes, False

        # Add remaining facts not selected by the agent, preserving deterministic fallback ordering.
        selected_metrics = {f.metric for f in curated}
        remainder = [f for f in facts if f.metric not in selected_metrics]
        remainder.sort(key=lambda f: f.wow_score, reverse=True)

        all_curated = curated + remainder
        all_curated = all_curated[:max_items]
        notes.append(f"Agent curated {len(curated)} facts using model {self.model}.")
        return all_curated, notes, True

    def _extract_response_text(self, payload: Dict[str, Any]) -> Optional[str]:
        if isinstance(payload.get("output_text"), str) and payload["output_text"].strip():
            return payload["output_text"].strip()

        output = payload.get("output")
        if not isinstance(output, list):
            return None

        chunks: List[str] = []
        for item in output:
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if not isinstance(content, list):
                continue
            for c in content:
                if not isinstance(c, dict):
                    continue
                text = c.get("text")
                if isinstance(text, str) and text.strip():
                    chunks.append(text.strip())

        if not chunks:
            return None
        return "\n".join(chunks)

    def _parse_json_object(self, text: str) -> Optional[Dict[str, Any]]:
        text = text.strip()
        try:
            return json.loads(text)
        except Exception:
            pass

        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        candidate = text[start:end + 1]
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
            return None
        except Exception:
            return None


# -----------------------------
# Output formatting
# -----------------------------
def facts_to_json(facts: List[Fact], notes: List[str], agent_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return {
        "generated_at": _now_iso(),
        "agent": agent_info,
        "facts": [asdict(f) for f in facts],
        "notes": notes,
    }


def script_pack(facts: List[Fact], max_items: int = 12) -> str:
    """
    “Versión hablada” lista para narración.
    """
    lines = []
    for i, f in enumerate(facts[:max_items], 1):
        lines.append(f"{i}. {f.claim}")
    return "\n".join(lines)


# -----------------------------
# CLI / main
# -----------------------------
def main():
    import argparse

    parser = argparse.ArgumentParser(description="Palomo Fact Agent POC (soccerdata + verify + wow facts)")
    parser.add_argument("--league", default=DEFAULT_LEAGUE, help="e.g. 'ENG-Premier League'")
    parser.add_argument("--season", type=int, default=DEFAULT_SEASON, help="e.g. 2025")
    parser.add_argument("--provider", choices=["auto", "fbref", "understat"], default="auto", help="Data provider inside soccerdata")
    parser.add_argument("--player", default=None, help="Player full name (must match uniquely)")
    parser.add_argument("--team", default=None, help="Team name (must match uniquely)")
    parser.add_argument("--stadium", default=None, help="Stadium name (Wikipedia title works best)")
    parser.add_argument("--agent-model", default=DEFAULT_AGENT_MODEL, help="OpenAI model id for the agent, e.g. 'gpt-5.2'")
    parser.add_argument("--openai-api-key", default=os.environ.get("OPENAI_API_KEY"), help="OpenAI API key (defaults to OPENAI_API_KEY env)")
    parser.add_argument("--disable-agent", action="store_true", help="Disable OpenAI curation and use deterministic ranking only")
    parser.add_argument("--out", default="facts.json", help="Output JSON path")
    args = parser.parse_args()

    client = DataClient(league=args.league, season=args.season, provider=args.provider)
    forge = FactForge(client)

    all_facts: List[Fact] = []
    all_notes: List[str] = []

    if args.player:
        pf, pn = forge.generate_player_facts(args.player)
        all_facts.extend(pf)
        all_notes.extend([f"[player] {x}" for x in pn])

    if args.team:
        tf, tn = forge.generate_team_facts(args.team)
        all_facts.extend(tf)
        all_notes.extend([f"[team] {x}" for x in tn])

    if args.stadium:
        sf, sn = forge.generate_stadium_facts(args.stadium)
        all_facts.extend(sf)
        all_notes.extend([f"[stadium] {x}" for x in sn])

    agent_used = False
    if not args.disable_agent:
        agent = OpenAIFactAgent(model=args.agent_model, api_key=args.openai_api_key)
        all_facts, an, agent_used = agent.curate_facts(all_facts)
        all_notes.extend([f"[agent] {x}" for x in an])

    # Final ranking across categories when no agent curation happened
    if not agent_used:
        all_facts.sort(key=lambda f: f.wow_score, reverse=True)

    payload = facts_to_json(
        all_facts,
        all_notes,
        agent_info={
            "provider": "openai",
            "model": args.agent_model,
            "enabled": not args.disable_agent,
            "used": agent_used,
            "data_provider": args.provider,
        },
    )
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"\nSaved: {args.out}")
    print(f"Agent model: {args.agent_model} | used={agent_used}")
    print("\n--- SCRIPT PACK ---")
    print(script_pack(all_facts))


if __name__ == "__main__":
    main()
