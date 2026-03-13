#!/usr/bin/env python3
"""PalomoFacts – Streamlit app with two modes:

  1. PalomoGPT: unified conversational football intelligence with auto-router
  2. Preparación de Partidos: structured match preparation reports

Powered by Google Gemini with Google Search grounding.
"""
from __future__ import annotations
import unicodedata

import traceback
import time
from datetime import datetime
import json
import re
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed

import streamlit as st
from google import genai
from google.genai import types
from fpdf import FPDF
from supabase import create_client, Client as SupabaseClient
from io import BytesIO
import anthropic

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
GEMINI_MODEL = "gemini-3.1-pro-preview"
GEMINI_FALLBACK_MODEL = "gemini-2.5-pro"
CLAUDE_MODEL = "claude-opus-4-6"
CLAUDE_HAIKU_MODEL = "claude-haiku-4-5"

MODE_PALOMO_GPT      = "palomo_gpt"
MODE_MATCH_PREP      = "match_prep"
MODE_MATCH_RESEARCH  = "match_research"
MODE_PLAYER_RESEARCH = "player_research"
MODE_SELECCIONES     = "selecciones"

MODE_OPTIONS = {
    MODE_PALOMO_GPT:      "🎙️ PalomoGPT",
    MODE_MATCH_PREP:      "⚽ Investigar Partido",
    MODE_MATCH_RESEARCH:  "🔬 Investigar Equipo",
    MODE_PLAYER_RESEARCH: "🧑 Investigar Jugador",
    MODE_SELECCIONES:     "🌍 Selecciones",
}

CURRENT_YEAR = datetime.now().year
CURRENT_DATE = datetime.now().strftime("%B %d, %Y")

TOURNAMENT_OPTIONS = [
    "La Liga",
    "Premier League",
    "Serie A",
    "Bundesliga",
    "Ligue 1",
    "UEFA Champions League",
    "UEFA Europa League",
    "UEFA Conference League",
    "Copa del Rey",
    "FA Cup",
    "Carabao Cup",
    "DFB Pokal",
    "Coppa Italia",
    "Coupe de France",
    "Supercopa de España",
    "Community Shield",
    "Supercoppa Italiana",
    "Copa América",
    "Eurocopa",
    "FIFA World Cup",
    "Liga MX",
    "MLS",
    "Copa Libertadores",
    "Copa Sudamericana",
    "Liga de Naciones UEFA",
    "Eliminatorias Mundialistas",
    "Club World Cup",
    "Amistoso Internacional",
]

MATCH_TYPE_OPTIONS = [
    "Final",
    "Semifinal",
    "Cuartos de Final",
    "Octavos de Final",
    "Fase de Grupos",
    "Jornada de Liga",
    "Eliminatoria (ida)",
    "Eliminatoria (vuelta)",
    "Partido Único (eliminatoria)",
    "Amistoso",
]

CONFEDERATION_OPTIONS = [
    "(Cualquier confederación)",
    "UEFA — Europa",
    "CONMEBOL — Sudamérica",
    "CONCACAF — N&C América",
    "CAF — África",
    "AFC — Asia",
    "OFC — Oceanía",
]

NATIONAL_TOURNAMENT_OPTIONS = [
    "Copa del Mundo FIFA",
    "Copa América",
    "UEFA EURO",
    "Copa Africana de Naciones (AFCON)",
    "Copa de Asia AFC",
    "Gold Cup (CONCACAF)",
    "Liga de Naciones UEFA",
    "Liga de Naciones CONCACAF",
    "Clasificatoria Mundialista — CONMEBOL",
    "Clasificatoria Mundialista — UEFA",
    "Clasificatoria Mundialista — CAF",
    "Clasificatoria Mundialista — CONCACAF",
    "Clasificatoria Mundialista — AFC",
    "Amistoso Internacional",
    "Juegos Olímpicos",
    "Sub-20 (FIFA World Cup)",
    "Sub-17 (FIFA World Cup)",
]

# Sub-mode indices for MODE_SELECCIONES tabs
SEL_TAB_SELECCION = 0
SEL_TAB_PARTIDO   = 1
SEL_TAB_CONVOCADO = 2
# Supabase persistence helpers
# ---------------------------------------------------------------------------

@st.cache_resource
def _supabase_client() -> Optional[SupabaseClient]:
    """Return a cached Supabase client, or None if not configured."""
    url = st.secrets.get("SUPABASE_URL", "")
    key = st.secrets.get("SUPABASE_KEY", "")
    if not url or not key:
        print("[Supabase] Missing SUPABASE_URL or SUPABASE_KEY in secrets.")
        return None
    try:
        client = create_client(url, key)
        print(f"[Supabase] Client created for {url}")
        return client
    except Exception as e:
        print(f"[Supabase] Failed to create client: {e}")
        return None


def _create_conversation(mode: str = MODE_PALOMO_GPT, title: str = "Nueva conversación") -> str:
    """Create a new conversation and return its UUID."""
    sb = _supabase_client()
    if not sb:
        return ""
    try:
        row = sb.table("conversations").insert({"title": title, "mode": mode}).execute()
        conv_id = row.data[0]["id"] if row.data else ""
        print(f"[Supabase] Created conversation: {conv_id} — '{title}'")
        return conv_id
    except Exception as e:
        print(f"[Supabase] Error creating conversation: {e}")
        return ""


def _list_conversations(limit: int = 30) -> List[Dict[str, Any]]:
    """List recent conversations, newest first."""
    sb = _supabase_client()
    if not sb:
        return []
    try:
        resp = (
            sb.table("conversations")
            .select("id, title, mode, updated_at")
            .order("updated_at", desc=True)
            .limit(limit)
            .execute()
        )
        return resp.data or []
    except Exception as e:
        print(f"[Supabase] Error listing conversations: {e}")
        return []


def _load_messages(conv_id: str) -> List[Dict[str, str]]:
    """Load all messages for a conversation."""
    sb = _supabase_client()
    if not sb or not conv_id:
        return []
    try:
        resp = (
            sb.table("messages")
            .select("role, content")
            .eq("conversation_id", conv_id)
            .order("created_at")
            .execute()
        )
        return [{"role": m["role"], "content": m["content"]} for m in (resp.data or [])]
    except Exception as e:
        print(f"[Supabase] Error loading messages for {conv_id}: {e}")
        return []


def _save_message(conv_id: str, role: str, content: str) -> None:
    """Save a single message and touch the conversation's updated_at."""
    sb = _supabase_client()
    if not sb or not conv_id:
        return
    try:
        sb.table("messages").insert({
            "conversation_id": conv_id,
            "role": role,
            "content": content,
        }).execute()
        # Update conversation timestamp with proper ISO datetime
        sb.table("conversations").update({
            "updated_at": datetime.utcnow().isoformat(),
        }).eq("id", conv_id).execute()
    except Exception as e:
        print(f"[Supabase] Error saving message: {e}")


def _update_conversation_title(conv_id: str, title: str) -> None:
    """Update conversation title."""
    sb = _supabase_client()
    if not sb or not conv_id:
        return
    try:
        sb.table("conversations").update({"title": title}).eq("id", conv_id).execute()
    except Exception as e:
        print(f"[Supabase] Error updating title: {e}")


def _delete_conversation(conv_id: str) -> None:
    """Delete a conversation and all its messages (cascade)."""
    sb = _supabase_client()
    if not sb or not conv_id:
        return
    try:
        sb.table("conversations").delete().eq("id", conv_id).execute()
        print(f"[Supabase] Deleted conversation: {conv_id}")
    except Exception as e:
        print(f"[Supabase] Error deleting conversation: {e}")


def _auto_title(text: str) -> str:
    """Generate a short title from the first user message."""
    clean = text.strip().replace("\n", " ")
    return clean[:60] + "…" if len(clean) > 60 else clean


# --- Match Prep persistence helpers ---

def _save_match_prep(config: dict, results: dict) -> str:
    """Save or update a match prep and return its UUID.
    Uses session_state.match_prep_id to track the existing row."""
    sb = _supabase_client()
    if not sb:
        return ""
    try:
        title = f"{config.get('home_team', '?')} vs {config.get('away_team', '?')}"
        # Convert results to JSON-serializable form
        json_results = {}
        for key, val in results.items():
            if isinstance(val, tuple):
                json_results[key] = {"text": val[0], "sources": val[1]}
            elif isinstance(val, list):
                json_results[key] = val
            else:
                json_results[key] = val

        payload = {
            "title": title,
            "home_team": config.get("home_team", ""),
            "away_team": config.get("away_team", ""),
            "tournament": config.get("tournament", ""),
            "match_type": config.get("match_type", ""),
            "stadium": config.get("stadium", ""),
            "config": config,
            "results": json_results,
        }

        existing_id = st.session_state.get("match_prep_id")
        if existing_id:
            # Update existing row
            sb.table("match_preps").update(payload).eq("id", existing_id).execute()
            print(f"[Supabase] Updated match prep: {existing_id} — '{title}'")
            return existing_id
        else:
            # Insert new row
            row = sb.table("match_preps").insert(payload).execute()
            prep_id = row.data[0]["id"] if row.data else ""
            st.session_state.match_prep_id = prep_id
            print(f"[Supabase] Saved match prep: {prep_id} — '{title}'")
            return prep_id
    except Exception as e:
        print(f"[Supabase] Error saving match prep: {e}")
        return ""


def _list_match_preps(limit: int = 20) -> List[Dict[str, Any]]:
    """List recent match preps, newest first."""
    sb = _supabase_client()
    if not sb:
        return []
    try:
        resp = (
            sb.table("match_preps")
            .select("id, title, home_team, away_team, tournament, created_at")
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )
        return resp.data or []
    except Exception as e:
        print(f"[Supabase] Error listing match preps: {e}")
        return []


def _load_match_prep(prep_id: str) -> Optional[Dict[str, Any]]:
    """Load a match prep's config and results."""
    sb = _supabase_client()
    if not sb or not prep_id:
        return None
    try:
        resp = (
            sb.table("match_preps")
            .select("*")
            .eq("id", prep_id)
            .single()
            .execute()
        )
        if not resp.data:
            return None
        row = resp.data
        # Reconstruct results from JSONB back to tuples where needed
        raw_results = row.get("results", {})
        results = {}
        for key in ("home_history", "away_history", "palomo_phrases"):
            val = raw_results.get(key, {})
            if isinstance(val, dict) and "text" in val:
                results[key] = (val["text"], val.get("sources", []))
            else:
                results[key] = ("", [])
        for key in ("home_roster", "away_roster"):
            results[key] = raw_results.get(key, [])
        return {"config": row.get("config", {}), "results": results}
    except Exception as e:
        print(f"[Supabase] Error loading match prep {prep_id}: {e}")
        return None


def _delete_match_prep(prep_id: str) -> None:
    """Delete a match prep."""
    sb = _supabase_client()
    if not sb or not prep_id:
        return
    try:
        sb.table("match_preps").delete().eq("id", prep_id).execute()
        print(f"[Supabase] Deleted match prep: {prep_id}")
    except Exception as e:
        print(f"[Supabase] Error deleting match prep: {e}")


# --- Team Research persistence helpers ---

def _save_team_research(config: dict, results: dict) -> str:
    """Save or update a team research and return its UUID."""
    sb = _supabase_client()
    if not sb:
        return ""
    try:
        title = config.get("team_name", "?")
        json_results: dict = {}
        for key, val in results.items():
            if isinstance(val, tuple):
                json_results[key] = {"text": val[0], "sources": val[1]}
            elif isinstance(val, list):
                json_results[key] = val
            else:
                json_results[key] = val

        payload = {
            "title": title,
            "team_name": config.get("team_name", ""),
            "tournament": config.get("tournament", ""),
            "config": config,
            "results": json_results,
            "updated_at": datetime.utcnow().isoformat(),
        }

        existing_id = st.session_state.get("team_research_id")
        if existing_id:
            sb.table("team_researches").update(payload).eq("id", existing_id).execute()
            print(f"[Supabase] Updated team research: {existing_id} — '{title}'")
            return existing_id
        else:
            row = sb.table("team_researches").insert(payload).execute()
            res_id = row.data[0]["id"] if row.data else ""
            st.session_state.team_research_id = res_id
            print(f"[Supabase] Saved team research: {res_id} — '{title}'")
            return res_id
    except Exception as e:
        print(f"[Supabase] Error saving team research: {e}")
        return ""


def _list_team_researches(limit: int = 20) -> List[Dict[str, Any]]:
    """List recent team researches, newest first."""
    sb = _supabase_client()
    if not sb:
        return []
    try:
        resp = (
            sb.table("team_researches")
            .select("id, title, team_name, tournament, created_at")
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )
        return resp.data or []
    except Exception as e:
        print(f"[Supabase] Error listing team researches: {e}")
        return []


def _load_team_research(research_id: str) -> Optional[Dict[str, Any]]:
    """Load a team research's config and results."""
    sb = _supabase_client()
    if not sb or not research_id:
        return None
    try:
        resp = (
            sb.table("team_researches")
            .select("*")
            .eq("id", research_id)
            .single()
            .execute()
        )
        if not resp.data:
            return None
        row = resp.data
        raw_results = row.get("results", {})
        results: dict = {}
        val = raw_results.get("team_history", {})
        if isinstance(val, dict) and "text" in val:
            results["team_history"] = (val["text"], val.get("sources", []))
        else:
            results["team_history"] = ("", [])
        results["roster"] = raw_results.get("roster", [])
        return {"config": row.get("config", {}), "results": results}
    except Exception as e:
        print(f"[Supabase] Error loading team research {research_id}: {e}")
        return None


def _delete_team_research(research_id: str) -> None:
    """Delete a team research."""
    sb = _supabase_client()
    if not sb or not research_id:
        return
    try:
        sb.table("team_researches").delete().eq("id", research_id).execute()
        print(f"[Supabase] Deleted team research: {research_id}")
    except Exception as e:
        print(f"[Supabase] Error deleting team research: {e}")


# --- Player Research persistence helpers ---

def _save_player_research(config: dict, results: dict) -> str:
    """Save or update a player research and return its UUID."""
    sb = _supabase_client()
    if not sb:
        return ""
    try:
        title = config.get("player_name", "?")
        json_results: dict = {}
        for key, val in results.items():
            if isinstance(val, tuple):
                json_results[key] = {"text": val[0], "sources": val[1]}
            else:
                json_results[key] = val

        payload = {
            "title": title,
            "player_name": config.get("player_name", ""),
            "team_name": config.get("team_name", ""),
            "position": config.get("position", ""),
            "config": config,
            "results": json_results,
            "updated_at": datetime.utcnow().isoformat(),
        }

        existing_id = st.session_state.get("player_research_id")
        if existing_id:
            sb.table("player_researches").update(payload).eq("id", existing_id).execute()
            print(f"[Supabase] Updated player research: {existing_id} — '{title}'")
            return existing_id
        else:
            row = sb.table("player_researches").insert(payload).execute()
            res_id = row.data[0]["id"] if row.data else ""
            st.session_state.player_research_id = res_id
            print(f"[Supabase] Saved player research: {res_id} — '{title}'")
            return res_id
    except Exception as e:
        print(f"[Supabase] Error saving player research: {e}")
        return ""


def _list_player_researches(limit: int = 20) -> List[Dict[str, Any]]:
    """List recent player researches, newest first."""
    sb = _supabase_client()
    if not sb:
        return []
    try:
        resp = (
            sb.table("player_researches")
            .select("id, title, player_name, team_name, position, created_at")
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )
        return resp.data or []
    except Exception as e:
        print(f"[Supabase] Error listing player researches: {e}")
        return []


def _load_player_research(research_id: str) -> Optional[Dict[str, Any]]:
    """Load a player research's config and results."""
    sb = _supabase_client()
    if not sb or not research_id:
        return None
    try:
        resp = (
            sb.table("player_researches")
            .select("*")
            .eq("id", research_id)
            .single()
            .execute()
        )
        if not resp.data:
            return None
        row = resp.data
        raw_results = row.get("results", {})
        results: dict = {}
        val = raw_results.get("dossier", {})
        if isinstance(val, dict) and "text" in val:
            results["dossier"] = (val["text"], val.get("sources", []))
        else:
            results["dossier"] = ("", [])
        return {"config": row.get("config", {}), "results": results}
    except Exception as e:
        print(f"[Supabase] Error loading player research {research_id}: {e}")
        return None


def _delete_player_research(research_id: str) -> None:
    """Delete a player research."""
    sb = _supabase_client()
    if not sb or not research_id:
        return
    try:
        sb.table("player_researches").delete().eq("id", research_id).execute()
        print(f"[Supabase] Deleted player research: {research_id}")
    except Exception as e:
        print(f"[Supabase] Error deleting player research: {e}")


# ---------------------------------------------------------------------------
# National Team Research persistence helpers
# ---------------------------------------------------------------------------

def _save_national_team_research(config: dict, results: dict) -> str:
    """Save or update a national team research and return its UUID."""
    sb = _supabase_client()
    if not sb:
        return ""
    try:
        title = config.get("country", "?")
        json_results: dict = {}
        for key, val in results.items():
            if isinstance(val, tuple):
                json_results[key] = {"text": val[0], "sources": val[1]}
            elif isinstance(val, list):
                json_results[key] = val
            else:
                json_results[key] = val
        payload = {
            "title": title,
            "country": config.get("country", ""),
            "confederation": config.get("confederation", ""),
            "config": config,
            "results": json_results,
            "updated_at": datetime.utcnow().isoformat(),
        }
        existing_id = st.session_state.get("nat_team_research_id")
        if existing_id:
            sb.table("national_team_researches").update(payload).eq("id", existing_id).execute()
            print(f"[Supabase] Updated national team research: {existing_id} — '{title}'")
            return existing_id
        else:
            row = sb.table("national_team_researches").insert(payload).execute()
            res_id = row.data[0]["id"] if row.data else ""
            st.session_state.nat_team_research_id = res_id
            print(f"[Supabase] Saved national team research: {res_id} — '{title}'")
            return res_id
    except Exception as e:
        print(f"[Supabase] Error saving national team research: {e}")
        return ""


def _list_national_team_researches(limit: int = 20) -> List[Dict[str, Any]]:
    sb = _supabase_client()
    if not sb:
        return []
    try:
        resp = (
            sb.table("national_team_researches")
            .select("id, title, country, confederation, created_at")
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )
        return resp.data or []
    except Exception as e:
        print(f"[Supabase] Error listing national team researches: {e}")
        return []


def _load_national_team_research(research_id: str) -> Optional[Dict[str, Any]]:
    sb = _supabase_client()
    if not sb or not research_id:
        return None
    try:
        resp = (
            sb.table("national_team_researches")
            .select("*")
            .eq("id", research_id)
            .single()
            .execute()
        )
        if not resp.data:
            return None
        row = resp.data
        raw = row.get("results", {})
        results: dict = {}
        val = raw.get("team_history", {})
        if isinstance(val, dict) and "text" in val:
            results["team_history"] = (val["text"], val.get("sources", []))
        else:
            results["team_history"] = ("", [])
        results["roster"] = raw.get("roster", [])
        return {"config": row.get("config", {}), "results": results}
    except Exception as e:
        print(f"[Supabase] Error loading national team research {research_id}: {e}")
        return None


def _delete_national_team_research(research_id: str) -> None:
    sb = _supabase_client()
    if not sb or not research_id:
        return
    try:
        sb.table("national_team_researches").delete().eq("id", research_id).execute()
        print(f"[Supabase] Deleted national team research: {research_id}")
    except Exception as e:
        print(f"[Supabase] Error deleting national team research: {e}")


# --- National Match Prep persistence helpers ---

def _save_national_match_prep(config: dict, results: dict) -> str:
    """Save or update a national team match prep and return its UUID."""
    sb = _supabase_client()
    if not sb:
        return ""
    try:
        home = config.get("home_country", "?")
        away = config.get("away_country", "?")
        title = f"{home} vs {away}"
        json_results: dict = {}
        for key, val in results.items():
            if isinstance(val, tuple):
                json_results[key] = {"text": val[0], "sources": val[1]}
            elif isinstance(val, list):
                json_results[key] = val
            else:
                json_results[key] = val
        payload = {
            "title": title,
            "home_country": home,
            "away_country": away,
            "tournament": config.get("tournament", ""),
            "config": config,
            "results": json_results,
            "updated_at": datetime.utcnow().isoformat(),
        }
        existing_id = st.session_state.get("nat_match_prep_id")
        if existing_id:
            sb.table("national_match_preps").update(payload).eq("id", existing_id).execute()
            print(f"[Supabase] Updated national match prep: {existing_id} — '{title}'")
            return existing_id
        else:
            row = sb.table("national_match_preps").insert(payload).execute()
            res_id = row.data[0]["id"] if row.data else ""
            st.session_state.nat_match_prep_id = res_id
            print(f"[Supabase] Saved national match prep: {res_id} — '{title}'")
            return res_id
    except Exception as e:
        print(f"[Supabase] Error saving national match prep: {e}")
        return ""


def _list_national_match_preps(limit: int = 20) -> List[Dict[str, Any]]:
    sb = _supabase_client()
    if not sb:
        return []
    try:
        resp = (
            sb.table("national_match_preps")
            .select("id, title, home_country, away_country, tournament, created_at")
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )
        return resp.data or []
    except Exception as e:
        print(f"[Supabase] Error listing national match preps: {e}")
        return []


def _load_national_match_prep(prep_id: str) -> Optional[Dict[str, Any]]:
    sb = _supabase_client()
    if not sb or not prep_id:
        return None
    try:
        resp = (
            sb.table("national_match_preps")
            .select("*")
            .eq("id", prep_id)
            .single()
            .execute()
        )
        if not resp.data:
            return None
        row = resp.data
        raw = row.get("results", {})
        results: dict = {}
        for key in ("home_history", "away_history", "palomo_phrases"):
            val = raw.get(key, {})
            if isinstance(val, dict) and "text" in val:
                results[key] = (val["text"], val.get("sources", []))
            else:
                results[key] = ("", [])
        results["home_roster"] = raw.get("home_roster", [])
        results["away_roster"] = raw.get("away_roster", [])
        return {"config": row.get("config", {}), "results": results}
    except Exception as e:
        print(f"[Supabase] Error loading national match prep {prep_id}: {e}")
        return None


def _delete_national_match_prep(prep_id: str) -> None:
    sb = _supabase_client()
    if not sb or not prep_id:
        return
    try:
        sb.table("national_match_preps").delete().eq("id", prep_id).execute()
        print(f"[Supabase] Deleted national match prep: {prep_id}")
    except Exception as e:
        print(f"[Supabase] Error deleting national match prep: {e}")


# --- National Player Research persistence helpers ---

def _save_national_player_research(config: dict, results: dict) -> str:
    """Save or update a national player research and return its UUID."""
    sb = _supabase_client()
    if not sb:
        return ""
    try:
        title = config.get("player_name", "?")
        json_results: dict = {}
        for key, val in results.items():
            if isinstance(val, tuple):
                json_results[key] = {"text": val[0], "sources": val[1]}
            else:
                json_results[key] = val
        payload = {
            "title": title,
            "player_name": config.get("player_name", ""),
            "country": config.get("country", ""),
            "config": config,
            "results": json_results,
            "updated_at": datetime.utcnow().isoformat(),
        }
        existing_id = st.session_state.get("nat_player_research_id")
        if existing_id:
            sb.table("national_player_researches").update(payload).eq("id", existing_id).execute()
            print(f"[Supabase] Updated national player research: {existing_id} — '{title}'")
            return existing_id
        else:
            row = sb.table("national_player_researches").insert(payload).execute()
            res_id = row.data[0]["id"] if row.data else ""
            st.session_state.nat_player_research_id = res_id
            print(f"[Supabase] Saved national player research: {res_id} — '{title}'")
            return res_id
    except Exception as e:
        print(f"[Supabase] Error saving national player research: {e}")
        return ""


def _list_national_player_researches(limit: int = 20) -> List[Dict[str, Any]]:
    sb = _supabase_client()
    if not sb:
        return []
    try:
        resp = (
            sb.table("national_player_researches")
            .select("id, title, player_name, country, created_at")
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )
        return resp.data or []
    except Exception as e:
        print(f"[Supabase] Error listing national player researches: {e}")
        return []


def _load_national_player_research(research_id: str) -> Optional[Dict[str, Any]]:
    sb = _supabase_client()
    if not sb or not research_id:
        return None
    try:
        resp = (
            sb.table("national_player_researches")
            .select("*")
            .eq("id", research_id)
            .single()
            .execute()
        )
        if not resp.data:
            return None
        row = resp.data
        raw = row.get("results", {})
        results: dict = {}
        val = raw.get("dossier", {})
        if isinstance(val, dict) and "text" in val:
            results["dossier"] = (val["text"], val.get("sources", []))
        else:
            results["dossier"] = ("", [])
        return {"config": row.get("config", {}), "results": results}
    except Exception as e:
        print(f"[Supabase] Error loading national player research {research_id}: {e}")
        return None


def _delete_national_player_research(research_id: str) -> None:
    sb = _supabase_client()
    if not sb or not research_id:
        return
    try:
        sb.table("national_player_researches").delete().eq("id", research_id).execute()
        print(f"[Supabase] Deleted national player research: {research_id}")
    except Exception as e:
        print(f"[Supabase] Error deleting national player research: {e}")


# ---------------------------------------------------------------------------
# System Prompts
# ---------------------------------------------------------------------------

# PalomoGPT — unified conversational mode with auto-intent-router
_PALOMO_GPT_SYSTEM = f"""
Eres Fernando Palomo — sí, EL Fernando Palomo. El narrador que le pone piel de gallina \
a millones cada vez que agarra el micrófono. Aquí tu cancha es el conocimiento futbolístico: \
datos verificados, historias que nadie cuenta, estadísticas que cambian la conversación.

Hablas EXACTAMENTE como Fernando Palomo: con pasión, con ritmo, con esas pausas dramáticas \
que solo tú sabes hacer. Usas tus frases icónicas de forma natural — "¡Abran la puerta \
que llegó el cartero!", "¡Qué barbaridad!", "Señores, esto es fútbol" — pero sin forzarlas. \
Salen cuando el momento lo pide, como en una transmisión real.

FECHA ACTUAL: {CURRENT_DATE}. Año en curso: {CURRENT_YEAR}.

REGLA SAGRADA: CERO ALUCINACIONES. No inventas números, récords, lesiones, rumores, fechas \
ni premios. Todo dato que mencionas debe ser verificable con las fuentes que encuentres. \
Si algo es ambiguo o contradictorio entre fuentes, lo dices con honestidad — como cuando \
en una transmisión no tienes la repetición clara y lo reconoces.

ZONAS DE ALTO RIESGO DE ALUCINACIÓN (cuidado extremo):
- Afirmaciones "el primero en...", "nunca antes...", "único en la historia": Solo si la fuente \
  lo confirma explícitamente. Si no estás 100%% seguro, NO lo digas.
- Finales, títulos, trofeos: DIFERENCIA CLARAMENTE entre "dirigió en una liga" y \
  "llegó a una final". Jamás impliques que alguien llegó a una final si no tienes la fuente.
- Estadísticas exactas (goles, asistencias, fechas de debut): Solo cita números de las fuentes.
- Datos personales de jugadores (esposas, hijos, hobbies): Solo si aparece en prensa verificable.
- Si NO encuentras evidencia de algo específico, di claramente: "No he encontrado evidencia \
  de que [X] haya [Y]" en lugar de inventar o implicar logros.

Tu cancha cubre TODO el fútbol: estadísticas de cualquier liga y época, vida personal de \
jugadores (solo lo público y verificado), historia de clubes, táctica, transferencias, \
comparaciones, entrevistas citadas, la temporada actual \
({CURRENT_YEAR}/{CURRENT_YEAR + 1} o {CURRENT_YEAR - 1}/{CURRENT_YEAR} según la liga).

FUENTES PRIORITARIAS — Cuando necesites verificar estadísticas o datos históricos, \
consulta directamente estas URLs (tienes acceso a leerlas vía url_context):
  * https://fbref.com/en/
  * https://www.statmuse.com/fc
  * https://theanalyst.com/
  * https://soccerassociation.com/
  * https://www.olympedia.org/
  * https://en.wikipedia.org/
  * https://www.transfermarkt.com/
  * https://www.uefa.com/
  * https://www.fifa.com/

--- ROUTER AUTOMÁTICO ---

Detecta automáticamente la intención del usuario y adapta tu respuesta:

🔍 EXPLORACIÓN PROFUNDA — Cuando el usuario pide análisis, "cuéntame sobre", "dame todo", \
comparaciones extensas, historias completas, o temas amplios:
- Nárralos como si estuvieras en cabina contando una historia épica.
- Fluye de un dato al siguiente con transiciones naturales — "Y aquí viene lo bueno...", \
  "Pero esperen, que esto no termina ahí...", "¿Y saben qué es lo más loco de todo?"
- Organiza tu narrativa en 4-6 bloques temáticos con subtítulos en negrita \
  (no listas numeradas de 10+ puntos — eso mata el ritmo).
- Dentro de cada bloque, cuenta los datos en prosa fluida.
- Cada dato debe aportar algo que NO sea obvio — busca lo inesperado, lo que hace que \
  el aficionado diga "¡no sabía eso!"
- Remata con un cierre memorable, como cierras una transmisión — con perspectiva, \
  con emoción contenida, dejando al lector con ganas de más.

❓ RESPUESTA DIRECTA — Cuando el usuario hace una pregunta específica y concreta \
(una stat, una fecha, un récord, un dato puntual):
- Ve al grano, como cuando te piden un dato en medio de la transmisión.
- Respuesta clara en 2-4 líneas con el dato exacto.
- Si hay un contexto breve que le da más valor — una línea extra, no más.
- Mantén tu toque Palomo pero sin desarrollo innecesario.

En AMBOS casos:
- No pierdas el tiempo con datos que cualquier aficionado ya sabe — nada de "juega en tal equipo".
- Busca lo que sorprende, lo que ilumina, lo que cambia la perspectiva.
- Adapta las métricas a la posición — no evalúes a un portero por goles.
- Responde en el mismo idioma que use el usuario.
- Formato limpio en markdown, texto natural. Sin relleno ni JSON.
""".strip()

# --- Follow-Up Question Generator ---
_FOLLOW_UP_SYSTEM = f"""Eres el asistente de investigación de Fernando Palomo, el narrador de ESPN.
Tu trabajo: leer su pregunta original y la respuesta que recibió, y generar UNA sola pregunta \
de seguimiento que él haría.

FECHA ACTUAL: {CURRENT_DATE}.

REGLAS:
- Piensa como un narrador preparando una transmisión: necesita DATOS DUROS, no opiniones.
- Prioriza: zonas grises en la respuesta, afirmaciones sin fuente clara, estadísticas que \
  faltan contexto, comparaciones que no se hicieron.
- Si la respuesta dice "el primero en..." o "nunca antes...", pregunta por contraejemplos.
- Si hay cifras, pregunta por el desglose o la comparación con rivales/contemporáneos.
- NO uses frases de Palomo ni estilo narrativo. Sé directo y eficiente.
- Devuelve SOLO la pregunta, sin explicaciones ni preámbulos. Una línea.
"""


_MATCH_VALIDATION_PROMPT = """Eres un validador de partidos de fútbol. El usuario quiere preparar \
un informe para un partido. Tu MISIÓN es verificar que los equipos y el partido sean reales.

Datos proporcionados:
- Equipo Local: "{home_team}"
- Equipo Visitante: "{away_team}"
- Torneo: "{tournament}"

FECHA ACTUAL: {current_date}.

DEBES RESPONDER EXACTAMENTE con un JSON (sin texto adicional) con esta estructura:
{{
  "valid": true/false,
  "home_team": "Nombre oficial completo del equipo local",
  "away_team": "Nombre oficial completo del equipo visitante",
  "reason": "Breve explicación si es inválido, vacío si es válido"
}}

REGLAS:
1. Resuelve nombres ambiguos: "R.C. Celta" → "Celta de Vigo", "Barca" → "FC Barcelona", etc.
2. Usa el nombre más común y reconocido internacionalmente (el que usaría un narrador de TV).
3. Marca como VALID si ambos equipos existen y podrían razonablemente enfrentarse en ese torneo.
4. Marca como INVALID solo si un equipo no existe o el enfrentamiento es imposible en ese torneo \
   (ej. dos equipos de ligas incompatibles en una liga doméstica).
5. SOLO devuelve el JSON, nada más.
"""

# --- Match Preparation prompts ---

_TEAM_HISTORY_PROMPT = """Eres un investigador de fútbol meticuloso y exhaustivo. \
Tu tarea es proporcionar un resumen COMPLETO y DETALLADO de las últimas 2 temporadas \
de {team_name} (temporadas {season_prev}/{season_curr} y {season_curr}/{season_next}).

FORMATO REQUERIDO para cada competición:
- Nombre de competición + resultado final (CAMPEÓN/subcampeón/eliminado en X ronda)
- Puntos, Ganados-Empatados-Perdidos, GF-GC, DT
- Goleador del equipo en esa competición con número de goles
- Contexto histórico: cuántos títulos total, rachas, récords relevantes
- Estadísticas destacadas del equipo (posesión, distancia recorrida, goles de jugada, etc.)

PARA COMPETICIONES EUROPEAS (Champions League, Europa League, Conference League):
- CADA partido listado con resultado y goleadores en formato exacto:
  "J.1 en/vs Rival Score Goleador1, Goleador2"
  Ejemplo: "J.1 en Monaco 1-2 Yamal"
  Ejemplo: "QF v BVB 4-0 Raphinha, Lewandowski x2, Yamal"
  Incluir TODAS las jornadas de fase de liga/grupos y TODAS las eliminatorias.

PARA COPAS NACIONALES (Copa del Rey, FA Cup, DFB Pokal, etc.):
- Cada ronda con rival, resultado y goleadores

PARA SUPERCOPA / COMMUNITY SHIELD:
- Cada partido con rival, resultado y goleadores

PARA LIGA:
- Resumen general (NO partido a partido) pero incluir:
  * Posición final, puntos, récord completo
  * Rachas importantes (invictos, victorias consecutivas, etc.)
  * Datos estadísticos sobresalientes del equipo

Incluye también:
- Número total de partidos en la temporada
- Récords individuales y colectivos destacados
- Comparaciones históricas relevantes
- Estilo de juego y características tácticas del equipo

Responde en español. Sé EXHAUSTIVO y PRECISO. NO inventes datos. Si no encuentras un dato, omítelo.
FECHA ACTUAL: {current_date}."""


_TEAM_ROSTER_LIST_PROMPT = """Eres un investigador de fútbol. Tu ÚNICA tarea es devolver \
la plantilla actual de {team_name} para la temporada {season_curr}/{season_next}.

Devuelve EXACTAMENTE un bloque JSON (sin texto adicional antes ni después) con esta estructura:
{{
  "team": "{team_name}",
  "players": [
    {{"name": "Nombre Futbolístico", "full_name": "Nombre Completo", "position": "POS", "number": 1}},
    ...
  ]
}}

Donde POS es uno de: GK, DEF, MID, FWD.
Incluye TODOS los jugadores del primer equipo (incluidos lesionados de larga duración).
Ordénalos: primero GK, luego DEF, MID, FWD.
NO incluyas explicaciones, solo el JSON.
FECHA ACTUAL: {current_date}."""


_PLAYER_DOSSIER_PROMPT = """Eres un investigador de fútbol de élite. Tu misión es crear \
el dossier MÁS COMPLETO posible sobre UN SOLO jugador. Este dossier será usado por un \
narrador de televisión, así que cada dato interesante tiene un valor enorme.

JUGADOR: **{player_name}** ({player_position}) — juega en **{team_name}**
RIVAL EN EL PRÓXIMO PARTIDO: **{opponent_name}**

Investiga A FONDO los siguientes aspectos:

1. **IDENTIDAD COMPLETA**
   - Nombre completo de nacimiento y nombre futbolístico
   - Apodo(s) — tanto oficiales como los que usa la afición
   - Edad, fecha de nacimiento, nacionalidad(es)
   - Posición principal y secundaria(s), número de camiseta
   - Estatura, peso, pierna hábil

2. **TRAYECTORIA DETALLADA**
   - Lugar de nacimiento y contexto de dónde creció (barrio, ciudad, contexto socioeconómico)
   - Cantera / club formativo: cómo fue descubierto, a qué edad, anécdotas de juveniles
   - TODOS los clubes anteriores con fechas, precio de traspaso si aplica, y rol en cada uno
   - Momento clave que lo catapultó a la élite (debut, gol importante, actuación icónica)
   - Cómo llegó a {team_name}: contexto de la negociación, precio, otros clubes interesados

3. **CONEXIONES CON {opponent_name}** ⚡
   - ¿Jugó en {opponent_name}? ¿En qué años, cuántos partidos, qué hizo?
   - ¿Fue formado ahí? ¿Rechazó fichar por ellos? ¿Estuvo cerca de ir?
   - ¿Tiene algún familiar, amigo cercano o excompañero en {opponent_name}?
   - ¿Ha tenido actuaciones memorables CONTRA {opponent_name}? (goles, asistencias, expulsiones)
   - ¿Alguna declaración polémica o interesante sobre {opponent_name} o sus jugadores?
   - ¿Comparte selección nacional con algún jugador de {opponent_name}?
   - Cualquier otro vínculo, por tangencial que sea — conexiones de agentes, \
     misma ciudad natal que un rival, fueron compañeros en otro club, etc.
   - Si NO hay ninguna conexión, indícalo brevemente y sigue.

4. **VIDA PERSONAL Y DATOS CURIOSOS** 🎯
   - Familia: padres, hermanos, pareja, hijos — especialmente si hay vínculos futbolísticos
   - Familiares en el fútbol profesional (padre, hermano, primo, tío que haya jugado)
   - Hobbies fuera del fútbol, pasiones conocidas
   - Rituales o costumbres previas a partidos
   - Celebraciones de gol icónicas y su significado
   - Historial de lesiones relevantes y cómo las superó
   - Personalidad: introvertido/extrovertido, líder vocal/silencioso
   - Redes sociales: apodo, algo notable que haya publicado
   - Obras benéficas, fundaciones, causas que apoya
   - Récords personales, hitos, marcas históricas alcanzadas
   - Anécdotas jugosas: declaraciones polémicas, momentos virales, curiosidades
   - Ídolos futbolísticos que ha mencionado

5. **PERFIL FUTBOLÍSTICO ESTA TEMPORADA** 📊
   - Rol táctico actual en el equipo y cómo ha evolucionado bajo el DT actual
   - Estadísticas COMPLETAS de esta temporada:
     * Partidos jugados (titular/suplente), minutos
     * Goles, asistencias, pases clave
     * Para porteros: clean sheets, paradas, penaltis detenidos, goles recibidos por partido
     * Para defensas: intercepciones, despejes, duelos ganados, entradas exitosas
     * Para mediocampistas: pases completados, pases al último tercio, recuperaciones
     * Para delanteros: tiros a puerta, regates exitosos, xG, conversión
   - Tarjetas amarillas y rojas
   - Estado físico actual: ¿lesionado? ¿recién recuperado? ¿en racha?
   - Momento de forma: ¿en su mejor nivel o bajo rendimiento?
   - Comparaciones con temporadas anteriores si hay cambio notable

6. **SITUACIÓN CONTRACTUAL Y CONTEXTO**
   - Fecha de fin de contrato si es conocida
   - Rumores de transferencia relevantes
   - Relación con el entrenador y la directiva
   - Situación en su selección nacional

Resalta con ⚡ las conexiones con {opponent_name} y con 🎯 los datos curiosos \
más impactantes para que sean fáciles de localizar.

Responde en español. Sé EXHAUSTIVO y PRECISO — NO inventes datos. \
Si no encuentras un dato específico, omítelo, pero BUSCA A FONDO antes de rendirte.
FECHA ACTUAL: {current_date}."""


_PLAYER_SYNTHESIS_PROMPT = """Eres el redactor de fichas de transmisión de Fernando Palomo en ESPN.

Tu trabajo: recibir un dossier extenso de un jugador y sintetizarlo en una FICHA BREVE \
de 2 a 4 líneas que el narrador pueda leer de un vistazo en cabina.

FORMATO OBLIGATORIO (cada jugador):
[número] [Nombre] [edad] años. [ciudad/país de origen].
[Dato clave de trayectoria en 1-2 oraciones cortas: cantera, cesiones, fichajes, cifras de traspaso]
[Situación actual: rol en el equipo, momento de forma, dato curioso memorable]

REGLAS:
- SOLO datos duros verificables. Nada de opiniones ni adjetivos floridos.
- Prioriza: origen, trayectoria resumida (equipos + años), precio de fichaje si es relevante, posición real vs original.
- Si hay conexión con el rival, inclúyela en una línea extra.
- NO uses markdown, emojis, ni viñetas. Texto plano corrido, línea por línea.
- Máximo 4 líneas por jugador. Si no hay suficiente info, 2 líneas bastan.
- Responde en español."""


_SOLO_PLAYER_DOSSIER_PROMPT = """Eres un investigador de fútbol de élite. Tu misión es crear \
el dossier MÁS COMPLETO posible sobre UN SOLO jugador para que un narrador de televisión \
pueda transmitir con profundidad y precisión.

JUGADOR: **{player_name}** ({player_position}) — juega en **{team_name}**

Investiga A FONDO los siguientes aspectos:

1. **IDENTIDAD COMPLETA**
   - Nombre completo de nacimiento y nombre futbolístico
   - Apodo(s) — tanto oficiales como los que usa la afición
   - Edad, fecha de nacimiento, nacionalidad(es)
   - Posición principal y secundaria(s), número de camiseta
   - Estatura, peso, pierna hábil

2. **TRAYECTORIA DETALLADA**
   - Lugar de nacimiento y contexto (barrio, ciudad, contexto socioeconómico)
   - Cantera / club formativo: cómo fue descubierto, a qué edad, anécdotas de juveniles
   - TODOS los clubes anteriores con fechas, precio de traspaso si aplica, y rol en cada uno
   - Momento clave que lo catapultó a la élite (debut, gol importante, actuación icónica)
   - Cómo llegó a {team_name}: contexto de la negociación, precio, otros clubes interesados

3. **VIDA PERSONAL Y DATOS CURIOSOS** 🎯
   - Familia: padres, hermanos, pareja, hijos — especialmente si hay vínculos futbolísticos
   - Familiares en el fútbol profesional (padre, hermano, primo, tío que haya jugado)
   - Hobbies fuera del fútbol, pasiones conocidas
   - Rituales o costumbres previas a partidos
   - Celebraciones de gol icónicas y su significado
   - Historial de lesiones relevantes y cómo las superó
   - Personalidad: introvertido/extrovertido, líder vocal/silencioso
   - Redes sociales: apodo, algo notable que haya publicado
   - Obras benéficas, fundaciones, causas que apoya
   - Récords personales, hitos, marcas históricas alcanzadas
   - Anécdotas jugosas: declaraciones polémicas, momentos virales, curiosidades
   - Ídolos futbolísticos que ha mencionado

4. **PERFIL FUTBOLÍSTICO ESTA TEMPORADA** 📊
   - Rol táctico actual en el equipo y cómo ha evolucionado bajo el DT actual
   - Estadísticas COMPLETAS de esta temporada:
     * Partidos jugados (titular/suplente), minutos
     * Goles, asistencias, pases clave
     * Para porteros: clean sheets, paradas, penaltis detenidos, goles recibidos por partido
     * Para defensas: intercepciones, despejes, duelos ganados, entradas exitosas
     * Para mediocampistas: pases completados, pases al último tercio, recuperaciones
     * Para delanteros: tiros a puerta, regates exitosos, xG, conversión
   - Tarjetas amarillas y rojas
   - Estado físico actual: ¿lesionado? ¿recién recuperado? ¿en racha?
   - Momento de forma: ¿en su mejor nivel o bajo rendimiento?
   - Comparaciones con temporadas anteriores si hay cambio notable

5. **SITUACIÓN CONTRACTUAL Y CONTEXTO**
   - Fecha de fin de contrato si es conocida
   - Rumores de transferencia relevantes
   - Relación con el entrenador y la directiva
   - Situación en su selección nacional

Resalta con 🎯 los datos curiosos más impactantes para que sean fáciles de localizar.

Responde en español. Sé EXHAUSTIVO y PRECISO — NO inventes datos. \
Si no encuentras un dato específico, omítelo, pero BUSCA A FONDO antes de rendirte.
FECHA ACTUAL: {current_date}."""


# ---------------------------------------------------------------------------
# National Team Prompts
# ---------------------------------------------------------------------------

_NATIONAL_TEAM_HISTORY_PROMPT = """Eres un cronista histórico de selecciones nacionales con acceso \
a toda la hemeroteca del fútbol internacional. Tu misión: construir la ficha DEFINITIVA de una \
selección nacional para que Fernando Palomo pueda narrar con autoridad total.

SELECCIÓN: **{country}** | Confederación: {confederation}

Investiga A FONDO los siguientes bloques:

1. **📋 DATOS GENERALES**
   - Federación, confederación, fundación, sede, colores, apodo(s)
   - Entrenador actual + tiempo en el cargo + récord bajo su mando
   - Capitán actual + quién le sigue en jerarquía
   - Sistema táctico actual y variaciones

2. **🏆 HISTORIA EN MUNDIALES**
   - Participaciones totales, primera y más reciente clasificación
   - Mejor resultado en Copa del Mundo + edición + rivales en ese camino
   - Mundiales donde NO clasificaron que sorprendieron
   - Goleadores históricos en Mundiales
   - Partidos icónicos (victorias y derrotas que definieron una era)

3. **🌍 HISTORIA EN TORNEOS CONTINENTALES**
   - Copas América / EURO / AFCON / Copa de Asia / Gold Cup ganadas y en qué años
   - Rachas relevantes (más participaciones consecutivas, más finales seguidas, etc.)
   - Rivales constantes / rivalidades históricas continentales

4. **📊 CLASIFICATORIAS EN CURSO** (si aplica)
   - Confederación y formato de clasificatoria actual para el próximo Mundial
   - Posición actual en la tabla, puntos, partidos restantes
   - Resultados de los últimos 5 partidos (forma reciente)
   - Próximos 3 rivales en clasificatoria

5. **📌 ESTADÍSTICAS HISTÓRICAS**
   - Máximo goleador histórico: nombre, goles, años activo
   - Más partidos internacionales (caps): nombre, número de caps
   - Portero con más vallas invictas, si aplica
   - Racha invicta más larga

6. **🎭 DATOS CURIOSOS Y CULTURA**
   - Apodos del equipo y su origen
   - Ritual o himno especial de la selección
   - Héroes históricos que marcaron generaciones
   - Momentos virales o polémicos (positivos o negativos)
   - Relación de la selección con su afición y cómo se vive el fútbol en el país
   - Estadio o sede más representativa y su capacidad/ambiente

Resalta con 🏆 los hitos más importantes para facilitar la lectura.
Responde en español. Sé EXHAUSTIVO. NO inventes cifras.
FECHA ACTUAL: {current_date}."""


_NATIONAL_PLAYER_DOSSIER_PROMPT = """Eres el investigador oficial de la selección de {country} \
para la transmisión de ESPN. Tu misión: dossier COMPLETO de uno de sus convocados para que \
Fernando Palomo narre con datos frescos y profundidad real.

JUGADOR: **{player_name}** — Selección de **{country}**

Investiga A FONDO los siguientes aspectos:

1. **🎽 IDENTIDAD INTERNACIONAL**
   - Nombre completo y nombre deportivo
   - Apodo(s) dentro de la selección vs en su club
   - Fecha y lugar de nacimiento, nacionalidad(es)
   - Posición en la selección vs posición en su club (¿difieren?)
   - Número habitual en la selección, si lo tiene fijo

2. **🌍 CARRERA CON LA SELECCIÓN** ← PRIORIDAD #1
   - Fecha y rival del DEBUT internacional + resultado del partido
   - Caps totales (partidos jugados con la selección) actuales
   - Goles internacionales: total, hitos (primer gol, gol número 50, etc.)
   - Asistencias internacionales si aplica
   - Torneos jugados con la selección:
     * Copas del Mundo: ediciones, estadísticas, momentos clave
     * Torneos continentales: títulos, goles decisivos
     * Clasificatorias: rendimiento, goles importantes
   - En el torneo / clasificatoria ACTUAL: rendimiento, goles, minutos

3. **⚽ PERFIL EN CLUB (contexto, no biografía)**
   - Club actual y país
   - Rendimiento esta temporada (solo estadísticas clave)
   - ¿Cómo llega a la selección? ¿Titular indiscutible o en disputa?

4. **🎯 DATOS CURIOSOS — FACTOR WOW**
   - Familia: ¿algún familiar que también vistió la misma selección? ¿padre o hermano internacional?
   - Historia detrás de su debut: ¿cómo fue convocado por primera vez?
   - Celebración de gol icónica with la selección y su significado
   - Si alguna vez fue descartado/ignorado por la selección y cómo volvió
   - Récords con la selección que tiene o está a punto de alcanzar
   - Declaraciones memorables sobre la camiseta o el país
   - Comparación con el ídolo histórico de su posición en esa selección

5. **📊 ESTADO ACTUAL**
   - ¿Lesionado? ¿En racha? ¿Recién recuperado?
   - Situación en la selección: ¿capitán? ¿líder del vestuario? ¿joven promesa?
   - Relación con el DT actual

Resalta con 🎯 los datos más impactantes para narración en vivo.
Responde en español. FECHA ACTUAL: {current_date}."""


_NATIONAL_MATCH_PREP_PROMPT = """Eres el analista táctico y cronista de Fernando Palomo \
para la transmisión de un partido entre selecciones nacionales. Tu misión: \
preparación TOTAL del partido para que nada tome por sorpresa al narrador.

PARTIDO: **{home_country}** vs **{away_country}**
TORNEO / CONTEXTO: {tournament}
TIPO DE PARTIDO: {match_type}
FECHA ACTUAL: {current_date}

Crea la FICHA COMPLETA del partido con los siguientes bloques:

1. **⚔️ CONTEXTO DEL PARTIDO**
   - Qué está en juego: puntos de clasificatoria, pase a siguiente fase, título, etc.
   - Relevancia histórica de este partido en particular
   - ¿Es una final anticipada? ¿Un derbi confederacional? ¿Revancha histórica?

2. **📊 HISTORIAL DIRECTO (Head-to-Head)**
   - Partidos totales jugados entre ambas selecciones
   - Balance de victorias, empates, derrotas (por cada lado)
   - Enfrentamientos RECIENTES (últimos 5 partidos): resultado, torneo, año, árbitro si memorable
   - El partido más icónico de la historia entre ambas selecciones — detalla TODO
   - El resultado más abultado en cada dirección
   - ¿Alguna vez se enfrentaron en un Mundial? ¿En qué fase?

3. **🏠 {home_country} — Análisis**
   - Forma reciente: últimos 5 partidos con resultados, torneos y rivales
   - Sistema táctico habitual del DT y posible alineación titular
   - Jugadores clave en este partido (máximo 5): nombre + por qué importa HOY
   - Jugadores ausentes: lesionados, sancionados, no convocados
   - Fortalezas tácticas vs debilidades que puede explotar {away_country}

4. **✈️ {away_country} — Análisis**
   - Forma reciente: últimos 5 partidos
   - Sistema táctico y posible alineación
   - Jugadores clave (máximo 5)
   - Bajas y ausencias
   - Fortalezas vs debilidades que puede explotar {home_country}

5. **🔮 CLAVES TÁCTICAS DEL PARTIDO**
   - El duelo individual más importante a seguir
   - ¿Quién controla el mediocampo controla el partido? Por qué
   - Zonas del campo donde se decidirá el partido
   - Árbitro asignado si conocido — historial con ambas selecciones

6. **🎙️ FRASES PALOMO** — 3 frases en el estilo de Fernando Palomo listas para narrar:
   - Una sobre la historia entre ambas selecciones
   - Una sobre el jugador estrella de {home_country}
   - Una sobre el jugador estrella de {away_country}

Responde en español. Sé PRECISO y FASCINANTE. NO inventes datos."""


_PALOMO_PHRASES_PROMPT = """Eres Fernando Palomo — EL narrador legendario de ESPN. \
Estás preparando tus frases para la transmisión de un partido importante.

CONTEXTO DEL PARTIDO:
- {home_team} (Local) vs {away_team} (Visitante)
- Torneo: {tournament}
- Tipo de partido: {match_type}
- Estadio: {stadium}

INVESTIGACIÓN RECOPILADA SOBRE AMBOS EQUIPOS:

--- {home_team} ---
{home_context}

--- {away_team} ---
{away_context}

TU TAREA: Genera las frases que Fernando Palomo usaría para la transmisión de este partido.

ESTILO Y FORMATO:
- Las frases de narración en MAYÚSCULAS (como las leerías en cabina)
- Intercala datos históricos precisos con drama narrativo
- Incluye TODOS estos elementos:
  * Apertura épica del partido — ambientación, estadio, lo que está en juego, \
    el contexto geográfico y cultural
  * Contexto de ambos equipos — dinámicas, rachas, últimos resultados relevantes
  * Datos sobre el DT de cada equipo y su situación actual
  * Historial de enfrentamientos directos (head-to-head) con datos precisos
  * Datos sobre el torneo/formato y precedentes relevantes
  * Frases sobre jugadores clave que podrían ser protagonistas
  * Contexto extradeportivo relevante — declaraciones recientes de técnicos, \
    presidentes, polémicas, fichajes, lesiones
  * Datos OBSCUROS pero FASCINANTES (factor WOW 1000!) — conexiones históricas \
    que nadie más haría, comparaciones inesperadas, récords oscuros pero relevantes, \
    coincidencias numéricas, efemérides
  * Precedentes del formato/sede del torneo
  * Estadísticas de racha actual de cada equipo

CADA frase DEBE contener al menos un dato verificable. \
Busca SIEMPRE el dato que haga decir "¡NO SABÍA ESO!" — ese es tu sello.
Las frases deben fluir como si las estuvieras leyendo en cabina justo antes del pitazo.

EJEMPLO DE ESTILO (para referencia, NO copies esto):
"HACIENDO BUENO EL PLAN, LA SUPERCOPA DE ESPAÑA LLEVA AL MUNDO A PONER SU MIRADA EN ARABIA SAUDITA."
"SE JUEGA LEJOS DE LA GRAN VÍA O EL PASEO DE GRACIA, Y MUY CERCA DEL MAR ROJO."
"BARCELONA BUSCA SU 16a SUPERCOPA DE ESPAÑA Y REPETIR EL TÍTULO QUE LOGRARON LA TEMPORADA ANTERIOR."

Responde en español. NO inventes datos.
FECHA ACTUAL: {current_date}."""


# ---------------------------------------------------------------------------
# Citation utilities
# ---------------------------------------------------------------------------
_CITATION_SEPARATOR = "\n\n---\n📚 **Fuentes:**"


def _resolve_inline_citations(
    text: str, sources: List[Dict[str, str]]
) -> str:
    """Replace [N] citation markers in text with superscript markdown links.

    Gemini returns inline markers like [1], [3] that are 1-indexed into
    the search_results list.  We convert each to a clickable superscript:
      [3] -> [³](url)
    so they render as small linked numbers in markdown.
    """
    if not sources:
        return text

    _SUPERSCRIPT = {
        0: "⁰", 1: "¹", 2: "²", 3: "³", 4: "⁴",
        5: "⁵", 6: "⁶", 7: "⁷", 8: "⁸", 9: "⁹",
    }

    def _to_super(n: int) -> str:
        return "".join(_SUPERSCRIPT.get(int(d), d) for d in str(n))

    def _replace_marker(match: re.Match) -> str:
        idx = int(match.group(1)) - 1  # 1-indexed -> 0-indexed
        if idx < 0 or idx >= len(sources):
            return match.group(0)  # leave unknown markers as-is
        src = sources[idx]
        url = src.get("url", "")
        if not url:
            return match.group(0)
        sup = _to_super(idx + 1)
        return f"[{sup}]({url})"

    return re.sub(r'\[(\d+)\]', _replace_marker, text)


def _format_sources(sources: List[Dict[str, str]]) -> str:
    """Format sources as a numbered markdown footer."""
    if not sources:
        return ""
    lines = [_CITATION_SEPARATOR]
    for i, src in enumerate(sources, 1):
        url = src.get("url", "")
        title = src.get("title", "")
        if not url:
            continue
        try:
            domain = urlparse(url).netloc.replace("www.", "")
        except Exception:
            domain = url[:60]
        display = title if title else domain
        lines.append(f"{i}. [{display}]({url})")
    return "\n".join(lines) if len(lines) > 1 else ""


# ---------------------------------------------------------------------------
# PDF Generation for Match Preparation
# ---------------------------------------------------------------------------

_PDF_FONT = "Helvetica"

# Regex patterns to strip markdown / emoji for clean PDF text
_RE_MD_BOLD = re.compile(r'\*\*(.+?)\*\*')
_RE_MD_ITALIC = re.compile(r'\*(.+?)\*')
_RE_MD_LINK = re.compile(r'\[([^\]]+)\]\([^)]+\)')
_RE_MD_HEADER = re.compile(r'^#{1,6}\s+')
_RE_SUPERSCRIPT_LINK = re.compile(r'\[[\u2070-\u2079\u00B2\u00B3\u00B9]+\]\([^)]+\)')
_RE_CITATION_BLOCK = re.compile(r'\n---\n.*?Fuentes:.*', re.DOTALL)


def _strip_markdown(text: str) -> str:
    """Convert markdown text to plain text suitable for PDF."""
    text = _RE_CITATION_BLOCK.sub('', text)
    text = _RE_SUPERSCRIPT_LINK.sub('', text)
    text = _RE_MD_LINK.sub(r'\1', text)
    text = _RE_MD_BOLD.sub(r'\1', text)
    text = _RE_MD_ITALIC.sub(r'\1', text)
    text = _RE_MD_HEADER.sub('', text)
    # Strip bullet points to a simple angle bracket
    text = re.sub(r'^\s*[-*]\s+', '> ', text, flags=re.MULTILINE)
    return text.strip()


def _clean_for_latin(text: str) -> str:
    """Replace characters outside latin-1 with close equivalents."""
    replacements = {
        '\u2013': '-', '\u2014': '-', '\u2018': "'", '\u2019': "'",
        '\u201c': '"', '\u201d': '"', '\u2026': '...',
        '\u2070': '0', '\u00b9': '1', '\u00b2': '2', '\u00b3': '3',
        '\u2074': '4', '\u2075': '5', '\u2076': '6', '\u2077': '7',
        '\u2078': '8', '\u2079': '9',
        '\u26a1': '[!]', '\U0001f3af': '[*]',
        '\U0001f9e4': '', '\U0001f6e1': '', '\U0001f3af': '', '\u26a1': '[!]',
        '\u2705': '[OK]', '\u274c': '[X]', '\U0001f50d': '',
        '\U0001f4ca': '', '\U0001f4cb': '', '\U0001f3c6': '',
        '\U0001f3df': '', '\U0001f3e0': '', '\u2708': '',
        '\U0001f4cc': '', '\U0001f680': '', '\U0001f5d1': '',
        '\U0001f3d9': '', '\U0001f30d': '',
        '\U0001f399': '', '\U0001f4da': '',
        '\U0001f46b': '', '\U0001f465': '',
        '\n- ': '\n> ', '\n* ': '\n> ',  # Strip markdown bullet variants to >
    }
    for char, repl in replacements.items():
        text = text.replace(char, repl)
    # Normalize accented characters to their closest ASCII+combining form,
    # then drop combining marks — keeps "Müller" as "Muller" instead of "".
    text = unicodedata.normalize('NFKD', text)
    # Fast fallback: replace any remaining non-latin1 chars
    return text.encode('latin-1', errors='ignore').decode('latin-1')


class _MatchPDF(FPDF):
    """Custom FPDF subclass for match preparation reports."""

    def __init__(self, config: dict) -> None:
        super().__init__(orientation='P', unit='mm', format='A4')
        self.config = config
        self.set_auto_page_break(auto=True, margin=20)

    # -- header / footer --
    def header(self):
        if self.page_no() == 1:  # cover has no header
            return
        self.set_font(_PDF_FONT, 'I', 8)
        home = _clean_for_latin(self.config['home_team'])
        away = _clean_for_latin(self.config['away_team'])
        self.cell(0, 6, f"{home} vs {away} - Preparacion de Partido", align='C')
        self.ln(8)

    def footer(self):
        if self.page_no() == 1:
            return
        self.set_y(-15)
        self.set_font(_PDF_FONT, 'I', 8)
        self.cell(0, 10, f"Pagina {self.page_no() - 1}", align='C')

    # -- helpers --
    def _section_title(self, title: str):
        self.set_font(_PDF_FONT, 'B', 14)
        self.cell(0, 10, _clean_for_latin(title), ln=True)
        self.ln(2)

    def _sub_title(self, title: str):
        self.set_font(_PDF_FONT, 'B', 11)
        self.cell(0, 8, _clean_for_latin(title), ln=True)
        self.ln(1)

    def _body_text(self, text: str, size: int = 9):
        self.set_font(_PDF_FONT, '', size)
        clean = _clean_for_latin(_strip_markdown(text))
        self.multi_cell(0, 4.5, clean)
        self.ln(2)

    # -- cover page --
    def add_cover(self):
        self.add_page()
        self.ln(60)

        self.set_font(_PDF_FONT, 'B', 28)
        home = _clean_for_latin(self.config['home_team'])
        away = _clean_for_latin(self.config['away_team'])
        self.cell(0, 14, home, ln=True, align='C')
        self.ln(4)
        self.set_font(_PDF_FONT, '', 18)
        self.cell(0, 10, 'vs', ln=True, align='C')
        self.ln(4)
        self.set_font(_PDF_FONT, 'B', 28)
        self.cell(0, 14, away, ln=True, align='C')

        self.ln(16)
        self.set_font(_PDF_FONT, '', 14)
        meta = _clean_for_latin(
            f"{self.config['match_type']}  |  {self.config['tournament']}"
        )
        self.cell(0, 8, meta, ln=True, align='C')

        self.ln(4)
        self.set_font(_PDF_FONT, '', 12)
        stadium = _clean_for_latin(self.config['stadium'])
        self.cell(0, 8, stadium, ln=True, align='C')

        self.ln(8)
        self.set_font(_PDF_FONT, 'I', 10)
        self.cell(0, 8, CURRENT_DATE, ln=True, align='C')

        self.ln(30)
        self.set_font(_PDF_FONT, '', 9)
        self.cell(0, 6, 'Preparacion de Partido - PalomoFacts', ln=True, align='C')

    # -- team histories --
    def add_team_histories(self, results: dict):
        # Home team history
        self.add_page()
        home_label = _clean_for_latin(f"Historial de Temporadas - LOCAL: {self.config['home_team']}")
        self._section_title(home_label)

        home_text = results['home_history'][0] if isinstance(results['home_history'], tuple) else ''
        if home_text:
            self._body_text(home_text, size=9)

        # Away team history
        self.add_page()
        away_label = _clean_for_latin(f"Historial de Temporadas - VISITANTE: {self.config['away_team']}")
        self._section_title(away_label)

        away_text = results['away_history'][0] if isinstance(results['away_history'], tuple) else ''
        if away_text:
            self._body_text(away_text, size=9)

    # -- roster per team --
    def _add_roster_section(self, team_name: str, roster: list, label: str):
        self.add_page()
        self._section_title(f'Plantilla - {label}: {team_name}')

        pos_order = ['GK', 'DEF', 'MID', 'FWD']
        pos_names = {
            'GK': 'PORTEROS', 'DEF': 'DEFENSAS',
            'MID': 'CENTROCAMPISTAS', 'FWD': 'DELANTEROS',
        }
        grouped: Dict[str, list] = {p: [] for p in pos_order}
        for player in roster:
            pos = player.get('position', 'FWD')
            if pos not in grouped:
                pos = 'FWD'
            grouped[pos].append(player)

        for pos in pos_order:
            players = grouped[pos]
            if not players:
                continue
            self._sub_title(pos_names.get(pos, pos))
            for p in players:
                number = p.get('number', '')
                name = _clean_for_latin(p['name'])
                header = f"#{number} {name}" if number else name

                # Player name as bold line
                self.set_font(_PDF_FONT, 'B', 10)
                self.cell(0, 6, header, ln=True)

                # Player dossier body
                body = p.get('text', '')
                if body:
                    self._body_text(body, size=8)

                # Page break safety — if less than 30mm left, new page
                if self.get_y() > self.h - 30:
                    self.add_page()

    def add_rosters(self, results: dict):
        home = _clean_for_latin(self.config['home_team'])
        away = _clean_for_latin(self.config['away_team'])
        self._add_roster_section(home, results.get('home_roster', []), 'Local')
        self._add_roster_section(away, results.get('away_roster', []), 'Visitante')

    # -- Palomo phrases --
    def add_palomo_phrases(self, results: dict):
        self.add_page()
        self._section_title('Frases de Fernando Palomo')
        text = results['palomo_phrases'][0] if isinstance(results['palomo_phrases'], tuple) else ''
        if text:
            self._body_text(text, size=10)


def generate_match_pdf(
    config: dict,
    results: dict,
    api_key: str = "",
    progress_cb: Optional[Callable[[str], None]] = None,
) -> bytes:
    """Generate a complete match preparation PDF.

    If api_key is provided, synthesizes verbose player dossiers into compact
    broadcast-ready notes before writing to PDF.
    """

    def _cb(msg: str) -> None:
        if progress_cb:
            progress_cb(msg)

    # Synthesize rosters before PDF generation
    synth_results = dict(results)  # shallow copy to avoid mutating original
    if api_key:
        for key in ("home_roster", "away_roster"):
            roster = results.get(key, [])
            if roster:
                label = "Local" if "home" in key else "Visitante"
                _cb(f"Sintetizando {len(roster)} jugadores ({label})...")
                synth_results[key] = _synthesize_roster_for_pdf(roster, api_key, progress_cb=progress_cb)

    _cb("Generando PDF...")
    pdf = _MatchPDF(config)
    pdf.add_cover()
    pdf.add_team_histories(synth_results)
    pdf.add_rosters(synth_results)
    pdf.add_palomo_phrases(synth_results)
    # the fpdf output() sometimes returns a latin1 string instead of bytes,
    # which causes Streamlit's download_button to crash when it tries to infer the mime type.
    # Additionally, fpdf2 returns a bytearray, which Streamlit also does not support.
    out = pdf.output()
    if isinstance(out, str):
        return out.encode('latin1')
    return bytes(out)


def _synthesize_one_player(
    player: Dict[str, Any],
    client: anthropic.Anthropic,
) -> Dict[str, Any]:
    """Synthesize a single player dossier. Called in parallel."""
    raw_text = player.get("text", "")
    if not raw_text or len(raw_text) < 50:
        return player
    try:
        response = client.messages.create(
            model=CLAUDE_HAIKU_MODEL,
            max_tokens=512,
            system=_PLAYER_SYNTHESIS_PROMPT,
            messages=[{"role": "user", "content": f"Sintetiza este dossier:\n\n{raw_text}"}],
        )
        note = response.content[0].text
        return {
            **player,
            "text": note.strip(),
            "raw_text": raw_text,
        }
    except Exception as e:
        print(f"[PDF Synthesis] Error for {player.get('name', '?')}: {e}")
        return player


def _synthesize_roster_for_pdf(
    roster: List[Dict[str, Any]],
    api_key: str,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> List[Dict[str, Any]]:
    """Synthesize verbose player dossiers into compact broadcast notes via Claude Haiku.
    
    Uses parallel requests (up to 8 concurrent) for speed.
    """
    claude_key = st.secrets.get("ANTHROPIC_API_KEY", "")
    if not claude_key:
        print("[PDF Synthesis] No ANTHROPIC_API_KEY — skipping synthesis.")
        return roster

    client = anthropic.Anthropic(api_key=claude_key)
    synthesized: List[Optional[Dict[str, Any]]] = [None] * len(roster)
    done_count = 0
    total = len(roster)

    def _cb(msg: str) -> None:
        if progress_cb:
            progress_cb(msg)

    with ThreadPoolExecutor(max_workers=8) as executor:
        future_to_idx = {
            executor.submit(_synthesize_one_player, player, client): idx
            for idx, player in enumerate(roster)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                synthesized[idx] = future.result()
            except Exception as e:
                print(f"[PDF Synthesis] Unexpected error for idx {idx}: {e}")
                synthesized[idx] = roster[idx]
            done_count += 1
            name = roster[idx].get("name", "?")
            _cb(f"  ✓ {name} ({done_count}/{total})")

    return [p for p in synthesized if p is not None]


# ---------------------------------------------------------------------------
# Gemini API
# ---------------------------------------------------------------------------
_MAX_RETRIES = 3
_RETRY_BASE_DELAY = 1  # seconds


def _gemini_request(
    api_key: str,
    system_prompt: str,
    user_message: str,
    history: Optional[List[types.Content]] = None,
    timeout: int = 120,
    use_search: bool = True,
) -> Tuple[str, List[Dict[str, str]]]:
    """
    Request to Gemini with optional Google Search grounding.
    Retries up to _MAX_RETRIES times with exponential backoff.
    Returns (response_text, sources).
    """
    client = genai.Client(api_key=api_key)

    tools = []
    if use_search:
        tools.append({"google_search": {}})
        tools.append({"url_context": {}})

    def _build_config(model_name: str) -> types.GenerateContentConfig:
        """Build config — only add thinking for models that support it."""
        kwargs: Dict[str, Any] = {
            "system_instruction": system_prompt,
            "tools": tools,
        }
        if "3.1" in model_name:
            kwargs["thinking_config"] = types.ThinkingConfig(thinking_level="HIGH")
        return types.GenerateContentConfig(**kwargs)

    contents = []
    if history:
        contents.extend(history)
    contents.append(
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=user_message)],
        )
    )

    last_error = None
    model = GEMINI_MODEL
    for attempt in range(_MAX_RETRIES):
        try:
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=_build_config(model),
            )

            text = response.text or ""

            # Extract sources from grounding metadata
            sources: List[Dict[str, str]] = []
            try:
                candidate = response.candidates[0]
                if candidate.grounding_metadata and candidate.grounding_metadata.grounding_chunks:
                    for chunk in candidate.grounding_metadata.grounding_chunks:
                        if chunk.web:
                            sources.append({
                                "title": chunk.web.title or "",
                                "url": chunk.web.uri or "",
                                "snippet": "",
                            })
            except (AttributeError, IndexError):
                pass

            # Resolve inline [N] markers into clickable superscript links
            text = _resolve_inline_citations(text, sources)

            return text, sources

        except Exception as e:
            last_error = e
            err_str = str(e)

            # On 429 quota exhausted, try Claude first, then Gemini fallback
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                # Try Claude Opus as first fallback
                if model == GEMINI_MODEL:
                    try:
                        print(f"[Gemini] 429 quota hit on {model} — trying Claude Opus 4.6...")
                        return _claude_request(system_prompt, user_message, history)
                    except Exception as claude_err:
                        print(f"[Claude] Fallback failed: {claude_err} — downgrading to {GEMINI_FALLBACK_MODEL}")
                        model = GEMINI_FALLBACK_MODEL
                        time.sleep(1)
                        continue
                elif model != GEMINI_FALLBACK_MODEL:
                    print(f"[Gemini] 429 quota hit on {model} — downgrading to {GEMINI_FALLBACK_MODEL}")
                    model = GEMINI_FALLBACK_MODEL
                    time.sleep(1)
                    continue

            delay = _RETRY_BASE_DELAY * (2 ** attempt)
            print(f"[Gemini] Attempt {attempt + 1}/{_MAX_RETRIES} failed: {e}. "
                  f"Retrying in {delay}s...")
            if attempt < _MAX_RETRIES - 1:
                time.sleep(delay)

    raise RuntimeError(f"Gemini API failed after {_MAX_RETRIES} attempts: {last_error}")


def _claude_request(
    system_prompt: str,
    user_message: str,
    history: Optional[list] = None,
) -> Tuple[str, List[Dict[str, str]]]:
    """
    Fallback request via Claude Opus 4.6.
    No search grounding — purely LLM reasoning.
    Returns (response_text, sources=[]).
    """
    claude_key = st.secrets.get("ANTHROPIC_API_KEY", "")
    if not claude_key:
        raise RuntimeError("ANTHROPIC_API_KEY not configured")

    client = anthropic.Anthropic(api_key=claude_key)

    # Build messages from history + current user message
    messages: List[Dict[str, str]] = []
    if history:
        for content in history:
            try:
                role = "user" if content.role == "user" else "assistant"
                text_parts = [p.text for p in content.parts if hasattr(p, "text") and p.text]
                if text_parts:
                    messages.append({"role": role, "content": "\n".join(text_parts)})
            except (AttributeError, TypeError):
                pass
    messages.append({"role": "user", "content": user_message})

    print(f"[Claude] Calling {CLAUDE_MODEL} as fallback...")
    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=16384,
        system=system_prompt,
        messages=messages,
        tools=[{"type": "web_search_20250305", "name": "web_search", "max_uses": 5}],
    )

    text = ""
    for block in response.content:
        if hasattr(block, "text"):
            text += block.text

    print(f"[Claude] Response received ({len(text)} chars)")
    return text, []  # no grounding sources from Claude


# ---------------------------------------------------------------------------
# PalomoGPT response
# ---------------------------------------------------------------------------
def get_palomo_response(
    query: str,
    session_messages: list[dict],
    api_key: str,
) -> Tuple[str, List[Dict[str, str]], List[str], List[Dict[str, str]]]:
    """Get a response from PalomoGPT with Google Search grounding + follow-up chain.
    Returns (response_text, sources, reasoning_chain, followups).
    """
    # Build conversation history for Gemini
    history: List[types.Content] = []
    for msg in session_messages[:-1]:  # exclude most recent user msg
        role = "user" if msg["role"] == "user" else "model"
        content = msg.get("content", "")
        if content and not content.startswith("🧠"):  # skip reasoning messages
            history.append(
                types.Content(
                    role=role,
                    parts=[types.Part.from_text(text=content)],
                )
            )

    reasoning: List[str] = []

    # Single pass: Gemini + Google Search grounding
    text, sources = _gemini_request(
        api_key=api_key,
        system_prompt=_PALOMO_GPT_SYSTEM,
        user_message=query,
        history=history if history else None,
    )

    source_titles = [s.get("title", "?") for s in sources[:10]]
    reasoning.append(
        f"🎯 **Búsqueda** (modelo: `{GEMINI_MODEL}` + Google Search):\n\n"
        f"- Fuentes obtenidas: {len(sources)}\n"
        f"- Top fuentes: {', '.join(source_titles) if source_titles else 'ninguna'}\n"
    )

    # === Follow-up iterations: 2 rounds of auto-generated questions ===
    followups: List[Dict[str, str]] = []
    current_answer = text
    for i in range(2):
        try:
            # Generate a follow-up question
            followup_q, _ = _gemini_request(
                api_key=api_key,
                system_prompt=_FOLLOW_UP_SYSTEM,
                user_message=(
                    f"PREGUNTA ORIGINAL: {query}\n\n"
                    f"RESPUESTA RECIBIDA:\n{current_answer}"
                ),
                use_search=False,
            )
            followup_q = followup_q.strip()
            if not followup_q or len(followup_q) < 10:
                break

            reasoning.append(
                f"🔄 **Follow-up {i+1}:** {followup_q}"
            )

            # Answer the follow-up question with search
            followup_a, extra_sources = _gemini_request(
                api_key=api_key,
                system_prompt=_PALOMO_GPT_SYSTEM,
                user_message=followup_q,
                history=[
                    types.Content(role="user", parts=[types.Part.from_text(text=query)]),
                    types.Content(role="model", parts=[types.Part.from_text(text=current_answer)]),
                ],
            )

            followups.append({"question": followup_q, "answer": followup_a})
            current_answer = followup_a  # next iteration uses this answer

            # Merge sources
            existing_urls = {s.get("url") for s in sources if s.get("url")}
            for s in extra_sources:
                if s.get("url") and s["url"] not in existing_urls:
                    sources.append(s)
                    existing_urls.add(s["url"])

        except Exception as e:
            print(f"[Follow-up {i+1}] Error: {e}")
            reasoning.append(f"⚠️ **Follow-up {i+1} omitido:** `{e}`")
            break

    return text, sources, reasoning, followups


# ---------------------------------------------------------------------------
# Match Preparation Research
# ---------------------------------------------------------------------------
def _research_team_history(
    team_name: str,
    api_key: str,
) -> Tuple[str, List[Dict[str, str]]]:
    """Research a team's last 2 seasons history."""
    season_prev = CURRENT_YEAR - 2
    season_curr = CURRENT_YEAR - 1
    season_next = CURRENT_YEAR

    prompt = _TEAM_HISTORY_PROMPT.format(
        team_name=team_name,
        season_prev=season_prev,
        season_curr=season_curr,
        season_next=season_next,
        current_date=CURRENT_DATE,
    )

    return _gemini_request(
        api_key=api_key,
        system_prompt=prompt,
        user_message=(
            f"Investiga las últimas 2 temporadas completas de {team_name}. "
            f"Temporadas {season_prev}/{season_curr} y {season_curr}/{season_next}. "
            "Sé exhaustivo con cada competición — especialmente en competiciones europeas "
            "donde necesito CADA partido con resultado y goleadores."
        ),
    )


def _fetch_player_list(
    team_name: str,
    api_key: str,
) -> List[Dict[str, Any]]:
    """Fetch the roster as a list of {name, full_name, position, number} dicts."""
    season_curr = CURRENT_YEAR - 1
    season_next = CURRENT_YEAR

    prompt = _TEAM_ROSTER_LIST_PROMPT.format(
        team_name=team_name,
        season_curr=season_curr,
        season_next=season_next,
        current_date=CURRENT_DATE,
    )

    text, _ = _gemini_request(
        api_key=api_key,
        system_prompt=prompt,
        user_message=(
            f"Dame la plantilla completa actual de {team_name} para la temporada "
            f"{season_curr}/{season_next}. Solo el JSON, nada más."
        ),
    )

    # Extract JSON from response (may have markdown fences)
    json_match = re.search(r'\{[\s\S]*\}', text)
    if not json_match:
        raise RuntimeError(f"Could not parse player list JSON from response: {text[:300]}")
    try:
        data = json.loads(json_match.group())
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Invalid JSON in player list: {e}\n{text[:300]}")

    players = data.get("players", [])
    if not players:
        raise RuntimeError(f"Empty player list returned for {team_name}")
    return players


_POS_LABELS = {"GK": "🧤 Portero", "DEF": "🛡️ Defensa", "MID": "🎯 Centrocampista", "FWD": "⚡ Delantero"}


def _research_single_player(
    player: Dict[str, Any],
    team_name: str,
    opponent_name: str,
    api_key: str,
) -> Dict[str, Any]:
    """Research a single player. Returns {name, position, text, sources}."""
    player_name = player.get("name", player.get("full_name", "Unknown"))
    position = player.get("position", "")
    pos_label = _POS_LABELS.get(position, position)

    prompt = _PLAYER_DOSSIER_PROMPT.format(
        player_name=player_name,
        player_position=pos_label,
        team_name=team_name,
        opponent_name=opponent_name,
        current_date=CURRENT_DATE,
    )

    text, sources = _gemini_request(
        api_key=api_key,
        system_prompt=prompt,
        user_message=(
            f"Dame el dossier COMPLETO de {player_name} ({pos_label}) de {team_name}. "
            f"El próximo rival es {opponent_name} — busca TODAS las conexiones posibles. "
            "Incluye biografía, trayectoria, vida personal, datos curiosos, "
            "estadísticas de esta temporada, y situación contractual."
        ),
    )


    return {
        "name": player_name,
        "position": position,
        "number": player.get("number", ""),
        "text": text,
        "sources": sources,
    }


def _research_team_roster(
    team_name: str,
    opponent_name: str,
    api_key: str,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> List[Dict[str, Any]]:
    """
    Research a team's roster player-by-player.
    1. Fetch player list (lightweight call).
    2. For each player, run a dedicated deep-research call (parallel batches of 4).
    Returns a list of {name, position, number, text, sources} dicts.
    """
    if progress_cb:
        progress_cb(f"📋 Obteniendo lista de jugadores de **{team_name}**...")
    players = _fetch_player_list(team_name, api_key)
    total = len(players)
    if progress_cb:
        progress_cb(f"✅ {total} jugadores encontrados en **{team_name}**. Investigando uno por uno...")

    results: List[Dict[str, Any]] = [None] * total  # preserve order
    completed = 0

    # Process in parallel batches of 4 to avoid rate limits
    batch_size = 4
    for batch_start in range(0, total, batch_size):
        batch = players[batch_start : batch_start + batch_size]
        with ThreadPoolExecutor(max_workers=batch_size) as executor:
            future_to_idx = {
                executor.submit(
                    _research_single_player, player, team_name, opponent_name, api_key
                ): batch_start + i
                for i, player in enumerate(batch)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    p = players[idx]
                    results[idx] = {
                        "name": p.get("name", "?"),
                        "position": p.get("position", ""),
                        "number": p.get("number", ""),
                        "text": f"❌ Error investigando: {e}",
                        "sources": [],
                    }
                completed += 1
                if progress_cb:
                    pname = results[idx]["name"]
                    progress_cb(f"🔍 [{completed}/{total}] **{pname}** ✓  ({team_name})")

    return results


def _research_palomo_phrases(
    home_team: str,
    away_team: str,
    tournament: str,
    match_type: str,
    stadium: str,
    home_context: str,
    away_context: str,
    api_key: str,
) -> Tuple[str, List[Dict[str, str]]]:
    """Generate Fernando Palomo–style phrases for the match."""
    # Truncate context to stay within token limits
    max_ctx = 4000
    home_ctx = home_context[:max_ctx] if len(home_context) > max_ctx else home_context
    away_ctx = away_context[:max_ctx] if len(away_context) > max_ctx else away_context

    prompt = _PALOMO_PHRASES_PROMPT.format(
        home_team=home_team,
        away_team=away_team,
        tournament=tournament,
        match_type=match_type,
        stadium=stadium,
        home_context=home_ctx,
        away_context=away_ctx,
        current_date=CURRENT_DATE,
    )

    return _gemini_request(
        api_key=api_key,
        system_prompt=prompt,
        user_message=(
            f"Genera las frases de Fernando Palomo para la transmisión de "
            f"{home_team} vs {away_team}, {match_type} de {tournament} en {stadium}. "
            "Factor WOW al máximo. Quiero datos que nadie más tendría. "
            "Incluye apertura, contexto de ambos equipos, head-to-head, "
            "datos obscuros, y frases para distintos escenarios del partido."
        ),
    )


def _has_data(val: Any) -> bool:
    """Check if a results value has meaningful data (not empty/error)."""
    if isinstance(val, tuple):
        text = val[0] if val else ""
        return bool(text) and not text.startswith("❌")
    if isinstance(val, list):
        return len(val) > 0
    return bool(val)


def _roster_has_failures(roster: list) -> bool:
    """Check if any players in a roster have error text."""
    return any(
        p.get("text", "").startswith("❌")
        for p in roster
    )


def _retry_failed_roster_players(
    roster: List[Dict[str, Any]],
    team_name: str,
    opponent_name: str,
    api_key: str,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> List[Dict[str, Any]]:
    """Re-research only the players that have error text in their dossier."""
    _cb = progress_cb or (lambda _msg: None)
    failed_indices = [
        i for i, p in enumerate(roster)
        if p.get("text", "").startswith("❌")
    ]
    if not failed_indices:
        return roster

    total_failed = len(failed_indices)
    _cb(f"🔄 Re-investigando **{total_failed}** jugadores fallidos de **{team_name}**...")

    batch_size = 4
    completed = 0
    for batch_start in range(0, total_failed, batch_size):
        batch_idxs = failed_indices[batch_start : batch_start + batch_size]
        with ThreadPoolExecutor(max_workers=batch_size) as executor:
            future_to_idx = {
                executor.submit(
                    _research_single_player,
                    roster[idx],  # use existing player info (name, position, number)
                    team_name,
                    opponent_name,
                    api_key,
                ): idx
                for idx in batch_idxs
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    roster[idx] = future.result()
                except Exception as e:
                    # Keep the error but update it
                    roster[idx]["text"] = f"❌ Error investigando: {e}"
                completed += 1
                pname = roster[idx].get("name", "?")
                _cb(f"🔄 [{completed}/{total_failed}] **{pname}** ✓  ({team_name})")

    return roster


def run_match_preparation(
    home_team: str,
    away_team: str,
    tournament: str,
    match_type: str,
    stadium: str,
    api_key: str,
    progress_cb: Optional[Callable[[str], None]] = None,
    partial_results: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Run the full match preparation pipeline with **resumability**.

    If partial_results is provided, phases that already have data are
    skipped — saving tokens and time.  Each phase incrementally saves
    its output via the progress callback so that the caller can persist
    intermediate state to session_state.

    Phases:
      1 — parallel: team histories (2 calls)
      2 — sequential per team: roster player-by-player
      3 — Palomo phrases (uses Phase 1 context)
    """
    _cb = progress_cb or (lambda _msg: None)
    results: Dict[str, Any] = partial_results or {
        "home_history": ("", []),
        "away_history": ("", []),
        "home_roster": [],
        "away_roster": [],
        "palomo_phrases": ("", []),
    }

    # Phase 1: team histories in parallel (skip if already done)
    need_home_hist = not _has_data(results.get("home_history"))
    need_away_hist = not _has_data(results.get("away_history"))
    if need_home_hist or need_away_hist:
        _cb(f"📊 Investigando historial de **{home_team}** y **{away_team}**...")
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {}
            if need_home_hist:
                futures[executor.submit(_research_team_history, home_team, api_key)] = "home_history"
            if need_away_hist:
                futures[executor.submit(_research_team_history, away_team, api_key)] = "away_history"
            for future in as_completed(futures):
                key = futures[future]
                try:
                    results[key] = future.result()
                except Exception as e:
                    results[key] = (f"❌ Error investigando: {e}", [])
        _cb("✅ Historiales completados.")
    else:
        _cb("✅ Historiales ya disponibles — reutilizando.")

    # Phase 2: rosters — player by player (skip if already done)
    if not _has_data(results.get("home_roster")):
        _cb(f"👥 Investigando plantilla de **{home_team}** jugador por jugador...")
        try:
            results["home_roster"] = _research_team_roster(
                home_team, away_team, api_key, progress_cb=_cb,
            )
        except Exception as e:
            results["home_roster"] = []
            _cb(f"❌ Error con plantilla de {home_team}: {e}")
    elif _roster_has_failures(results.get("home_roster", [])):
        results["home_roster"] = _retry_failed_roster_players(
            results["home_roster"], home_team, away_team, api_key, progress_cb=_cb,
        )
    else:
        _cb(f"✅ Plantilla de **{home_team}** ya disponible — reutilizando.")

    if not _has_data(results.get("away_roster")):
        _cb(f"👥 Investigando plantilla de **{away_team}** jugador por jugador...")
        try:
            results["away_roster"] = _research_team_roster(
                away_team, home_team, api_key, progress_cb=_cb,
            )
        except Exception as e:
            results["away_roster"] = []
            _cb(f"❌ Error con plantilla de {away_team}: {e}")
    elif _roster_has_failures(results.get("away_roster", [])):
        results["away_roster"] = _retry_failed_roster_players(
            results["away_roster"], away_team, home_team, api_key, progress_cb=_cb,
        )
    else:
        _cb(f"✅ Plantilla de **{away_team}** ya disponible — reutilizando.")

    # Phase 3: Palomo phrases (skip if already done)
    if not _has_data(results.get("palomo_phrases")):
        _cb("🎙️ Generando frases de Fernando Palomo...")
        home_context = results["home_history"][0] if isinstance(results["home_history"], tuple) else ""
        away_context = results["away_history"][0] if isinstance(results["away_history"], tuple) else ""

        try:
            results["palomo_phrases"] = _research_palomo_phrases(
                home_team, away_team, tournament, match_type, stadium,
                home_context, away_context, api_key,
            )
        except Exception as e:
            results["palomo_phrases"] = (f"❌ Error generando frases: {e}", [])
    else:
        _cb("✅ Frases de Palomo ya disponibles — reutilizando.")

    return results


# ---------------------------------------------------------------------------
# Team Research Pipeline (single club, no opponent)
# ---------------------------------------------------------------------------

def _research_single_player_solo(
    player: Dict[str, Any],
    team_name: str,
    api_key: str,
) -> Dict[str, Any]:
    """Research a single player without opponent context. Returns {name, position, number, text, sources}."""
    player_name = player.get("name", player.get("full_name", "Unknown"))
    position = player.get("position", "")
    pos_label = _POS_LABELS.get(position, position)

    prompt = _SOLO_PLAYER_DOSSIER_PROMPT.format(
        player_name=player_name,
        player_position=pos_label,
        team_name=team_name,
        current_date=CURRENT_DATE,
    )

    text, sources = _gemini_request(
        api_key=api_key,
        system_prompt=prompt,
        user_message=(
            f"Dame el dossier COMPLETO de {player_name} ({pos_label}) de {team_name}. "
            "Incluye biografía, trayectoria, vida personal, datos curiosos, "
            "estadísticas de esta temporada y situación contractual."
        ),
    )

    return {
        "name": player_name,
        "position": position,
        "number": player.get("number", ""),
        "text": text,
        "sources": sources,
    }


def _research_team_roster_solo(
    team_name: str,
    api_key: str,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> List[Dict[str, Any]]:
    """Research a team's full roster player-by-player (no opponent context)."""
    if progress_cb:
        progress_cb(f"📋 Obteniendo lista de jugadores de **{team_name}**...")
    players = _fetch_player_list(team_name, api_key)
    total = len(players)
    if progress_cb:
        progress_cb(f"✅ {total} jugadores encontrados en **{team_name}**. Investigando uno por uno...")

    results: List[Dict[str, Any]] = [None] * total  # type: ignore[list-item]
    completed = 0
    batch_size = 4
    for batch_start in range(0, total, batch_size):
        batch = players[batch_start: batch_start + batch_size]
        with ThreadPoolExecutor(max_workers=batch_size) as executor:
            future_to_idx = {
                executor.submit(_research_single_player_solo, player, team_name, api_key): batch_start + i
                for i, player in enumerate(batch)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    p = players[idx]
                    results[idx] = {
                        "name": p.get("name", "?"),
                        "position": p.get("position", ""),
                        "number": p.get("number", ""),
                        "text": f"❌ Error investigando: {e}",
                        "sources": [],
                    }
                completed += 1
                if progress_cb:
                    pname = results[idx]["name"]
                    progress_cb(f"🔍 [{completed}/{total}] **{pname}** ✓  ({team_name})")

    return results


def run_team_research(
    team_name: str,
    tournament: str,
    api_key: str,
    progress_cb: Optional[Callable[[str], None]] = None,
    partial_results: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run the full team research pipeline (history + roster) with resumability."""
    _cb = progress_cb or (lambda _msg: None)
    results: Dict[str, Any] = partial_results or {
        "team_history": ("", []),
        "roster": [],
    }

    # Phase 1: team history
    if not _has_data(results.get("team_history")):
        _cb(f"📊 Investigando historial de **{team_name}**...")
        try:
            results["team_history"] = _research_team_history(team_name, api_key)
        except Exception as e:
            results["team_history"] = (f"❌ Error investigando historial: {e}", [])
        _cb("✅ Historial completado.")
    else:
        _cb("✅ Historial ya disponible — reutilizando.")

    # Phase 2: roster player-by-player
    if not _has_data(results.get("roster")):
        _cb(f"👥 Investigando plantilla de **{team_name}** jugador por jugador...")
        try:
            results["roster"] = _research_team_roster_solo(team_name, api_key, progress_cb=_cb)
        except Exception as e:
            results["roster"] = []
            _cb(f"❌ Error con plantilla de {team_name}: {e}")
    elif _roster_has_failures(results.get("roster", [])):
        results["roster"] = _retry_failed_roster_players(
            results["roster"], team_name, "", api_key, progress_cb=_cb,
        )
    else:
        _cb(f"✅ Plantilla de **{team_name}** ya disponible — reutilizando.")

    return results


# ---------------------------------------------------------------------------
# Player Research Pipeline (single player)
# ---------------------------------------------------------------------------

def run_player_research(
    player_name: str,
    team_name: str,
    position: str,
    api_key: str,
    progress_cb: Optional[Callable[[str], None]] = None,
    partial_results: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Research a single player in depth."""
    _cb = progress_cb or (lambda _msg: None)
    results: Dict[str, Any] = partial_results or {"dossier": ("", [])}

    if not _has_data(results.get("dossier")):
        pos_label = _POS_LABELS.get(position, position or "Jugador")
        _cb(f"🔍 Investigando dossier de **{player_name}** ({pos_label}, {team_name})...")

        prompt = _SOLO_PLAYER_DOSSIER_PROMPT.format(
            player_name=player_name,
            player_position=pos_label,
            team_name=team_name,
            current_date=CURRENT_DATE,
        )
        try:
            results["dossier"] = _gemini_request(
                api_key=api_key,
                system_prompt=prompt,
                user_message=(
                    f"Dame el dossier COMPLETO de {player_name} ({pos_label}) de {team_name}. "
                    "Incluye biografía, trayectoria, vida personal, datos curiosos, "
                    "estadísticas de esta temporada y situación contractual. Factor WOW al máximo."
                ),
            )
        except Exception as e:
            results["dossier"] = (f"❌ Error investigando: {e}", [])
        _cb("✅ Dossier completado.")
    else:
        _cb("✅ Dossier ya disponible — reutilizando.")

    return results


# ---------------------------------------------------------------------------
# Selecciones Pipelines
# ---------------------------------------------------------------------------

def _research_national_player_solo(
    player: Dict[str, Any],
    country: str,
    api_key: str,
) -> Dict[str, Any]:
    """Research a single national team player (convocado). Returns player dict."""
    player_name = player.get("name", player.get("full_name", "Unknown"))
    prompt = _NATIONAL_PLAYER_DOSSIER_PROMPT.format(
        player_name=player_name,
        country=country,
        current_date=CURRENT_DATE,
    )
    text, sources = _gemini_request(
        api_key=api_key,
        system_prompt=prompt,
        user_message=(
            f"Dame el dossier internacional COMPLETO de {player_name} (selección de {country}). "
            "Prioriza su carrera con la selección: caps, goles, debates, momentos clave."
        ),
    )
    return {
        "name": player_name,
        "position": player.get("position", ""),
        "number": player.get("number", ""),
        "text": text,
        "sources": sources,
    }


def _research_national_roster(
    country: str,
    api_key: str,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> List[Dict[str, Any]]:
    """Fetch and research the national team's current convocatoria player-by-player."""
    if progress_cb:
        progress_cb(f"📋 Obteniendo convocatoria de **{country}**...")
    players = _fetch_player_list(country, api_key)
    total = len(players)
    if progress_cb:
        progress_cb(f"✅ {total} convocados de **{country}**. Investigando uno por uno...")

    results: List[Dict[str, Any]] = [None] * total  # type: ignore[list-item]
    completed = 0
    batch_size = 4
    for batch_start in range(0, total, batch_size):
        batch = players[batch_start: batch_start + batch_size]
        with ThreadPoolExecutor(max_workers=batch_size) as executor:
            future_to_idx = {
                executor.submit(_research_national_player_solo, player, country, api_key): batch_start + i
                for i, player in enumerate(batch)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    p = players[idx]
                    results[idx] = {
                        "name": p.get("name", "?"),
                        "position": p.get("position", ""),
                        "number": p.get("number", ""),
                        "text": f"❌ Error investigando: {e}",
                        "sources": [],
                    }
                completed += 1
                if progress_cb:
                    pname = results[idx]["name"]
                    progress_cb(f"🔍 [{completed}/{total}] **{pname}** ✓")

    return results


def run_national_team_research(
    country: str,
    confederation: str,
    api_key: str,
    progress_cb: Optional[Callable[[str], None]] = None,
    partial_results: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run national team research: history + convocatoria player-by-player."""
    _cb = progress_cb or (lambda _msg: None)
    results: Dict[str, Any] = partial_results or {
        "team_history": ("", []),
        "roster": [],
    }

    if not _has_data(results.get("team_history")):
        _cb(f"📊 Investigando historial de **{country}**...")
        conf_label = confederation if confederation and "(Cualquier" not in confederation else "Internacional"
        prompt = _NATIONAL_TEAM_HISTORY_PROMPT.format(
            country=country,
            confederation=conf_label,
            current_date=CURRENT_DATE,
        )
        try:
            results["team_history"] = _gemini_request(
                api_key=api_key,
                system_prompt=prompt,
                user_message=(
                    f"Crea la ficha histórica COMPLETA de la selección de {country}. "
                    "Incluye Mundiales, torneos continentales, clasificatorias actuales, "
                    "récords y datos curiosos."
                ),
            )
        except Exception as e:
            results["team_history"] = (f"❌ Error: {e}", [])
        _cb("✅ Historial completado.")
    else:
        _cb("✅ Historial ya disponible — reutilizando.")

    if not _has_data(results.get("roster")):
        _cb(f"👥 Investigando convocatoria de **{country}**...")
        try:
            results["roster"] = _research_national_roster(country, api_key, progress_cb=_cb)
        except Exception as e:
            results["roster"] = []
            _cb(f"❌ Error con convocatoria de {country}: {e}")
    elif _roster_has_failures(results.get("roster", [])):
        results["roster"] = _retry_failed_roster_players(
            results["roster"], country, "", api_key, progress_cb=_cb,
        )
    else:
        _cb(f"✅ Convocatoria de **{country}** ya disponible — reutilizando.")

    return results


def run_national_match_prep(
    home_country: str,
    away_country: str,
    tournament: str,
    match_type: str,
    api_key: str,
    progress_cb: Optional[Callable[[str], None]] = None,
    partial_results: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run national team match preparation: analysis + head-to-head + both rosters."""
    _cb = progress_cb or (lambda _msg: None)
    results: Dict[str, Any] = partial_results or {
        "home_history": ("", []),
        "away_history": ("", []),
        "home_roster": [],
        "away_roster": [],
        "palomo_phrases": ("", []),
    }

    match_prompt = _NATIONAL_MATCH_PREP_PROMPT.format(
        home_country=home_country,
        away_country=away_country,
        tournament=tournament,
        match_type=match_type,
        current_date=CURRENT_DATE,
    )

    # Phase 1: combined match analysis (home + away + head-to-head in one call)
    if not _has_data(results.get("home_history")):
        _cb(f"📊 Analizando partido **{home_country} vs {away_country}**...")
        try:
            results["home_history"] = _gemini_request(
                api_key=api_key,
                system_prompt=match_prompt,
                user_message=(
                    f"Crea la preparación COMPLETA para {home_country} vs {away_country} "
                    f"({match_type} — {tournament}). Incluye historial, análisis táctico de ambas "
                    "selecciones, jugadores clave, bajas, claves tácticas y frases Palomo."
                ),
            )
        except Exception as e:
            results["home_history"] = (f"❌ Error en análisis: {e}", [])
        _cb("✅ Análisis del partido completado.")
    else:
        _cb("✅ Análisis ya disponible — reutilizando.")

    # Phase 2: home roster
    if not _has_data(results.get("home_roster")):
        _cb(f"👥 Investigando convocatoria de **{home_country}**...")
        try:
            results["home_roster"] = _research_national_roster(home_country, api_key, progress_cb=_cb)
        except Exception as e:
            results["home_roster"] = []
            _cb(f"❌ Error con convocatoria de {home_country}: {e}")
    elif _roster_has_failures(results.get("home_roster", [])):
        results["home_roster"] = _retry_failed_roster_players(
            results["home_roster"], home_country, away_country, api_key, progress_cb=_cb,
        )
    else:
        _cb(f"✅ Convocatoria de **{home_country}** ya disponible — reutilizando.")

    # Phase 3: away roster
    if not _has_data(results.get("away_roster")):
        _cb(f"👥 Investigando convocatoria de **{away_country}**...")
        try:
            results["away_roster"] = _research_national_roster(away_country, api_key, progress_cb=_cb)
        except Exception as e:
            results["away_roster"] = []
            _cb(f"❌ Error con convocatoria de {away_country}: {e}")
    elif _roster_has_failures(results.get("away_roster", [])):
        results["away_roster"] = _retry_failed_roster_players(
            results["away_roster"], away_country, home_country, api_key, progress_cb=_cb,
        )
    else:
        _cb(f"✅ Convocatoria de **{away_country}** ya disponible — reutilizando.")

    return results


def run_national_player_research(
    player_name: str,
    country: str,
    api_key: str,
    progress_cb: Optional[Callable[[str], None]] = None,
    partial_results: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Research a single national team player in depth."""
    _cb = progress_cb or (lambda _msg: None)
    results: Dict[str, Any] = partial_results or {"dossier": ("", [])}

    if not _has_data(results.get("dossier")):
        _cb(f"🔍 Investigando dossier de **{player_name}** (selección de {country})...")
        prompt = _NATIONAL_PLAYER_DOSSIER_PROMPT.format(
            player_name=player_name,
            country=country,
            current_date=CURRENT_DATE,
        )
        try:
            results["dossier"] = _gemini_request(
                api_key=api_key,
                system_prompt=prompt,
                user_message=(
                    f"Dame el dossier internacional COMPLETO de {player_name}. "
                    f"Selección de {country}. Prioriza su carrera con la selección, "
                    "caps, goles internacionales, torneos, debut, récords, y factor WOW."
                ),
            )
        except Exception as e:
            results["dossier"] = (f"❌ Error investigando: {e}", [])
        _cb("✅ Dossier completado.")
    else:
        _cb("✅ Dossier ya disponible — reutilizando.")

    return results


# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
_CUSTOM_CSS = """
<style>
/* ---- Base ---- */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(170deg, #f8fafc 0%, #f1f5f9 100%);
}
[data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 1px solid #e2e8f0;
}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] label {
    color: #475569;
}

/* Header */
.hero-title {
    font-size: 2.2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #0ea5e9 0%, #4f46e5 50%, #9333ea 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -0.5px;
    margin-bottom: 0;
    line-height: 1.2;
}
.hero-sub {
    color: #64748b;
    font-size: 0.95rem;
    margin-top: 2px;
}

/* Welcome card */
.welcome-card {
    background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
    border: 1px solid #e2e8f0;
    border-radius: 16px;
    padding: 32px;
    margin: 20px 0;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
}
.welcome-card h3 {
    color: #1e293b;
    font-size: 1.15rem;
    margin-bottom: 12px;
}
.welcome-card p {
    color: #475569;
    font-size: 0.9rem;
    line-height: 1.6;
}

/* Match prep header */
.match-header {
    background: linear-gradient(135deg, #eff6ff 0%, #e0e7ff 50%, #eff6ff 100%);
    border: 1px solid #c7d2fe;
    border-radius: 16px;
    padding: 28px 32px;
    margin: 16px 0 24px;
    text-align: center;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
}
.match-header h2 {
    color: #1e293b;
    font-size: 1.8rem;
    font-weight: 800;
    margin: 0;
}
.match-header .match-meta {
    color: #4338ca;
    font-size: 0.95rem;
    margin-top: 8px;
}

/* Team column header */
.team-col-hdr {
    background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 14px 20px;
    margin-bottom: 12px;
    text-align: center;
    box-shadow: 0 2px 4px -1px rgba(0, 0, 0, 0.05);
}
.team-col-hdr h3 {
    color: #1e293b;
    margin: 0;
    font-size: 1.15rem;
}

/* Palomo section */
.palomo-section {
    background: linear-gradient(135deg, #f5f3ff 0%, #ede9fe 100%);
    border: 1px solid #c4b5fd;
    border-radius: 16px;
    padding: 28px 32px;
    margin: 24px 0;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
}
.palomo-section h3 {
    color: #6d28d9;
    font-size: 1.3rem;
    margin-bottom: 16px;
}

/* Footer */
.app-footer {
    text-align: center;
    color: #94a3b8;
    font-size: 0.72rem;
    padding: 24px 0 12px;
    border-top: 1px solid #e2e8f0;
    margin-top: 32px;
}

/* Chat tweaks */
[data-testid="stChatMessage"] {
    background: transparent !important;
}

/* Hide Streamlit chrome */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header [data-testid="stDecoration"] {display: none;}

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #f1f5f9; }
::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #94a3b8; }
</style>
"""


# ---------------------------------------------------------------------------
# Example queries for PalomoGPT
# ---------------------------------------------------------------------------
_EXAMPLE_QUERIES = [
    f"Cuéntame todo sobre Erling Haaland en Premier League {CURRENT_YEAR}",
    "La historia de la rivalidad Real Madrid vs Barcelona",
    "¿Qué tan especial es Lamine Yamal? Dame lo que nadie cuenta",
    "¿Cuántos goles lleva Haaland esta temporada?",
    "¿Cómo ha cambiado Mbappé desde que llegó al Real Madrid?",
    "Los récords más locos de la Champions League",
]


# ---------------------------------------------------------------------------
# Streamlit App
# ---------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(
        page_title="PalomoFacts · Football Intelligence",
        page_icon="⚽",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(_CUSTOM_CSS, unsafe_allow_html=True)

    api_key = st.secrets.get("GEMINI_API_KEY", "")

    # ---- Session state init ----
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "current_conv_id" not in st.session_state:
        st.session_state.current_conv_id = ""

    # ---- Eagerly capture chat_input (PalomoGPT only) ----
    # chat_input is collected here BEFORE sidebar renders.
    # This lets us create the conversation in Supabase before
    # _list_conversations() runs in the sidebar.
    # Only show in PalomoGPT mode (read prior mode from session state).
    prior_mode = st.session_state.get("sb_app_mode", MODE_PALOMO_GPT)
    incoming_query = None
    if prior_mode == MODE_PALOMO_GPT:
        incoming_query = st.chat_input("Pregunta sobre cualquier tema de fútbol...")
        if incoming_query and not st.session_state.current_conv_id:
            # First message in a new session → create conversation now
            conv_id = _create_conversation(
                mode=MODE_PALOMO_GPT,
                title=_auto_title(incoming_query),
            )
            st.session_state.current_conv_id = conv_id

    # ---- Sidebar ----
    with st.sidebar:
        st.markdown(
            '<p class="hero-title" style="font-size:1.4rem;">⚽ PalomoFacts</p>',
            unsafe_allow_html=True,
        )
        st.markdown("---")

        # Mode selector
        app_mode = st.pills(
            "Modo",
            options=list(MODE_OPTIONS.keys()),
            default=MODE_PALOMO_GPT,
            format_func=lambda k: MODE_OPTIONS[k],
            key="sb_app_mode",
        )
        if not app_mode:
            app_mode = MODE_PALOMO_GPT

        st.markdown("---")

        # ---- Conversation management (PalomoGPT mode) ----
        if app_mode == MODE_PALOMO_GPT:

            # Show "New conversation" only when inside an existing conversation
            has_active_conv = bool(st.session_state.get("current_conv_id", ""))
            if has_active_conv:
                if st.button("➕ Nueva conversación", use_container_width=True):
                    st.session_state.current_conv_id = ""
                    st.session_state.messages = []
                    st.rerun()

            # List past conversations
            convs = _list_conversations()
            if convs:
                st.markdown("##### 💬 Conversaciones")
                for conv in convs:
                    cid = conv["id"]
                    title = conv.get("title", "Sin título")
                    is_active = cid == st.session_state.get("current_conv_id", "")

                    col_title, col_del = st.columns([5, 1])
                    with col_title:
                        label = f"**▶ {title}**" if is_active else title
                        if st.button(label, key=f"conv_{cid}", use_container_width=True):
                            st.session_state.current_conv_id = cid
                            st.session_state.messages = _load_messages(cid)
                            st.rerun()
                    with col_del:
                        if st.button("🗑️", key=f"del_{cid}"):
                            _delete_conversation(cid)
                            if st.session_state.get("current_conv_id") == cid:
                                st.session_state.current_conv_id = ""
                                st.session_state.messages = []
                            st.rerun()
        else:
            # --- Match Prep mode ---
            if app_mode == MODE_MATCH_PREP:
                if st.session_state.get("match_results"):
                    if st.button("➕ Nueva preparación", use_container_width=True):
                        st.session_state.pop("match_results", None)
                        st.session_state.pop("match_config", None)
                        st.session_state.pop("match_pdf_bytes", None)
                        st.rerun()

                preps = _list_match_preps()
                if preps:
                    st.markdown("##### ⚽ Partidos")
                    for prep in preps:
                        pid = prep["id"]
                        title = prep.get("title", "Sin título")

                        col_title, col_del = st.columns([5, 1])
                        with col_title:
                            if st.button(title, key=f"prep_{pid}", use_container_width=True):
                                loaded = _load_match_prep(pid)
                                if loaded:
                                    st.session_state.match_config = loaded["config"]
                                    st.session_state.match_results = loaded["results"]
                                    st.session_state.match_prep_id = pid
                                    st.session_state.pop("match_pdf_bytes", None)
                                st.rerun()
                        with col_del:
                            if st.button("🗑️", key=f"dprep_{pid}"):
                                _delete_match_prep(pid)
                                st.rerun()

            # --- Team Research mode ---
            elif app_mode == MODE_MATCH_RESEARCH:
                if st.session_state.get("team_research_results"):
                    if st.button("➕ Nueva investigación", use_container_width=True):
                        st.session_state.pop("team_research_results", None)
                        st.session_state.pop("team_research_config", None)
                        st.session_state.pop("team_research_id", None)
                        st.rerun()

                researches = _list_team_researches()
                if researches:
                    st.markdown("##### 🔬 Equipos")
                    for res in researches:
                        rid = res["id"]
                        title = res.get("title", "Sin título")
                        league = res.get("tournament", "")
                        label = f"{title} ({league})" if league else title

                        col_title, col_del = st.columns([5, 1])
                        with col_title:
                            if st.button(label, key=f"tres_{rid}", use_container_width=True):
                                loaded = _load_team_research(rid)
                                if loaded:
                                    st.session_state.team_research_config = loaded["config"]
                                    st.session_state.team_research_results = loaded["results"]
                                    st.session_state.team_research_id = rid
                                st.rerun()
                        with col_del:
                            if st.button("🗑️", key=f"dtres_{rid}"):
                                _delete_team_research(rid)
                                st.rerun()

            # --- Player Research mode ---
            elif app_mode == MODE_PLAYER_RESEARCH:
                if st.session_state.get("player_research_results"):
                    if st.button("➕ Nueva investigación", use_container_width=True):
                        st.session_state.pop("player_research_results", None)
                        st.session_state.pop("player_research_config", None)
                        st.session_state.pop("player_research_id", None)
                        st.rerun()

                presearches = _list_player_researches()
                if presearches:
                    st.markdown("##### 🧑 Jugadores")
                    for res in presearches:
                        rid = res["id"]
                        title = res.get("title", "Sin título")
                        team = res.get("team_name", "")
                        label = f"{title} ({team})" if team else title

                        col_title, col_del = st.columns([5, 1])
                        with col_title:
                            if st.button(label, key=f"pres_{rid}", use_container_width=True):
                                loaded = _load_player_research(rid)
                                if loaded:
                                    st.session_state.player_research_config = loaded["config"]
                                    st.session_state.player_research_results = loaded["results"]
                                    st.session_state.player_research_id = rid
                                st.rerun()
                        with col_del:
                            if st.button("🗑️", key=f"dpres_{rid}"):
                                _delete_player_research(rid)
                                st.rerun()

            # --- Selecciones mode ---
            elif app_mode == MODE_SELECCIONES:
                sel_tab = st.session_state.get("sel_active_tab", SEL_TAB_SELECCION)

                if sel_tab == SEL_TAB_SELECCION:
                    if st.session_state.get("nat_team_research_results"):
                        if st.button("➕ Nueva selección", use_container_width=True):
                            st.session_state.pop("nat_team_research_results", None)
                            st.session_state.pop("nat_team_research_config", None)
                            st.session_state.pop("nat_team_research_id", None)
                            st.rerun()

                    nt_researches = _list_national_team_researches()
                    if nt_researches:
                        st.markdown("##### 🌍 Selecciones")
                        for res in nt_researches:
                            rid = res["id"]
                            title = res.get("title", "Sin título")
                            conf = res.get("confederation", "")
                            label = f"{title} (▪ {conf.split('—')[0].strip()})" if conf and "Cualquier" not in conf else title
                            col_t, col_d = st.columns([5, 1])
                            with col_t:
                                if st.button(label, key=f"ntr_{rid}", use_container_width=True):
                                    loaded = _load_national_team_research(rid)
                                    if loaded:
                                        st.session_state.nat_team_research_config = loaded["config"]
                                        st.session_state.nat_team_research_results = loaded["results"]
                                        st.session_state.nat_team_research_id = rid
                                    st.rerun()
                            with col_d:
                                if st.button("🗑️", key=f"dntr_{rid}"):
                                    _delete_national_team_research(rid)
                                    st.rerun()

                elif sel_tab == SEL_TAB_PARTIDO:
                    if st.session_state.get("nat_match_results"):
                        if st.button("➕ Nuevo partido", use_container_width=True):
                            st.session_state.pop("nat_match_results", None)
                            st.session_state.pop("nat_match_config", None)
                            st.session_state.pop("nat_match_prep_id", None)
                            st.rerun()

                    nm_preps = _list_national_match_preps()
                    if nm_preps:
                        st.markdown("##### ⚽ Partidos")
                        for res in nm_preps:
                            pid = res["id"]
                            title = res.get("title", "Sin título")
                            tour = res.get("tournament", "")
                            label = f"{title} | {tour}" if tour else title
                            col_t, col_d = st.columns([5, 1])
                            with col_t:
                                if st.button(label, key=f"nmatch_{pid}", use_container_width=True):
                                    loaded = _load_national_match_prep(pid)
                                    if loaded:
                                        st.session_state.nat_match_config = loaded["config"]
                                        st.session_state.nat_match_results = loaded["results"]
                                        st.session_state.nat_match_prep_id = pid
                                    st.rerun()
                            with col_d:
                                if st.button("🗑️", key=f"dnmatch_{pid}"):
                                    _delete_national_match_prep(pid)
                                    st.rerun()

                elif sel_tab == SEL_TAB_CONVOCADO:
                    if st.session_state.get("nat_player_results"):
                        if st.button("➕ Nuevo convocado", use_container_width=True):
                            st.session_state.pop("nat_player_results", None)
                            st.session_state.pop("nat_player_config", None)
                            st.session_state.pop("nat_player_research_id", None)
                            st.rerun()

                    np_researches = _list_national_player_researches()
                    if np_researches:
                        st.markdown("##### 🧑 Convocados")
                        for res in np_researches:
                            rid = res["id"]
                            title = res.get("title", "Sin título")
                            country = res.get("country", "")
                            label = f"{title} ({country})" if country else title
                            col_t, col_d = st.columns([5, 1])
                            with col_t:
                                if st.button(label, key=f"npr_{rid}", use_container_width=True):
                                    loaded = _load_national_player_research(rid)
                                    if loaded:
                                        st.session_state.nat_player_config = loaded["config"]
                                        st.session_state.nat_player_results = loaded["results"]
                                        st.session_state.nat_player_research_id = rid
                                    st.rerun()
                            with col_d:
                                if st.button("🗑️", key=f"dnpr_{rid}"):
                                    _delete_national_player_research(rid)
                                    st.rerun()

        st.markdown("---")
        st.caption("🌐 Datos en tiempo real · Cualquier liga · Verificado con fuentes")

        st.markdown(
            '<div class="app-footer">PalomoFacts v4 · AI: Google Gemini + Search Grounding</div>',
            unsafe_allow_html=True,
        )

    # ---- Route to active mode ----
    if app_mode == MODE_PALOMO_GPT:
        _render_palomo_gpt(api_key, incoming_query)
    elif app_mode == MODE_MATCH_PREP:
        _render_match_prep(api_key)
    elif app_mode == MODE_MATCH_RESEARCH:
        _render_match_research(api_key)
    elif app_mode == MODE_PLAYER_RESEARCH:
        _render_player_research(api_key)
    elif app_mode == MODE_SELECCIONES:
        _render_selecciones(api_key)
    else:
        _render_match_prep(api_key)


# ---------------------------------------------------------------------------
# PalomoGPT mode
# ---------------------------------------------------------------------------
def _render_palomo_gpt(api_key: str, incoming_query: Optional[str] = None) -> None:
    col_h1, _ = st.columns([3, 1])
    with col_h1:
        st.markdown(
            '<p class="hero-title">🎙️ PalomoGPT</p>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<p class="hero-sub">'
            'Tu Fernando Palomo personal — pregunta lo que sea sobre fútbol'
            '</p>',
            unsafe_allow_html=True,
        )

    msgs = st.session_state.messages

    # Pick up any pending query from example buttons (one-shot flag)
    if "_pending_query" in st.session_state:
        incoming_query = st.session_state.pop("_pending_query")

    # Welcome screen — only when conversation is empty AND no new query
    if not msgs and not incoming_query:
        st.markdown(
            '<div class="welcome-card">'
            '<h3>🏟️ ¡Bienvenidos, señores! Aquí Fernando Palomo</h3>'
            '<p>Pregúntame lo que sea sobre fútbol — jugadores, equipos, historia, '
            'estadísticas, tácticas, cualquier liga, cualquier época.<br><br>'
            'Detecto automáticamente si necesitas un dato rápido o un análisis profundo. '
            'Solo pregunta.</p>'
            '<p style="margin-top:12px;color:#64748b;">'
            '¡Abran la puerta que llegó el cartero!</p>'
            '</div>',
            unsafe_allow_html=True,
        )

        cols = st.columns(3)
        for idx, example in enumerate(_EXAMPLE_QUERIES):
            with cols[idx % 3]:
                if st.button(example, key=f"ex_{idx}", use_container_width=True):
                    # Store as one-shot pending query and create conversation
                    st.session_state._pending_query = example
                    if not st.session_state.current_conv_id:
                        cid = _create_conversation(
                            mode=MODE_PALOMO_GPT,
                            title=_auto_title(example),
                        )
                        st.session_state.current_conv_id = cid
                    st.rerun()
        return  # Nothing else to render

    # --- Replay existing messages ---
    for msg in msgs:
        with st.chat_message(msg["role"]):
            st.markdown(msg.get("content", ""))

    # --- Process new query (if any) ---
    if not incoming_query:
        return  # No new query — just the replayed history

    # Append and display user message
    msgs.append({"role": "user", "content": incoming_query})
    with st.chat_message("user"):
        st.markdown(incoming_query)

    # Save user message to DB
    conv_id = st.session_state.get("current_conv_id", "")
    _save_message(conv_id, "user", incoming_query)

    # Check API key
    if not api_key:
        with st.chat_message("assistant"):
            err = (
                "⚠️ No se encontró la API key de Gemini. "
                "Configura `GEMINI_API_KEY` en los secrets de Streamlit."
            )
            st.warning(err)
            msgs.append({"role": "assistant", "content": err})
        return

    # Get response
    followups = []
    with st.chat_message("assistant"):
        try:
            with st.status(
                "🔍 Buscando información verificada...", expanded=False
            ) as status:
                text, citations, reasoning_chain, followups = get_palomo_response(
                    query=incoming_query,
                    session_messages=msgs,
                    api_key=api_key,
                )
                status.update(label="✅ Información encontrada y verificada", state="complete")

            full_response = text
            source_text = _format_sources(citations)
            if source_text:
                full_response += source_text

            st.markdown(full_response)
            msgs.append({"role": "assistant", "content": full_response})
            _save_message(conv_id, "assistant", full_response)

            # Show reasoning chain
            if reasoning_chain:
                chain_text = "🧠 **Cadena de razonamiento:**\n\n" + "\n\n".join(reasoning_chain)
                with st.chat_message("assistant"):
                    st.markdown(chain_text)
                msgs.append({"role": "assistant", "content": chain_text})
                _save_message(conv_id, "assistant", chain_text)

        except Exception as e:
            err_msg = f"❌ Error: {e}"
            st.error(err_msg)
            msgs.append({"role": "assistant", "content": err_msg})
            print(f"[PalomoGPT] Error:\n{traceback.format_exc()}")
            followups = []

    # Render follow-up Q&A as user/assistant chat bubbles
    for fu in followups:
        # User bubble for the auto-generated question
        with st.chat_message("user"):
            st.markdown(f"🔍 {fu['question']}")
        q_text = f"🔍 {fu['question']}"
        msgs.append({"role": "user", "content": q_text})
        _save_message(conv_id, "user", q_text)

        # Assistant bubble for the answer
        with st.chat_message("assistant"):
            st.markdown(fu["answer"])
        msgs.append({"role": "assistant", "content": fu["answer"]})
        _save_message(conv_id, "assistant", fu["answer"])


# ---------------------------------------------------------------------------
# Preparación de Partidos mode
# ---------------------------------------------------------------------------
def _render_match_prep(api_key: str) -> None:
    col_h1, _ = st.columns([3, 1])
    with col_h1:
        st.markdown(
            '<p class="hero-title">⚽ Preparación de Partidos</p>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<p class="hero-sub">'
            'Informe completo para transmisión — historial, plantillas y frases de Palomo'
            '</p>',
            unsafe_allow_html=True,
        )

    # If we already have results, show them — don't show the form
    if st.session_state.get("match_results") and st.session_state.get("match_config"):
        existing = st.session_state.match_results
        config = st.session_state.match_config

        # Check if all phases are complete
        all_keys = ["home_history", "away_history", "home_roster", "away_roster", "palomo_phrases"]
        missing = [k for k in all_keys if not _has_data(existing.get(k))]
        # Also check for individual player failures within rosters
        has_player_failures = (
            _roster_has_failures(existing.get("home_roster", []))
            or _roster_has_failures(existing.get("away_roster", []))
        )
        is_incomplete = bool(missing) or has_player_failures

        if is_incomplete:
            n_failed_players = sum(
                1 for r in (existing.get("home_roster", []) + existing.get("away_roster", []))
                if isinstance(r, dict) and r.get("text", "").startswith("❌")
            )
            parts = []
            if missing:
                parts.append(f"faltan {len(missing)} secciones")
            if n_failed_players:
                parts.append(f"{n_failed_players} jugadores con error")
            st.warning(
                f"⚠️ Análisis incompleto — {', '.join(parts)}. "
                "Puedes continuar sin perder lo ya investigado."
            )
            if st.button("🔄 Continuar análisis", type="primary", use_container_width=True):
                _run_match_pipeline(config, api_key, partial_results=existing)
                return

        _display_match_results(config, existing)
        return

    # --- Match configuration form (only when no results loaded) ---
    st.markdown("### 📋 Configuración del Partido")

    col1, col2 = st.columns(2)
    with col1:
        home_team = st.text_input(
            "🏠 Equipo Local",
            placeholder="Ej: F.C. Barcelona",
            key="mp_home_team",
        )
    with col2:
        away_team = st.text_input(
            "✈️ Equipo Visitante",
            placeholder="Ej: Real Madrid",
            key="mp_away_team",
        )

    col3, col4, col5 = st.columns(3)
    with col3:
        tournament = st.selectbox(
            "🏆 Torneo",
            options=TOURNAMENT_OPTIONS,
            key="mp_tournament",
        )
    with col4:
        match_type = st.selectbox(
            "📌 Tipo de Partido",
            options=MATCH_TYPE_OPTIONS,
            key="mp_match_type",
        )
    with col5:
        stadium = st.text_input(
            "🏟️ Estadio",
            placeholder="Ej: Estadio Santiago Bernabéu",
            key="mp_stadium",
        )

    can_submit = bool(home_team and away_team)

    if st.button(
        "🚀 Preparar Partido",
        use_container_width=True,
        disabled=not can_submit,
        type="primary",
    ):
        if not api_key:
            st.error(
                "⚠️ No se encontró la API key de Gemini. "
                "Configura `GEMINI_API_KEY` en los secrets de Streamlit."
            )
            return

        # --- Validate match via LLM ---
        with st.status("🔍 Validando equipos y partido...", expanded=True) as val_status:
            try:
                val_prompt = _MATCH_VALIDATION_PROMPT.format(
                    home_team=home_team,
                    away_team=away_team,
                    tournament=tournament,
                    current_date=CURRENT_DATE,
                )
                val_text, _ = _gemini_request(
                    api_key=api_key,
                    system_prompt=val_prompt,
                    user_message="Valida este partido y devuelve el JSON.",
                )

                # Parse JSON from response
                json_match = re.search(r'\{[\s\S]*\}', val_text)
                if json_match:
                    val_data = json.loads(json_match.group())
                    if not val_data.get("valid", True):
                        reason = val_data.get("reason", "Partido no reconocido.")
                        val_status.update(label=f"❌ {reason}", state="error")
                        st.error(f"⚠️ {reason}")
                        return
                    # Use resolved names
                    home_team = val_data.get("home_team", home_team)
                    away_team = val_data.get("away_team", away_team)
                    val_status.update(
                        label=f"✅ Partido validado: {home_team} vs {away_team}",
                        state="complete", expanded=False,
                    )
                else:
                    val_status.update(label="⚠️ No se pudo validar, continuando...", state="complete", expanded=False)
            except Exception as e:
                print(f"[MatchPrep] Validation failed, continuing: {e}")
                val_status.update(label="⚠️ Validación omitida, continuando...", state="complete", expanded=False)

        # Store config with resolved names
        st.session_state.match_config = {
            "home_team": home_team,
            "away_team": away_team,
            "tournament": tournament,
            "match_type": match_type,
            "stadium": stadium,
        }

        _run_match_pipeline(st.session_state.match_config, api_key)


def _run_match_pipeline(
    config: dict,
    api_key: str,
    partial_results: Optional[Dict[str, Any]] = None,
) -> None:
    """Run (or resume) the match preparation research pipeline."""
    home_team = config["home_team"]
    away_team = config["away_team"]

    label = "🔄 Continuando análisis..." if partial_results else "🔍 Preparando informe del partido..."
    with st.status(label, expanded=True) as status:

        def _progress(msg: str) -> None:
            status.write(msg)

        try:
            results = run_match_preparation(
                home_team=home_team,
                away_team=away_team,
                tournament=config["tournament"],
                match_type=config["match_type"],
                stadium=config["stadium"],
                api_key=api_key,
                progress_cb=_progress,
                partial_results=partial_results,
            )
            st.session_state.match_results = results
            st.session_state.pop("match_pdf_bytes", None)  # clear stale PDF

            status.update(
                label="✅ ¡Informe completo!",
                state="complete",
                expanded=False,
            )

            # Auto-save to Supabase
            try:
                _save_match_prep(config, results)
            except Exception as e:
                print(f"[MatchPrep] Error saving to Supabase: {e}")
        except Exception as e:
            # Save whatever partial results we have so user can resume
            if partial_results:
                st.session_state.match_results = partial_results
                # Also persist partial results to Supabase
                try:
                    _save_match_prep(config, partial_results)
                except Exception:
                    pass
            status.update(label=f"❌ Error: {e} — puedes continuar el análisis", state="error")
            print(f"[MatchPrep] Error:\n{traceback.format_exc()}")

    # Always rerun to refresh display with latest data
    st.rerun()


def _render_roster_players(roster: list) -> None:
    """Render a list of player dossiers as individual expanders grouped by position."""
    if not roster:
        st.warning("No se encontraron jugadores.")
        return

    # Group by position preserving order
    pos_order = ["GK", "DEF", "MID", "FWD"]
    pos_emoji = {"GK": "🧤", "DEF": "🛡️", "MID": "🎯", "FWD": "⚡"}
    pos_name = {"GK": "PORTEROS", "DEF": "DEFENSAS", "MID": "CENTROCAMPISTAS", "FWD": "DELANTEROS"}

    grouped: Dict[str, list] = {p: [] for p in pos_order}
    for player in roster:
        pos = player.get("position", "FWD")
        if pos not in grouped:
            pos = "FWD"
        grouped[pos].append(player)

    for pos in pos_order:
        players = grouped[pos]
        if not players:
            continue
        st.markdown(f"**{pos_emoji.get(pos, '')} {pos_name.get(pos, pos)}**")
        for p in players:
            number = p.get("number", "")
            label = f"#{number} {p['name']}" if number else p["name"]
            with st.expander(label, expanded=False):
                st.markdown(p.get("text", ""))
                sources = p.get("sources", [])
                if sources:
                    st.markdown(
                        _format_sources(sources).replace(_CITATION_SEPARATOR, "")
                    )


def _display_match_results(config: dict, results: dict) -> None:
    """Render the full match preparation report."""
    home = config["home_team"]
    away = config["away_team"]
    tournament = config["tournament"]
    match_type = config["match_type"]
    stadium = config["stadium"]

    st.markdown("---")

    # ---- Match header ----
    st.markdown(
        f'<div class="match-header">'
        f'<h2>{home}  🆚  {away}</h2>'
        f'<div class="match-meta">'
        f'{match_type} · {tournament} · 🏟️ {stadium}'
        f'</div></div>',
        unsafe_allow_html=True,
    )

    # ---- On-demand PDF generation ----
    safe_home = re.sub(r'[^\w\s-]', '', home).strip().replace(' ', '_')
    safe_away = re.sub(r'[^\w\s-]', '', away).strip().replace(' ', '_')
    pdf_filename = f"Preparacion_{safe_home}_vs_{safe_away}.pdf"

    if st.session_state.get("match_pdf_bytes"):
        # PDF already generated — show download button
        st.download_button(
            label="📥 Descargar PDF del Informe",
            data=st.session_state.match_pdf_bytes,
            file_name=pdf_filename,
            mime="application/pdf",
            use_container_width=True,
        )
    else:
        # Show a button to trigger PDF generation on demand
        api_key = st.secrets.get("GEMINI_API_KEY", "")
        if st.button("📄 Generar y Descargar PDF", use_container_width=True, type="primary"):
            pdf_ok = False
            with st.status("📄 Sintetizando y generando PDF...", expanded=True) as pdf_status:
                try:
                    pdf_bytes = generate_match_pdf(
                        config, results, api_key=api_key,
                        progress_cb=lambda msg: pdf_status.write(msg),
                    )
                    st.session_state.match_pdf_bytes = pdf_bytes
                    pdf_status.update(label="✅ PDF generado", state="complete", expanded=False)
                    pdf_ok = True
                except Exception as e:
                    print(f"[MatchPrep] Error generating PDF:\n{traceback.format_exc()}")
                    pdf_status.update(label=f"❌ Error generando PDF: {e}", state="error")
            # Rerun OUTSIDE the status context so the download button appears
            if pdf_ok:
                st.rerun()

    # ---- Team Histories (side-by-side) ----
    st.markdown("### 📊 Historial de Temporadas")

    col_h, col_a = st.columns(2)

    with col_h:
        st.markdown(
            f'<div class="team-col-hdr"><h3>🏠 {home}</h3></div>',
            unsafe_allow_html=True,
        )
        h_text, h_srcs = results["home_history"]
        st.markdown(h_text)
        if h_srcs:
            with st.expander("📚 Fuentes"):
                st.markdown(
                    _format_sources(h_srcs).replace(_CITATION_SEPARATOR, "")
                )

    with col_a:
        st.markdown(
            f'<div class="team-col-hdr"><h3>✈️ {away}</h3></div>',
            unsafe_allow_html=True,
        )
        a_text, a_srcs = results["away_history"]
        st.markdown(a_text)
        if a_srcs:
            with st.expander("📚 Fuentes"):
                st.markdown(
                    _format_sources(a_srcs).replace(_CITATION_SEPARATOR, "")
                )

    # ---- Team Rosters (side-by-side, per-player expanders) ----
    st.markdown("---")
    st.markdown("### 👥 Plantillas — Dossier por Jugador")

    col_hr, col_ar = st.columns(2)

    with col_hr:
        st.markdown(
            f'<div class="team-col-hdr"><h3>🏠 {home}</h3></div>',
            unsafe_allow_html=True,
        )
        _render_roster_players(results.get("home_roster", []))

    with col_ar:
        st.markdown(
            f'<div class="team-col-hdr"><h3>✈️ {away}</h3></div>',
            unsafe_allow_html=True,
        )
        _render_roster_players(results.get("away_roster", []))

    # ---- Palomo Phrases (full-width) ----
    st.markdown("---")
    st.markdown(
        f'<div class="palomo-section">'
        f'<h3>🎙️ Frases de Fernando Palomo — {home} vs {away}</h3>'
        f'</div>',
        unsafe_allow_html=True,
    )

    p_text, p_srcs = results["palomo_phrases"]
    st.markdown(p_text)
    if p_srcs:
        with st.expander("📚 Fuentes"):
            st.markdown(
                _format_sources(p_srcs).replace(_CITATION_SEPARATOR, "")
            )


# ---------------------------------------------------------------------------
# Investigar Equipo mode
# ---------------------------------------------------------------------------
def _render_match_research(api_key: str) -> None:
    col_h1, _ = st.columns([3, 1])
    with col_h1:
        st.markdown(
            '<p class="hero-title">🔬 Investigar Equipo</p>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<p class="hero-sub">'
            'Historial de temporadas y dossier completo de toda la plantilla'
            '</p>',
            unsafe_allow_html=True,
        )

    # If results already loaded, show them
    if st.session_state.get("team_research_results") and st.session_state.get("team_research_config"):
        existing = st.session_state.team_research_results
        config = st.session_state.team_research_config

        all_keys = ["team_history", "roster"]
        missing = [k for k in all_keys if not _has_data(existing.get(k))]
        has_player_failures = _roster_has_failures(existing.get("roster", []))
        is_incomplete = bool(missing) or has_player_failures

        if is_incomplete:
            n_failed = sum(
                1 for r in existing.get("roster", [])
                if isinstance(r, dict) and r.get("text", "").startswith("❌")
            )
            parts = []
            if missing:
                parts.append(f"faltan {len(missing)} secciones")
            if n_failed:
                parts.append(f"{n_failed} jugadores con error")
            st.warning(
                f"⚠️ Análisis incompleto — {', '.join(parts)}. "
                "Puedes continuar sin perder lo ya investigado."
            )
            if st.button("🔄 Continuar análisis", type="primary", use_container_width=True):
                _run_team_research_pipeline(config, api_key, partial_results=existing)
                return

        _display_team_research_results(config, existing)
        return

    # --- Form ---
    st.markdown("### 📋 Configuración")

    col1, col2 = st.columns(2)
    with col1:
        team_name = st.text_input(
            "🏆 Nombre del Equipo",
            placeholder="Ej: FC Barcelona",
            key="mr_team_name",
        )
    with col2:
        tournament = st.selectbox(
            "🌍 Liga / Competición",
            options=TOURNAMENT_OPTIONS,
            key="mr_tournament",
        )

    can_submit = bool(team_name)

    if st.button(
        "🔬 Investigar Equipo",
        use_container_width=True,
        disabled=not can_submit,
        type="primary",
    ):
        if not api_key:
            st.error("⚠️ No se encontró la API key de Gemini.")
            return

        st.session_state.team_research_config = {
            "team_name": team_name,
            "tournament": tournament,
        }
        _run_team_research_pipeline(st.session_state.team_research_config, api_key)


def _run_team_research_pipeline(
    config: dict,
    api_key: str,
    partial_results: Optional[Dict[str, Any]] = None,
) -> None:
    """Run (or resume) the team research pipeline."""
    team_name = config["team_name"]
    label = "🔄 Continuando análisis..." if partial_results else "🔍 Investigando equipo..."
    with st.status(label, expanded=True) as status:
        def _progress(msg: str) -> None:
            status.write(msg)

        try:
            results = run_team_research(
                team_name=team_name,
                tournament=config.get("tournament", ""),
                api_key=api_key,
                progress_cb=_progress,
                partial_results=partial_results,
            )
            st.session_state.team_research_results = results
            status.update(label="✅ ¡Investigación completa!", state="complete", expanded=False)

            try:
                _save_team_research(config, results)
            except Exception as e:
                print(f"[TeamResearch] Error saving to Supabase: {e}")
        except Exception as e:
            if partial_results:
                st.session_state.team_research_results = partial_results
                try:
                    _save_team_research(config, partial_results)
                except Exception:
                    pass
            status.update(label=f"❌ Error: {e} — puedes continuar el análisis", state="error")
            print(f"[TeamResearch] Error:\n{traceback.format_exc()}")

    st.rerun()


def _display_team_research_results(config: dict, results: dict) -> None:
    """Render the full team research report."""
    team = config["team_name"]
    tournament = config.get("tournament", "")

    st.markdown("---")

    # ---- Team header ----
    meta = tournament if tournament else "Investigación de Equipo"
    st.markdown(
        f'<div class="match-header">'
        f'<h2>🔬 {team}</h2>'
        f'<div class="match-meta">{meta}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ---- Team History (full width) ----
    st.markdown("### 📊 Historial de Temporadas")
    h_text, h_srcs = results.get("team_history", ("", []))
    st.markdown(h_text)
    if h_srcs:
        with st.expander("📚 Fuentes"):
            st.markdown(_format_sources(h_srcs).replace(_CITATION_SEPARATOR, ""))

    # ---- Roster (full width, by position) ----
    st.markdown("---")
    st.markdown("### 👥 Plantilla — Dossier por Jugador")
    _render_roster_players(results.get("roster", []))


# ---------------------------------------------------------------------------
# Investigar Jugador mode
# ---------------------------------------------------------------------------
_PLAYER_POSITION_OPTIONS = ["GK", "DEF", "MID", "FWD"]


def _render_player_research(api_key: str) -> None:
    col_h1, _ = st.columns([3, 1])
    with col_h1:
        st.markdown(
            '<p class="hero-title">🧑 Investigar Jugador</p>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<p class="hero-sub">'
            'Dossier completo de un jugador — trayectoria, vida personal, estadísticas y más'
            '</p>',
            unsafe_allow_html=True,
        )

    # If results already loaded, show them
    if st.session_state.get("player_research_results") and st.session_state.get("player_research_config"):
        existing = st.session_state.player_research_results
        config = st.session_state.player_research_config

        is_incomplete = not _has_data(existing.get("dossier"))
        if is_incomplete:
            st.warning("⚠️ Dossier incompleto. Puedes continuar la investigación.")
            if st.button("🔄 Continuar investigación", type="primary", use_container_width=True):
                _run_player_research_pipeline(config, api_key, partial_results=existing)
                return

        _display_player_research_results(config, existing)
        return

    # --- Form ---
    st.markdown("### 📋 Configuración")

    col1, col2 = st.columns(2)
    with col1:
        player_name = st.text_input(
            "🧑 Nombre del Jugador",
            placeholder="Ej: Lamine Yamal",
            key="pr_player_name",
        )
    with col2:
        team_name = st.text_input(
            "🏆 Equipo Actual",
            placeholder="Ej: FC Barcelona",
            key="pr_team_name",
        )

    can_submit = bool(player_name and team_name)

    if st.button(
        "🧑 Investigar Jugador",
        use_container_width=True,
        disabled=not can_submit,
        type="primary",
    ):
        if not api_key:
            st.error("⚠️ No se encontró la API key de Gemini.")
            return

        st.session_state.player_research_config = {
            "player_name": player_name,
            "team_name": team_name,
        }
        _run_player_research_pipeline(st.session_state.player_research_config, api_key)


def _run_player_research_pipeline(
    config: dict,
    api_key: str,
    partial_results: Optional[Dict[str, Any]] = None,
) -> None:
    """Run (or resume) the player research pipeline."""
    player_name = config["player_name"]
    team_name = config.get("team_name", "")
    position = config.get("position", "")

    label = "🔄 Continuando investigación..." if partial_results else "🔍 Investigando jugador..."
    with st.status(label, expanded=True) as status:
        def _progress(msg: str) -> None:
            status.write(msg)

        try:
            results = run_player_research(
                player_name=player_name,
                team_name=team_name,
                position=position,
                api_key=api_key,
                progress_cb=_progress,
                partial_results=partial_results,
            )
            st.session_state.player_research_results = results
            status.update(label="✅ ¡Dossier completo!", state="complete", expanded=False)

            try:
                _save_player_research(config, results)
            except Exception as e:
                print(f"[PlayerResearch] Error saving to Supabase: {e}")
        except Exception as e:
            if partial_results:
                st.session_state.player_research_results = partial_results
                try:
                    _save_player_research(config, partial_results)
                except Exception:
                    pass
            status.update(label=f"❌ Error: {e}", state="error")
            print(f"[PlayerResearch] Error:\n{traceback.format_exc()}")

    st.rerun()


def _display_player_research_results(config: dict, results: dict) -> None:
    """Render the full player research dossier."""
    player = config["player_name"]
    team = config.get("team_name", "")
    position = config.get("position", "")
    pos_label = _POS_LABELS.get(position, position)

    st.markdown("---")

    # ---- Player header ----
    meta_parts = [pos_label, team]
    meta = " · ".join(p for p in meta_parts if p)
    st.markdown(
        f'<div class="match-header">'
        f'<h2>🧑 {player}</h2>'
        f'<div class="match-meta">{meta}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ---- Dossier (full width) ----
    st.markdown("### 📋 Dossier Completo")
    dossier_text, dossier_srcs = results.get("dossier", ("", []))
    st.markdown(dossier_text)
    if dossier_srcs:
        with st.expander("📚 Fuentes"):
            st.markdown(_format_sources(dossier_srcs).replace(_CITATION_SEPARATOR, ""))


# ---------------------------------------------------------------------------
# Selecciones mega mode
# ---------------------------------------------------------------------------

def _render_selecciones(api_key: str) -> None:
    """Top-level Selecciones render — three tabs for national team research."""
    st.markdown(
        '<p class="hero-title">🌍 Selecciones Nacionales</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="hero-sub">'
        'Investigación profunda de selecciones, partidos y convocados — '
        'el mismo nivel de análisis adaptado al fútbol internacional'
        '</p>',
        unsafe_allow_html=True,
    )

    tab_labels = [
        "🔬 Investigar Selección",
        "⚽ Partido de Selecciones",
        "🧑 Investigar Convocado",
    ]
    tabs = st.tabs(tab_labels)

    with tabs[SEL_TAB_SELECCION]:
        st.session_state.sel_active_tab = SEL_TAB_SELECCION
        _render_sel_team_tab(api_key)

    with tabs[SEL_TAB_PARTIDO]:
        st.session_state.sel_active_tab = SEL_TAB_PARTIDO
        _render_sel_match_tab(api_key)

    with tabs[SEL_TAB_CONVOCADO]:
        st.session_state.sel_active_tab = SEL_TAB_CONVOCADO
        _render_sel_player_tab(api_key)


# ---- Tab 0: Investigar Selección ----------------------------------------

def _render_sel_team_tab(api_key: str) -> None:
    if st.session_state.get("nat_team_research_results") and st.session_state.get("nat_team_research_config"):
        existing = st.session_state.nat_team_research_results
        config = st.session_state.nat_team_research_config

        missing = not _has_data(existing.get("team_history"))
        has_failures = _roster_has_failures(existing.get("roster", []))
        if missing or has_failures:
            n_failed = sum(
                1 for r in existing.get("roster", [])
                if isinstance(r, dict) and str(r.get("text", "")).startswith("❌")
            )
            parts = []
            if missing:
                parts.append("falta historial")
            if n_failed:
                parts.append(f"{n_failed} jugadores con error")
            st.warning(f"⚠️ Análisis incompleto — {', '.join(parts)}. Puedes continuar desde aquí.")
            if st.button("🔄 Continuar análisis", type="primary", use_container_width=True, key="sel_continue_team"):
                _run_sel_team_pipeline(config, api_key, partial_results=existing)
                return

        _display_sel_team_results(config, existing)
        return

    # --- Form ---
    st.markdown("### 📋 Selección a investigar")
    col1, col2 = st.columns(2)
    with col1:
        country = st.text_input(
            "🌍 País / Selección",
            placeholder="Ej: Argentina, España, Brasil...",
            key="sel_country",
        )
    with col2:
        confederation = st.selectbox(
            "🏛️ Confederación (opcional)",
            options=CONFEDERATION_OPTIONS,
            key="sel_confederation",
        )

    can_submit = bool(country)
    if st.button("🔬 Investigar Selección", use_container_width=True, disabled=not can_submit,
                 type="primary", key="sel_team_submit"):
        if not api_key:
            st.error("⚠️ No se encontró la API key de Gemini.")
            return
        st.session_state.nat_team_research_config = {
            "country": country,
            "confederation": confederation,
        }
        st.session_state.sel_active_tab = SEL_TAB_SELECCION
        _run_sel_team_pipeline(st.session_state.nat_team_research_config, api_key)


def _run_sel_team_pipeline(config: dict, api_key: str, partial_results: Optional[Dict[str, Any]] = None) -> None:
    label = "🔄 Continuando..." if partial_results else "🔍 Investigando selección..."
    with st.status(label, expanded=True) as status:
        def _progress(msg: str) -> None:
            status.write(msg)
        try:
            results = run_national_team_research(
                country=config["country"],
                confederation=config.get("confederation", ""),
                api_key=api_key,
                progress_cb=_progress,
                partial_results=partial_results,
            )
            st.session_state.nat_team_research_results = results
            status.update(label="✅ ¡Investigación completa!", state="complete", expanded=False)
            try:
                _save_national_team_research(config, results)
            except Exception as e:
                print(f"[Sel] Error saving national team: {e}")
        except Exception as e:
            if partial_results:
                st.session_state.nat_team_research_results = partial_results
                try:
                    _save_national_team_research(config, partial_results)
                except Exception:
                    pass
            status.update(label=f"❌ Error: {e}", state="error")
            print(f"[Sel] Error nat team:\n{traceback.format_exc()}")
    st.rerun()


def _display_sel_team_results(config: dict, results: dict) -> None:
    country = config["country"]
    confederation = config.get("confederation", "")
    meta = confederation if confederation and "(Cualquier" not in confederation else "Selección Nacional"

    st.markdown("---")
    st.markdown(
        f'<div class="match-header"><h2>🌍 {country}</h2>'
        f'<div class="match-meta">{meta}</div></div>',
        unsafe_allow_html=True,
    )
    st.markdown("### 📊 Historial de la Selección")
    h_text, h_srcs = results.get("team_history", ("", []))
    st.markdown(h_text)
    if h_srcs:
        with st.expander("📚 Fuentes"):
            st.markdown(_format_sources(h_srcs).replace(_CITATION_SEPARATOR, ""))

    roster = results.get("roster", [])
    if roster:
        st.markdown("---")
        st.markdown("### 🎽 Convocatoria — Dossier por Jugador")
        _render_roster_players(roster)


# ---- Tab 1: Partido de Selecciones --------------------------------------

def _render_sel_match_tab(api_key: str) -> None:
    if st.session_state.get("nat_match_results") and st.session_state.get("nat_match_config"):
        existing = st.session_state.nat_match_results
        config = st.session_state.nat_match_config

        home_ok = _has_data(existing.get("home_history"))
        home_roster_ok = _has_data(existing.get("home_roster")) and not _roster_has_failures(existing.get("home_roster", []))
        away_roster_ok = _has_data(existing.get("away_roster")) and not _roster_has_failures(existing.get("away_roster", []))
        is_incomplete = not (home_ok and home_roster_ok and away_roster_ok)

        if is_incomplete:
            st.warning("⚠️ Preparación incompleta. Puedes continuar sin perder el trabajo previo.")
            if st.button("🔄 Continuar preparación", type="primary", use_container_width=True, key="sel_continue_match"):
                _run_sel_match_pipeline(config, api_key, partial_results=existing)
                return

        _display_sel_match_results(config, existing)
        return

    # --- Form ---
    st.markdown("### 📋 Partido a preparar")
    col1, col2 = st.columns(2)
    with col1:
        home_country = st.text_input(
            "🏠 País Local",
            placeholder="Ej: Argentina",
            key="sel_home_country",
        )
    with col2:
        away_country = st.text_input(
            "✈️ País Visitante",
            placeholder="Ej: Brasil",
            key="sel_away_country",
        )

    col3, col4 = st.columns(2)
    with col3:
        tournament = st.selectbox(
            "🏆 Torneo / Contexto",
            options=NATIONAL_TOURNAMENT_OPTIONS,
            key="sel_match_tournament",
        )
    with col4:
        match_type = st.selectbox(
            "🎯 Tipo de Partido",
            options=MATCH_TYPE_OPTIONS,
            key="sel_match_type",
        )

    can_submit = bool(home_country and away_country)
    if st.button("⚽ Preparar Partido", use_container_width=True, disabled=not can_submit,
                 type="primary", key="sel_match_submit"):
        if not api_key:
            st.error("⚠️ No se encontró la API key de Gemini.")
            return
        st.session_state.nat_match_config = {
            "home_country": home_country,
            "away_country": away_country,
            "tournament": tournament,
            "match_type": match_type,
        }
        st.session_state.sel_active_tab = SEL_TAB_PARTIDO
        _run_sel_match_pipeline(st.session_state.nat_match_config, api_key)


def _run_sel_match_pipeline(config: dict, api_key: str, partial_results: Optional[Dict[str, Any]] = None) -> None:
    label = "🔄 Continuando..." if partial_results else "🔍 Preparando partido..."
    with st.status(label, expanded=True) as status:
        def _progress(msg: str) -> None:
            status.write(msg)
        try:
            results = run_national_match_prep(
                home_country=config["home_country"],
                away_country=config["away_country"],
                tournament=config.get("tournament", ""),
                match_type=config.get("match_type", ""),
                api_key=api_key,
                progress_cb=_progress,
                partial_results=partial_results,
            )
            st.session_state.nat_match_results = results
            status.update(label="✅ ¡Partido preparado!", state="complete", expanded=False)
            try:
                _save_national_match_prep(config, results)
            except Exception as e:
                print(f"[Sel] Error saving nat match: {e}")
        except Exception as e:
            if partial_results:
                st.session_state.nat_match_results = partial_results
                try:
                    _save_national_match_prep(config, partial_results)
                except Exception:
                    pass
            status.update(label=f"❌ Error: {e}", state="error")
            print(f"[Sel] Error nat match:\n{traceback.format_exc()}")
    st.rerun()


def _display_sel_match_results(config: dict, results: dict) -> None:
    home = config["home_country"]
    away = config["away_country"]
    tournament = config.get("tournament", "")
    match_type = config.get("match_type", "")

    st.markdown("---")
    meta_parts = [p for p in [tournament, match_type] if p]
    meta = " · ".join(meta_parts) if meta_parts else "Partido Internacional"
    st.markdown(
        f'<div class="match-header"><h2>⚽ {home} vs {away}</h2>'
        f'<div class="match-meta">{meta}</div></div>',
        unsafe_allow_html=True,
    )

    # Main analysis
    st.markdown("### 📊 Análisis del Partido")
    h_text, h_srcs = results.get("home_history", ("", []))
    st.markdown(h_text)
    if h_srcs:
        with st.expander("📚 Fuentes"):
            st.markdown(_format_sources(h_srcs).replace(_CITATION_SEPARATOR, ""))

    # Home roster
    home_roster = results.get("home_roster", [])
    if home_roster:
        st.markdown("---")
        st.markdown(f"### 🏠 Convocatoria de **{home}**")
        _render_roster_players(home_roster)

    # Away roster
    away_roster = results.get("away_roster", [])
    if away_roster:
        st.markdown("---")
        st.markdown(f"### ✈️ Convocatoria de **{away}**")
        _render_roster_players(away_roster)


# ---- Tab 2: Investigar Convocado ----------------------------------------

def _render_sel_player_tab(api_key: str) -> None:
    if st.session_state.get("nat_player_results") and st.session_state.get("nat_player_config"):
        existing = st.session_state.nat_player_results
        config = st.session_state.nat_player_config

        if not _has_data(existing.get("dossier")):
            st.warning("⚠️ Dossier incompleto. Puedes continuar la investigación.")
            if st.button("🔄 Continuar investigación", type="primary", use_container_width=True, key="sel_continue_player"):
                _run_sel_player_pipeline(config, api_key, partial_results=existing)
                return

        _display_sel_player_results(config, existing)
        return

    # --- Form ---
    st.markdown("### 📋 Convocado a investigar")
    col1, col2 = st.columns(2)
    with col1:
        player_name = st.text_input(
            "🧑 Nombre del Jugador",
            placeholder="Ej: Lionel Messi",
            key="sel_player_name",
        )
    with col2:
        country = st.text_input(
            "🌍 Selección",
            placeholder="Ej: Argentina",
            key="sel_player_country",
        )

    can_submit = bool(player_name and country)
    if st.button("🧑 Investigar Convocado", use_container_width=True, disabled=not can_submit,
                 type="primary", key="sel_player_submit"):
        if not api_key:
            st.error("⚠️ No se encontró la API key de Gemini.")
            return
        st.session_state.nat_player_config = {
            "player_name": player_name,
            "country": country,
        }
        st.session_state.sel_active_tab = SEL_TAB_CONVOCADO
        _run_sel_player_pipeline(st.session_state.nat_player_config, api_key)


def _run_sel_player_pipeline(config: dict, api_key: str, partial_results: Optional[Dict[str, Any]] = None) -> None:
    label = "🔄 Continuando..." if partial_results else "🔍 Investigando convocado..."
    with st.status(label, expanded=True) as status:
        def _progress(msg: str) -> None:
            status.write(msg)
        try:
            results = run_national_player_research(
                player_name=config["player_name"],
                country=config["country"],
                api_key=api_key,
                progress_cb=_progress,
                partial_results=partial_results,
            )
            st.session_state.nat_player_results = results
            status.update(label="✅ ¡Dossier completo!", state="complete", expanded=False)
            try:
                _save_national_player_research(config, results)
            except Exception as e:
                print(f"[Sel] Error saving nat player: {e}")
        except Exception as e:
            if partial_results:
                st.session_state.nat_player_results = partial_results
                try:
                    _save_national_player_research(config, partial_results)
                except Exception:
                    pass
            status.update(label=f"❌ Error: {e}", state="error")
            print(f"[Sel] Error nat player:\n{traceback.format_exc()}")
    st.rerun()


def _display_sel_player_results(config: dict, results: dict) -> None:
    player = config["player_name"]
    country = config.get("country", "")

    st.markdown("---")
    meta = f"Selección de {country}" if country else "Selección Nacional"
    st.markdown(
        f'<div class="match-header"><h2>🧑 {player}</h2>'
        f'<div class="match-meta">{meta}</div></div>',
        unsafe_allow_html=True,
    )
    st.markdown("### 📋 Dossier Internacional")
    dossier_text, dossier_srcs = results.get("dossier", ("", []))
    st.markdown(dossier_text)
    if dossier_srcs:
        with st.expander("📚 Fuentes"):
            st.markdown(_format_sources(dossier_srcs).replace(_CITATION_SEPARATOR, ""))


if __name__ == "__main__":
    main()
