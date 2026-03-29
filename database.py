"""PalomoFacts — Supabase persistence layer."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

import streamlit as st
from supabase import create_client, Client as SupabaseClient

from config import (
    MODE_PALOMO_GPT,
    USAGE_BACKFILL,
    USAGE_RUNTIME,
    _TEAM_RESEARCH_MAX_AGE_DAYS,
)
from metrics import (
    _backfill_run_id,
    _deserialize_text_result,
    _empty_token_usage,
    _estimate_workflow_cost,
    _has_data,
    _init_workflow_metrics,
    _normalize_roster_entries,
    _normalize_token_usage,
    _relative_window_start,
    _serialize_result_value,
)


# ---------------------------------------------------------------------------
# Supabase client
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


# ---------------------------------------------------------------------------
# Conversation CRUD
# ---------------------------------------------------------------------------

def _create_conversation(mode: str = MODE_PALOMO_GPT, title: str = "Nueva conversación") -> str:
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
    sb = _supabase_client()
    if not sb or not conv_id:
        return
    try:
        sb.table("messages").insert({
            "conversation_id": conv_id,
            "role": role,
            "content": content,
        }).execute()
        sb.table("conversations").update({
            "updated_at": datetime.utcnow().isoformat(),
        }).eq("id", conv_id).execute()
    except Exception as e:
        print(f"[Supabase] Error saving message: {e}")


def _update_conversation_title(conv_id: str, title: str) -> None:
    sb = _supabase_client()
    if not sb or not conv_id:
        return
    try:
        sb.table("conversations").update({"title": title}).eq("id", conv_id).execute()
    except Exception as e:
        print(f"[Supabase] Error updating title: {e}")


def _delete_conversation(conv_id: str) -> None:
    sb = _supabase_client()
    if not sb or not conv_id:
        return
    try:
        sb.table("conversations").delete().eq("id", conv_id).execute()
        print(f"[Supabase] Deleted conversation: {conv_id}")
    except Exception as e:
        print(f"[Supabase] Error deleting conversation: {e}")


def _auto_title(text: str) -> str:
    clean = text.strip().replace("\n", " ")
    return clean[:60] + "…" if len(clean) > 60 else clean


# ---------------------------------------------------------------------------
# Match Prep persistence
# ---------------------------------------------------------------------------

def _save_match_prep(config: dict, results: dict) -> str:
    sb = _supabase_client()
    if not sb:
        return ""
    try:
        title = f"{config.get('home_team', '?')} vs {config.get('away_team', '?')}"
        json_results = {}
        for key, val in results.items():
            json_results[key] = _serialize_result_value(val)

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
            sb.table("match_preps").update(payload).eq("id", existing_id).execute()
            print(f"[Supabase] Updated match prep: {existing_id} — '{title}'")
            return existing_id
        else:
            row = sb.table("match_preps").insert(payload).execute()
            prep_id = row.data[0]["id"] if row.data else ""
            st.session_state.match_prep_id = prep_id
            print(f"[Supabase] Saved match prep: {prep_id} — '{title}'")
            return prep_id
    except Exception as e:
        print(f"[Supabase] Error saving match prep: {e}")
        return ""


def _list_match_preps(limit: int = 20) -> List[Dict[str, Any]]:
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
        raw_results = row.get("results", {})
        results = {}
        for key in ("home_history", "away_history", "home_coach", "away_coach", "palomo_phrases"):
            results[key] = _deserialize_text_result(raw_results.get(key, {}))
        for key in ("home_roster", "away_roster"):
            results[key] = _normalize_roster_entries(raw_results.get(key, []))
        results["workflow_metrics"] = _init_workflow_metrics(
            "match_preparation",
            raw_results.get("workflow_metrics"),
        )
        return {"config": row.get("config", {}), "results": results}
    except Exception as e:
        print(f"[Supabase] Error loading match prep {prep_id}: {e}")
        return None


def _delete_match_prep(prep_id: str) -> None:
    sb = _supabase_client()
    if not sb or not prep_id:
        return
    try:
        sb.table("match_preps").delete().eq("id", prep_id).execute()
        print(f"[Supabase] Deleted match prep: {prep_id}")
    except Exception as e:
        print(f"[Supabase] Error deleting match prep: {e}")


# ---------------------------------------------------------------------------
# Team Research persistence
# ---------------------------------------------------------------------------

def _save_team_research(config: dict, results: dict) -> str:
    sb = _supabase_client()
    if not sb:
        return ""
    try:
        title = config.get("team_name", "?")
        json_results: dict = {}
        for key, val in results.items():
            json_results[key] = _serialize_result_value(val)

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
        results["team_history"] = _deserialize_text_result(raw_results.get("team_history", {}))
        results["coach"] = _deserialize_text_result(raw_results.get("coach", {}))
        results["roster"] = _normalize_roster_entries(raw_results.get("roster", []))
        results["workflow_metrics"] = _init_workflow_metrics(
            "team_research",
            raw_results.get("workflow_metrics"),
        )
        return {"config": row.get("config", {}), "results": results}
    except Exception as e:
        print(f"[Supabase] Error loading team research {research_id}: {e}")
        return None


def _delete_team_research(research_id: str) -> None:
    sb = _supabase_client()
    if not sb or not research_id:
        return
    try:
        sb.table("team_researches").delete().eq("id", research_id).execute()
        print(f"[Supabase] Deleted team research: {research_id}")
    except Exception as e:
        print(f"[Supabase] Error deleting team research: {e}")


def _find_existing_team_research(team_name: str) -> Optional[Dict[str, Any]]:
    sb = _supabase_client()
    if not sb or not team_name:
        return None
    try:
        resp = (
            sb.table("team_researches")
            .select("*")
            .eq("team_name", team_name)
            .order("updated_at", desc=True)
            .limit(1)
            .execute()
        )
        if not resp.data:
            return None
        row = resp.data[0]

        updated_str = row.get("updated_at", "")
        if updated_str:
            try:
                updated_at = datetime.fromisoformat(updated_str.replace("Z", "+00:00"))
                age_days = (datetime.utcnow().replace(tzinfo=updated_at.tzinfo) - updated_at).days
                if age_days > _TEAM_RESEARCH_MAX_AGE_DAYS:
                    print(f"[Supabase] Team research for '{team_name}' is {age_days}d old — skipping reuse")
                    return None
            except (ValueError, TypeError):
                pass

        raw_results = row.get("results", {})
        results: dict = {}
        results["team_history"] = _deserialize_text_result(raw_results.get("team_history", {}))
        results["coach"] = _deserialize_text_result(raw_results.get("coach", {}))
        results["roster"] = _normalize_roster_entries(raw_results.get("roster", []))
        return results
    except Exception as e:
        print(f"[Supabase] Error finding team research for '{team_name}': {e}")
        return None


def _auto_save_team_from_match(
    team_name: str,
    tournament: str,
    match_results: dict,
    side: str,
) -> None:
    sb = _supabase_client()
    if not sb:
        return
    try:
        team_results = {
            "team_history": match_results.get(f"{side}_history"),
            "coach": match_results.get(f"{side}_coach"),
            "roster": match_results.get(f"{side}_roster"),
        }
        if not _has_data(team_results.get("team_history")):
            return

        json_results: dict = {}
        for key, val in team_results.items():
            json_results[key] = _serialize_result_value(val)

        payload = {
            "title": team_name,
            "team_name": team_name,
            "tournament": tournament,
            "config": {"team_name": team_name, "tournament": tournament},
            "results": json_results,
            "updated_at": datetime.utcnow().isoformat(),
        }

        existing = (
            sb.table("team_researches")
            .select("id")
            .eq("team_name", team_name)
            .order("updated_at", desc=True)
            .limit(1)
            .execute()
        )
        if existing.data:
            existing_id = existing.data[0]["id"]
            sb.table("team_researches").update(payload).eq("id", existing_id).execute()
            print(f"[Supabase] Auto-updated team research from match: '{team_name}' ({existing_id})")
        else:
            sb.table("team_researches").insert(payload).execute()
            print(f"[Supabase] Auto-saved team research from match: '{team_name}'")
    except Exception as e:
        print(f"[Supabase] Error auto-saving team from match: {e}")


# ---------------------------------------------------------------------------
# Player Research persistence
# ---------------------------------------------------------------------------

def _save_player_research(config: dict, results: dict) -> str:
    sb = _supabase_client()
    if not sb:
        return ""
    try:
        title = config.get("player_name", "?")
        json_results: dict = {}
        for key, val in results.items():
            json_results[key] = _serialize_result_value(val)

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
        results["dossier"] = _deserialize_text_result(raw_results.get("dossier", {}))
        results["workflow_metrics"] = _init_workflow_metrics(
            "player_research",
            raw_results.get("workflow_metrics"),
        )
        return {"config": row.get("config", {}), "results": results}
    except Exception as e:
        print(f"[Supabase] Error loading player research {research_id}: {e}")
        return None


def _delete_player_research(research_id: str) -> None:
    sb = _supabase_client()
    if not sb or not research_id:
        return
    try:
        sb.table("player_researches").delete().eq("id", research_id).execute()
        print(f"[Supabase] Deleted player research: {research_id}")
    except Exception as e:
        print(f"[Supabase] Error deleting player research: {e}")


# ---------------------------------------------------------------------------
# National Team Research persistence
# ---------------------------------------------------------------------------

def _save_national_team_research(config: dict, results: dict) -> str:
    sb = _supabase_client()
    if not sb:
        return ""
    try:
        title = config.get("country", "?")
        json_results: dict = {}
        for key, val in results.items():
            json_results[key] = _serialize_result_value(val)
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
        results["team_history"] = _deserialize_text_result(raw.get("team_history", {}))
        results["coach"] = _deserialize_text_result(raw.get("coach", {}))
        results["roster"] = _normalize_roster_entries(raw.get("roster", []))
        results["workflow_metrics"] = _init_workflow_metrics(
            "national_team_research",
            raw.get("workflow_metrics"),
        )
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


# ---------------------------------------------------------------------------
# National Match Prep persistence
# ---------------------------------------------------------------------------

def _save_national_match_prep(config: dict, results: dict) -> str:
    sb = _supabase_client()
    if not sb:
        return ""
    try:
        home = config.get("home_country", "?")
        away = config.get("away_country", "?")
        title = f"{home} vs {away}"
        json_results: dict = {}
        for key, val in results.items():
            json_results[key] = _serialize_result_value(val)
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
            results[key] = _deserialize_text_result(raw.get(key, {}))
        results["home_roster"] = _normalize_roster_entries(raw.get("home_roster", []))
        results["away_roster"] = _normalize_roster_entries(raw.get("away_roster", []))
        results["workflow_metrics"] = _init_workflow_metrics(
            "national_match_prep",
            raw.get("workflow_metrics"),
        )
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


# ---------------------------------------------------------------------------
# National Player Research persistence
# ---------------------------------------------------------------------------

def _save_national_player_research(config: dict, results: dict) -> str:
    sb = _supabase_client()
    if not sb:
        return ""
    try:
        title = config.get("player_name", "?")
        json_results: dict = {}
        for key, val in results.items():
            json_results[key] = _serialize_result_value(val)
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
        results["dossier"] = _deserialize_text_result(raw.get("dossier", {}))
        results["workflow_metrics"] = _init_workflow_metrics(
            "national_player_research",
            raw.get("workflow_metrics"),
        )
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
# Usage Run persistence
# ---------------------------------------------------------------------------

def _fetch_table_rows(
    table_name: str,
    columns: str,
    order_column: str = "created_at",
    descending: bool = False,
    gte_created_at: Optional[str] = None,
    page_size: int = 500,
) -> List[Dict[str, Any]]:
    sb = _supabase_client()
    if not sb:
        return []

    all_rows: List[Dict[str, Any]] = []
    start = 0
    while True:
        try:
            query = (
                sb.table(table_name)
                .select(columns)
                .order(order_column, desc=descending)
                .range(start, start + page_size - 1)
            )
            if gte_created_at:
                query = query.gte("created_at", gte_created_at)
            resp = query.execute()
        except Exception as e:
            print(f"[Supabase] Error fetching {table_name}: {e}")
            break

        batch = resp.data or []
        all_rows.extend(batch)
        if len(batch) < page_size:
            break
        start += page_size

    return all_rows


def _usage_run_exists(run_id: str) -> bool:
    sb = _supabase_client()
    if not sb or not run_id:
        return False
    try:
        resp = sb.table("usage_runs").select("run_id").eq("run_id", run_id).limit(1).execute()
        return bool(resp.data)
    except Exception as e:
        print(f"[Supabase] Error checking usage run {run_id}: {e}")
        return False


def _save_usage_run(
    run_id: str,
    source_type: str,
    source_id: str,
    workflow: str,
    title: str,
    subject: str,
    metrics: Any,
    ingest_source: str = USAGE_RUNTIME,
    created_at: Optional[str] = None,
) -> bool:
    sb = _supabase_client()
    if not sb or not run_id:
        return False

    normalized = _init_workflow_metrics(workflow, metrics)
    payload = {
        "run_id": run_id,
        "source_type": source_type,
        "source_id": source_id or None,
        "workflow": str(normalized.get("workflow") or workflow),
        "title": title or "",
        "subject": subject or "",
        "totals": normalized.get("totals", _empty_token_usage()),
        "steps": normalized.get("steps", []),
        "estimated_cost_usd": _estimate_workflow_cost(normalized),
        "ingest_source": ingest_source,
    }
    if created_at:
        payload["created_at"] = created_at

    try:
        sb.table("usage_runs").upsert(payload, on_conflict="run_id").execute()
        return True
    except Exception as e:
        print(f"[Supabase] Error saving usage run {run_id}: {e}")
        return False


def _load_usage_runs(filter_key: str = "all") -> List[Dict[str, Any]]:
    lower_bound = _relative_window_start(filter_key)
    rows = _fetch_table_rows(
        table_name="usage_runs",
        columns="id, run_id, source_type, source_id, workflow, title, subject, totals, steps, estimated_cost_usd, created_at, ingest_source",
        order_column="created_at",
        descending=True,
        gte_created_at=lower_bound,
    )
    normalized_rows: List[Dict[str, Any]] = []
    for row in rows:
        normalized_rows.append(
            {
                **row,
                "totals": _normalize_token_usage(row.get("totals")),
                "steps": [dict(step) for step in row.get("steps", []) if isinstance(step, dict)],
                "estimated_cost_usd": float(row.get("estimated_cost_usd", 0.0) or 0.0),
            }
        )
    return normalized_rows


def _persist_usage_run_safe(
    run_id: str,
    source_type: str,
    source_id: str,
    workflow: str,
    title: str,
    subject: str,
    metrics: Any,
    ingest_source: str = USAGE_RUNTIME,
    created_at: Optional[str] = None,
    allow_empty: bool = False,
) -> None:
    normalized = _init_workflow_metrics(workflow, metrics)
    if not allow_empty and not normalized.get("steps"):
        return
    try:
        _save_usage_run(
            run_id=run_id,
            source_type=source_type,
            source_id=source_id,
            workflow=workflow,
            title=title,
            subject=subject,
            metrics=normalized,
            ingest_source=ingest_source,
            created_at=created_at,
        )
    except Exception as e:
        print(f"[Usage] Failed to persist usage for {run_id}: {e}")


# ---------------------------------------------------------------------------
# Backfill specs
# ---------------------------------------------------------------------------

BACKFILL_SPECS = [
    {
        "table": "match_preps",
        "select": "id, title, home_team, away_team, created_at, results",
        "source_type": "match_prep",
        "workflow": "match_preparation",
        "title_fn": lambda row: str(row.get("title") or f"{row.get('home_team', '?')} vs {row.get('away_team', '?')}"),
        "subject_fn": lambda row: f"{row.get('home_team', '?')} vs {row.get('away_team', '?')}",
    },
    {
        "table": "team_researches",
        "select": "id, title, team_name, created_at, results",
        "source_type": "team_research",
        "workflow": "team_research",
        "title_fn": lambda row: str(row.get("title") or row.get("team_name") or "Equipo"),
        "subject_fn": lambda row: str(row.get("team_name") or ""),
    },
    {
        "table": "player_researches",
        "select": "id, title, player_name, team_name, created_at, results",
        "source_type": "player_research",
        "workflow": "player_research",
        "title_fn": lambda row: str(row.get("title") or row.get("player_name") or "Jugador"),
        "subject_fn": lambda row: (
            f"{row.get('player_name', '')} · {row.get('team_name', '')}".strip(" ·")
        ),
    },
    {
        "table": "national_team_researches",
        "select": "id, title, country, created_at, results",
        "source_type": "national_team_research",
        "workflow": "national_team_research",
        "title_fn": lambda row: str(row.get("title") or row.get("country") or "Selección"),
        "subject_fn": lambda row: str(row.get("country") or ""),
    },
    {
        "table": "national_match_preps",
        "select": "id, title, home_country, away_country, created_at, results",
        "source_type": "national_match_prep",
        "workflow": "national_match_prep",
        "title_fn": lambda row: str(row.get("title") or f"{row.get('home_country', '?')} vs {row.get('away_country', '?')}"),
        "subject_fn": lambda row: f"{row.get('home_country', '?')} vs {row.get('away_country', '?')}",
    },
    {
        "table": "national_player_researches",
        "select": "id, title, player_name, country, created_at, results",
        "source_type": "national_player_research",
        "workflow": "national_player_research",
        "title_fn": lambda row: str(row.get("title") or row.get("player_name") or "Convocado"),
        "subject_fn": lambda row: (
            f"{row.get('player_name', '')} · {row.get('country', '')}".strip(" ·")
        ),
    },
]


def _backfill_usage_runs() -> Dict[str, int]:
    processed = 0
    empty = 0
    existing = 0
    for spec in BACKFILL_SPECS:
        rows = _fetch_table_rows(
            table_name=spec["table"],
            columns=spec["select"],
            order_column="created_at",
            descending=False,
        )
        for row in rows:
            source_id = str(row.get("id") or "")
            metrics = _init_workflow_metrics(spec["workflow"], (row.get("results") or {}).get("workflow_metrics"))
            if not metrics.get("steps"):
                empty += 1
                continue
            run_id = _backfill_run_id(spec["source_type"], source_id)
            if _usage_run_exists(run_id):
                existing += 1
                continue
            _save_usage_run(
                run_id=run_id,
                source_type=spec["source_type"],
                source_id=source_id,
                workflow=spec["workflow"],
                title=spec["title_fn"](row),
                subject=spec["subject_fn"](row),
                metrics=metrics,
                ingest_source=USAGE_BACKFILL,
                created_at=row.get("created_at"),
            )
            processed += 1
    return {"processed": processed, "empty": empty, "existing": existing}
