#!/usr/bin/env python3
"""PalomoFacts – Streamlit app with two modes:

  1. PalomoGPT: unified conversational football intelligence with auto-router
  2. Preparación de Partidos: structured match preparation reports

Powered by Google Gemini with Google Search grounding.
"""
from __future__ import annotations
import hashlib
import unicodedata
import uuid

import traceback
import time
from datetime import datetime, timedelta, timezone
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

MODE_PALOMO_GPT  = "palomo_gpt"
MODE_CLUB        = "club"
MODE_SELECCION   = "seleccion"

# Aliases so existing sidebar/routing references keep working
MODE_MATCH_PREP      = MODE_CLUB
MODE_MATCH_RESEARCH  = MODE_CLUB
MODE_PLAYER_RESEARCH = MODE_CLUB
MODE_SELECCIONES     = MODE_SELECCION

MODE_OPTIONS = {
    MODE_PALOMO_GPT: "🎙️ PalomoGPT",
    MODE_CLUB:       "🏠 Investigar Club",
    MODE_SELECCION:  "🌍 Investigar Selección",
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

# Shared tab indices — identical for both Club and Selección
TAB_PARTIDO  = 0   # ⚽ Partido
TAB_EQUIPO   = 1   # 🔬 Equipo / Selección
TAB_JUGADOR  = 2   # 🧑 Jugador / Convocado

# Aliases for existing Seleccion render functions
SEL_TAB_SELECCION = TAB_EQUIPO
SEL_TAB_PARTIDO   = TAB_PARTIDO
SEL_TAB_CONVOCADO = TAB_JUGADOR

# Club tab session-state key
CLUB_ACTIVE_TAB_KEY = "club_active_tab"
SEL_ACTIVE_TAB_KEY  = "sel_active_tab"

DASHBOARD_ACCESS_STATE_KEY = "dashboard_access_granted"
DASHBOARD_FILTER_OPTIONS = {
    "all": "All time",
    "7d": "7d",
    "30d": "30d",
    "90d": "90d",
}

USAGE_RUNTIME = "runtime"
USAGE_BACKFILL = "backfill"

_DEFAULT_PRICING = {
    "input_per_million": 0.0,
    "input_long_per_million": 0.0,
    "output_per_million": 0.0,
    "output_long_per_million": 0.0,
    "search_per_unit": 0.0,
    "long_context_threshold": 200_000,
}

MODEL_PRICING = {
    GEMINI_MODEL: {
        "input_per_million": 2.0,
        "input_long_per_million": 4.0,
        "output_per_million": 12.0,
        "output_long_per_million": 18.0,
        "search_per_unit": 14.0 / 1000.0,
        "long_context_threshold": 200_000,
    },
    GEMINI_FALLBACK_MODEL: {
        "input_per_million": 1.25,
        "input_long_per_million": 2.50,
        "output_per_million": 10.0,
        "output_long_per_million": 15.0,
        "search_per_unit": 35.0 / 1000.0,
        "long_context_threshold": 200_000,
    },
    CLAUDE_MODEL: {
        "input_per_million": 5.0,
        "input_long_per_million": 5.0,
        "output_per_million": 25.0,
        "output_long_per_million": 25.0,
        "search_per_unit": 10.0 / 1000.0,
        "long_context_threshold": 200_000,
    },
    CLAUDE_HAIKU_MODEL: {
        "input_per_million": 1.0,
        "input_long_per_million": 1.0,
        "output_per_million": 5.0,
        "output_long_per_million": 5.0,
        "search_per_unit": 0.0,
        "long_context_threshold": 200_000,
    },
}

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


def _empty_token_usage(model: str = "", provider: str = "") -> Dict[str, Any]:
    """Return a normalized empty token usage payload."""
    return {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "grounding_requests": 0,
        "model": model,
        "provider": provider,
    }


def _normalize_token_usage(tokens: Any) -> Dict[str, Any]:
    """Normalize token usage payloads so old and new runs share one shape."""
    normalized = _empty_token_usage()
    if not isinstance(tokens, dict):
        return normalized

    normalized["input_tokens"] = int(tokens.get("input_tokens", 0) or 0)
    normalized["output_tokens"] = int(tokens.get("output_tokens", 0) or 0)
    normalized["total_tokens"] = int(tokens.get("total_tokens", 0) or 0)
    normalized["grounding_requests"] = int(tokens.get("grounding_requests", 0) or 0)
    normalized["model"] = str(tokens.get("model", "") or "")
    normalized["provider"] = str(tokens.get("provider", "") or "")

    if not normalized["total_tokens"]:
        normalized["total_tokens"] = normalized["input_tokens"] + normalized["output_tokens"]

    return normalized


def _add_token_usage(total: Dict[str, Any], tokens: Any) -> Dict[str, Any]:
    """Accumulate a normalized token payload into a running total."""
    usage = _normalize_token_usage(tokens)
    total["input_tokens"] = int(total.get("input_tokens", 0) or 0) + usage["input_tokens"]
    total["output_tokens"] = int(total.get("output_tokens", 0) or 0) + usage["output_tokens"]
    total["total_tokens"] = int(total.get("total_tokens", 0) or 0) + usage["total_tokens"]
    total["grounding_requests"] = int(total.get("grounding_requests", 0) or 0) + usage["grounding_requests"]
    if usage.get("model") and not total.get("model"):
        total["model"] = usage["model"]
    if usage.get("provider") and not total.get("provider"):
        total["provider"] = usage["provider"]
    return total


def _init_workflow_metrics(workflow: str, existing: Any = None) -> Dict[str, Any]:
    """Return a workflow metrics container with recomputed totals."""
    metrics = existing if isinstance(existing, dict) else {}
    steps = metrics.get("steps", [])
    if not isinstance(steps, list):
        steps = []

    totals = _empty_token_usage()
    if steps:
        for step in steps:
            _add_token_usage(totals, step)
    else:
        totals = _normalize_token_usage(metrics.get("totals"))

    return {
        "workflow": str(metrics.get("workflow") or workflow),
        "steps": steps,
        "totals": totals,
    }


def _ensure_workflow_metrics(results: Dict[str, Any], workflow: str) -> Dict[str, Any]:
    """Attach workflow metrics to a results dict if missing."""
    metrics = _init_workflow_metrics(workflow, results.get("workflow_metrics"))
    results["workflow_metrics"] = metrics
    return metrics


def _record_workflow_step(
    metrics: Dict[str, Any],
    step: str,
    label: str,
    tokens: Any,
    entity: str = "",
) -> None:
    """Append one measured step to the workflow metrics payload."""
    usage = _normalize_token_usage(tokens)
    entry: Dict[str, Any] = {
        "step": step,
        "label": label,
        **usage,
    }
    if entity:
        entry["entity"] = entity

    metrics.setdefault("steps", []).append(entry)
    _add_token_usage(metrics.setdefault("totals", _empty_token_usage()), usage)


def _merge_workflow_metrics(target: Dict[str, Any], incoming: Any) -> Dict[str, Any]:
    """Merge another workflow metrics payload into the target tracker."""
    source = _init_workflow_metrics(target.get("workflow", "workflow"), incoming)
    for step in source.get("steps", []):
        if isinstance(step, dict):
            target.setdefault("steps", []).append(dict(step))
            _add_token_usage(target.setdefault("totals", _empty_token_usage()), step)
    return target


def _serialize_result_value(value: Any) -> Any:
    """Serialize tuples and nested token payloads for Supabase JSON storage."""
    if isinstance(value, tuple):
        text = value[0] if len(value) > 0 else ""
        sources = value[1] if len(value) > 1 else []
        payload = {
            "text": text,
            "sources": sources,
            "tokens": _normalize_token_usage(value[2] if len(value) > 2 else None),
        }
        return payload

    if isinstance(value, list):
        serialized: List[Any] = []
        for item in value:
            if isinstance(item, dict):
                normalized_item = dict(item)
                if "tokens" in normalized_item or "text" in normalized_item:
                    normalized_item["tokens"] = _normalize_token_usage(normalized_item.get("tokens"))
                serialized.append(normalized_item)
            else:
                serialized.append(item)
        return serialized

    return value


def _deserialize_text_result(value: Any) -> Tuple[str, List[Dict[str, str]], Dict[str, Any]]:
    """Load a persisted text result while staying compatible with older rows."""
    if isinstance(value, dict) and "text" in value:
        return (
            value.get("text", ""),
            value.get("sources", []),
            _normalize_token_usage(value.get("tokens")),
        )

    if isinstance(value, tuple):
        text = value[0] if len(value) > 0 else ""
        sources = value[1] if len(value) > 1 else []
        tokens = _normalize_token_usage(value[2] if len(value) > 2 else None)
        return text, sources, tokens

    return "", [], _empty_token_usage()


def _normalize_roster_entries(value: Any) -> List[Dict[str, Any]]:
    """Normalize roster entries so player token payloads are always present."""
    if not isinstance(value, list):
        return []

    normalized: List[Dict[str, Any]] = []
    for item in value:
        if isinstance(item, dict):
            row = dict(item)
            row["tokens"] = _normalize_token_usage(row.get("tokens"))
            normalized.append(row)
    return normalized


def _unpack_text_result(value: Any) -> Tuple[str, List[Dict[str, str]], Dict[str, Any]]:
    """Unpack either in-memory tuples or persisted dict payloads."""
    return _deserialize_text_result(value)


def _aggregate_workflow_metrics(metrics: Any) -> List[Dict[str, Any]]:
    """Roll up raw call telemetry by workflow step for easier analysis."""
    normalized = _init_workflow_metrics("workflow", metrics)
    aggregated: Dict[str, Dict[str, Any]] = {}

    for step in normalized.get("steps", []):
        if not isinstance(step, dict):
            continue
        step_key = str(step.get("step") or step.get("label") or "workflow.step")
        label = str(step.get("label") or step_key)
        row = aggregated.setdefault(
            step_key,
            {
                "step": step_key,
                "label": label,
                "calls": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "grounding_requests": 0,
                "models": [],
                "examples": [],
            },
        )
        row["calls"] += 1
        row["input_tokens"] += int(step.get("input_tokens", 0) or 0)
        row["output_tokens"] += int(step.get("output_tokens", 0) or 0)
        row["total_tokens"] += int(step.get("total_tokens", 0) or 0)
        row["grounding_requests"] += int(step.get("grounding_requests", 0) or 0)

        model = str(step.get("model", "") or "")
        if model and model not in row["models"]:
            row["models"].append(model)

        entity = str(step.get("entity", "") or "")
        if entity and entity not in row["examples"] and len(row["examples"]) < 3:
            row["examples"].append(entity)

    return sorted(aggregated.values(), key=lambda item: item["total_tokens"], reverse=True)


def _format_metric_number(value: Any) -> str:
    """Pretty-print integer metrics in the UI."""
    return f"{int(value or 0):,}"


def _build_markdown_table(headers: List[str], rows: List[List[str]]) -> str:
    """Render a lightweight markdown table without extra dependencies."""
    if not rows:
        return ""

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _render_workflow_metrics(metrics: Any, title: str = "📈 Uso de tokens") -> None:
    """Render workflow totals, per-step rollups, and top expensive calls."""
    normalized = _init_workflow_metrics("workflow", metrics)
    steps = normalized.get("steps", [])
    if not steps:
        return

    totals = normalized.get("totals", _empty_token_usage())
    grouped_rows = _aggregate_workflow_metrics(normalized)
    top_calls = sorted(
        [step for step in steps if isinstance(step, dict)],
        key=lambda item: int(item.get("total_tokens", 0) or 0),
        reverse=True,
    )[:5]

    with st.expander(title, expanded=False):
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Input", _format_metric_number(totals.get("input_tokens")))
        col2.metric("Output", _format_metric_number(totals.get("output_tokens")))
        col3.metric("Total", _format_metric_number(totals.get("total_tokens")))
        col4.metric("Search", _format_metric_number(totals.get("grounding_requests")))

        if grouped_rows:
            st.markdown("**Por paso**")
            grouped_table = _build_markdown_table(
                ["Paso", "Llamadas", "Input", "Output", "Total", "Modelo"],
                [
                    [
                        row["label"],
                        _format_metric_number(row["calls"]),
                        _format_metric_number(row["input_tokens"]),
                        _format_metric_number(row["output_tokens"]),
                        _format_metric_number(row["total_tokens"]),
                        ", ".join(row["models"]) if row["models"] else "n/a",
                    ]
                    for row in grouped_rows
                ],
            )
            if grouped_table:
                st.markdown(grouped_table)

        if top_calls:
            st.markdown("**Llamadas más costosas**")
            calls_table = _build_markdown_table(
                ["Llamada", "Total", "Modelo"],
                [
                    [
                        (
                            f"{step.get('label', step.get('step', 'Paso'))} · {step.get('entity')}"
                            if step.get("entity")
                            else str(step.get("label", step.get("step", "Paso")))
                        ),
                        _format_metric_number(step.get("total_tokens")),
                        str(step.get("model", "") or "n/a"),
                    ]
                    for step in top_calls
                ],
            )
            if calls_table:
                st.markdown(calls_table)


def _utcnow_iso() -> str:
    """Return an ISO timestamp in UTC."""
    return datetime.now(timezone.utc).isoformat()


def _new_usage_run_id(prefix: str) -> str:
    """Create a unique idempotency key for one billable execution."""
    return f"{prefix}:{uuid.uuid4().hex}"


def _backfill_run_id(source_type: str, source_id: str) -> str:
    """Create a deterministic run id so historical backfills can be retried safely."""
    digest = hashlib.sha256(f"{source_type}:{source_id}".encode("utf-8")).hexdigest()[:24]
    return f"backfill:{source_type}:{digest}"


def _workflow_step_count(metrics: Any) -> int:
    """Return the number of recorded workflow steps."""
    normalized = _init_workflow_metrics("workflow", metrics)
    return len([step for step in normalized.get("steps", []) if isinstance(step, dict)])


def _slice_workflow_metrics(metrics: Any, start_index: int = 0) -> Dict[str, Any]:
    """Return only the workflow steps added after a known step index."""
    normalized = _init_workflow_metrics("workflow", metrics)
    steps = normalized.get("steps", [])
    sliced_steps = [dict(step) for step in steps[start_index:] if isinstance(step, dict)]
    sliced = {
        "workflow": str(normalized.get("workflow") or "workflow"),
        "steps": sliced_steps,
        "totals": _empty_token_usage(),
    }
    for step in sliced_steps:
        _add_token_usage(sliced["totals"], step)
    return sliced


def _pricing_for_model(model: str) -> Dict[str, float]:
    """Return pricing metadata for a tracked model."""
    pricing = MODEL_PRICING.get(model)
    if pricing:
        return pricing
    return dict(_DEFAULT_PRICING)


def _estimate_usage_cost(tokens: Any) -> float:
    """Estimate the raw provider cost for one normalized usage payload."""
    usage = _normalize_token_usage(tokens)
    pricing = _pricing_for_model(usage.get("model", ""))
    threshold = int(pricing.get("long_context_threshold", 200_000) or 200_000)
    is_long_context = usage["input_tokens"] > threshold

    input_rate = pricing["input_long_per_million"] if is_long_context else pricing["input_per_million"]
    output_rate = pricing["output_long_per_million"] if is_long_context else pricing["output_per_million"]

    cost = 0.0
    cost += (usage["input_tokens"] / 1_000_000.0) * input_rate
    cost += (usage["output_tokens"] / 1_000_000.0) * output_rate
    cost += usage["grounding_requests"] * float(pricing.get("search_per_unit", 0.0) or 0.0)
    return round(cost, 6)


def _estimate_workflow_cost(metrics: Any) -> float:
    """Estimate total raw provider cost for a workflow metrics payload."""
    normalized = _init_workflow_metrics("workflow", metrics)
    return round(
        sum(_estimate_usage_cost(step) for step in normalized.get("steps", []) if isinstance(step, dict)),
        6,
    )


def _coerce_datetime(value: Any) -> Optional[datetime]:
    """Parse a Supabase datetime string into a timezone-aware UTC datetime."""
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if not value:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        parsed = datetime.fromisoformat(text)
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _relative_window_start(filter_key: str) -> Optional[str]:
    """Return the ISO lower bound for a dashboard date filter."""
    days_map = {"7d": 7, "30d": 30, "90d": 90}
    days = days_map.get(filter_key)
    if not days:
        return None
    return (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()


def _format_cost(value: Any) -> str:
    """Format raw USD cost for dashboard display."""
    amount = float(value or 0.0)
    if amount >= 100:
        return f"${amount:,.2f}"
    if amount >= 1:
        return f"${amount:,.4f}"
    return f"${amount:,.6f}"


def _extract_gemini_search_units(response: Any, model_name: str, use_search: bool) -> int:
    """Estimate billable Gemini grounding units from grounding metadata."""
    if not use_search:
        return 0

    try:
        candidate = response.candidates[0]
    except (AttributeError, IndexError, TypeError):
        return 0

    grounding = getattr(candidate, "grounding_metadata", None)
    if not grounding:
        return 0

    queries = getattr(grounding, "web_search_queries", None)
    if queries is None and hasattr(grounding, "get"):
        queries = grounding.get("web_search_queries")
    query_count = len(list(queries or []))

    chunks = getattr(grounding, "grounding_chunks", None)
    has_chunks = bool(chunks)
    has_search_entry = bool(getattr(grounding, "search_entry_point", None))
    has_grounding_signal = query_count > 0 or has_chunks or has_search_entry

    if model_name.startswith("gemini-3"):
        if query_count > 0:
            return query_count
        return 1 if has_grounding_signal else 0

    return 1 if has_grounding_signal else 0


def _extract_claude_search_units(response: Any) -> int:
    """Read Claude server-side web search usage when available."""
    usage = getattr(response, "usage", None)
    server_tool_use = getattr(usage, "server_tool_use", None)
    if server_tool_use is None and isinstance(usage, dict):
        server_tool_use = usage.get("server_tool_use")

    if isinstance(server_tool_use, dict):
        return int(server_tool_use.get("web_search_requests", 0) or 0)

    return int(getattr(server_tool_use, "web_search_requests", 0) or 0)


def _fetch_table_rows(
    table_name: str,
    columns: str,
    order_column: str = "created_at",
    descending: bool = False,
    gte_created_at: Optional[str] = None,
    page_size: int = 500,
) -> List[Dict[str, Any]]:
    """Fetch all rows from a Supabase table using page ranges."""
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
    """Return whether a usage run already exists in the ledger."""
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
    """Persist one usage run to the ledger, using run_id for idempotency."""
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
    """Load usage ledger rows for the dashboard."""
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


def _aggregate_usage_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate top-level dashboard KPIs across usage rows."""
    totals = _empty_token_usage()
    total_cost = 0.0
    for row in rows:
        _add_token_usage(totals, row.get("totals"))
        total_cost += float(row.get("estimated_cost_usd", 0.0) or 0.0)
    return {
        "runs": len(rows),
        "totals": totals,
        "estimated_cost_usd": round(total_cost, 6),
    }


def _aggregate_usage_by_model(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Aggregate usage step metrics by model and provider."""
    grouped: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for row in rows:
        for step in row.get("steps", []):
            usage = _normalize_token_usage(step)
            key = (usage.get("provider", ""), usage.get("model", ""))
            current = grouped.setdefault(
                key,
                {
                    "provider": usage.get("provider", "") or "n/a",
                    "model": usage.get("model", "") or "n/a",
                    "calls": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "grounding_requests": 0,
                    "estimated_cost_usd": 0.0,
                },
            )
            current["calls"] += 1
            current["input_tokens"] += usage["input_tokens"]
            current["output_tokens"] += usage["output_tokens"]
            current["total_tokens"] += usage["total_tokens"]
            current["grounding_requests"] += usage["grounding_requests"]
            current["estimated_cost_usd"] += _estimate_usage_cost(usage)

    return sorted(
        grouped.values(),
        key=lambda item: (float(item["estimated_cost_usd"]), int(item["total_tokens"])),
        reverse=True,
    )


def _aggregate_usage_by_workflow(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Aggregate usage ledger rows by workflow and source type."""
    grouped: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for row in rows:
        key = (str(row.get("workflow") or ""), str(row.get("source_type") or ""))
        current = grouped.setdefault(
            key,
            {
                "workflow": row.get("workflow", "") or "n/a",
                "source_type": row.get("source_type", "") or "n/a",
                "runs": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "grounding_requests": 0,
                "estimated_cost_usd": 0.0,
            },
        )
        current["runs"] += 1
        totals = _normalize_token_usage(row.get("totals"))
        current["input_tokens"] += totals["input_tokens"]
        current["output_tokens"] += totals["output_tokens"]
        current["total_tokens"] += totals["total_tokens"]
        current["grounding_requests"] += totals["grounding_requests"]
        current["estimated_cost_usd"] += float(row.get("estimated_cost_usd", 0.0) or 0.0)

    return sorted(
        grouped.values(),
        key=lambda item: (float(item["estimated_cost_usd"]), int(item["total_tokens"])),
        reverse=True,
    )


def _serialize_usage_recent_rows(rows: List[Dict[str, Any]], limit: int = 15) -> List[Dict[str, Any]]:
    """Build the recent-runs table shown in the admin dashboard."""
    recent: List[Dict[str, Any]] = []
    for row in rows[:limit]:
        parsed = _coerce_datetime(row.get("created_at"))
        models: List[str] = []
        for step in row.get("steps", []):
            model = str(step.get("model", "") or "")
            if model and model not in models:
                models.append(model)
        recent.append(
            {
                "When": parsed.astimezone().strftime("%Y-%m-%d %H:%M") if parsed else str(row.get("created_at", "")),
                "Workflow": str(row.get("workflow") or "n/a"),
                "Type": str(row.get("source_type") or "n/a"),
                "Title": str(row.get("title") or row.get("subject") or "n/a"),
                "Models": ", ".join(models) if models else "n/a",
                "Search": _format_metric_number(_normalize_token_usage(row.get("totals")).get("grounding_requests")),
                "Cost": _format_cost(row.get("estimated_cost_usd")),
            }
        )
    return recent


def _backfill_usage_runs() -> Dict[str, int]:
    """Backfill recoverable structured workflow history into the usage ledger."""
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
    """Persist usage without letting telemetry failures break the user flow."""
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
        results["team_history"] = _deserialize_text_result(raw_results.get("team_history", {}))
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

REGLA SAGRADA: CERO ALUCINACIONES. No inventas números, goles, partidos, resultados, récords \
ni goleadores en ligas. El usuario ha notado que inventas marcadores (ej. El Clásico o Derbis) \
o goleadores falsos porque quieres dar información detallada aunque no sepas la respuesta. \
ESTO ESTÁ ESTRICTAMENTE PROHIBIDO. Si no estás *absoluta y matemáticamente seguro* del marcador final \
de un evento histórico y quién metió los goles, ¡NO LO DIGAS! Di "En un duelo histórico donde \
ganaron" en lugar de "Ganaron 3-1 con goles de X, Y y Z" y quedar en ridículo porque inventaste \
a los goleadores. TODO dato de estadísticas debe ser 100% verificable.

ZONAS DE ALTO RIESGO DE ALUCINACIÓN (cuidado extremo, debes cuestionarte antes de responder):
- MARCADORES Y GOLEADORES EXACTOS DE PARTIDOS CLAVE: Exige verificación extrema.
- Afirmaciones "el primero en...", "nunca antes...", "único en la historia": Solo si la fuente \
  lo confirma explícitamente. Si no estás 100%% seguro, NO lo digas.
- Finales, títulos, trofeos: DIFERENCIA CLARAMENTE entre "dirigió en una liga" y \
  "llegó a una final". Jamás impliques que alguien llegó a una final si no tienes la fuente.
- Estadísticas exactas (goles, asistencias, fechas de debut, caps en selección): Solo cita números seguros.
- Si NO encuentras evidencia de algo específico, di: "No he encontrado un registro certero \
  de ese partido/goleador" en lugar de inventar para quedar bien.

Tu cancha cubre TODO el fútbol: estadísticas estrictas de cualquier liga y época, vida personal de \
jugadores, historia de clubes, táctica, transferencias, la temporada actual \
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

⚠️ REGLA ESTRICTA DE PRECISIÓN (CERO ALUCINACIONES) ⚠️
Antes de escribir CUALQUIER marcador final o CUALQUIER nombre de un goleador, VERIFICA en tu \
conocimiento base si estás 100% seguro de ese dato específico.
- Si NO estás 100% seguro de quién metió el gol, escribe "Goleadores no confirmados" en lugar de adivinar.
- Si NO estás 100% seguro del marcador exacto de un partido, omite el partido o indica "Resultado no confirmado".
- ES PREFERIBLE MOSTRAR MENOS PARTIDOS O DATOS OMITIDOS QUE UN DATO INVENTADO.
- Tómate tu tiempo internamente para contrastar las cifras antes de emitir la respuesta.

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

PARA RECOPAS / SUPERCOPA / COMMUNITY SHIELD:
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

Responde en español. Sé EXHAUSTIVO, PRECISO Y TOTALMENTE VERÍDICO. NO inventes datos. \
Si no encuentras un dato con absoluta certeza, omítelo.
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

⚠️ REGLA ESTRICTA DE PRECISIÓN (CERO ALUCINACIONES) ⚠️
Antes de escribir CUALQUIER cifra, marcador, goleador o récord, VERIFÍCALO internamente.
- Si no estás 100% seguro del resultado de un partido icónico, confírmalo o no lo detalles.
- Si mencionas a un goleador histórico, asegúrate de que el número de goles es correcto y actualizado a {current_date}.
- NUNCA inventes estadísticas de "Caps" ni de goles internacionales.
- Es preferible proporcionar menos datos a proporcionar datos inventados.

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
   - Goleadores históricos en Mundiales (Asegura NO alucinar los goles precisos)
   - Partidos icónicos (victorias y derrotas que definieron una era)

3. **🌍 HISTORIA EN TORNEOS CONTINENTALES**
   - Copas América / EURO / AFCON / Copa de Asia / Gold Cup ganadas y en qué años
   - Rachas relevantes (más participaciones consecutivas, más finales seguidas, etc.)
   - Rivales constantes / rivalidades históricas continentales

4. **📊 CLASIFICATORIAS EN CURSO** (si aplica)
   - Confederación y formato de clasificatoria actual para el próximo Mundial
   - Posición actual en la tabla, puntos, partidos restantes
   - Resultados de los últimos 5 partidos (forma reciente)

5. **📌 ESTADÍSTICAS HISTÓRICAS**
   - Máximo goleador histórico: nombre, goles (NÚMERO EXACTO ACTUALIZADO), años activo
   - Más partidos internacionales (caps): nombre, número exacto
   - Portero con más vallas invictas, si aplica
   - Racha invicta más larga

6. **🎭 DATOS CURIOSOS Y CULTURA**
   - Apodos del equipo y su origen
   - Ritual o himno especial de la selección
   - Héroes históricos que marcaron generaciones
   - Momentos virales o polémicos
   - Relación con su afición y el fútbol en el país

Resalta con 🏆 los hitos más importantes.
Responde en español. Sé EXHAUSTIVO. NO INVENTES CIFRAS bajo NINGÚN concepto.
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

⚠️ REGLA ESTRICTA DE PRECISIÓN (CERO ALUCINACIONES) ⚠️
Al listar historiales o marcadores previos: NO inventes resultados engañosos.
Al listar goleadores del historial reciente: DEBES estar absoluta y matemáticamente seguro de \
quién anotó cada gol en ese partido específico.
Si tienes la MÁS MÍNIMA DUDA del marcador exacto de un enfrentamiento previo histórico, no pongas \
el número engañoso, pon el dato del torneo sin el marcador o usa términos generales como "empate".
ES MIL VECES MEJOR LA OMISIÓN QUE UNA METIDA DE PATA GIGANTE EN TELEVISIÓN EN VIVO A CAUSA DE ALUCINACIONES.

PARTIDO: **{home_country}** vs **{away_country}**
TORNEO / CONTEXTO: {tournament}
TIPO DE PARTIDO: {match_type}
FECHA ACTUAL: {current_date}

Crea la FICHA COMPLETA del partido con los siguientes bloques:

1. **⚔️ CONTEXTO DEL PARTIDO**
   - Qué está en juego: puntos de clasificatoria, pase a siguiente fase, título, etc.
   - Relevancia histórica de este partido en particular
   - ¿Es una final anticipada? ¿Un derbi confederacional? ¿Revancha histórica?

2. **📊 HISTORIAL DIRECTO ESTRICTO (Head-to-Head)**
   - Partidos totales jugados entre ambas selecciones
   - Balance de victorias, empates, derrotas (por cada lado)
   - Enfrentamientos RECIENTES (últimos 5 partidos verificables): resultado EXACTO, torneo, año
   - El partido más icónico de la historia entre ambas selecciones — relátalo sin inventar goles al aire.
   - El resultado más abultado en cada dirección (confirmación obligatoria)
   - ¿Alguna vez se enfrentaron en un Mundial o en otra gran competición?

3. **🏠 {home_country} — Análisis**
   - Forma reciente: últimos 5 partidos
   - Sistema táctico habitual del DT y posible alineación titular
   - Jugadores clave en este partido (máximo 5): nombre + por qué importa HOY

4. **✈️ {away_country} — Análisis**
   - Forma reciente: últimos 5 partidos
   - Sistema táctico y posible alineación
   - Jugadores clave (máximo 5)

5. **🔮 CLAVES TÁCTICAS DEL PARTIDO**
   - El duelo individual más importante a seguir
   - Zonas del campo donde se decidirá el partido
   - Árbitro asignado (si se sabe)

6. **🎙️ FRASES PALOMO** — 3 frases en el estilo de Fernando Palomo listas para narrar:
   - Una sobre la historia entre ambas selecciones
   - Una sobre el jugador estrella de {home_country}
   - Una sobre el jugador estrella de {away_country}

Responde en español. Sé PRECISO, CLARO Y TOTALMENTE VERÍDICO. NO INVENTES NADA."""


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
) -> Tuple[bytes, Dict[str, Any]]:
    """Generate a complete match preparation PDF.

    If api_key is provided, synthesizes verbose player dossiers into compact
    broadcast-ready notes before writing to PDF.
    """

    def _cb(msg: str) -> None:
        if progress_cb:
            progress_cb(msg)

    # Synthesize rosters before PDF generation
    synth_results = dict(results)  # shallow copy to avoid mutating original
    export_metrics = _init_workflow_metrics("match_pdf_export")
    if api_key:
        for key in ("home_roster", "away_roster"):
            roster = results.get(key, [])
            if roster:
                label = "Local" if "home" in key else "Visitante"
                _cb(f"Sintetizando {len(roster)} jugadores ({label})...")
                synthesized_roster, roster_metrics = _synthesize_roster_for_pdf(
                    roster,
                    api_key,
                    progress_cb=progress_cb,
                    team_label=label,
                )
                synth_results[key] = synthesized_roster
                _merge_workflow_metrics(export_metrics, roster_metrics)

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
        return out.encode('latin1'), export_metrics
    return bytes(out), export_metrics


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
        token_usage = _empty_token_usage(model=CLAUDE_HAIKU_MODEL, provider="anthropic")
        if hasattr(response, "usage") and response.usage:
            token_usage["input_tokens"] = int(getattr(response.usage, "input_tokens", 0) or 0)
            token_usage["output_tokens"] = int(getattr(response.usage, "output_tokens", 0) or 0)
            token_usage["total_tokens"] = token_usage["input_tokens"] + token_usage["output_tokens"]
        return {
            **player,
            "text": note.strip(),
            "raw_text": raw_text,
            "_pdf_synthesis_tokens": token_usage,
        }
    except Exception as e:
        print(f"[PDF Synthesis] Error for {player.get('name', '?')}: {e}")
        return player


def _synthesize_roster_for_pdf(
    roster: List[Dict[str, Any]],
    api_key: str,
    progress_cb: Optional[Callable[[str], None]] = None,
    team_label: str = "",
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Synthesize verbose player dossiers into compact broadcast notes via Claude Haiku.
    
    Uses parallel requests (up to 8 concurrent) for speed.
    """
    claude_key = st.secrets.get("ANTHROPIC_API_KEY", "")
    if not claude_key:
        print("[PDF Synthesis] No ANTHROPIC_API_KEY — skipping synthesis.")
        return roster, _init_workflow_metrics("match_pdf_export")

    client = anthropic.Anthropic(api_key=claude_key)
    synthesized: List[Optional[Dict[str, Any]]] = [None] * len(roster)
    done_count = 0
    total = len(roster)
    export_metrics = _init_workflow_metrics("match_pdf_export")

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

    cleaned: List[Dict[str, Any]] = []
    for player in [p for p in synthesized if p is not None]:
        tokens = _normalize_token_usage(player.get("_pdf_synthesis_tokens"))
        if tokens["total_tokens"] or tokens["grounding_requests"]:
            _record_workflow_step(
                export_metrics,
                "match_pdf_export.player_synthesis",
                f"PDF synthesis ({team_label})" if team_label else "PDF synthesis",
                tokens,
                entity=str(player.get("name", "") or ""),
            )
        cleaned_player = dict(player)
        cleaned_player.pop("_pdf_synthesis_tokens", None)
        cleaned.append(cleaned_player)

    return cleaned, export_metrics


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
) -> Tuple[str, List[Dict[str, str]], Dict[str, Any]]:
    """
    Request to Gemini with optional Google Search grounding.
    Retries up to _MAX_RETRIES times with exponential backoff.
    Returns (response_text, sources, token_usage).
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
            token_usage = _empty_token_usage(model=model, provider="google")
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                token_usage["input_tokens"] = int(getattr(response.usage_metadata, "prompt_token_count", 0) or 0)
                token_usage["output_tokens"] = int(getattr(response.usage_metadata, "candidates_token_count", 0) or 0)
                token_usage["total_tokens"] = int(getattr(response.usage_metadata, "total_token_count", 0) or 0)
                if not token_usage["total_tokens"]:
                    token_usage["total_tokens"] = token_usage["input_tokens"] + token_usage["output_tokens"]
            token_usage["grounding_requests"] = _extract_gemini_search_units(response, model, use_search)

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

            return text, sources, token_usage

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
) -> Tuple[str, List[Dict[str, str]], Dict[str, Any]]:
    """
    Fallback request via Claude Opus 4.6.
    No search grounding — purely LLM reasoning.
    Returns (response_text, sources=[], token_usage).
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
    token_usage = _empty_token_usage(model=CLAUDE_MODEL, provider="anthropic")
    if hasattr(response, "usage") and response.usage:
        token_usage["input_tokens"] = int(getattr(response.usage, "input_tokens", 0) or 0)
        token_usage["output_tokens"] = int(getattr(response.usage, "output_tokens", 0) or 0)
        token_usage["total_tokens"] = token_usage["input_tokens"] + token_usage["output_tokens"]
        token_usage["grounding_requests"] = _extract_claude_search_units(response)
    return text, [], token_usage  # no grounding sources from Claude


# ---------------------------------------------------------------------------
# PalomoGPT response
# ---------------------------------------------------------------------------
def get_palomo_response(
    query: str,
    session_messages: list[dict],
    api_key: str,
) -> Tuple[str, List[Dict[str, str]], List[str], List[Dict[str, str]], Dict[str, Any]]:
    """Get a response from PalomoGPT with Google Search grounding + follow-up chain.
    Returns (response_text, sources, reasoning_chain, followups, workflow_metrics).
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
    workflow_metrics = _init_workflow_metrics("palomo_gpt")

    # Single pass: Gemini + Google Search grounding
    text, sources, tokens_main = _gemini_request(
        api_key=api_key,
        system_prompt=_PALOMO_GPT_SYSTEM,
        user_message=query,
        history=history if history else None,
    )
    _record_workflow_step(
        workflow_metrics,
        "palomo.main_answer",
        "Respuesta principal",
        tokens_main,
        entity=query[:80],
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
            followup_q, _, tokens_fq = _gemini_request(
                api_key=api_key,
                system_prompt=_FOLLOW_UP_SYSTEM,
                user_message=(
                    f"PREGUNTA ORIGINAL: {query}\n\n"
                    f"RESPUESTA RECIBIDA:\n{current_answer}"
                ),
                use_search=False,
            )
            _record_workflow_step(
                workflow_metrics,
                "palomo.followup_question",
                "Pregunta de seguimiento",
                tokens_fq,
                entity=f"Ronda {i + 1}",
            )
            followup_q = followup_q.strip()
            if not followup_q or len(followup_q) < 10:
                break

            reasoning.append(
                f"🔄 **Follow-up {i+1}:** {followup_q}"
            )

            # Answer the follow-up question with search
            followup_a, extra_sources, tokens_fa = _gemini_request(
                api_key=api_key,
                system_prompt=_PALOMO_GPT_SYSTEM,
                user_message=followup_q,
                history=[
                    types.Content(role="user", parts=[types.Part.from_text(text=query)]),
                    types.Content(role="model", parts=[types.Part.from_text(text=current_answer)]),
                ],
            )
            _record_workflow_step(
                workflow_metrics,
                "palomo.followup_answer",
                "Respuesta de seguimiento",
                tokens_fa,
                entity=followup_q[:80],
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

    return text, sources, reasoning, followups, workflow_metrics


# ---------------------------------------------------------------------------
# Match Preparation Research
# ---------------------------------------------------------------------------
def _research_team_history(
    team_name: str,
    api_key: str,
) -> Tuple[str, List[Dict[str, str]], Dict[str, Any]]:
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
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Fetch the roster as a list of {name, full_name, position, number} dicts."""
    season_curr = CURRENT_YEAR - 1
    season_next = CURRENT_YEAR

    prompt = _TEAM_ROSTER_LIST_PROMPT.format(
        team_name=team_name,
        season_curr=season_curr,
        season_next=season_next,
        current_date=CURRENT_DATE,
    )

    text, _, tokens = _gemini_request(
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
    return players, tokens


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

    text, sources, tokens = _gemini_request(
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
        "tokens": tokens,
    }


def _research_team_roster(
    team_name: str,
    opponent_name: str,
    api_key: str,
    progress_cb: Optional[Callable[[str], None]] = None,
    workflow_metrics: Optional[Dict[str, Any]] = None,
    step_prefix: str = "roster",
) -> List[Dict[str, Any]]:
    """
    Research a team's roster player-by-player.
    1. Fetch player list (lightweight call).
    2. For each player, run a dedicated deep-research call (parallel batches of 4).
    Returns a list of {name, position, number, text, sources} dicts.
    """
    if progress_cb:
        progress_cb(f"📋 Obteniendo lista de jugadores de **{team_name}**...")
    players, list_tokens = _fetch_player_list(team_name, api_key)
    if workflow_metrics is not None:
        _record_workflow_step(
            workflow_metrics,
            f"{step_prefix}.fetch_player_list",
            f"{team_name}: lista de jugadores",
            list_tokens,
        )
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
                        "tokens": _empty_token_usage(),
                    }
                if workflow_metrics is not None:
                    _record_workflow_step(
                        workflow_metrics,
                        f"{step_prefix}.player_dossier",
                        f"{team_name}: dossier de jugador",
                        results[idx].get("tokens"),
                        entity=str(results[idx].get("name", "?")),
                    )
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
) -> Tuple[str, List[Dict[str, str]], Dict[str, Any]]:
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
    research_fn: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    workflow_metrics: Optional[Dict[str, Any]] = None,
    step_prefix: str = "roster",
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
    player_fetch = research_fn
    if player_fetch is None:
        if opponent_name:
            player_fetch = lambda player: _research_single_player(player, team_name, opponent_name, api_key)
        else:
            player_fetch = lambda player: _research_single_player_solo(player, team_name, api_key)

    batch_size = 4
    completed = 0
    for batch_start in range(0, total_failed, batch_size):
        batch_idxs = failed_indices[batch_start : batch_start + batch_size]
        with ThreadPoolExecutor(max_workers=batch_size) as executor:
            future_to_idx = {
                executor.submit(player_fetch, roster[idx]): idx
                for idx in batch_idxs
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    roster[idx] = future.result()
                except Exception as e:
                    # Keep the error but update it
                    roster[idx]["text"] = f"❌ Error investigando: {e}"
                    roster[idx]["tokens"] = _empty_token_usage()
                if workflow_metrics is not None:
                    _record_workflow_step(
                        workflow_metrics,
                        f"{step_prefix}.player_dossier",
                        f"{team_name}: dossier de jugador",
                        roster[idx].get("tokens"),
                        entity=str(roster[idx].get("name", "?")),
                    )
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
    initial_metrics: Optional[Dict[str, Any]] = None,
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
    workflow_metrics = _ensure_workflow_metrics(results, "match_preparation")
    if initial_metrics:
        _merge_workflow_metrics(workflow_metrics, initial_metrics)

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
                    label = home_team if key == "home_history" else away_team
                    _, _, tokens = _unpack_text_result(results[key])
                    _record_workflow_step(
                        workflow_metrics,
                        f"match_prep.{key}",
                        f"{label}: historial",
                        tokens,
                    )
                except Exception as e:
                    results[key] = (f"❌ Error investigando: {e}", [], _empty_token_usage())
        _cb("✅ Historiales completados.")
    else:
        _cb("✅ Historiales ya disponibles — reutilizando.")

    # Phase 2: rosters — player by player (skip if already done)
    if not _has_data(results.get("home_roster")):
        _cb(f"👥 Investigando plantilla de **{home_team}** jugador por jugador...")
        try:
            results["home_roster"] = _research_team_roster(
                home_team,
                away_team,
                api_key,
                progress_cb=_cb,
                workflow_metrics=workflow_metrics,
                step_prefix="match_prep.home_roster",
            )
        except Exception as e:
            results["home_roster"] = []
            _cb(f"❌ Error con plantilla de {home_team}: {e}")
    elif _roster_has_failures(results.get("home_roster", [])):
        results["home_roster"] = _retry_failed_roster_players(
            results["home_roster"],
            home_team,
            away_team,
            api_key,
            progress_cb=_cb,
            workflow_metrics=workflow_metrics,
            step_prefix="match_prep.home_roster",
        )
    else:
        _cb(f"✅ Plantilla de **{home_team}** ya disponible — reutilizando.")

    if not _has_data(results.get("away_roster")):
        _cb(f"👥 Investigando plantilla de **{away_team}** jugador por jugador...")
        try:
            results["away_roster"] = _research_team_roster(
                away_team,
                home_team,
                api_key,
                progress_cb=_cb,
                workflow_metrics=workflow_metrics,
                step_prefix="match_prep.away_roster",
            )
        except Exception as e:
            results["away_roster"] = []
            _cb(f"❌ Error con plantilla de {away_team}: {e}")
    elif _roster_has_failures(results.get("away_roster", [])):
        results["away_roster"] = _retry_failed_roster_players(
            results["away_roster"],
            away_team,
            home_team,
            api_key,
            progress_cb=_cb,
            workflow_metrics=workflow_metrics,
            step_prefix="match_prep.away_roster",
        )
    else:
        _cb(f"✅ Plantilla de **{away_team}** ya disponible — reutilizando.")

    # Phase 3: Palomo phrases (skip if already done)
    if not _has_data(results.get("palomo_phrases")):
        _cb("🎙️ Generando frases de Fernando Palomo...")
        home_context, _, _ = _unpack_text_result(results.get("home_history"))
        away_context, _, _ = _unpack_text_result(results.get("away_history"))

        try:
            results["palomo_phrases"] = _research_palomo_phrases(
                home_team, away_team, tournament, match_type, stadium,
                home_context, away_context, api_key,
            )
            _, _, phrase_tokens = _unpack_text_result(results["palomo_phrases"])
            _record_workflow_step(
                workflow_metrics,
                "match_prep.palomo_phrases",
                "Frases de Palomo",
                phrase_tokens,
            )
        except Exception as e:
            results["palomo_phrases"] = (f"❌ Error generando frases: {e}", [], _empty_token_usage())
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

    text, sources, tokens = _gemini_request(
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
        "tokens": tokens,
    }


def _research_team_roster_solo(
    team_name: str,
    api_key: str,
    progress_cb: Optional[Callable[[str], None]] = None,
    workflow_metrics: Optional[Dict[str, Any]] = None,
    step_prefix: str = "roster",
) -> List[Dict[str, Any]]:
    """Research a team's full roster player-by-player (no opponent context)."""
    if progress_cb:
        progress_cb(f"📋 Obteniendo lista de jugadores de **{team_name}**...")
    players, list_tokens = _fetch_player_list(team_name, api_key)
    if workflow_metrics is not None:
        _record_workflow_step(
            workflow_metrics,
            f"{step_prefix}.fetch_player_list",
            f"{team_name}: lista de jugadores",
            list_tokens,
        )
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
                        "tokens": _empty_token_usage(),
                    }
                if workflow_metrics is not None:
                    _record_workflow_step(
                        workflow_metrics,
                        f"{step_prefix}.player_dossier",
                        f"{team_name}: dossier de jugador",
                        results[idx].get("tokens"),
                        entity=str(results[idx].get("name", "?")),
                    )
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
    workflow_metrics = _ensure_workflow_metrics(results, "team_research")

    # Phase 1: team history
    if not _has_data(results.get("team_history")):
        _cb(f"📊 Investigando historial de **{team_name}**...")
        try:
            results["team_history"] = _research_team_history(team_name, api_key)
            _, _, tokens = _unpack_text_result(results["team_history"])
            _record_workflow_step(
                workflow_metrics,
                "team_research.team_history",
                "Historial del equipo",
                tokens,
                entity=team_name,
            )
        except Exception as e:
            results["team_history"] = (f"❌ Error investigando historial: {e}", [], _empty_token_usage())
        _cb("✅ Historial completado.")
    else:
        _cb("✅ Historial ya disponible — reutilizando.")

    # Phase 2: roster player-by-player
    if not _has_data(results.get("roster")):
        _cb(f"👥 Investigando plantilla de **{team_name}** jugador por jugador...")
        try:
            results["roster"] = _research_team_roster_solo(
                team_name,
                api_key,
                progress_cb=_cb,
                workflow_metrics=workflow_metrics,
                step_prefix="team_research.roster",
            )
        except Exception as e:
            results["roster"] = []
            _cb(f"❌ Error con plantilla de {team_name}: {e}")
    elif _roster_has_failures(results.get("roster", [])):
        results["roster"] = _retry_failed_roster_players(
            results["roster"],
            team_name,
            "",
            api_key,
            progress_cb=_cb,
            workflow_metrics=workflow_metrics,
            step_prefix="team_research.roster",
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
    workflow_metrics = _ensure_workflow_metrics(results, "player_research")

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
            _, _, tokens = _unpack_text_result(results["dossier"])
            _record_workflow_step(
                workflow_metrics,
                "player_research.dossier",
                "Dossier del jugador",
                tokens,
                entity=player_name,
            )
        except Exception as e:
            results["dossier"] = (f"❌ Error investigando: {e}", [], _empty_token_usage())
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
    text, sources, tokens = _gemini_request(
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
        "tokens": tokens,
    }


def _research_national_roster(
    country: str,
    api_key: str,
    progress_cb: Optional[Callable[[str], None]] = None,
    workflow_metrics: Optional[Dict[str, Any]] = None,
    step_prefix: str = "roster",
) -> List[Dict[str, Any]]:
    """Fetch and research the national team's current convocatoria player-by-player."""
    if progress_cb:
        progress_cb(f"📋 Obteniendo convocatoria de **{country}**...")
    players, list_tokens = _fetch_player_list(country, api_key)
    if workflow_metrics is not None:
        _record_workflow_step(
            workflow_metrics,
            f"{step_prefix}.fetch_player_list",
            f"{country}: lista de convocados",
            list_tokens,
        )
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
                        "tokens": _empty_token_usage(),
                    }
                if workflow_metrics is not None:
                    _record_workflow_step(
                        workflow_metrics,
                        f"{step_prefix}.player_dossier",
                        f"{country}: dossier de convocado",
                        results[idx].get("tokens"),
                        entity=str(results[idx].get("name", "?")),
                    )
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
    workflow_metrics = _ensure_workflow_metrics(results, "national_team_research")

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
            _, _, tokens = _unpack_text_result(results["team_history"])
            _record_workflow_step(
                workflow_metrics,
                "national_team_research.team_history",
                "Historial de la selección",
                tokens,
                entity=country,
            )
        except Exception as e:
            results["team_history"] = (f"❌ Error: {e}", [], _empty_token_usage())
        _cb("✅ Historial completado.")
    else:
        _cb("✅ Historial ya disponible — reutilizando.")

    if not _has_data(results.get("roster")):
        _cb(f"👥 Investigando convocatoria de **{country}**...")
        try:
            results["roster"] = _research_national_roster(
                country,
                api_key,
                progress_cb=_cb,
                workflow_metrics=workflow_metrics,
                step_prefix="national_team_research.roster",
            )
        except Exception as e:
            results["roster"] = []
            _cb(f"❌ Error con convocatoria de {country}: {e}")
    elif _roster_has_failures(results.get("roster", [])):
        results["roster"] = _retry_failed_roster_players(
            results["roster"],
            country,
            "",
            api_key,
            progress_cb=_cb,
            research_fn=lambda player: _research_national_player_solo(player, country, api_key),
            workflow_metrics=workflow_metrics,
            step_prefix="national_team_research.roster",
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
    workflow_metrics = _ensure_workflow_metrics(results, "national_match_prep")

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
            _, _, tokens = _unpack_text_result(results["home_history"])
            _record_workflow_step(
                workflow_metrics,
                "national_match_prep.match_analysis",
                "Análisis del partido",
                tokens,
                entity=f"{home_country} vs {away_country}",
            )
        except Exception as e:
            results["home_history"] = (f"❌ Error en análisis: {e}", [], _empty_token_usage())
        _cb("✅ Análisis del partido completado.")
    else:
        _cb("✅ Análisis ya disponible — reutilizando.")

    # Phase 2: home roster
    if not _has_data(results.get("home_roster")):
        _cb(f"👥 Investigando convocatoria de **{home_country}**...")
        try:
            results["home_roster"] = _research_national_roster(
                home_country,
                api_key,
                progress_cb=_cb,
                workflow_metrics=workflow_metrics,
                step_prefix="national_match_prep.home_roster",
            )
        except Exception as e:
            results["home_roster"] = []
            _cb(f"❌ Error con convocatoria de {home_country}: {e}")
    elif _roster_has_failures(results.get("home_roster", [])):
        results["home_roster"] = _retry_failed_roster_players(
            results["home_roster"],
            home_country,
            away_country,
            api_key,
            progress_cb=_cb,
            research_fn=lambda player: _research_national_player_solo(player, home_country, api_key),
            workflow_metrics=workflow_metrics,
            step_prefix="national_match_prep.home_roster",
        )
    else:
        _cb(f"✅ Convocatoria de **{home_country}** ya disponible — reutilizando.")

    # Phase 3: away roster
    if not _has_data(results.get("away_roster")):
        _cb(f"👥 Investigando convocatoria de **{away_country}**...")
        try:
            results["away_roster"] = _research_national_roster(
                away_country,
                api_key,
                progress_cb=_cb,
                workflow_metrics=workflow_metrics,
                step_prefix="national_match_prep.away_roster",
            )
        except Exception as e:
            results["away_roster"] = []
            _cb(f"❌ Error con convocatoria de {away_country}: {e}")
    elif _roster_has_failures(results.get("away_roster", [])):
        results["away_roster"] = _retry_failed_roster_players(
            results["away_roster"],
            away_country,
            home_country,
            api_key,
            progress_cb=_cb,
            research_fn=lambda player: _research_national_player_solo(player, away_country, api_key),
            workflow_metrics=workflow_metrics,
            step_prefix="national_match_prep.away_roster",
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
    workflow_metrics = _ensure_workflow_metrics(results, "national_player_research")

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
            _, _, tokens = _unpack_text_result(results["dossier"])
            _record_workflow_step(
                workflow_metrics,
                "national_player_research.dossier",
                "Dossier del convocado",
                tokens,
                entity=player_name,
            )
        except Exception as e:
            results["dossier"] = (f"❌ Error investigando: {e}", [], _empty_token_usage())
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
def _render_root_page() -> None:
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
            # ================================================================
            # MODE_CLUB sidebar — context per active club tab
            # ================================================================
            if app_mode == MODE_CLUB:
                st.markdown("### 📚 Historial")

                # ⚽ Partido history
                preps = _list_match_preps()
                if preps or st.session_state.get("match_results"):
                    with st.expander("⚽ Partidos", expanded=True):
                        if st.button("➕ Nuevo partido", use_container_width=True):
                            st.session_state.pop("match_results", None)
                            st.session_state.pop("match_config", None)
                            st.session_state.pop("match_pdf_bytes", None)
                            st.session_state.club_active_tab = TAB_PARTIDO
                            st.rerun()
                        for prep in preps:
                            pid = prep["id"]
                            title = prep.get("title", "Sin título")
                            col_t, col_d = st.columns([5, 1])
                            with col_t:
                                if st.button(title, key=f"prep_{pid}", use_container_width=True):
                                    loaded = _load_match_prep(pid)
                                    if loaded:
                                        st.session_state.match_config = loaded["config"]
                                        st.session_state.match_results = loaded["results"]
                                        st.session_state.match_prep_id = pid
                                        st.session_state.pop("match_pdf_bytes", None)
                                        st.session_state.club_active_tab = TAB_PARTIDO
                                    st.rerun()
                            with col_d:
                                if st.button("🗑️", key=f"dprep_{pid}"):
                                    _delete_match_prep(pid)
                                    st.rerun()

                # 🔬 Equipo history
                researches = _list_team_researches()
                if researches or st.session_state.get("team_research_results"):
                    with st.expander("🔬 Equipos", expanded=False):
                        if st.button("➕ Nuevo equipo", use_container_width=True):
                            st.session_state.pop("team_research_results", None)
                            st.session_state.pop("team_research_config", None)
                            st.session_state.pop("team_research_id", None)
                            st.session_state.club_active_tab = TAB_EQUIPO
                            st.rerun()

                        for res in researches:
                            rid = res["id"]
                            title = res.get("title", "Sin título")
                            league = res.get("tournament", "")
                            label = f"{title} ({league})" if league else title
                            col_t, col_d = st.columns([5, 1])
                            with col_t:
                                if st.button(label, key=f"tres_{rid}", use_container_width=True):
                                    loaded = _load_team_research(rid)
                                    if loaded:
                                        st.session_state.team_research_config = loaded["config"]
                                        st.session_state.team_research_results = loaded["results"]
                                        st.session_state.team_research_id = rid
                                        st.session_state.club_active_tab = TAB_EQUIPO
                                    st.rerun()
                            with col_d:
                                if st.button("🗑️", key=f"dtres_{rid}"):
                                    _delete_team_research(rid)
                                    st.rerun()

                # 🧑 Jugador history
                presearches = _list_player_researches()
                if presearches or st.session_state.get("player_research_results"):
                    with st.expander("🧑 Jugadores", expanded=False):
                        if st.button("➕ Nuevo jugador", use_container_width=True):
                            st.session_state.pop("player_research_results", None)
                            st.session_state.pop("player_research_config", None)
                            st.session_state.pop("player_research_id", None)
                            st.session_state.club_active_tab = TAB_JUGADOR
                            st.rerun()
                        for res in presearches:
                            rid = res["id"]
                            title = res.get("title", "Sin título")
                            team = res.get("team_name", "")
                            label = f"{title} ({team})" if team else title
                            col_t, col_d = st.columns([5, 1])
                            with col_t:
                                if st.button(label, key=f"pres_{rid}", use_container_width=True):
                                    loaded = _load_player_research(rid)
                                    if loaded:
                                        st.session_state.player_research_config = loaded["config"]
                                        st.session_state.player_research_results = loaded["results"]
                                        st.session_state.player_research_id = rid
                                        st.session_state.club_active_tab = TAB_JUGADOR
                                    st.rerun()
                            with col_d:
                                if st.button("🗑️", key=f"dpres_{rid}"):
                                    _delete_player_research(rid)
                                    st.rerun()

            # ================================================================
            # MODE_SELECCION sidebar — context per active seleccion tab
            # ================================================================
            elif app_mode == MODE_SELECCION:
                st.markdown("### 📚 Historial")

                # ⚽ Partidos history
                preps = _list_national_match_preps()
                if preps or st.session_state.get("nat_match_results"):
                    with st.expander("⚽ Partidos", expanded=True):
                        if st.button("➕ Nuevo partido", use_container_width=True, key="new_sel_match"):
                            st.session_state.pop("nat_match_results", None)
                            st.session_state.pop("nat_match_config", None)
                            st.session_state.pop("nat_match_prep_id", None)
                            st.session_state.sel_active_tab = TAB_PARTIDO
                            st.rerun()
                        for prep in preps:
                            pid = prep["id"]
                            title = prep.get("title", "Sin título")
                            col_t, col_d = st.columns([5, 1])
                            with col_t:
                                if st.button(title, key=f"nprep_{pid}", use_container_width=True):
                                    loaded = _load_national_match_prep(pid)
                                    if loaded:
                                        st.session_state.nat_match_config = loaded["config"]
                                        st.session_state.nat_match_results = loaded["results"]
                                        st.session_state.nat_match_prep_id = pid
                                        st.session_state.sel_active_tab = TAB_PARTIDO
                                    st.rerun()
                            with col_d:
                                if st.button("🗑️", key=f"dnprep_{pid}"):
                                    _delete_national_match_prep(pid)
                                    st.rerun()

                # 🔬 Selección history
                researches = _list_national_team_researches()
                if researches or st.session_state.get("nat_team_research_results"):
                    with st.expander("🔬 Selecciones", expanded=False):
                        if st.button("➕ Nueva selección", use_container_width=True, key="new_sel_team"):
                            st.session_state.pop("nat_team_research_results", None)
                            st.session_state.pop("nat_team_research_config", None)
                            st.session_state.pop("nat_team_research_id", None)
                            st.session_state.sel_active_tab = TAB_EQUIPO
                            st.rerun()
                        for res in researches:
                            rid = res["id"]
                            title = res.get("title", "Sin título")
                            conf = res.get("confederation", "")
                            label = f"{title} ({conf.split('—')[0].strip()})" if conf and "Cualquier" not in conf else title
                            col_t, col_d = st.columns([5, 1])
                            with col_t:
                                if st.button(label, key=f"ntres_{rid}", use_container_width=True):
                                    loaded = _load_national_team_research(rid)
                                    if loaded:
                                        st.session_state.nat_team_research_config = loaded["config"]
                                        st.session_state.nat_team_research_results = loaded["results"]
                                        st.session_state.nat_team_research_id = rid
                                        st.session_state.sel_active_tab = TAB_EQUIPO
                                    st.rerun()
                            with col_d:
                                if st.button("🗑️", key=f"dntres_{rid}"):
                                    _delete_national_team_research(rid)
                                    st.rerun()

                # 🧑 Convocado history
                presearches = _list_national_player_researches()
                if presearches or st.session_state.get("nat_player_results"):
                    with st.expander("🧑 Convocados", expanded=False):
                        if st.button("➕ Nuevo convocado", use_container_width=True, key="new_sel_player"):
                            st.session_state.pop("nat_player_results", None)
                            st.session_state.pop("nat_player_config", None)
                            st.session_state.pop("nat_player_research_id", None)
                            st.session_state.sel_active_tab = TAB_JUGADOR
                            st.rerun()
                        for res in presearches:
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
    elif app_mode == MODE_CLUB:
        _render_club(api_key)
    elif app_mode == MODE_SELECCION:
        _render_seleccion(api_key)
    else:
        _render_club(api_key)


def _dashboard_access_granted() -> bool:
    """Render the admin access gate and return whether the dashboard is unlocked."""
    expected_key = str(st.secrets.get("DASHBOARD_ACCESS_KEY", "") or "")
    if st.session_state.get(DASHBOARD_ACCESS_STATE_KEY):
        return True

    st.markdown(
        '<p class="hero-title">📊 Usage Dashboard</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="hero-sub">Hidden admin view for raw provider cost, tokens, and web-search usage.</p>',
        unsafe_allow_html=True,
    )

    if not expected_key:
        st.error("Configura `DASHBOARD_ACCESS_KEY` en los secrets de Streamlit para habilitar este dashboard.")
        return False

    with st.form("dashboard_access_form", clear_on_submit=False):
        access_key = st.text_input("Access key", type="password")
        submitted = st.form_submit_button("Enter dashboard", use_container_width=True, type="primary")
        if submitted:
            if access_key == expected_key:
                st.session_state[DASHBOARD_ACCESS_STATE_KEY] = True
                st.rerun()
            st.error("Clave incorrecta.")
    return False


def _render_dashboard_page() -> None:
    """Render the hidden admin dashboard at /dashboard."""
    if not _dashboard_access_granted():
        return

    st.markdown(
        '<p class="hero-title">📊 Usage Dashboard</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="hero-sub">Overall app usage across models, workflows, web search, and raw provider cost.</p>',
        unsafe_allow_html=True,
    )

    controls_col, backfill_col, logout_col = st.columns([3, 2, 1])
    with controls_col:
        filter_key = st.pills(
            "Window",
            options=list(DASHBOARD_FILTER_OPTIONS.keys()),
            default="all",
            format_func=lambda key: DASHBOARD_FILTER_OPTIONS[key],
            key="dashboard_window",
        )
        if not filter_key:
            filter_key = "all"
    with backfill_col:
        st.markdown("<div style='height: 1.9rem;'></div>", unsafe_allow_html=True)
        if st.button("Backfill history", use_container_width=True):
            result = _backfill_usage_runs()
            st.success(
                f"Backfill inserted {result['processed']} records, skipped {result['existing']} already in the ledger, "
                f"and ignored {result['empty']} without metrics."
            )
    with logout_col:
        st.markdown("<div style='height: 1.9rem;'></div>", unsafe_allow_html=True)
        if st.button("Lock", use_container_width=True):
            st.session_state[DASHBOARD_ACCESS_STATE_KEY] = False
            st.rerun()

    usage_rows = _load_usage_runs(filter_key)
    aggregate = _aggregate_usage_rows(usage_rows)
    totals = aggregate["totals"]
    by_model = _aggregate_usage_by_model(usage_rows)
    by_workflow = _aggregate_usage_by_workflow(usage_rows)
    recent_rows = _serialize_usage_recent_rows(usage_rows, limit=15)

    metric_cols = st.columns(6)
    metric_cols[0].metric("Raw Cost", _format_cost(aggregate["estimated_cost_usd"]))
    metric_cols[1].metric("Runs", _format_metric_number(aggregate["runs"]))
    metric_cols[2].metric("Input", _format_metric_number(totals["input_tokens"]))
    metric_cols[3].metric("Output", _format_metric_number(totals["output_tokens"]))
    metric_cols[4].metric("Total", _format_metric_number(totals["total_tokens"]))
    metric_cols[5].metric("Web Search", _format_metric_number(totals["grounding_requests"]))

    st.caption(
        "Note: older structured workflows can be backfilled from saved `workflow_metrics`, "
        "but historical PalomoGPT traffic and older PDF exports from before this release were not persisted."
    )

    if not usage_rows:
        st.info("No usage data found yet. Run the app or use backfill to populate the ledger.")
        return

    st.markdown("### By Model")
    model_rows = [
        {
            "Provider": row["provider"],
            "Model": row["model"],
            "Calls": _format_metric_number(row["calls"]),
            "Input": _format_metric_number(row["input_tokens"]),
            "Output": _format_metric_number(row["output_tokens"]),
            "Total": _format_metric_number(row["total_tokens"]),
            "Search": _format_metric_number(row["grounding_requests"]),
            "Cost": _format_cost(row["estimated_cost_usd"]),
        }
        for row in by_model
    ]
    st.dataframe(model_rows, use_container_width=True, hide_index=True)

    st.markdown("### By Workflow")
    workflow_rows = [
        {
            "Workflow": row["workflow"],
            "Type": row["source_type"],
            "Runs": _format_metric_number(row["runs"]),
            "Input": _format_metric_number(row["input_tokens"]),
            "Output": _format_metric_number(row["output_tokens"]),
            "Total": _format_metric_number(row["total_tokens"]),
            "Search": _format_metric_number(row["grounding_requests"]),
            "Cost": _format_cost(row["estimated_cost_usd"]),
        }
        for row in by_workflow
    ]
    st.dataframe(workflow_rows, use_container_width=True, hide_index=True)

    st.markdown("### Recent Runs")
    st.dataframe(recent_rows, use_container_width=True, hide_index=True)


def _build_streamlit_page(
    page_callable: Callable[[], None],
    title: str,
    url_path: Optional[str] = None,
    default: bool = False,
    hidden: bool = False,
):
    """Create a Streamlit page, using hidden visibility when supported."""
    kwargs: Dict[str, Any] = {
        "title": title,
        "default": default,
    }
    if url_path is not None:
        kwargs["url_path"] = url_path
    if hidden:
        kwargs["visibility"] = "hidden"

    try:
        return st.Page(page_callable, **kwargs)
    except TypeError:
        # Older Streamlit builds may not support the `visibility` argument.
        kwargs.pop("visibility", None)
        return st.Page(page_callable, **kwargs)


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
    workflow_metrics = _init_workflow_metrics("palomo_gpt")
    with st.chat_message("assistant"):
        try:
            with st.status(
                "🔍 Buscando información verificada...", expanded=False
            ) as status:
                text, citations, reasoning_chain, followups, workflow_metrics = get_palomo_response(
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
            _render_workflow_metrics(workflow_metrics, "📈 Uso de tokens de esta respuesta")
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

    _persist_usage_run_safe(
        run_id=_new_usage_run_id("palomo"),
        source_type="conversation",
        source_id=conv_id,
        workflow="palomo_gpt",
        title=_auto_title(incoming_query),
        subject=incoming_query,
        metrics=workflow_metrics,
    )


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
            validation_metrics = _init_workflow_metrics("match_preparation")
            try:
                val_prompt = _MATCH_VALIDATION_PROMPT.format(
                    home_team=home_team,
                    away_team=away_team,
                    tournament=tournament,
                    current_date=CURRENT_DATE,
                )
                val_text, _, val_tokens = _gemini_request(
                    api_key=api_key,
                    system_prompt=val_prompt,
                    user_message="Valida este partido y devuelve el JSON.",
                )
                _record_workflow_step(
                    validation_metrics,
                    "match_validation",
                    "Validación del partido",
                    val_tokens,
                    entity=f"{home_team} vs {away_team}",
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

        _run_match_pipeline(
            st.session_state.match_config,
            api_key,
            initial_metrics=validation_metrics,
        )


def _run_match_pipeline(
    config: dict,
    api_key: str,
    partial_results: Optional[Dict[str, Any]] = None,
    initial_metrics: Optional[Dict[str, Any]] = None,
) -> None:
    """Run (or resume) the match preparation research pipeline."""
    home_team = config["home_team"]
    away_team = config["away_team"]
    baseline_step_count = _workflow_step_count(
        partial_results.get("workflow_metrics") if isinstance(partial_results, dict) else None
    )
    match_title = f"{home_team} vs {away_team}"
    match_subject = " · ".join(
        [part for part in [config.get("tournament", ""), config.get("match_type", ""), config.get("stadium", "")] if part]
    )

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
                initial_metrics=initial_metrics,
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
                saved_id = _save_match_prep(config, results)
                _persist_usage_run_safe(
                    run_id=_new_usage_run_id("match-prep"),
                    source_type="match_prep",
                    source_id=saved_id,
                    workflow="match_preparation",
                    title=match_title,
                    subject=match_subject or match_title,
                    metrics=_slice_workflow_metrics(results.get("workflow_metrics"), baseline_step_count),
                )
            except Exception as e:
                print(f"[MatchPrep] Error saving to Supabase: {e}")
        except Exception as e:
            # Save whatever partial results we have so user can resume
            if partial_results:
                st.session_state.match_results = partial_results
                # Also persist partial results to Supabase
                try:
                    saved_id = _save_match_prep(config, partial_results)
                    _persist_usage_run_safe(
                        run_id=_new_usage_run_id("match-prep"),
                        source_type="match_prep",
                        source_id=saved_id,
                        workflow="match_preparation",
                        title=match_title,
                        subject=match_subject or match_title,
                        metrics=_slice_workflow_metrics(
                            partial_results.get("workflow_metrics"),
                            baseline_step_count,
                        ),
                    )
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
                    pdf_bytes, pdf_metrics = generate_match_pdf(
                        config, results, api_key=api_key,
                        progress_cb=lambda msg: pdf_status.write(msg),
                    )
                    st.session_state.match_pdf_bytes = pdf_bytes
                    _persist_usage_run_safe(
                        run_id=_new_usage_run_id("match-pdf"),
                        source_type="match_pdf_export",
                        source_id=str(st.session_state.get("match_prep_id", "") or ""),
                        workflow="match_pdf_export",
                        title=f"{home} vs {away}",
                        subject=pdf_filename,
                        metrics=pdf_metrics,
                        allow_empty=True,
                    )
                    pdf_status.update(label="✅ PDF generado", state="complete", expanded=False)
                    pdf_ok = True
                except Exception as e:
                    print(f"[MatchPrep] Error generating PDF:\n{traceback.format_exc()}")
                    pdf_status.update(label=f"❌ Error generando PDF: {e}", state="error")
            # Rerun OUTSIDE the status context so the download button appears
            if pdf_ok:
                st.rerun()

    _render_workflow_metrics(results.get("workflow_metrics"), "📈 Uso de tokens del informe")

    # ---- Team Histories (side-by-side) ----
    st.markdown("### 📊 Historial de Temporadas")

    col_h, col_a = st.columns(2)

    with col_h:
        st.markdown(
            f'<div class="team-col-hdr"><h3>🏠 {home}</h3></div>',
            unsafe_allow_html=True,
        )
        h_text, h_srcs, _ = _unpack_text_result(results.get("home_history"))
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
        a_text, a_srcs, _ = _unpack_text_result(results.get("away_history"))
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

    p_text, p_srcs, _ = _unpack_text_result(results.get("palomo_phrases"))
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
    baseline_step_count = _workflow_step_count(
        partial_results.get("workflow_metrics") if isinstance(partial_results, dict) else None
    )
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
                saved_id = _save_team_research(config, results)
                _persist_usage_run_safe(
                    run_id=_new_usage_run_id("team-research"),
                    source_type="team_research",
                    source_id=saved_id,
                    workflow="team_research",
                    title=team_name,
                    subject=config.get("tournament", "") or team_name,
                    metrics=_slice_workflow_metrics(results.get("workflow_metrics"), baseline_step_count),
                )
            except Exception as e:
                print(f"[TeamResearch] Error saving to Supabase: {e}")
        except Exception as e:
            if partial_results:
                st.session_state.team_research_results = partial_results
                try:
                    saved_id = _save_team_research(config, partial_results)
                    _persist_usage_run_safe(
                        run_id=_new_usage_run_id("team-research"),
                        source_type="team_research",
                        source_id=saved_id,
                        workflow="team_research",
                        title=team_name,
                        subject=config.get("tournament", "") or team_name,
                        metrics=_slice_workflow_metrics(
                            partial_results.get("workflow_metrics"),
                            baseline_step_count,
                        ),
                    )
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

    _render_workflow_metrics(results.get("workflow_metrics"), "📈 Uso de tokens de la investigación")

    # ---- Team History (full width) ----
    st.markdown("### 📊 Historial de Temporadas")
    h_text, h_srcs, _ = _unpack_text_result(results.get("team_history"))
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


    url_path: Optional[str] = None,
    if url_path is not None:
        kwargs["url_path"] = url_path
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
    baseline_step_count = _workflow_step_count(
        partial_results.get("workflow_metrics") if isinstance(partial_results, dict) else None
    )

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
                saved_id = _save_player_research(config, results)
                _persist_usage_run_safe(
                    run_id=_new_usage_run_id("player-research"),
                    source_type="player_research",
                    source_id=saved_id,
                    workflow="player_research",
                    title=player_name,
                    subject=" · ".join([part for part in [team_name, position] if part]) or player_name,
                    metrics=_slice_workflow_metrics(results.get("workflow_metrics"), baseline_step_count),
                )
            except Exception as e:
                print(f"[PlayerResearch] Error saving to Supabase: {e}")
        except Exception as e:
            if partial_results:
                st.session_state.player_research_results = partial_results
                try:
                    saved_id = _save_player_research(config, partial_results)
                    _persist_usage_run_safe(
                        run_id=_new_usage_run_id("player-research"),
                        source_type="player_research",
                        source_id=saved_id,
                        workflow="player_research",
                        title=player_name,
                        subject=" · ".join([part for part in [team_name, position] if part]) or player_name,
                        metrics=_slice_workflow_metrics(
                            partial_results.get("workflow_metrics"),
                            baseline_step_count,
                        ),
                    )
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

    _render_workflow_metrics(results.get("workflow_metrics"), "📈 Uso de tokens del dossier")

    # ---- Dossier (full width) ----
    st.markdown("### 📋 Dossier Completo")
    dossier_text, dossier_srcs, _ = _unpack_text_result(results.get("dossier"))
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
        _render_sel_team_tab(api_key)

    with tabs[SEL_TAB_PARTIDO]:
        _render_sel_match_tab(api_key)

    with tabs[SEL_TAB_CONVOCADO]:
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
        _run_sel_team_pipeline(st.session_state.nat_team_research_config, api_key)


def _run_sel_team_pipeline(config: dict, api_key: str, partial_results: Optional[Dict[str, Any]] = None) -> None:
    baseline_step_count = _workflow_step_count(
        partial_results.get("workflow_metrics") if isinstance(partial_results, dict) else None
    )
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
                saved_id = _save_national_team_research(config, results)
                _persist_usage_run_safe(
                    run_id=_new_usage_run_id("nat-team"),
                    source_type="national_team_research",
                    source_id=saved_id,
                    workflow="national_team_research",
                    title=str(config.get("country", "") or "Selección"),
                    subject=str(config.get("confederation", "") or config.get("country", "")),
                    metrics=_slice_workflow_metrics(results.get("workflow_metrics"), baseline_step_count),
                )
            except Exception as e:
                print(f"[Sel] Error saving national team: {e}")
        except Exception as e:
            if partial_results:
                st.session_state.nat_team_research_results = partial_results
                try:
                    saved_id = _save_national_team_research(config, partial_results)
                    _persist_usage_run_safe(
                        run_id=_new_usage_run_id("nat-team"),
                        source_type="national_team_research",
                        source_id=saved_id,
                        workflow="national_team_research",
                        title=str(config.get("country", "") or "Selección"),
                        subject=str(config.get("confederation", "") or config.get("country", "")),
                        metrics=_slice_workflow_metrics(
                            partial_results.get("workflow_metrics"),
                            baseline_step_count,
                        ),
                    )
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
    _render_workflow_metrics(results.get("workflow_metrics"), "📈 Uso de tokens de la investigación")
    st.markdown("### 📊 Historial de la Selección")
    h_text, h_srcs, _ = _unpack_text_result(results.get("team_history"))
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
        _run_sel_match_pipeline(st.session_state.nat_match_config, api_key)


def _run_sel_match_pipeline(config: dict, api_key: str, partial_results: Optional[Dict[str, Any]] = None) -> None:
    baseline_step_count = _workflow_step_count(
        partial_results.get("workflow_metrics") if isinstance(partial_results, dict) else None
    )
    title = f"{config.get('home_country', '?')} vs {config.get('away_country', '?')}"
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
                saved_id = _save_national_match_prep(config, results)
                _persist_usage_run_safe(
                    run_id=_new_usage_run_id("nat-match"),
                    source_type="national_match_prep",
                    source_id=saved_id,
                    workflow="national_match_prep",
                    title=title,
                    subject=" · ".join(
                        [part for part in [config.get("tournament", ""), config.get("match_type", "")] if part]
                    ) or title,
                    metrics=_slice_workflow_metrics(results.get("workflow_metrics"), baseline_step_count),
                )
            except Exception as e:
                print(f"[Sel] Error saving nat match: {e}")
        except Exception as e:
            if partial_results:
                st.session_state.nat_match_results = partial_results
                try:
                    saved_id = _save_national_match_prep(config, partial_results)
                    _persist_usage_run_safe(
                        run_id=_new_usage_run_id("nat-match"),
                        source_type="national_match_prep",
                        source_id=saved_id,
                        workflow="national_match_prep",
                        title=title,
                        subject=" · ".join(
                            [part for part in [config.get("tournament", ""), config.get("match_type", "")] if part]
                        ) or title,
                        metrics=_slice_workflow_metrics(
                            partial_results.get("workflow_metrics"),
                            baseline_step_count,
                        ),
                    )
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

    _render_workflow_metrics(results.get("workflow_metrics"), "📈 Uso de tokens del partido")

    # Main analysis
    st.markdown("### 📊 Análisis del Partido")
    h_text, h_srcs, _ = _unpack_text_result(results.get("home_history"))
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
        _run_sel_player_pipeline(st.session_state.nat_player_config, api_key)


def _run_sel_player_pipeline(config: dict, api_key: str, partial_results: Optional[Dict[str, Any]] = None) -> None:
    baseline_step_count = _workflow_step_count(
        partial_results.get("workflow_metrics") if isinstance(partial_results, dict) else None
    )
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
                saved_id = _save_national_player_research(config, results)
                _persist_usage_run_safe(
                    run_id=_new_usage_run_id("nat-player"),
                    source_type="national_player_research",
                    source_id=saved_id,
                    workflow="national_player_research",
                    title=str(config.get("player_name", "") or "Convocado"),
                    subject=str(config.get("country", "") or config.get("player_name", "")),
                    metrics=_slice_workflow_metrics(results.get("workflow_metrics"), baseline_step_count),
                )
            except Exception as e:
                print(f"[Sel] Error saving nat player: {e}")
        except Exception as e:
            if partial_results:
                st.session_state.nat_player_results = partial_results
                try:
                    saved_id = _save_national_player_research(config, partial_results)
                    _persist_usage_run_safe(
                        run_id=_new_usage_run_id("nat-player"),
                        source_type="national_player_research",
                        source_id=saved_id,
                        workflow="national_player_research",
                        title=str(config.get("player_name", "") or "Convocado"),
                        subject=str(config.get("country", "") or config.get("player_name", "")),
                        metrics=_slice_workflow_metrics(
                            partial_results.get("workflow_metrics"),
                            baseline_step_count,
                        ),
                    )
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
    _render_workflow_metrics(results.get("workflow_metrics"), "📈 Uso de tokens del dossier")
    st.markdown("### 📋 Dossier Internacional")
    dossier_text, dossier_srcs, _ = _unpack_text_result(results.get("dossier"))
    st.markdown(dossier_text)
    if dossier_srcs:
        with st.expander("📚 Fuentes"):
            st.markdown(_format_sources(dossier_srcs).replace(_CITATION_SEPARATOR, ""))


# ---------------------------------------------------------------------------
# Top-level mode wrappers — new symmetric Club / Selección design
# ---------------------------------------------------------------------------

def _render_club(api_key: str) -> None:
    """🏠 Investigar Club — three equal tabs: ⚽ Partido | 🔬 Equipo | 🧑 Jugador."""
    st.markdown(
        '<p class="hero-title">🏠 Investigar Club</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="hero-sub">'
        'Preparación de partido, investigación de equipo y dossier de jugadores — '
        'todo el análisis de fútbol de clubes en un solo lugar'
        '</p>',
        unsafe_allow_html=True,
    )

    tabs_dict = {
        TAB_PARTIDO: "⚽ Partido",
        TAB_EQUIPO: "🔬 Equipo",
        TAB_JUGADOR: "🧑 Jugador",
    }

    if "club_active_tab" not in st.session_state:
        st.session_state.club_active_tab = TAB_PARTIDO

    selected_tab = st.pills(
        "Nivel de Análisis",
        options=list(tabs_dict.keys()),
        format_func=lambda k: tabs_dict[k],
        key="club_active_tab",
        label_visibility="collapsed"
    )

    if selected_tab is None:
        selected_tab = TAB_PARTIDO

    if selected_tab == TAB_PARTIDO:
        _render_match_prep(api_key)
    elif selected_tab == TAB_EQUIPO:
        _render_match_research(api_key)
    elif selected_tab == TAB_JUGADOR:
        _render_player_research(api_key)


def _render_seleccion(api_key: str) -> None:
    """🌍 Investigar Selección — three equal tabs: ⚽ Partido | 🔬 Selección | 🧑 Convocado."""
    st.markdown(
        '<p class="hero-title">🌍 Investigar Selección</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="hero-sub">'
        'Partido de selecciones, historial de tu selección y dossier de convocados — '
        'el mismo nivel de análisis adaptado al fútbol internacional'
        '</p>',
        unsafe_allow_html=True,
    )

    tabs_dict = {
        TAB_PARTIDO: "⚽ Partido",
        TAB_EQUIPO: "🔬 Selección",
        TAB_JUGADOR: "🧑 Convocado",
    }

    if "sel_active_tab" not in st.session_state:
        st.session_state.sel_active_tab = TAB_PARTIDO

    selected_tab = st.pills(
        "Nivel de Análisis",
        options=list(tabs_dict.keys()),
        format_func=lambda k: tabs_dict[k],
        key="sel_active_tab",
        label_visibility="collapsed"
    )

    if selected_tab is None:
        selected_tab = TAB_PARTIDO

    if selected_tab == TAB_PARTIDO:
        _render_sel_match_tab(api_key)
    elif selected_tab == TAB_EQUIPO:
        _render_sel_team_tab(api_key)
    elif selected_tab == TAB_JUGADOR:
        _render_sel_player_tab(api_key)


def main() -> None:
    """Configure the app shell and route between the root app and hidden dashboard."""
    st.set_page_config(
        page_title="PalomoFacts · Football Intelligence",
        page_icon="⚽",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(_CUSTOM_CSS, unsafe_allow_html=True)

    if not hasattr(st, "Page") or not hasattr(st, "navigation"):
        st.warning(
            "This build of Streamlit does not support `st.Page` / `st.navigation`, "
            "so `/dashboard` routing will require a Streamlit upgrade."
        )
        _render_root_page()
        return

    root_page = _build_streamlit_page(
        _render_root_page,
        title="PalomoFacts",
        default=True,
    )
    dashboard_page = _build_streamlit_page(
        _render_dashboard_page,
        title="Usage Dashboard",
        url_path="dashboard",
        hidden=True,
    )

    navigation = st.navigation([root_page, dashboard_page], position="hidden")
    navigation.run()


if __name__ == "__main__":
    main()
