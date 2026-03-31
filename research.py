"""PalomoFacts — research pipelines (match prep, team, player, national)."""
from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional, Tuple

from google.genai import types

from config import CURRENT_DATE, CURRENT_YEAR, GEMINI_MODEL
from prompts import (
    COACH_DOSSIER_PROMPT,
    FOLLOW_UP_SYSTEM,
    MATCH_VALIDATION_PROMPT,
    NATIONAL_MATCH_PREP_PROMPT,
    NATIONAL_PLAYER_DOSSIER_PROMPT,
    NATIONAL_ROSTER_LIST_PROMPT,
    NATIONAL_TEAM_HISTORY_PROMPT,
    OPPONENT_CONNECTION_PROMPT,
    PALOMO_GPT_SYSTEM,
    PALOMO_PHRASES_PROMPT,
    SOLO_PLAYER_DOSSIER_PROMPT,
    TEAM_HISTORY_PROMPT,
    TEAM_ROSTER_LIST_PROMPT,
)
from metrics import (
    _add_token_usage,
    _empty_token_usage,
    _ensure_workflow_metrics,
    _has_data,
    _init_workflow_metrics,
    _merge_workflow_metrics,
    _record_workflow_step,
    _unpack_text_result,
)
from api import _gemini_request
from database import _find_existing_team_research


# ---------------------------------------------------------------------------
# Sofascore formation crawler helper
# ---------------------------------------------------------------------------

def _crawl_formations(team_name: str, limit: int = 5) -> list[dict]:
    """Crawl recent formations from Sofascore. Returns list of dicts with
    formation, players, and image_bytes.  Non-blocking: returns [] on any error."""
    import tempfile, shutil
    try:
        from sofascore_formations_crawler import crawl_team_lineups, MatchLineup
        from dataclasses import asdict
        from pathlib import Path
    except ImportError:
        return []
    tmp_dir = None
    try:
        tmp_dir = Path(tempfile.mkdtemp(prefix="formations_"))
        lineups = crawl_team_lineups(
            team_query=team_name,
            limit=limit,
            output_dir=tmp_dir,
            headless=True,
        )
        results: list[dict] = []
        for lu in lineups:
            entry = asdict(lu)
            img_path = Path(entry.get("image_path", ""))
            if img_path.exists():
                entry["image_bytes"] = img_path.read_bytes()
            else:
                entry["image_bytes"] = None
            entry.pop("image_path", None)
            entry.pop("raw_image_path", None)
            results.append(entry)
        return results
    except Exception as exc:
        print(f"[formations] Error crawling {team_name}: {exc}")
        return []
    finally:
        if tmp_dir and tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Position labels
# ---------------------------------------------------------------------------

_POS_LABELS = {"GK": "🧤 Portero", "DEF": "🛡️ Defensa", "MID": "🎯 Centrocampista", "FWD": "⚡ Delantero"}


# ---------------------------------------------------------------------------
# PalomoGPT response chain
# ---------------------------------------------------------------------------

def get_palomo_response(
    query: str,
    session_messages: list[dict],
    api_key: str,
) -> Tuple[str, List[Dict[str, str]], List[str], List[Dict[str, str]], Dict[str, Any]]:
    history: List[types.Content] = []
    for msg in session_messages[:-1]:
        role = "user" if msg["role"] == "user" else "model"
        content = msg.get("content", "")
        if content and not content.startswith("🧠"):
            history.append(
                types.Content(
                    role=role,
                    parts=[types.Part.from_text(text=content)],
                )
            )

    reasoning: List[str] = []
    workflow_metrics = _init_workflow_metrics("palomo_gpt")

    text, sources, tokens_main = _gemini_request(
        api_key=api_key,
        system_prompt=PALOMO_GPT_SYSTEM,
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

    followups: List[Dict[str, str]] = []
    current_answer = text
    for i in range(2):
        try:
            followup_q, _, tokens_fq = _gemini_request(
                api_key=api_key,
                system_prompt=FOLLOW_UP_SYSTEM,
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

            followup_a, extra_sources, tokens_fa = _gemini_request(
                api_key=api_key,
                system_prompt=PALOMO_GPT_SYSTEM,
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
            current_answer = followup_a

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
# Team history / coach / roster helpers
# ---------------------------------------------------------------------------

def _research_team_history(
    team_name: str,
    api_key: str,
) -> Tuple[str, List[Dict[str, str]], Dict[str, Any]]:
    season_prev = CURRENT_YEAR - 2
    season_curr = CURRENT_YEAR - 1
    season_next = CURRENT_YEAR

    prompt = TEAM_HISTORY_PROMPT.format(
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


def _research_coach(
    team_name: str,
    api_key: str,
    is_womens: bool = False,
) -> Tuple[str, List[Dict[str, str]], Dict[str, Any]]:
    womens_context = " (equipo femenino)" if is_womens else ""
    prompt = COACH_DOSSIER_PROMPT.format(
        team_name=team_name,
        womens_context=womens_context,
        current_date=CURRENT_DATE,
    )
    subject = f"equipo femenino de {team_name}" if is_womens else team_name
    is_national = "selección" in team_name.lower()
    if is_national:
        user_msg = (
            f"Dame el dossier COMPLETO del seleccionador ACTUAL de {subject} "
            f"a fecha {CURRENT_DATE}. Si hubo un cambio reciente de seleccionador, "
            "asegúrate de que sea el NUEVO, no el anterior. "
            "Incluye su carrera, récord con la selección, táctica, situación contractual y datos curiosos."
        )
    else:
        user_msg = (
            f"Dame el dossier COMPLETO del entrenador actual del {subject}. "
            "Incluye su carrera, récord en el club, táctica, situación contractual y datos curiosos."
        )
    return _gemini_request(
        api_key=api_key,
        system_prompt=prompt,
        user_message=user_msg,
    )


def _fetch_player_list(
    team_name: str,
    api_key: str,
    _retries: int = 2,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    season_curr = CURRENT_YEAR - 1
    season_next = CURRENT_YEAR

    prompt = TEAM_ROSTER_LIST_PROMPT.format(
        team_name=team_name,
        season_curr=season_curr,
        season_next=season_next,
        current_date=CURRENT_DATE,
    )

    last_err: Optional[Exception] = None
    for attempt in range(_retries):
        try:
            text, _, tokens = _gemini_request(
                api_key=api_key,
                system_prompt=prompt,
                user_message=(
                    f"Dame la plantilla completa actual de {team_name} para la temporada "
                    f"{season_curr}/{season_next}. Solo el JSON, nada más."
                ),
            )

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
        except Exception as e:
            last_err = e
            if attempt < _retries - 1:
                print(f"[Roster] _fetch_player_list attempt {attempt+1} failed for {team_name}: {e}  — retrying...")
                time.sleep(2)
    raise last_err  # type: ignore[misc]


def _fetch_national_player_list(
    country: str,
    api_key: str,
    _retries: int = 2,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    prompt = NATIONAL_ROSTER_LIST_PROMPT.format(
        country=country,
        current_date=CURRENT_DATE,
    )

    last_err: Optional[Exception] = None
    for attempt in range(_retries):
        try:
            text, _, tokens = _gemini_request(
                api_key=api_key,
                system_prompt=prompt,
                user_message=(
                    f"Dame la ÚLTIMA CONVOCATORIA OFICIAL publicada de la selección de {country}. "
                    f"Necesito la lista real publicada por la federación, NO una lista estimada. "
                    f"Incluye el nombre del seleccionador ACTUAL y para qué ventana/torneo fue. "
                    f"Fecha de hoy: {CURRENT_DATE}. Solo el JSON, nada más."
                ),
            )

            json_match = re.search(r'\{[\s\S]*\}', text)
            if not json_match:
                raise RuntimeError(f"Could not parse national roster JSON from response: {text[:300]}")
            try:
                data = json.loads(json_match.group())
            except json.JSONDecodeError as e:
                raise RuntimeError(f"Invalid JSON in national roster: {e}\n{text[:300]}")

            players = data.get("players", [])
            if not players:
                raise RuntimeError(f"Empty convocatoria returned for {country}")
            return players, tokens
        except Exception as e:
            last_err = e
            if attempt < _retries - 1:
                print(f"[Roster] _fetch_national_player_list attempt {attempt+1} failed for {country}: {e}  — retrying...")
                time.sleep(2)
    raise last_err  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Single player research (with opponent context)
# ---------------------------------------------------------------------------

def _research_single_player(
    player: Dict[str, Any],
    team_name: str,
    opponent_name: str,
    api_key: str,
) -> Dict[str, Any]:
    base = _research_single_player_solo(player, team_name, api_key)

    player_name = base["name"]
    position = base["position"]
    pos_label = _POS_LABELS.get(position, position)

    conn_prompt = OPPONENT_CONNECTION_PROMPT.format(
        player_name=player_name,
        player_position=pos_label,
        team_name=team_name,
        opponent_name=opponent_name,
        current_date=CURRENT_DATE,
    )

    try:
        conn_text, conn_sources, conn_tokens = _gemini_request(
            api_key=api_key,
            system_prompt=conn_prompt,
            user_message=(
                f"Investiga TODAS las conexiones de {player_name} con {opponent_name}. "
                "Incluye historial en ese club, partidos contra ellos, vínculos personales, "
                "compañeros de selección y cualquier otra relación."
            ),
        )
    except Exception as e:
        conn_text = f"⚡ No se pudo investigar conexiones con {opponent_name}: {e}"
        conn_sources = []
        conn_tokens = _empty_token_usage()

    merged_tokens = _empty_token_usage()
    for key in ("input_tokens", "output_tokens", "total_tokens", "grounding_requests"):
        merged_tokens[key] = base.get("tokens", {}).get(key, 0) + conn_tokens.get(key, 0)

    return {
        "name": player_name,
        "position": position,
        "number": base.get("number", ""),
        "text": base["text"],
        "opponent_text": conn_text,
        "sources": base.get("sources", []) + conn_sources,
        "tokens": merged_tokens,
    }


def _research_single_player_solo(
    player: Dict[str, Any],
    team_name: str,
    api_key: str,
) -> Dict[str, Any]:
    player_name = player.get("name", player.get("full_name", "Unknown"))
    position = player.get("position", "")
    pos_label = _POS_LABELS.get(position, position)

    prompt = SOLO_PLAYER_DOSSIER_PROMPT.format(
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


# ---------------------------------------------------------------------------
# Roster research
# ---------------------------------------------------------------------------

_PLAYER_FUTURE_TIMEOUT = 300  # 5 min max per player


def _research_team_roster(
    team_name: str,
    opponent_name: str,
    api_key: str,
    progress_cb: Optional[Callable[[str], None]] = None,
    workflow_metrics: Optional[Dict[str, Any]] = None,
    step_prefix: str = "roster",
    on_batch_complete: Optional[Callable[[List[Dict[str, Any]]], None]] = None,
) -> List[Dict[str, Any]]:
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
                    results[idx] = future.result(timeout=_PLAYER_FUTURE_TIMEOUT)
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
        # Save after each batch so partial roster survives crashes
        if on_batch_complete:
            on_batch_complete([r for r in results if r is not None])

    return results


def _research_team_roster_solo(
    team_name: str,
    api_key: str,
    progress_cb: Optional[Callable[[str], None]] = None,
    workflow_metrics: Optional[Dict[str, Any]] = None,
    step_prefix: str = "roster",
    on_batch_complete: Optional[Callable[[List[Dict[str, Any]]], None]] = None,
) -> List[Dict[str, Any]]:
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
                    results[idx] = future.result(timeout=_PLAYER_FUTURE_TIMEOUT)
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
        if on_batch_complete:
            on_batch_complete([r for r in results if r is not None])

    return results


# ---------------------------------------------------------------------------
# Palomo phrases
# ---------------------------------------------------------------------------

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
    max_ctx = 4000
    home_ctx = home_context[:max_ctx] if len(home_context) > max_ctx else home_context
    away_ctx = away_context[:max_ctx] if len(away_context) > max_ctx else away_context

    prompt = PALOMO_PHRASES_PROMPT.format(
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


# ---------------------------------------------------------------------------
# Roster failure handling
# ---------------------------------------------------------------------------

def _roster_has_failures(roster: list) -> bool:
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


# ---------------------------------------------------------------------------
# Match preparation pipeline
# ---------------------------------------------------------------------------

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
    is_womens: bool = False,
    on_phase_complete: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    _cb = progress_cb or (lambda _msg: None)
    _save = on_phase_complete or (lambda _r: None)
    results: Dict[str, Any] = partial_results or {
        "home_history": ("", []),
        "away_history": ("", []),
        "home_coach": ("", []),
        "away_coach": ("", []),
        "home_roster": [],
        "away_roster": [],
        "palomo_phrases": ("", []),
    }
    workflow_metrics = _ensure_workflow_metrics(results, "match_preparation")
    if initial_metrics:
        _merge_workflow_metrics(workflow_metrics, initial_metrics)

    womens_ctx = " (equipo femenino)" if is_womens else ""
    research_home = f"{home_team}{womens_ctx}"
    research_away = f"{away_team}{womens_ctx}"

    if not partial_results:
        for side, team in [("home", research_home), ("away", research_away)]:
            try:
                existing = _find_existing_team_research(team)
                if existing:
                    if not _has_data(results.get(f"{side}_history")):
                        results[f"{side}_history"] = existing.get("team_history", ("", []))
                    _cb(f"♻️ Reutilizando historial de **{team}** (plantel y DT se investigan de nuevo)")
            except Exception as e:
                print(f"[MatchPrep] Error pre-loading team data for {team}: {e}")

    # Phase 1: team histories
    need_home_hist = not _has_data(results.get("home_history"))
    need_away_hist = not _has_data(results.get("away_history"))
    if need_home_hist or need_away_hist:
        _cb(f"📊 Investigando historial de **{research_home}** y **{research_away}**...")
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {}
            if need_home_hist:
                futures[executor.submit(_research_team_history, research_home, api_key)] = "home_history"
            if need_away_hist:
                futures[executor.submit(_research_team_history, research_away, api_key)] = "away_history"
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
    _save(results)

    # Phase 1.5: coach research
    need_home_coach = not _has_data(results.get("home_coach"))
    need_away_coach = not _has_data(results.get("away_coach"))
    if need_home_coach or need_away_coach:
        _cb(f"🎯 Investigando entrenadores de **{research_home}** y **{research_away}**...")
        with ThreadPoolExecutor(max_workers=2) as executor:
            coach_futures = {}
            if need_home_coach:
                coach_futures[executor.submit(_research_coach, research_home, api_key, is_womens=is_womens)] = "home_coach"
            if need_away_coach:
                coach_futures[executor.submit(_research_coach, research_away, api_key, is_womens=is_womens)] = "away_coach"
            for future in as_completed(coach_futures):
                key = coach_futures[future]
                try:
                    results[key] = future.result()
                    label = home_team if key == "home_coach" else away_team
                    _, _, tokens = _unpack_text_result(results[key])
                    _record_workflow_step(
                        workflow_metrics,
                        f"match_prep.{key}",
                        f"{label}: entrenador",
                        tokens,
                    )
                except Exception as e:
                    results[key] = (f"❌ Error investigando entrenador: {e}", [], _empty_token_usage())
        _cb("✅ Entrenadores completados.")
    else:
        _cb("✅ Entrenadores ya disponibles — reutilizando.")
    _save(results)

    # Phase 2: rosters
    if not _has_data(results.get("home_roster")):
        _cb(f"👥 Investigando plantilla de **{research_home}** jugador por jugador...")
        def _on_home_batch(partial_roster: list) -> None:
            results["home_roster"] = partial_roster
            _save(results)
        try:
            results["home_roster"] = _research_team_roster(
                research_home, research_away, api_key,
                progress_cb=_cb, workflow_metrics=workflow_metrics,
                step_prefix="match_prep.home_roster",
                on_batch_complete=_on_home_batch,
            )
        except Exception as e:
            results["home_roster"] = []
            _cb(f"❌ Error con plantilla de {home_team}: {e}")
    elif _roster_has_failures(results.get("home_roster", [])):
        results["home_roster"] = _retry_failed_roster_players(
            results["home_roster"], research_home, research_away, api_key,
            progress_cb=_cb, workflow_metrics=workflow_metrics,
            step_prefix="match_prep.home_roster",
        )
    else:
        _cb(f"✅ Plantilla de **{home_team}** ya disponible — reutilizando.")
    _save(results)

    if not _has_data(results.get("away_roster")):
        _cb(f"👥 Investigando plantilla de **{research_away}** jugador por jugador...")
        def _on_away_batch(partial_roster: list) -> None:
            results["away_roster"] = partial_roster
            _save(results)
        try:
            results["away_roster"] = _research_team_roster(
                research_away, research_home, api_key,
                progress_cb=_cb, workflow_metrics=workflow_metrics,
                step_prefix="match_prep.away_roster",
                on_batch_complete=_on_away_batch,
            )
        except Exception as e:
            results["away_roster"] = []
            _cb(f"❌ Error con plantilla de {research_away}: {e}")
    elif _roster_has_failures(results.get("away_roster", [])):
        results["away_roster"] = _retry_failed_roster_players(
            results["away_roster"], research_away, research_home, api_key,
            progress_cb=_cb, workflow_metrics=workflow_metrics,
            step_prefix="match_prep.away_roster",
        )
    else:
        _cb(f"✅ Plantilla de **{research_away}** ya disponible — reutilizando.")
    _save(results)

    # Phase 3: Palomo phrases
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
    _save(results)

    # --- Formations (Sofascore) ---
    if not results.get("home_formations"):
        _cb(f"⚽ Buscando formaciones recientes de **{home_team}**…")
        results["home_formations"] = _crawl_formations(home_team, limit=5)
    if not results.get("away_formations"):
        _cb(f"⚽ Buscando formaciones recientes de **{away_team}**…")
        results["away_formations"] = _crawl_formations(away_team, limit=5)

    return results


# ---------------------------------------------------------------------------
# Team Research pipeline
# ---------------------------------------------------------------------------

def run_team_research(
    team_name: str,
    tournament: str,
    api_key: str,
    progress_cb: Optional[Callable[[str], None]] = None,
    partial_results: Optional[Dict[str, Any]] = None,
    is_womens: bool = False,
    on_phase_complete: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    _cb = progress_cb or (lambda _msg: None)
    _save = on_phase_complete or (lambda _r: None)
    results: Dict[str, Any] = partial_results or {
        "team_history": ("", []),
        "coach": ("", []),
        "roster": [],
    }
    workflow_metrics = _ensure_workflow_metrics(results, "team_research")
    womens_ctx = " (equipo femenino)" if is_womens else ""
    research_name = f"{team_name}{womens_ctx}"

    if not _has_data(results.get("team_history")):
        _cb(f"📊 Investigando historial de **{research_name}**...")
        try:
            results["team_history"] = _research_team_history(research_name, api_key)
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
    _save(results)

    if not _has_data(results.get("coach")):
        _cb(f"🎯 Investigando entrenador de **{research_name}**...")
        try:
            results["coach"] = _research_coach(research_name, api_key, is_womens=is_womens)
            _, _, tokens = _unpack_text_result(results["coach"])
            _record_workflow_step(
                workflow_metrics,
                "team_research.coach",
                "Entrenador del equipo",
                tokens,
                entity=team_name,
            )
        except Exception as e:
            results["coach"] = (f"❌ Error investigando entrenador: {e}", [], _empty_token_usage())
        _cb("✅ Entrenador completado.")
    else:
        _cb("✅ Entrenador ya disponible — reutilizando.")
    _save(results)

    if not _has_data(results.get("roster")):
        _cb(f"👥 Investigando plantilla de **{research_name}** jugador por jugador...")
        def _on_roster_batch(partial_roster: list) -> None:
            results["roster"] = partial_roster
            _save(results)
        try:
            results["roster"] = _research_team_roster_solo(
                research_name, api_key,
                progress_cb=_cb, workflow_metrics=workflow_metrics,
                step_prefix="team_research.roster",
                on_batch_complete=_on_roster_batch,
            )
        except Exception as e:
            results["roster"] = []
            _cb(f"❌ Error con plantilla de {team_name}: {e}")
    elif _roster_has_failures(results.get("roster", [])):
        results["roster"] = _retry_failed_roster_players(
            results["roster"], research_name, "", api_key,
            progress_cb=_cb, workflow_metrics=workflow_metrics,
            step_prefix="team_research.roster",
        )
    else:
        _cb(f"✅ Plantilla de **{research_name}** ya disponible — reutilizando.")
    _save(results)

    # --- Formations (Sofascore) ---
    if not results.get("formations"):
        _cb(f"⚽ Buscando formaciones recientes de **{research_name}**…")
        results["formations"] = _crawl_formations(research_name, limit=5)

    return results


# ---------------------------------------------------------------------------
# Player Research pipeline
# ---------------------------------------------------------------------------

def run_player_research(
    player_name: str,
    team_name: str,
    position: str,
    api_key: str,
    progress_cb: Optional[Callable[[str], None]] = None,
    partial_results: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    _cb = progress_cb or (lambda _msg: None)
    results: Dict[str, Any] = partial_results or {"dossier": ("", [])}
    workflow_metrics = _ensure_workflow_metrics(results, "player_research")

    if not _has_data(results.get("dossier")):
        pos_label = _POS_LABELS.get(position, position or "Jugador")
        _cb(f"🔍 Investigando dossier de **{player_name}** ({pos_label}, {team_name})...")

        prompt = SOLO_PLAYER_DOSSIER_PROMPT.format(
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
# Selecciones pipelines
# ---------------------------------------------------------------------------

def _research_national_player_solo(
    player: Dict[str, Any],
    country: str,
    api_key: str,
) -> Dict[str, Any]:
    player_name = player.get("name", player.get("full_name", "Unknown"))
    prompt = NATIONAL_PLAYER_DOSSIER_PROMPT.format(
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
    on_batch_complete: Optional[Callable[[List[Dict[str, Any]]], None]] = None,
) -> List[Dict[str, Any]]:
    if progress_cb:
        progress_cb(f"📋 Obteniendo última convocatoria de **{country}**...")
    players, list_tokens = _fetch_national_player_list(country, api_key)
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
                    results[idx] = future.result(timeout=_PLAYER_FUTURE_TIMEOUT)
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
        if on_batch_complete:
            on_batch_complete([r for r in results if r is not None])

    return results


def run_national_team_research(
    country: str,
    confederation: str,
    api_key: str,
    progress_cb: Optional[Callable[[str], None]] = None,
    partial_results: Optional[Dict[str, Any]] = None,
    on_phase_complete: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    _cb = progress_cb or (lambda _msg: None)
    _save = on_phase_complete or (lambda _r: None)
    results: Dict[str, Any] = partial_results or {
        "team_history": ("", []),
        "coach": ("", []),
        "roster": [],
    }
    workflow_metrics = _ensure_workflow_metrics(results, "national_team_research")

    if not _has_data(results.get("team_history")):
        _cb(f"📊 Investigando historial de **{country}**...")
        conf_label = confederation if confederation and "(Cualquier" not in confederation else "Internacional"
        prompt = NATIONAL_TEAM_HISTORY_PROMPT.format(
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
    _save(results)

    if not _has_data(results.get("coach")):
        _cb(f"🎯 Investigando seleccionador de **{country}**...")
        try:
            results["coach"] = _research_coach(f"selección de {country}", api_key)
            _, _, tokens = _unpack_text_result(results["coach"])
            _record_workflow_step(
                workflow_metrics,
                "national_team_research.coach",
                "Seleccionador",
                tokens,
                entity=country,
            )
        except Exception as e:
            results["coach"] = (f"❌ Error investigando seleccionador: {e}", [], _empty_token_usage())
        _cb("✅ Seleccionador completado.")
    else:
        _cb("✅ Seleccionador ya disponible — reutilizando.")
    _save(results)

    if not _has_data(results.get("roster")):
        _cb(f"👥 Investigando convocatoria de **{country}**...")
        def _on_nat_roster_batch(partial_roster: list) -> None:
            results["roster"] = partial_roster
            _save(results)
        try:
            results["roster"] = _research_national_roster(
                country, api_key,
                progress_cb=_cb, workflow_metrics=workflow_metrics,
                step_prefix="national_team_research.roster",
                on_batch_complete=_on_nat_roster_batch,
            )
        except Exception as e:
            results["roster"] = []
            _cb(f"❌ Error con convocatoria de {country}: {e}")
    elif _roster_has_failures(results.get("roster", [])):
        results["roster"] = _retry_failed_roster_players(
            results["roster"], country, "", api_key,
            progress_cb=_cb,
            research_fn=lambda player: _research_national_player_solo(player, country, api_key),
            workflow_metrics=workflow_metrics,
            step_prefix="national_team_research.roster",
        )
    else:
        _cb(f"✅ Convocatoria de **{country}** ya disponible — reutilizando.")
    _save(results)

    # --- Formations (Sofascore) ---
    if not results.get("formations"):
        _cb(f"⚽ Buscando formaciones recientes de **{country}**…")
        results["formations"] = _crawl_formations(country, limit=5)

    return results


def run_national_match_prep(
    home_country: str,
    away_country: str,
    tournament: str,
    match_type: str,
    api_key: str,
    progress_cb: Optional[Callable[[str], None]] = None,
    partial_results: Optional[Dict[str, Any]] = None,
    is_womens: bool = False,
    on_phase_complete: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    _cb = progress_cb or (lambda _msg: None)
    _save = on_phase_complete or (lambda _r: None)
    results: Dict[str, Any] = partial_results or {
        "home_history": ("", []),
        "away_history": ("", []),
        "home_roster": [],
        "away_roster": [],
        "palomo_phrases": ("", []),
    }
    workflow_metrics = _ensure_workflow_metrics(results, "national_match_prep")

    womens_ctx = " (selección femenina)" if is_womens else ""
    research_home = f"{home_country}{womens_ctx}"
    research_away = f"{away_country}{womens_ctx}"

    match_prompt = NATIONAL_MATCH_PREP_PROMPT.format(
        home_country=home_country,
        away_country=away_country,
        tournament=tournament,
        match_type=match_type,
        current_date=CURRENT_DATE,
    )

    if not _has_data(results.get("home_history")):
        _cb(f"📊 Analizando partido **{research_home} vs {research_away}**...")
        try:
            results["home_history"] = _gemini_request(
                api_key=api_key,
                system_prompt=match_prompt,
                user_message=(
                    f"Crea la preparación COMPLETA para {research_home} vs {research_away} "
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
                entity=f"{research_home} vs {research_away}",
            )
        except Exception as e:
            results["home_history"] = (f"❌ Error en análisis: {e}", [], _empty_token_usage())
        _cb("✅ Análisis del partido completado.")
    else:
        _cb("✅ Análisis ya disponible — reutilizando.")
    _save(results)

    if not _has_data(results.get("home_roster")):
        _cb(f"👥 Investigando convocatoria de **{research_home}**...")
        def _on_nat_home_batch(partial_roster: list) -> None:
            results["home_roster"] = partial_roster
            _save(results)
        try:
            results["home_roster"] = _research_national_roster(
                research_home, api_key,
                progress_cb=_cb, workflow_metrics=workflow_metrics,
                step_prefix="national_match_prep.home_roster",
                on_batch_complete=_on_nat_home_batch,
            )
        except Exception as e:
            results["home_roster"] = []
            _cb(f"❌ Error con convocatoria de {research_home}: {e}")
    elif _roster_has_failures(results.get("home_roster", [])):
        results["home_roster"] = _retry_failed_roster_players(
            results["home_roster"], research_home, research_away, api_key,
            progress_cb=_cb,
            research_fn=lambda player: _research_national_player_solo(player, research_home, api_key),
            workflow_metrics=workflow_metrics,
            step_prefix="national_match_prep.home_roster",
        )
    else:
        _cb(f"✅ Convocatoria de **{research_home}** ya disponible — reutilizando.")
    _save(results)

    if not _has_data(results.get("away_roster")):
        _cb(f"👥 Investigando convocatoria de **{research_away}**...")
        def _on_nat_away_batch(partial_roster: list) -> None:
            results["away_roster"] = partial_roster
            _save(results)
        try:
            results["away_roster"] = _research_national_roster(
                research_away, api_key,
                progress_cb=_cb, workflow_metrics=workflow_metrics,
                step_prefix="national_match_prep.away_roster",
                on_batch_complete=_on_nat_away_batch,
            )
        except Exception as e:
            results["away_roster"] = []
            _cb(f"❌ Error con convocatoria de {research_away}: {e}")
    elif _roster_has_failures(results.get("away_roster", [])):
        results["away_roster"] = _retry_failed_roster_players(
            results["away_roster"], research_away, research_home, api_key,
            progress_cb=_cb,
            research_fn=lambda player: _research_national_player_solo(player, research_away, api_key),
            workflow_metrics=workflow_metrics,
            step_prefix="national_match_prep.away_roster",
        )
    else:
        _cb(f"✅ Convocatoria de **{research_away}** ya disponible — reutilizando.")
    _save(results)

    # --- Formations (Sofascore) ---
    if not results.get("home_formations"):
        _cb(f"⚽ Buscando formaciones recientes de **{research_home}**…")
        results["home_formations"] = _crawl_formations(research_home, limit=5)
    if not results.get("away_formations"):
        _cb(f"⚽ Buscando formaciones recientes de **{research_away}**…")
        results["away_formations"] = _crawl_formations(research_away, limit=5)

    return results


def run_national_player_research(
    player_name: str,
    country: str,
    api_key: str,
    progress_cb: Optional[Callable[[str], None]] = None,
    partial_results: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    _cb = progress_cb or (lambda _msg: None)
    results: Dict[str, Any] = partial_results or {"dossier": ("", [])}
    workflow_metrics = _ensure_workflow_metrics(results, "national_player_research")

    if not _has_data(results.get("dossier")):
        _cb(f"🔍 Investigando dossier de **{player_name}** (selección de {country})...")
        prompt = NATIONAL_PLAYER_DOSSIER_PROMPT.format(
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
