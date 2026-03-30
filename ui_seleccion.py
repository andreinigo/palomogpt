"""PalomoFacts — Selecciones (national teams) UI."""
from __future__ import annotations

import traceback
from typing import Any, Dict, Optional

import streamlit as st

from config import (
    CONFEDERATION_OPTIONS,
    MATCH_TYPE_OPTIONS,
    NATIONAL_TOURNAMENT_OPTIONS,
    TAB_EQUIPO,
    TAB_JUGADOR,
    TAB_PARTIDO,
)
from api import _CITATION_SEPARATOR, _format_sources
from database import (
    _persist_usage_run_safe,
    _save_national_match_prep,
    _save_national_player_research,
    _save_national_team_research,
)
from metrics import (
    _has_data,
    _new_usage_run_id,
    _slice_workflow_metrics,
    _unpack_text_result,
    _workflow_step_count,
)
from research import (
    _roster_has_failures,
    run_national_match_prep,
    run_national_player_research,
    run_national_team_research,
)
from ui_components import _render_formations, _render_roster_players, _render_workflow_metrics


# ---------------------------------------------------------------------------
# Tab 0 — Investigar Selección
# ---------------------------------------------------------------------------

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

    is_womens_sel = st.toggle(
        "♀️ Fútbol Femenino",
        key="sel_is_womens",
        help="Activa para investigar la selección femenina",
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
            "is_womens": is_womens_sel,
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

    coach_text, coach_srcs, _ = _unpack_text_result(results.get("coach"))
    if coach_text and not coach_text.startswith("❌"):
        st.markdown("---")
        st.markdown("### 🎯 Seleccionador")
        st.markdown(coach_text)
        if coach_srcs:
            with st.expander("📚 Fuentes"):
                st.markdown(_format_sources(coach_srcs).replace(_CITATION_SEPARATOR, ""))

    roster = results.get("roster", [])
    if roster:
        st.markdown("---")
        st.markdown("### 🎽 Convocatoria — Dossier por Jugador")
        _render_roster_players(roster, expand_key=f"sel_{config.get('country', 'sel')}")

    # Formations
    team_fm = results.get("formations", [])
    if team_fm:
        st.markdown("---")
        _render_formations(team_fm, team_label=country)


# ---------------------------------------------------------------------------
# Tab 1 — Partido de Selecciones
# ---------------------------------------------------------------------------

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

    is_womens_match = st.toggle(
        "♀️ Fútbol Femenino",
        key="sel_match_is_womens",
        help="Activa para preparar partido de selecciones femeninas",
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
            "is_womens": is_womens_match,
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
                is_womens=config.get("is_womens", False),
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

    st.markdown("### 📊 Análisis del Partido")
    h_text, h_srcs, _ = _unpack_text_result(results.get("home_history"))
    st.markdown(h_text)
    if h_srcs:
        with st.expander("📚 Fuentes"):
            st.markdown(_format_sources(h_srcs).replace(_CITATION_SEPARATOR, ""))

    home_roster = results.get("home_roster", [])
    if home_roster:
        st.markdown("---")
        st.markdown(f"### 🏠 Convocatoria de **{home}**")
        _render_roster_players(home_roster, expand_key=f"nmp_home_{config.get('home_country', 'h')}")

    away_roster = results.get("away_roster", [])
    if away_roster:
        st.markdown("---")
        st.markdown(f"### ✈️ Convocatoria de **{away}**")
        _render_roster_players(away_roster, expand_key=f"nmp_away_{config.get('away_country', 'a')}")

    # Formations
    home_fm = results.get("home_formations", [])
    away_fm = results.get("away_formations", [])
    if home_fm or away_fm:
        st.markdown("---")
        if home_fm:
            _render_formations(home_fm, team_label=home)
        if away_fm:
            _render_formations(away_fm, team_label=away)


# ---------------------------------------------------------------------------
# Tab 2 — Investigar Convocado
# ---------------------------------------------------------------------------

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
# Top-level Selección wrapper
# ---------------------------------------------------------------------------

def _render_seleccion(api_key: str) -> None:
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
        label_visibility="collapsed",
    )

    if selected_tab is None:
        selected_tab = TAB_PARTIDO

    if selected_tab == TAB_PARTIDO:
        _render_sel_match_tab(api_key)
    elif selected_tab == TAB_EQUIPO:
        _render_sel_team_tab(api_key)
    elif selected_tab == TAB_JUGADOR:
        _render_sel_player_tab(api_key)
