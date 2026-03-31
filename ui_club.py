"""PalomoFacts — Club mode UI (match prep, team research, player research)."""
from __future__ import annotations

import json
import re
import traceback
from typing import Any, Callable, Dict, List, Optional

import streamlit as st

from config import (
    CURRENT_DATE,
    MATCH_TYPE_OPTIONS,
    TAB_EQUIPO,
    TAB_JUGADOR,
    TAB_PARTIDO,
    TOURNAMENT_OPTIONS,
)
from api import _CITATION_SEPARATOR, _format_sources
from database import (
    _auto_save_team_from_match,
    _persist_usage_run_safe,
    _save_match_prep,
    _save_player_research,
    _save_team_research,
)
from metrics import (
    _has_data,
    _init_workflow_metrics,
    _new_usage_run_id,
    _record_workflow_step,
    _slice_workflow_metrics,
    _unpack_text_result,
    _workflow_step_count,
)
from api import _gemini_request
from pdf_gen import generate_match_pdf
from prompts import MATCH_VALIDATION_PROMPT
from research import (
    _POS_LABELS,
    _roster_has_failures,
    fill_roster_gaps,
    run_match_preparation,
    run_player_research,
    run_team_research,
)
from ui_components import _render_formations, _render_roster_players, _render_workflow_metrics


# ---------------------------------------------------------------------------
# Match Prep
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

    if st.session_state.get("match_results") and st.session_state.get("match_config"):
        existing = st.session_state.match_results
        config = st.session_state.match_config

        all_keys = ["home_history", "away_history", "home_roster", "away_roster", "palomo_phrases"]
        missing = [k for k in all_keys if not _has_data(existing.get(k))]
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

    # --- Match configuration form ---
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

    is_womens = st.toggle(
        "♀️ Fútbol Femenino",
        key="mp_is_womens",
        help="Activa para investigar equipos femeninos",
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

        with st.status("🔍 Validando equipos y partido...", expanded=True) as val_status:
            validation_metrics = _init_workflow_metrics("match_preparation")
            try:
                val_prompt = MATCH_VALIDATION_PROMPT.format(
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

                json_match = re.search(r'\{[\s\S]*\}', val_text)
                if json_match:
                    val_data = json.loads(json_match.group())
                    if not val_data.get("valid", True):
                        reason = val_data.get("reason", "Partido no reconocido.")
                        val_status.update(label=f"❌ {reason}", state="error")
                        st.error(f"⚠️ {reason}")
                        return
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

        st.session_state.match_config = {
            "home_team": home_team,
            "away_team": away_team,
            "tournament": tournament,
            "match_type": match_type,
            "stadium": stadium,
            "is_womens": is_womens,
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

        def _on_phase(partial: dict) -> None:
            st.session_state.match_results = partial
            try:
                _save_match_prep(config, partial)
            except Exception:
                pass

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
                is_womens=config.get("is_womens", False),
                on_phase_complete=_on_phase,
            )
            st.session_state.match_results = results
            st.session_state.pop("match_pdf_bytes", None)

            status.update(
                label="✅ ¡Informe completo!",
                state="complete",
                expanded=False,
            )

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

            try:
                tournament = config.get("tournament", "")
                _auto_save_team_from_match(home_team, tournament, results, "home")
                _auto_save_team_from_match(away_team, tournament, results, "away")
            except Exception as e:
                print(f"[MatchPrep] Error auto-saving team data: {e}")
        except Exception as e:
            error_results = st.session_state.get("match_results") or partial_results
            if error_results:
                st.session_state.match_results = error_results
                try:
                    saved_id = _save_match_prep(config, error_results)
                    _persist_usage_run_safe(
                        run_id=_new_usage_run_id("match-prep"),
                        source_type="match_prep",
                        source_id=saved_id,
                        workflow="match_preparation",
                        title=match_title,
                        subject=match_subject or match_title,
                        metrics=_slice_workflow_metrics(
                            error_results.get("workflow_metrics"),
                            baseline_step_count,
                        ),
                    )
                except Exception:
                    pass
            status.update(label=f"❌ Error: {e} — puedes continuar el análisis", state="error")
            print(f"[MatchPrep] Error:\n{traceback.format_exc()}")

    st.rerun()


def _display_match_results(config: dict, results: dict) -> None:
    home = config["home_team"]
    away = config["away_team"]
    tournament = config["tournament"]
    match_type = config["match_type"]
    stadium = config["stadium"]

    st.markdown("---")

    st.markdown(
        f'<div class="match-header">'
        f'<h2>{home}  🆚  {away}</h2>'
        f'<div class="match-meta">'
        f'{match_type} · {tournament} · 🏟️ {stadium}'
        f'</div></div>',
        unsafe_allow_html=True,
    )

    # PDF download
    safe_home = re.sub(r'[^\w\s-]', '', home).strip().replace(' ', '_')
    safe_away = re.sub(r'[^\w\s-]', '', away).strip().replace(' ', '_')
    pdf_filename = f"Preparacion_{safe_home}_vs_{safe_away}.pdf"

    if st.session_state.get("match_pdf_bytes"):
        st.download_button(
            label="📥 Descargar PDF del Informe",
            data=st.session_state.match_pdf_bytes,
            file_name=pdf_filename,
            mime="application/pdf",
            use_container_width=True,
        )
    else:
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
            if pdf_ok:
                st.rerun()

    _render_workflow_metrics(results.get("workflow_metrics"), "📈 Uso de tokens del informe")

    # Team Histories
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
                st.markdown(_format_sources(h_srcs).replace(_CITATION_SEPARATOR, ""))

    with col_a:
        st.markdown(
            f'<div class="team-col-hdr"><h3>✈️ {away}</h3></div>',
            unsafe_allow_html=True,
        )
        a_text, a_srcs, _ = _unpack_text_result(results.get("away_history"))
        st.markdown(a_text)
        if a_srcs:
            with st.expander("📚 Fuentes"):
                st.markdown(_format_sources(a_srcs).replace(_CITATION_SEPARATOR, ""))

    # Coaches
    home_coach_text, home_coach_srcs, _ = _unpack_text_result(results.get("home_coach"))
    away_coach_text, away_coach_srcs, _ = _unpack_text_result(results.get("away_coach"))
    if home_coach_text or away_coach_text:
        st.markdown("---")
        st.markdown("### 🎯 Entrenadores")
        col_hc, col_ac = st.columns(2)
        with col_hc:
            st.markdown(
                f'<div class="team-col-hdr"><h3>🏠 {home}</h3></div>',
                unsafe_allow_html=True,
            )
            st.markdown(home_coach_text)
            if home_coach_srcs:
                with st.expander("📚 Fuentes"):
                    st.markdown(_format_sources(home_coach_srcs).replace(_CITATION_SEPARATOR, ""))
        with col_ac:
            st.markdown(
                f'<div class="team-col-hdr"><h3>✈️ {away}</h3></div>',
                unsafe_allow_html=True,
            )
            st.markdown(away_coach_text)
            if away_coach_srcs:
                with st.expander("📚 Fuentes"):
                    st.markdown(_format_sources(away_coach_srcs).replace(_CITATION_SEPARATOR, ""))

    # Rosters
    st.markdown("---")
    st.markdown("### 👥 Plantillas — Dossier por Jugador")
    col_hr, col_ar = st.columns(2)

    with col_hr:
        st.markdown(
            f'<div class="team-col-hdr"><h3>🏠 {home}</h3></div>',
            unsafe_allow_html=True,
        )
        _render_roster_players(results.get("home_roster", []), expand_key=f"mp_home_{config.get('home_team', 'h')}")

    with col_ar:
        st.markdown(
            f'<div class="team-col-hdr"><h3>✈️ {away}</h3></div>',
            unsafe_allow_html=True,
        )
        _render_roster_players(results.get("away_roster", []), expand_key=f"mp_away_{config.get('away_team', 'a')}")

    # Fill roster gaps buttons
    col_btn_h, col_btn_a = st.columns(2)
    with col_btn_h:
        if st.button("🔍 Completar plantilla local", key="fill_home_roster", use_container_width=True):
            _fill_match_roster_gaps(config, results, "home", st.secrets.get("GEMINI_API_KEY", ""))
            return
    with col_btn_a:
        if st.button("🔍 Completar plantilla visitante", key="fill_away_roster", use_container_width=True):
            _fill_match_roster_gaps(config, results, "away", st.secrets.get("GEMINI_API_KEY", ""))
            return

    # Palomo Phrases
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
            st.markdown(_format_sources(p_srcs).replace(_CITATION_SEPARATOR, ""))

    # Formations
    home_fm = results.get("home_formations", [])
    away_fm = results.get("away_formations", [])
    if home_fm or away_fm:
        st.markdown("---")
        if home_fm:
            _render_formations(home_fm, team_label=home)
        if away_fm:
            _render_formations(away_fm, team_label=away)


def _fill_match_roster_gaps(config: dict, results: dict, side: str, api_key: str) -> None:
    """Fill missing player dossiers for one side of a match prep."""
    roster_key = f"{side}_roster"
    if side == "home":
        team = config.get("home_team", "?")
        opponent = config.get("away_team", "?")
    else:
        team = config.get("away_team", "?")
        opponent = config.get("home_team", "?")

    existing_roster = results.get(roster_key, [])
    with st.status(f"🔍 Completando plantilla de **{team}**...", expanded=True) as status:
        def _progress(msg: str) -> None:
            status.write(msg)

        def _on_batch(partial: list) -> None:
            results[roster_key] = partial
            st.session_state.match_results = results
            try:
                _save_match_prep(config, results)
            except Exception:
                pass

        try:
            results[roster_key] = fill_roster_gaps(
                team, opponent, api_key,
                existing_roster=existing_roster,
                progress_cb=_progress,
                on_batch_complete=_on_batch,
            )
            st.session_state.match_results = results
            _save_match_prep(config, results)
            status.update(label=f"✅ Plantilla de {team} completada", state="complete")
        except Exception as e:
            status.update(label=f"❌ Error: {e}", state="error")
            print(f"[FillGaps] Error:\n{traceback.format_exc()}")
    st.rerun()


# ---------------------------------------------------------------------------
# Team Research
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

    if st.session_state.get("team_research_results") and st.session_state.get("team_research_config"):
        existing = st.session_state.team_research_results
        config = st.session_state.team_research_config

        all_keys = ["team_history", "coach", "roster"]
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

    is_womens = st.toggle(
        "♀️ Fútbol Femenino",
        key="mr_is_womens",
        help="Activa para investigar el equipo femenino",
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
            "is_womens": is_womens,
        }
        _run_team_research_pipeline(st.session_state.team_research_config, api_key)


def _run_team_research_pipeline(
    config: dict,
    api_key: str,
    partial_results: Optional[Dict[str, Any]] = None,
) -> None:
    team_name = config["team_name"]
    baseline_step_count = _workflow_step_count(
        partial_results.get("workflow_metrics") if isinstance(partial_results, dict) else None
    )
    label = "🔄 Continuando análisis..." if partial_results else "🔍 Investigando equipo..."
    with st.status(label, expanded=True) as status:
        def _progress(msg: str) -> None:
            status.write(msg)

        def _on_phase(partial: dict) -> None:
            st.session_state.team_research_results = partial
            try:
                _save_team_research(config, partial)
            except Exception:
                pass

        try:
            results = run_team_research(
                team_name=team_name,
                tournament=config.get("tournament", ""),
                api_key=api_key,
                progress_cb=_progress,
                partial_results=partial_results,
                is_womens=config.get("is_womens", False),
                on_phase_complete=_on_phase,
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
            error_results = st.session_state.get("team_research_results") or partial_results
            if error_results:
                st.session_state.team_research_results = error_results
                try:
                    saved_id = _save_team_research(config, error_results)
                    _persist_usage_run_safe(
                        run_id=_new_usage_run_id("team-research"),
                        source_type="team_research",
                        source_id=saved_id,
                        workflow="team_research",
                        title=team_name,
                        subject=config.get("tournament", "") or team_name,
                        metrics=_slice_workflow_metrics(
                            error_results.get("workflow_metrics"),
                            baseline_step_count,
                        ),
                    )
                except Exception:
                    pass
            status.update(label=f"❌ Error: {e} — puedes continuar el análisis", state="error")
            print(f"[TeamResearch] Error:\n{traceback.format_exc()}")

    st.rerun()


def _display_team_research_results(config: dict, results: dict) -> None:
    team = config["team_name"]
    tournament = config.get("tournament", "")

    st.markdown("---")

    meta = tournament if tournament else "Investigación de Equipo"
    st.markdown(
        f'<div class="match-header">'
        f'<h2>🔬 {team}</h2>'
        f'<div class="match-meta">{meta}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    _render_workflow_metrics(results.get("workflow_metrics"), "📈 Uso de tokens de la investigación")

    st.markdown("### 📊 Historial de Temporadas")
    h_text, h_srcs, _ = _unpack_text_result(results.get("team_history"))
    st.markdown(h_text)
    if h_srcs:
        with st.expander("📚 Fuentes"):
            st.markdown(_format_sources(h_srcs).replace(_CITATION_SEPARATOR, ""))

    coach_text, coach_srcs, _ = _unpack_text_result(results.get("coach"))
    if coach_text and not coach_text.startswith("❌"):
        st.markdown("---")
        st.markdown("### 🎯 Entrenador")
        st.markdown(coach_text)
        if coach_srcs:
            with st.expander("📚 Fuentes"):
                st.markdown(_format_sources(coach_srcs).replace(_CITATION_SEPARATOR, ""))

    st.markdown("---")
    st.markdown("### 👥 Plantilla — Dossier por Jugador")
    _render_roster_players(results.get("roster", []), expand_key=f"team_{config.get('team_name', 'team')}")

    if st.button("🔍 Completar plantilla", key="fill_team_roster", use_container_width=True):
        _fill_team_roster_gaps(config, results, st.secrets.get("GEMINI_API_KEY", ""))
        return

    # Formations
    team_fm = results.get("formations", [])
    if team_fm:
        st.markdown("---")
        _render_formations(team_fm, team_label=team)


def _fill_team_roster_gaps(config: dict, results: dict, api_key: str) -> None:
    """Fill missing player dossiers for a team research."""
    team = config["team_name"]
    existing_roster = results.get("roster", [])
    with st.status(f"🔍 Completando plantilla de **{team}**...", expanded=True) as status:
        def _progress(msg: str) -> None:
            status.write(msg)

        def _on_batch(partial: list) -> None:
            results["roster"] = partial
            st.session_state.team_research_results = results
            try:
                _save_team_research(config, results)
            except Exception:
                pass

        try:
            results["roster"] = fill_roster_gaps(
                team, "", api_key,
                existing_roster=existing_roster,
                progress_cb=_progress,
                on_batch_complete=_on_batch,
            )
            st.session_state.team_research_results = results
            _save_team_research(config, results)
            status.update(label=f"✅ Plantilla de {team} completada", state="complete")
        except Exception as e:
            status.update(label=f"❌ Error: {e}", state="error")
            print(f"[FillGaps] Error:\n{traceback.format_exc()}")
    st.rerun()


# ---------------------------------------------------------------------------
# Player Research
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
            error_results = st.session_state.get("player_research_results") or partial_results
            if error_results:
                st.session_state.player_research_results = error_results
                try:
                    saved_id = _save_player_research(config, error_results)
                    _persist_usage_run_safe(
                        run_id=_new_usage_run_id("player-research"),
                        source_type="player_research",
                        source_id=saved_id,
                        workflow="player_research",
                        title=player_name,
                        subject=" · ".join([part for part in [team_name, position] if part]) or player_name,
                        metrics=_slice_workflow_metrics(
                            error_results.get("workflow_metrics"),
                            baseline_step_count,
                        ),
                    )
                except Exception:
                    pass
            status.update(label=f"❌ Error: {e}", state="error")
            print(f"[PlayerResearch] Error:\n{traceback.format_exc()}")

    st.rerun()


def _display_player_research_results(config: dict, results: dict) -> None:
    player = config["player_name"]
    team = config.get("team_name", "")
    position = config.get("position", "")
    pos_label = _POS_LABELS.get(position, position)

    st.markdown("---")

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

    st.markdown("### 📋 Dossier Completo")
    dossier_text, dossier_srcs, _ = _unpack_text_result(results.get("dossier"))
    st.markdown(dossier_text)
    if dossier_srcs:
        with st.expander("📚 Fuentes"):
            st.markdown(_format_sources(dossier_srcs).replace(_CITATION_SEPARATOR, ""))


# ---------------------------------------------------------------------------
# Top-level Club wrapper
# ---------------------------------------------------------------------------

def _render_club(api_key: str) -> None:
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
        label_visibility="collapsed",
    )

    if selected_tab is None:
        selected_tab = TAB_PARTIDO

    if selected_tab == TAB_PARTIDO:
        _render_match_prep(api_key)
    elif selected_tab == TAB_EQUIPO:
        _render_match_research(api_key)
    elif selected_tab == TAB_JUGADOR:
        _render_player_research(api_key)
