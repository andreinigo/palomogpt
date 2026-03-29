"""PalomoFacts — app shell, sidebar routing, and Streamlit page configuration."""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import streamlit as st

from config import (
    CUSTOM_CSS,
    MODE_CLUB,
    MODE_OPTIONS,
    MODE_PALOMO_GPT,
    MODE_SELECCION,
    TAB_EQUIPO,
    TAB_JUGADOR,
    TAB_PARTIDO,
)
from database import (
    _auto_title,
    _create_conversation,
    _delete_conversation,
    _delete_match_prep,
    _delete_national_match_prep,
    _delete_national_player_research,
    _delete_national_team_research,
    _delete_player_research,
    _delete_team_research,
    _list_conversations,
    _list_match_preps,
    _list_national_match_preps,
    _list_national_player_researches,
    _list_national_team_researches,
    _list_player_researches,
    _list_team_researches,
    _load_match_prep,
    _load_messages,
    _load_national_match_prep,
    _load_national_player_research,
    _load_national_team_research,
    _load_player_research,
    _load_team_research,
)
from ui_club import _render_club
from ui_dashboard import _render_dashboard_page
from ui_palomo import _render_palomo_gpt
from ui_seleccion import _render_seleccion


# ---------------------------------------------------------------------------
# Root page — sidebar + mode routing
# ---------------------------------------------------------------------------

def _render_root_page() -> None:
    api_key = st.secrets.get("GEMINI_API_KEY", "")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "current_conv_id" not in st.session_state:
        st.session_state.current_conv_id = ""

    # Eagerly capture chat_input for PalomoGPT mode
    prior_mode = st.session_state.get("sb_app_mode", MODE_PALOMO_GPT)
    incoming_query = None
    if prior_mode == MODE_PALOMO_GPT:
        incoming_query = st.chat_input("Pregunta sobre cualquier tema de fútbol...")
        if incoming_query and not st.session_state.current_conv_id:
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

        if app_mode == MODE_PALOMO_GPT:
            has_active_conv = bool(st.session_state.get("current_conv_id", ""))
            if has_active_conv:
                if st.button("➕ Nueva conversación", use_container_width=True):
                    st.session_state.current_conv_id = ""
                    st.session_state.messages = []
                    st.rerun()

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
            # Club sidebar
            # ================================================================
            if app_mode == MODE_CLUB:
                st.markdown("### 📚 Historial")

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
            # Selección sidebar
            # ================================================================
            elif app_mode == MODE_SELECCION:
                st.markdown("### 📚 Historial")

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


# ---------------------------------------------------------------------------
# Page builder helper
# ---------------------------------------------------------------------------

def _build_streamlit_page(
    page_callable: Callable[[], None],
    title: str,
    url_path: Optional[str] = None,
    default: bool = False,
    hidden: bool = False,
):
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
        kwargs.pop("visibility", None)
        return st.Page(page_callable, **kwargs)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title="PalomoFacts · Football Intelligence",
        page_icon="⚽",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

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
