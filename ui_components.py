"""PalomoFacts — shared UI helpers (roster display, workflow metrics)."""
from __future__ import annotations

from typing import Any, Dict

import streamlit as st

from api import _CITATION_SEPARATOR, _format_sources
from metrics import (
    _aggregate_workflow_metrics,
    _build_markdown_table,
    _empty_token_usage,
    _format_metric_number,
    _init_workflow_metrics,
)


# ---------------------------------------------------------------------------
# Workflow metrics expander
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Roster player expanders
# ---------------------------------------------------------------------------

def _render_roster_players(roster: list, expand_key: str = "roster") -> None:
    """Render player dossiers as expanders grouped by position."""
    if not roster:
        st.warning("No se encontraron jugadores.")
        return

    anchor_id = f"roster_anchor_{expand_key}"
    st.markdown(f'<div id="{anchor_id}"></div>', unsafe_allow_html=True)

    expand_state_key = f"expand_all_{expand_key}"
    if expand_state_key not in st.session_state:
        st.session_state[expand_state_key] = False

    btn_col1, btn_col2, _ = st.columns([1, 1, 4])
    with btn_col1:
        if st.button("⬇️ Expandir todos", key=f"expand_btn_{expand_key}", use_container_width=True):
            st.session_state[expand_state_key] = True
            st.rerun()
    with btn_col2:
        if st.button("⬆️ Colapsar todos", key=f"collapse_btn_{expand_key}", use_container_width=True):
            st.session_state[expand_state_key] = False
            st.rerun()

    expand_all = st.session_state.get(expand_state_key, False)

    pos_order = ["GK", "DEF", "MID", "FWD"]
    pos_emoji = {"GK": "🧤", "DEF": "🛡️", "MID": "🎯", "FWD": "⚡"}
    pos_name = {"GK": "PORTEROS", "DEF": "DEFENSAS", "MID": "CENTROCAMPISTAS", "FWD": "DELANTEROS"}

    grouped: Dict[str, list] = {p: [] for p in pos_order}
    for player in roster:
        pos = player.get("position", "FWD")
        if pos not in grouped:
            pos = "FWD"
        grouped[pos].append(player)

    player_idx = 0
    for pos in pos_order:
        players = grouped[pos]
        if not players:
            continue
        st.markdown(f"**{pos_emoji.get(pos, '')} {pos_name.get(pos, pos)}**")
        for p in players:
            number = p.get("number", "")
            label = f"#{number} {p['name']}" if number else p["name"]
            with st.expander(label, expanded=expand_all):
                st.markdown(p.get("text", ""))
                opponent_text = p.get("opponent_text", "")
                if opponent_text:
                    st.markdown("---")
                    st.markdown("**⚡ Conexiones con el rival**")
                    st.markdown(opponent_text)
                sources = p.get("sources", [])
                if sources:
                    st.markdown(
                        _format_sources(sources).replace(_CITATION_SEPARATOR, "")
                    )
                st.markdown("---")
                if st.button(
                    "⬆️ Volver al listado",
                    key=f"back_top_{expand_key}_{player_idx}",
                    use_container_width=True,
                ):
                    st.session_state[expand_state_key] = False
                    st.session_state[f"_scroll_to_{expand_key}"] = True
                    st.rerun()
            player_idx += 1

    scroll_key = f"_scroll_to_{expand_key}"
    if st.session_state.pop(scroll_key, False):
        st.markdown(
            f"""<script>
            var el = document.getElementById('{anchor_id}');
            if (el) {{ el.scrollIntoView({{behavior: 'smooth', block: 'start'}}); }}
            </script>""",
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Formation images display
# ---------------------------------------------------------------------------

def _render_formations(formations: list, team_label: str = "") -> None:
    """Render Sofascore formation data with summary."""
    if not formations:
        return

    from collections import Counter

    freq = Counter(f.get("formation") or "N/A" for f in formations)
    freq_str = ", ".join(f"**{fmt}** ({cnt}x)" for fmt, cnt in freq.most_common())
    header = f"⚽ Parado Táctico — {team_label}" if team_label else "⚽ Parado Táctico"
    st.markdown(f"### {header}")
    st.markdown(f"Formaciones recientes: {freq_str}")

    cols = st.columns(min(len(formations), 3))
    for idx, fm in enumerate(formations):
        with cols[idx % len(cols)]:
            img = fm.get("image_bytes")
            if img and isinstance(img, str):
                import base64
                img = base64.b64decode(img)
            if img:
                caption = (
                    f"{fm.get('target_team', '')} vs {fm.get('opponent', '')} "
                    f"({fm.get('match_date', '?')}) — {fm.get('formation', 'N/A')}"
                )
                st.image(img, caption=caption, use_container_width=True)
            # Always show a visible card with formation info
            formation = fm.get("formation", "N/A")
            opponent = fm.get("opponent", "")
            match_date = fm.get("match_date", "?")
            target_side = fm.get("target_side", "")
            side_label = " (L)" if target_side == "left" else " (V)" if target_side == "right" else ""
            st.markdown(
                f"**{formation}**{side_label} vs {opponent}  \n"
                f"📅 {match_date}"
            )
            with st.expander(f"Jugadores — {formation}"):
                for p in fm.get("players", []):
                    st.markdown(f"- {p}")
