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
# Formation pitch diagram (SVG)
# ---------------------------------------------------------------------------

def _formation_svg(lineup_grid: list[dict], formation: str = "") -> str:
    """Return an SVG half-pitch with players placed by API-Football grid data.

    Grid format from API: ``"row:col"`` where row 1 = GK (bottom),
    higher rows = further up the pitch.  Column numbers indicate lateral
    position within each row.
    """
    from xml.sax.saxutils import escape as _esc

    if not lineup_grid:
        return ""

    W, H = 260, 360
    PX, PY, PW, PH = 15, 15, 230, 320
    CX = PX + PW / 2
    LC = "rgba(255,255,255,0.35)"

    # --- parse grid positions ---
    rows: dict[int, list[tuple[int, dict]]] = {}
    for p in lineup_grid:
        g = p.get("grid") or ""
        if ":" not in g:
            continue
        r, c = int(g.split(":")[0]), int(g.split(":")[1])
        rows.setdefault(r, []).append((c, p))

    if not rows:
        return ""

    max_row = max(rows)
    parts: list[str] = []

    # --- SVG root ---
    parts.append(
        f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg"'
        f' style="width:100%;max-width:{W}px;display:block;margin:0 auto;'
        f'font-family:-apple-system,system-ui,BlinkMacSystemFont,sans-serif">'
    )

    # Background
    parts.append(f'<rect width="{W}" height="{H}" rx="10" fill="#1a472a"/>')

    # Grass stripes
    sw = PW / 6
    for i in range(0, 6, 2):
        parts.append(
            f'<rect x="{PX + i * sw:.0f}" y="{PY}" '
            f'width="{sw:.0f}" height="{PH}" fill="rgba(255,255,255,0.025)"/>'
        )

    # Pitch outline
    parts.append(
        f'<rect x="{PX}" y="{PY}" width="{PW}" height="{PH}" '
        f'fill="none" stroke="{LC}" stroke-width="1.5" rx="1"/>'
    )

    # Centre line (top edge = halfway line)
    parts.append(
        f'<line x1="{PX}" y1="{PY}" x2="{PX + PW}" y2="{PY}" '
        f'stroke="{LC}" stroke-width="1.5"/>'
    )

    # Centre circle arc
    rc = 42
    parts.append(
        f'<path d="M{CX - rc} {PY} A{rc} {rc} 0 0 1 {CX + rc} {PY}" '
        f'fill="none" stroke="{LC}" stroke-width="1.5"/>'
    )
    parts.append(f'<circle cx="{CX}" cy="{PY}" r="3" fill="{LC}"/>')

    # Penalty area
    paw, pah = 152, 72
    pax, pay = CX - paw / 2, PY + PH - pah
    parts.append(
        f'<rect x="{pax:.0f}" y="{pay:.0f}" width="{paw}" height="{pah}" '
        f'fill="none" stroke="{LC}" stroke-width="1.5"/>'
    )

    # Goal area
    gaw, gah = 66, 26
    gax, gay = CX - gaw / 2, PY + PH - gah
    parts.append(
        f'<rect x="{gax:.0f}" y="{gay:.0f}" width="{gaw}" height="{gah}" '
        f'fill="none" stroke="{LC}" stroke-width="1.5"/>'
    )

    # Penalty spot
    parts.append(f'<circle cx="{CX}" cy="{pay + 18:.0f}" r="2" fill="{LC}"/>')

    # Penalty arc
    ar = 28
    parts.append(
        f'<path d="M{CX - ar} {pay:.0f} A{ar} {ar} 0 0 0 {CX + ar} {pay:.0f}" '
        f'fill="none" stroke="{LC}" stroke-width="1.5"/>'
    )

    # Goal
    gw = 42
    parts.append(
        f'<rect x="{CX - gw / 2:.0f}" y="{PY + PH:.0f}" width="{gw}" height="8" '
        f'rx="2" fill="rgba(255,255,255,0.12)" stroke="{LC}" stroke-width="1"/>'
    )

    # --- place players ---
    CR = 15            # circle radius
    mx = 30            # horizontal margin
    mt, mb = 38, 48   # top/bottom margin within pitch

    # global max columns across all rows — used as the reference grid width
    global_mc = max(max(c for c, _ in plist) for plist in rows.values())

    for rn, plist in sorted(rows.items()):
        frac = 0.0 if max_row <= 1 else (rn - 1) / (max_row - 1)
        y = PY + PH - mb - frac * (PH - mt - mb)

        spl = sorted(plist, key=lambda x: x[0])
        n_in_row = len(spl)

        for col, p in spl:
            if global_mc <= 1:
                x = CX
            elif n_in_row == 1:
                x = CX
            else:
                # Centre the row's players within the global grid width
                # e.g. 2 players in a 4-wide grid → positioned at cols ~2 and ~3
                row_mc = max(c for c, _ in spl)
                offset = (global_mc - row_mc) / 2
                x = PX + mx + (col - 1 + offset) / (global_mc - 1) * (PW - 2 * mx)

            num = p.get("number", "")
            nm = p.get("name", "")
            sur = (nm.split() or [""])[-1]
            if len(sur) > 11:
                sur = sur[:10] + "."
            sur = _esc(sur)

            parts.append(
                f'<circle cx="{x:.0f}" cy="{y:.0f}" r="{CR}" '
                f'fill="white" fill-opacity="0.92"/>'
            )
            parts.append(
                f'<text x="{x:.0f}" y="{y + 4:.0f}" text-anchor="middle" '
                f'font-size="12" font-weight="700" fill="#1a472a">{num}</text>'
            )
            parts.append(
                f'<text x="{x:.0f}" y="{y + CR + 11:.0f}" text-anchor="middle" '
                f'font-size="8.5" font-weight="600" fill="white" '
                f'fill-opacity="0.9">{sur}</text>'
            )

    # Formation label
    if formation:
        parts.append(
            f'<text x="{CX}" y="{H - 6}" text-anchor="middle" '
            f'font-size="12" font-weight="700" '
            f'fill="rgba(255,255,255,0.6)">{_esc(formation)}</text>'
        )

    parts.append("</svg>")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Formation rendering
# ---------------------------------------------------------------------------

def _render_formations(formations: list, team_label: str = "") -> None:
    """Render formation data with pitch diagrams and summary."""
    if not formations:
        return

    import base64
    from collections import Counter

    freq = Counter(f.get("formation") or "N/A" for f in formations)
    freq_str = ", ".join(f"**{fmt}** ({cnt}x)" for fmt, cnt in freq.most_common())
    header = f"⚽ Parado Táctico — {team_label}" if team_label else "⚽ Parado Táctico"
    st.markdown(f"### {header}")
    st.markdown(f"Formaciones recientes: {freq_str}")

    cols = st.columns(min(len(formations), 3))
    for idx, fm in enumerate(formations):
        with cols[idx % len(cols)]:
            # Pitch diagram from API-Football grid data
            lineup_grid = fm.get("lineup_grid")
            if lineup_grid:
                svg = _formation_svg(lineup_grid, fm.get("formation", ""))
                if svg:
                    b64 = base64.b64encode(svg.encode()).decode()
                    st.markdown(
                        f'<img src="data:image/svg+xml;base64,{b64}" '
                        f'style="width:100%">',
                        unsafe_allow_html=True,
                    )
            # Text card
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
