"""PalomoFacts — PalomoGPT chat UI."""
from __future__ import annotations

import traceback
from typing import Any, Dict, List, Optional

import streamlit as st

from config import CURRENT_YEAR, MODE_PALOMO_GPT
from api import _format_sources
from database import (
    _auto_title,
    _create_conversation,
    _persist_usage_run_safe,
    _save_message,
)
from metrics import _init_workflow_metrics, _new_usage_run_id
from research import get_palomo_response
from ui_components import _render_workflow_metrics

# ---------------------------------------------------------------------------
# Example queries shown on the welcome screen
# ---------------------------------------------------------------------------

_EXAMPLE_QUERIES = [
    f"Cuéntame todo sobre Erling Haaland en Premier League {CURRENT_YEAR}",
    "La historia de la rivalidad Real Madrid vs Barcelona",
    "Lamine Yamal: trayectoria, estadísticas y datos curiosos",
    "¿Cuántos goles lleva Haaland esta temporada?",
    "¿Cómo ha cambiado Mbappé desde que llegó al Real Madrid?",
    "Los récords más locos de la Champions League",
]


# ---------------------------------------------------------------------------
# PalomoGPT renderer
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
                    st.session_state._pending_query = example
                    if not st.session_state.current_conv_id:
                        cid = _create_conversation(
                            mode=MODE_PALOMO_GPT,
                            title=_auto_title(example),
                        )
                        st.session_state.current_conv_id = cid
                    st.rerun()
        return

    # --- Replay existing messages ---
    for msg in msgs:
        with st.chat_message(msg["role"]):
            st.markdown(msg.get("content", ""))

    # --- Process new query (if any) ---
    if not incoming_query:
        return

    msgs.append({"role": "user", "content": incoming_query})
    with st.chat_message("user"):
        st.markdown(incoming_query)

    conv_id = st.session_state.get("current_conv_id", "")
    _save_message(conv_id, "user", incoming_query)

    if not api_key:
        with st.chat_message("assistant"):
            err = (
                "⚠️ No se encontró la API key de Gemini. "
                "Configura `GEMINI_API_KEY` en los secrets de Streamlit."
            )
            st.warning(err)
            msgs.append({"role": "assistant", "content": err})
        return

    followups: List[Dict[str, str]] = []
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

    for fu in followups:
        with st.chat_message("user"):
            st.markdown(f"🔍 {fu['question']}")
        q_text = f"🔍 {fu['question']}"
        msgs.append({"role": "user", "content": q_text})
        _save_message(conv_id, "user", q_text)

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
