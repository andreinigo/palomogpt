"""PalomoFacts — Admin usage dashboard (hidden /dashboard page)."""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import streamlit as st

from config import DASHBOARD_ACCESS_STATE_KEY, DASHBOARD_FILTER_OPTIONS
from database import _backfill_usage_runs, _load_usage_runs
from metrics import (
    _aggregate_usage_by_model,
    _aggregate_usage_by_workflow,
    _aggregate_usage_rows,
    _format_cost,
    _format_metric_number,
    _serialize_usage_recent_rows,
)


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

    if not usage_rows:
        st.info("No usage data found yet. Run the app or use backfill to populate the ledger.")
        return

    st.caption(
        "Note: older structured workflows can be backfilled from saved `workflow_metrics`, "
        "but historical PalomoGPT traffic and older PDF exports from before this release were not persisted."
    )

    by_model = _aggregate_usage_by_model(usage_rows)
    by_workflow = _aggregate_usage_by_workflow(usage_rows)

    # By Workflow
    st.markdown("### By Workflow")
    st.caption("Click a row to filter everything below. Click again to clear.")
    wf_event = st.dataframe(
        [
            {
                "Workflow": r["workflow"],
                "Type": r["source_type"],
                "Runs": _format_metric_number(r["runs"]),
                "Input": _format_metric_number(r["input_tokens"]),
                "Output": _format_metric_number(r["output_tokens"]),
                "Total": _format_metric_number(r["total_tokens"]),
                "Search": _format_metric_number(r["grounding_requests"]),
                "Cost": _format_cost(r["estimated_cost_usd"]),
            }
            for r in by_workflow
        ],
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key="dash_wf_table",
    )

    # By Model
    st.markdown("### By Model")
    st.caption("Click a row to filter everything below. Click again to clear.")
    model_event = st.dataframe(
        [
            {
                "Provider": r["provider"],
                "Model": r["model"],
                "Calls": _format_metric_number(r["calls"]),
                "Input": _format_metric_number(r["input_tokens"]),
                "Output": _format_metric_number(r["output_tokens"]),
                "Total": _format_metric_number(r["total_tokens"]),
                "Search": _format_metric_number(r["grounding_requests"]),
                "Cost": _format_cost(r["estimated_cost_usd"]),
            }
            for r in by_model
        ],
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key="dash_model_table",
    )

    # Resolve active filter
    filtered_rows = usage_rows
    filter_label: Optional[str] = None

    wf_sel = wf_event.selection.rows
    model_sel = model_event.selection.rows

    if wf_sel and wf_sel[0] < len(by_workflow):
        sel = by_workflow[wf_sel[0]]
        filtered_rows = [
            r for r in usage_rows
            if r.get("workflow") == sel["workflow"] and r.get("source_type") == sel["source_type"]
        ]
        filter_label = f"workflow **{sel['workflow']}** ({sel['source_type']})"
    elif model_sel and model_sel[0] < len(by_model):
        sel = by_model[model_sel[0]]
        filtered_rows = [
            r for r in usage_rows
            if any(
                str(s.get("model", "")) == sel["model"] and str(s.get("provider", "")) == sel["provider"]
                for s in r.get("steps", [])
            )
        ]
        filter_label = f"model **{sel['model']}** ({sel['provider']})"

    # KPI metrics
    st.markdown("---")
    if filter_label:
        st.info(f"🔍 {len(filtered_rows):,} / {len(usage_rows):,} runs — {filter_label}. Click same row to clear.")
    else:
        st.caption(f"All {len(usage_rows):,} runs. Click any row above to filter.")

    agg = _aggregate_usage_rows(filtered_rows)
    totals = agg["totals"]
    metric_cols = st.columns(6)
    metric_cols[0].metric("Raw Cost", _format_cost(agg["estimated_cost_usd"]))
    metric_cols[1].metric("Runs", _format_metric_number(agg["runs"]))
    metric_cols[2].metric("Input", _format_metric_number(totals["input_tokens"]))
    metric_cols[3].metric("Output", _format_metric_number(totals["output_tokens"]))
    metric_cols[4].metric("Total", _format_metric_number(totals["total_tokens"]))
    metric_cols[5].metric("Web Search", _format_metric_number(totals["grounding_requests"]))

    # Recent Runs
    st.markdown("### Recent Runs")
    st.dataframe(_serialize_usage_recent_rows(filtered_rows, limit=50), use_container_width=True, hide_index=True)
