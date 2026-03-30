"""PalomoFacts — token usage tracking, workflow metrics, cost estimation."""
from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from config import MODEL_PRICING, _DEFAULT_PRICING


# ---------------------------------------------------------------------------
# Token Usage Helpers
# ---------------------------------------------------------------------------

def _empty_token_usage(model: str = "", provider: str = "") -> Dict[str, Any]:
    return {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "grounding_requests": 0,
        "model": model,
        "provider": provider,
    }


def _normalize_token_usage(tokens: Any) -> Dict[str, Any]:
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


# ---------------------------------------------------------------------------
# Workflow Metrics
# ---------------------------------------------------------------------------

def _init_workflow_metrics(workflow: str, existing: Any = None) -> Dict[str, Any]:
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
    source = _init_workflow_metrics(target.get("workflow", "workflow"), incoming)
    for step in source.get("steps", []):
        if isinstance(step, dict):
            target.setdefault("steps", []).append(dict(step))
            _add_token_usage(target.setdefault("totals", _empty_token_usage()), step)
    return target


# ---------------------------------------------------------------------------
# Serialization Helpers
# ---------------------------------------------------------------------------

def _serialize_result_value(value: Any) -> Any:
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
                normalized_item = {}
                for k, v in item.items():
                    if isinstance(v, (bytes, bytearray)):
                        import base64
                        normalized_item[k] = base64.b64encode(v).decode("ascii")
                    else:
                        normalized_item[k] = v
                if "tokens" in normalized_item or "text" in normalized_item:
                    normalized_item["tokens"] = _normalize_token_usage(normalized_item.get("tokens"))
                serialized.append(normalized_item)
            else:
                serialized.append(item)
        return serialized

    if isinstance(value, (bytes, bytearray)):
        import base64
        return base64.b64encode(value).decode("ascii")

    return value


def _deserialize_text_result(value: Any) -> Tuple[str, List[Dict[str, str]], Dict[str, Any]]:
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
    return _deserialize_text_result(value)


# ---------------------------------------------------------------------------
# Aggregation & Display Helpers
# ---------------------------------------------------------------------------

def _aggregate_workflow_metrics(metrics: Any) -> List[Dict[str, Any]]:
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
    return f"{int(value or 0):,}"


def _build_markdown_table(headers: List[str], rows: List[List[str]]) -> str:
    if not rows:
        return ""

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Timestamps & IDs
# ---------------------------------------------------------------------------

def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_usage_run_id(prefix: str) -> str:
    return f"{prefix}:{uuid.uuid4().hex}"


def _backfill_run_id(source_type: str, source_id: str) -> str:
    digest = hashlib.sha256(f"{source_type}:{source_id}".encode("utf-8")).hexdigest()[:24]
    return f"backfill:{source_type}:{digest}"


def _workflow_step_count(metrics: Any) -> int:
    normalized = _init_workflow_metrics("workflow", metrics)
    return len([step for step in normalized.get("steps", []) if isinstance(step, dict)])


def _slice_workflow_metrics(metrics: Any, start_index: int = 0) -> Dict[str, Any]:
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


# ---------------------------------------------------------------------------
# Pricing & Cost
# ---------------------------------------------------------------------------

def _pricing_for_model(model: str) -> Dict[str, float]:
    pricing = MODEL_PRICING.get(model)
    if pricing:
        return pricing
    return dict(_DEFAULT_PRICING)


def _estimate_usage_cost(tokens: Any) -> float:
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
    normalized = _init_workflow_metrics("workflow", metrics)
    return round(
        sum(_estimate_usage_cost(step) for step in normalized.get("steps", []) if isinstance(step, dict)),
        6,
    )


def _coerce_datetime(value: Any) -> Optional[datetime]:
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
    days_map = {"7d": 7, "30d": 30, "90d": 90}
    days = days_map.get(filter_key)
    if not days:
        return None
    return (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()


def _format_cost(value: Any) -> str:
    amount = float(value or 0.0)
    if amount >= 100:
        return f"${amount:,.2f}"
    if amount >= 1:
        return f"${amount:,.4f}"
    return f"${amount:,.6f}"


# ---------------------------------------------------------------------------
# Grounding / Search Unit Extraction
# ---------------------------------------------------------------------------

def _extract_gemini_search_units(response: Any, model_name: str, use_search: bool) -> int:
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
    usage = getattr(response, "usage", None)
    server_tool_use = getattr(usage, "server_tool_use", None)
    if server_tool_use is None and isinstance(usage, dict):
        server_tool_use = usage.get("server_tool_use")

    if isinstance(server_tool_use, dict):
        return int(server_tool_use.get("web_search_requests", 0) or 0)

    return int(getattr(server_tool_use, "web_search_requests", 0) or 0)


# ---------------------------------------------------------------------------
# Dashboard Aggregation
# ---------------------------------------------------------------------------

def _aggregate_usage_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
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


# ---------------------------------------------------------------------------
# Data Validation Helpers
# ---------------------------------------------------------------------------

def _has_data(val: Any) -> bool:
    if isinstance(val, tuple):
        text = val[0] if val else ""
        return bool(text) and not text.startswith("❌")
    if isinstance(val, list):
        return len(val) > 0
    return bool(val)
