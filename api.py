"""PalomoFacts — Gemini and Claude API request layer + citation utilities."""
from __future__ import annotations

import re
import time
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import streamlit as st
from google import genai
from google.genai import types
import anthropic

from config import (
    CLAUDE_MODEL,
    GEMINI_FALLBACK_MODEL,
    GEMINI_MODEL,
)
from metrics import (
    _empty_token_usage,
    _extract_claude_search_units,
    _extract_gemini_search_units,
)


# ---------------------------------------------------------------------------
# Citation utilities
# ---------------------------------------------------------------------------

_CITATION_SEPARATOR = "\n\n---\n📚 **Fuentes:**"


def _resolve_inline_citations(
    text: str, sources: List[Dict[str, str]]
) -> str:
    if not sources:
        return text

    _SUPERSCRIPT = {
        0: "⁰", 1: "¹", 2: "²", 3: "³", 4: "⁴",
        5: "⁵", 6: "⁶", 7: "⁷", 8: "⁸", 9: "⁹",
    }

    def _to_super(n: int) -> str:
        return "".join(_SUPERSCRIPT.get(int(d), d) for d in str(n))

    def _replace_marker(match: re.Match) -> str:
        idx = int(match.group(1)) - 1
        if idx < 0 or idx >= len(sources):
            return match.group(0)
        src = sources[idx]
        url = src.get("url", "")
        if not url:
            return match.group(0)
        sup = _to_super(idx + 1)
        return f"[{sup}]({url})"

    return re.sub(r'\[(\d+)\]', _replace_marker, text)


def _format_sources(sources: List[Dict[str, str]]) -> str:
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
# Gemini API
# ---------------------------------------------------------------------------

_MAX_RETRIES = 3
_RETRY_BASE_DELAY = 1


def _gemini_request(
    api_key: str,
    system_prompt: str,
    user_message: str,
    history: Optional[List[types.Content]] = None,
    timeout: int = 120,
    use_search: bool = True,
) -> Tuple[str, List[Dict[str, str]], Dict[str, Any]]:
    client = genai.Client(
        api_key=api_key,
        http_options={"timeout": timeout * 1000},
    )

    tools = []
    if use_search:
        tools.append({"google_search": {}})
        tools.append({"url_context": {}})

    def _build_config(model_name: str) -> types.GenerateContentConfig:
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
            if not text.strip():
                print(f"[Gemini] Empty response on attempt {attempt + 1}/{_MAX_RETRIES} — retrying...")
                raise RuntimeError("Gemini returned empty response")
            token_usage = _empty_token_usage(model=model, provider="google")
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                um = response.usage_metadata
                input_tok = int(getattr(um, "prompt_token_count", 0) or 0)
                candidates_tok = int(getattr(um, "candidates_token_count", 0) or 0)
                thoughts_tok = int(getattr(um, "thoughts_token_count", 0) or 0)
                total_tok = int(getattr(um, "total_token_count", 0) or 0)
                output_tok = thoughts_tok + candidates_tok
                if total_tok and not output_tok:
                    output_tok = max(0, total_tok - input_tok)
                token_usage["input_tokens"] = input_tok
                token_usage["output_tokens"] = output_tok
                token_usage["total_tokens"] = total_tok if total_tok else (input_tok + output_tok)
            token_usage["grounding_requests"] = _extract_gemini_search_units(response, model, use_search)

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

            text = _resolve_inline_citations(text, sources)

            return text, sources, token_usage

        except Exception as e:
            last_error = e
            err_str = str(e)

            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
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

    # Last resort: try Claude before giving up entirely
    try:
        print(f"[Gemini] All {_MAX_RETRIES} retries exhausted — trying Claude as last resort...")
        return _claude_request(system_prompt, user_message, history)
    except Exception as claude_err:
        print(f"[Claude] Last-resort fallback also failed: {claude_err}")

    raise RuntimeError(f"Gemini API failed after {_MAX_RETRIES} attempts: {last_error}")


# ---------------------------------------------------------------------------
# Claude API
# ---------------------------------------------------------------------------

def _claude_request(
    system_prompt: str,
    user_message: str,
    history: Optional[list] = None,
) -> Tuple[str, List[Dict[str, str]], Dict[str, Any]]:
    claude_key = st.secrets.get("ANTHROPIC_API_KEY", "")
    if not claude_key:
        raise RuntimeError("ANTHROPIC_API_KEY not configured")

    client = anthropic.Anthropic(api_key=claude_key)

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
        input_tok = int(getattr(response.usage, "input_tokens", 0) or 0)
        output_tok = int(getattr(response.usage, "output_tokens", 0) or 0)
        cache_creation = int(getattr(response.usage, "cache_creation_input_tokens", 0) or 0)
        effective_input = input_tok + round(cache_creation * 0.25)
        token_usage["input_tokens"] = effective_input
        token_usage["output_tokens"] = output_tok
        token_usage["total_tokens"] = effective_input + output_tok
        token_usage["grounding_requests"] = _extract_claude_search_units(response)
    return text, [], token_usage
