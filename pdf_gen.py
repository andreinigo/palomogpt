"""PalomoFacts — PDF generation for match preparation reports."""
from __future__ import annotations

import re
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional, Tuple

import anthropic
import streamlit as st
from fpdf import FPDF

from config import CLAUDE_HAIKU_MODEL, CURRENT_DATE
from metrics import (
    _empty_token_usage,
    _init_workflow_metrics,
    _merge_workflow_metrics,
    _normalize_token_usage,
    _record_workflow_step,
)
from prompts import PLAYER_SYNTHESIS_PROMPT

# ---------------------------------------------------------------------------
# Constants & compiled regexes
# ---------------------------------------------------------------------------

_PDF_FONT = "Helvetica"

_RE_MD_BOLD = re.compile(r'\*\*(.+?)\*\*')
_RE_MD_ITALIC = re.compile(r'\*(.+?)\*')
_RE_MD_LINK = re.compile(r'\[([^\]]+)\]\([^)]+\)')
_RE_MD_HEADER = re.compile(r'^#{1,6}\s+')
_RE_SUPERSCRIPT_LINK = re.compile(r'\[[\u2070-\u2079\u00B2\u00B3\u00B9]+\]\([^)]+\)')
_RE_CITATION_BLOCK = re.compile(r'\n---\n.*?Fuentes:.*', re.DOTALL)


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

def _strip_markdown(text: str) -> str:
    """Convert markdown text to plain text suitable for PDF."""
    text = _RE_CITATION_BLOCK.sub('', text)
    text = _RE_SUPERSCRIPT_LINK.sub('', text)
    text = _RE_MD_LINK.sub(r'\1', text)
    text = _RE_MD_BOLD.sub(r'\1', text)
    text = _RE_MD_ITALIC.sub(r'\1', text)
    text = _RE_MD_HEADER.sub('', text)
    text = re.sub(r'^\s*[-*]\s+', '> ', text, flags=re.MULTILINE)
    return text.strip()


def _clean_for_latin(text: str) -> str:
    """Replace characters outside latin-1 with close equivalents."""
    replacements = {
        '\u2013': '-', '\u2014': '-', '\u2018': "'", '\u2019': "'",
        '\u201c': '"', '\u201d': '"', '\u2026': '...',
        '\u2070': '0', '\u00b9': '1', '\u00b2': '2', '\u00b3': '3',
        '\u2074': '4', '\u2075': '5', '\u2076': '6', '\u2077': '7',
        '\u2078': '8', '\u2079': '9',
        '\u26a1': '[!]', '\U0001f3af': '[*]',
        '\U0001f9e4': '', '\U0001f6e1': '', '\U0001f3af': '', '\u26a1': '[!]',
        '\u2705': '[OK]', '\u274c': '[X]', '\U0001f50d': '',
        '\U0001f4ca': '', '\U0001f4cb': '', '\U0001f3c6': '',
        '\U0001f3df': '', '\U0001f3e0': '', '\u2708': '',
        '\U0001f4cc': '', '\U0001f680': '', '\U0001f5d1': '',
        '\U0001f3d9': '', '\U0001f30d': '',
        '\U0001f399': '', '\U0001f4da': '',
        '\U0001f46b': '', '\U0001f465': '',
        '\n- ': '\n> ', '\n* ': '\n> ',
    }
    for char, repl in replacements.items():
        text = text.replace(char, repl)
    text = unicodedata.normalize('NFKD', text)
    return text.encode('latin-1', errors='ignore').decode('latin-1')


# ---------------------------------------------------------------------------
# _MatchPDF subclass
# ---------------------------------------------------------------------------

class _MatchPDF(FPDF):
    """Custom FPDF subclass for match preparation reports."""

    def __init__(self, config: dict) -> None:
        super().__init__(orientation='P', unit='mm', format='A4')
        self.config = config
        self.set_auto_page_break(auto=True, margin=20)

    def header(self):
        if self.page_no() == 1:
            return
        self.set_font(_PDF_FONT, 'I', 8)
        home = _clean_for_latin(self.config['home_team'])
        away = _clean_for_latin(self.config['away_team'])
        self.cell(0, 6, f"{home} vs {away} - Preparacion de Partido", align='C')
        self.ln(8)

    def footer(self):
        if self.page_no() == 1:
            return
        self.set_y(-15)
        self.set_font(_PDF_FONT, 'I', 8)
        self.cell(0, 10, f"Pagina {self.page_no() - 1}", align='C')

    def _section_title(self, title: str):
        self.set_font(_PDF_FONT, 'B', 14)
        self.cell(0, 10, _clean_for_latin(title), ln=True)
        self.ln(2)

    def _sub_title(self, title: str):
        self.set_font(_PDF_FONT, 'B', 11)
        self.cell(0, 8, _clean_for_latin(title), ln=True)
        self.ln(1)

    def _body_text(self, text: str, size: int = 9):
        self.set_font(_PDF_FONT, '', size)
        clean = _clean_for_latin(_strip_markdown(text))
        self.multi_cell(0, 4.5, clean)
        self.ln(2)

    def add_cover(self):
        self.add_page()
        self.ln(60)

        self.set_font(_PDF_FONT, 'B', 28)
        home = _clean_for_latin(self.config['home_team'])
        away = _clean_for_latin(self.config['away_team'])
        self.cell(0, 14, home, ln=True, align='C')
        self.ln(4)
        self.set_font(_PDF_FONT, '', 18)
        self.cell(0, 10, 'vs', ln=True, align='C')
        self.ln(4)
        self.set_font(_PDF_FONT, 'B', 28)
        self.cell(0, 14, away, ln=True, align='C')

        self.ln(16)
        self.set_font(_PDF_FONT, '', 14)
        meta = _clean_for_latin(
            f"{self.config['match_type']}  |  {self.config['tournament']}"
        )
        self.cell(0, 8, meta, ln=True, align='C')

        self.ln(4)
        self.set_font(_PDF_FONT, '', 12)
        stadium = _clean_for_latin(self.config['stadium'])
        self.cell(0, 8, stadium, ln=True, align='C')

        self.ln(8)
        self.set_font(_PDF_FONT, 'I', 10)
        self.cell(0, 8, CURRENT_DATE, ln=True, align='C')

        self.ln(30)
        self.set_font(_PDF_FONT, '', 9)
        self.cell(0, 6, 'Preparacion de Partido - PalomoFacts', ln=True, align='C')

    def add_team_histories(self, results: dict):
        self.add_page()
        home_label = _clean_for_latin(f"Historial de Temporadas - LOCAL: {self.config['home_team']}")
        self._section_title(home_label)
        home_text = results['home_history'][0] if isinstance(results['home_history'], tuple) else ''
        if home_text:
            self._body_text(home_text, size=9)

        self.add_page()
        away_label = _clean_for_latin(f"Historial de Temporadas - VISITANTE: {self.config['away_team']}")
        self._section_title(away_label)
        away_text = results['away_history'][0] if isinstance(results['away_history'], tuple) else ''
        if away_text:
            self._body_text(away_text, size=9)

    def _add_roster_section(self, team_name: str, roster: list, label: str):
        self.add_page()
        self._section_title(f'Plantilla - {label}: {team_name}')

        pos_order = ['GK', 'DEF', 'MID', 'FWD']
        pos_names = {
            'GK': 'PORTEROS', 'DEF': 'DEFENSAS',
            'MID': 'CENTROCAMPISTAS', 'FWD': 'DELANTEROS',
        }
        grouped: Dict[str, list] = {p: [] for p in pos_order}
        for player in roster:
            pos = player.get('position', 'FWD')
            if pos not in grouped:
                pos = 'FWD'
            grouped[pos].append(player)

        for pos in pos_order:
            players = grouped[pos]
            if not players:
                continue
            self._sub_title(pos_names.get(pos, pos))
            for p in players:
                number = p.get('number', '')
                name = _clean_for_latin(p['name'])
                header = f"#{number} {name}" if number else name

                self.set_font(_PDF_FONT, 'B', 10)
                self.cell(0, 6, header, ln=True)

                body = p.get('text', '')
                if body:
                    self._body_text(body, size=8)

                if self.get_y() > self.h - 30:
                    self.add_page()

    def add_rosters(self, results: dict):
        home = _clean_for_latin(self.config['home_team'])
        away = _clean_for_latin(self.config['away_team'])
        self._add_roster_section(home, results.get('home_roster', []), 'Local')
        self._add_roster_section(away, results.get('away_roster', []), 'Visitante')

    def add_palomo_phrases(self, results: dict):
        self.add_page()
        self._section_title('Frases de Fernando Palomo')
        text = results['palomo_phrases'][0] if isinstance(results['palomo_phrases'], tuple) else ''
        if text:
            self._body_text(text, size=10)


# ---------------------------------------------------------------------------
# Claude Haiku synthesis helpers
# ---------------------------------------------------------------------------

def _synthesize_one_player(
    player: Dict[str, Any],
    client: anthropic.Anthropic,
) -> Dict[str, Any]:
    raw_text = player.get("text", "")
    if not raw_text or len(raw_text) < 50:
        return player
    try:
        response = client.messages.create(
            model=CLAUDE_HAIKU_MODEL,
            max_tokens=512,
            system=PLAYER_SYNTHESIS_PROMPT,
            messages=[{"role": "user", "content": f"Sintetiza este dossier:\n\n{raw_text}"}],
        )
        note = response.content[0].text
        token_usage = _empty_token_usage(model=CLAUDE_HAIKU_MODEL, provider="anthropic")
        if hasattr(response, "usage") and response.usage:
            input_tok = int(getattr(response.usage, "input_tokens", 0) or 0)
            output_tok = int(getattr(response.usage, "output_tokens", 0) or 0)
            cache_creation = int(getattr(response.usage, "cache_creation_input_tokens", 0) or 0)
            effective_input = input_tok + round(cache_creation * 0.25)
            token_usage["input_tokens"] = effective_input
            token_usage["output_tokens"] = output_tok
            token_usage["total_tokens"] = effective_input + output_tok
        return {
            **player,
            "text": note.strip(),
            "raw_text": raw_text,
            "_pdf_synthesis_tokens": token_usage,
        }
    except Exception as e:
        print(f"[PDF Synthesis] Error for {player.get('name', '?')}: {e}")
        return player


def _synthesize_roster_for_pdf(
    roster: List[Dict[str, Any]],
    api_key: str,
    progress_cb: Optional[Callable[[str], None]] = None,
    team_label: str = "",
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    claude_key = st.secrets.get("ANTHROPIC_API_KEY", "")
    if not claude_key:
        print("[PDF Synthesis] No ANTHROPIC_API_KEY — skipping synthesis.")
        return roster, _init_workflow_metrics("match_pdf_export")

    client = anthropic.Anthropic(api_key=claude_key)
    synthesized: List[Optional[Dict[str, Any]]] = [None] * len(roster)
    done_count = 0
    total = len(roster)
    export_metrics = _init_workflow_metrics("match_pdf_export")

    def _cb(msg: str) -> None:
        if progress_cb:
            progress_cb(msg)

    with ThreadPoolExecutor(max_workers=8) as executor:
        future_to_idx = {
            executor.submit(_synthesize_one_player, player, client): idx
            for idx, player in enumerate(roster)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                synthesized[idx] = future.result()
            except Exception as e:
                print(f"[PDF Synthesis] Unexpected error for idx {idx}: {e}")
                synthesized[idx] = roster[idx]
            done_count += 1
            name = roster[idx].get("name", "?")
            _cb(f"  ✓ {name} ({done_count}/{total})")

    cleaned: List[Dict[str, Any]] = []
    for player in [p for p in synthesized if p is not None]:
        tokens = _normalize_token_usage(player.get("_pdf_synthesis_tokens"))
        if tokens["total_tokens"] or tokens["grounding_requests"]:
            _record_workflow_step(
                export_metrics,
                "match_pdf_export.player_synthesis",
                f"PDF synthesis ({team_label})" if team_label else "PDF synthesis",
                tokens,
                entity=str(player.get("name", "") or ""),
            )
        cleaned_player = dict(player)
        cleaned_player.pop("_pdf_synthesis_tokens", None)
        cleaned.append(cleaned_player)

    return cleaned, export_metrics


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_match_pdf(
    config: dict,
    results: dict,
    api_key: str = "",
    progress_cb: Optional[Callable[[str], None]] = None,
) -> Tuple[bytes, Dict[str, Any]]:
    def _cb(msg: str) -> None:
        if progress_cb:
            progress_cb(msg)

    synth_results = dict(results)
    export_metrics = _init_workflow_metrics("match_pdf_export")
    if api_key:
        for key in ("home_roster", "away_roster"):
            roster = results.get(key, [])
            if roster:
                label = "Local" if "home" in key else "Visitante"
                _cb(f"Sintetizando {len(roster)} jugadores ({label})...")
                synthesized_roster, roster_metrics = _synthesize_roster_for_pdf(
                    roster,
                    api_key,
                    progress_cb=progress_cb,
                    team_label=label,
                )
                synth_results[key] = synthesized_roster
                _merge_workflow_metrics(export_metrics, roster_metrics)

    _cb("Generando PDF...")
    pdf = _MatchPDF(config)
    pdf.add_cover()
    pdf.add_team_histories(synth_results)
    pdf.add_rosters(synth_results)
    pdf.add_palomo_phrases(synth_results)

    out = pdf.output()
    if isinstance(out, str):
        return out.encode('latin1'), export_metrics
    return bytes(out), export_metrics
