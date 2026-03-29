"""PalomoFacts — shared constants, option lists, pricing, and custom CSS."""
from __future__ import annotations

from datetime import datetime

# ---------------------------------------------------------------------------
# Model Constants
# ---------------------------------------------------------------------------
GEMINI_MODEL = "gemini-3.1-pro-preview"
GEMINI_FALLBACK_MODEL = "gemini-2.5-pro"
CLAUDE_MODEL = "claude-opus-4-6"
CLAUDE_HAIKU_MODEL = "claude-haiku-4-5"

# ---------------------------------------------------------------------------
# Mode Constants
# ---------------------------------------------------------------------------
MODE_PALOMO_GPT  = "palomo_gpt"
MODE_CLUB        = "club"
MODE_SELECCION   = "seleccion"

# Aliases so existing sidebar/routing references keep working
MODE_MATCH_PREP      = MODE_CLUB
MODE_MATCH_RESEARCH  = MODE_CLUB
MODE_PLAYER_RESEARCH = MODE_CLUB
MODE_SELECCIONES     = MODE_SELECCION

MODE_OPTIONS = {
    MODE_PALOMO_GPT: "🎙️ PalomoGPT",
    MODE_CLUB:       "🏠 Investigar Club",
    MODE_SELECCION:  "🌍 Investigar Selección",
}

# ---------------------------------------------------------------------------
# Date Constants
# ---------------------------------------------------------------------------
CURRENT_YEAR = datetime.now().year
CURRENT_DATE = datetime.now().strftime("%B %d, %Y")

# ---------------------------------------------------------------------------
# Tab Constants
# ---------------------------------------------------------------------------
# Shared tab indices — identical for both Club and Selección
TAB_PARTIDO  = 0   # ⚽ Partido
TAB_EQUIPO   = 1   # 🔬 Equipo / Selección
TAB_JUGADOR  = 2   # 🧑 Jugador / Convocado

# Aliases for existing Seleccion render functions
SEL_TAB_SELECCION = TAB_EQUIPO
SEL_TAB_PARTIDO   = TAB_PARTIDO
SEL_TAB_CONVOCADO = TAB_JUGADOR

# Club tab session-state key
CLUB_ACTIVE_TAB_KEY = "club_active_tab"
SEL_ACTIVE_TAB_KEY  = "sel_active_tab"

# ---------------------------------------------------------------------------
# Tournament / Match Options
# ---------------------------------------------------------------------------
TOURNAMENT_OPTIONS = [
    "La Liga",
    "Premier League",
    "Serie A",
    "Bundesliga",
    "Ligue 1",
    "UEFA Champions League",
    "UEFA Europa League",
    "UEFA Conference League",
    "Copa del Rey",
    "FA Cup",
    "Carabao Cup",
    "DFB Pokal",
    "Coppa Italia",
    "Coupe de France",
    "Supercopa de España",
    "Community Shield",
    "Supercoppa Italiana",
    "Copa América",
    "Eurocopa",
    "FIFA World Cup",
    "Liga MX",
    "MLS",
    "Copa Libertadores",
    "Copa Sudamericana",
    "Liga de Naciones UEFA",
    "Eliminatorias Mundialistas",
    "Club World Cup",
    "Amistoso Internacional",
]

MATCH_TYPE_OPTIONS = [
    "Final",
    "Semifinal",
    "Cuartos de Final",
    "Octavos de Final",
    "Fase de Grupos",
    "Jornada de Liga",
    "Eliminatoria (ida)",
    "Eliminatoria (vuelta)",
    "Partido Único (eliminatoria)",
    "Amistoso",
]

CONFEDERATION_OPTIONS = [
    "(Cualquier confederación)",
    "UEFA — Europa",
    "CONMEBOL — Sudamérica",
    "CONCACAF — N&C América",
    "CAF — África",
    "AFC — Asia",
    "OFC — Oceanía",
]

NATIONAL_TOURNAMENT_OPTIONS = [
    "Copa del Mundo FIFA",
    "Copa América",
    "UEFA EURO",
    "Copa Africana de Naciones (AFCON)",
    "Copa de Asia AFC",
    "Gold Cup (CONCACAF)",
    "Liga de Naciones UEFA",
    "Liga de Naciones CONCACAF",
    "Clasificatoria Mundialista — CONMEBOL",
    "Clasificatoria Mundialista — UEFA",
    "Clasificatoria Mundialista — CAF",
    "Clasificatoria Mundialista — CONCACAF",
    "Clasificatoria Mundialista — AFC",
    "Amistoso Internacional",
    "Juegos Olímpicos",
    "Sub-20 (FIFA World Cup)",
    "Sub-17 (FIFA World Cup)",
]

# ---------------------------------------------------------------------------
# Dashboard Constants
# ---------------------------------------------------------------------------
DASHBOARD_ACCESS_STATE_KEY = "dashboard_access_granted"
DASHBOARD_FILTER_OPTIONS = {
    "all": "All time",
    "7d": "7d",
    "30d": "30d",
    "90d": "90d",
}

USAGE_RUNTIME = "runtime"
USAGE_BACKFILL = "backfill"

# ---------------------------------------------------------------------------
# Pricing Constants
# ---------------------------------------------------------------------------
_DEFAULT_PRICING = {
    "input_per_million": 0.0,
    "input_long_per_million": 0.0,
    "output_per_million": 0.0,
    "output_long_per_million": 0.0,
    "search_per_unit": 0.0,
    "long_context_threshold": 200_000,
}

MODEL_PRICING = {
    GEMINI_MODEL: {
        "input_per_million": 2.0,
        "input_long_per_million": 4.0,
        "output_per_million": 12.0,
        "output_long_per_million": 18.0,
        "search_per_unit": 14.0 / 1000.0,
        "long_context_threshold": 200_000,
    },
    GEMINI_FALLBACK_MODEL: {
        "input_per_million": 1.25,
        "input_long_per_million": 2.50,
        "output_per_million": 10.0,
        "output_long_per_million": 15.0,
        "search_per_unit": 35.0 / 1000.0,
        "long_context_threshold": 200_000,
    },
    CLAUDE_MODEL: {
        "input_per_million": 5.0,
        "input_long_per_million": 5.0,
        "output_per_million": 25.0,
        "output_long_per_million": 25.0,
        "search_per_unit": 10.0 / 1000.0,
        "long_context_threshold": 200_000,
    },
    CLAUDE_HAIKU_MODEL: {
        "input_per_million": 1.0,
        "input_long_per_million": 1.0,
        "output_per_million": 5.0,
        "output_long_per_million": 5.0,
        "search_per_unit": 0.0,
        "long_context_threshold": 200_000,
    },
}

# Team research reuse staleness threshold
_TEAM_RESEARCH_MAX_AGE_DAYS = 30

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
/* ---- Base ---- */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(170deg, #f8fafc 0%, #f1f5f9 100%);
}
[data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 1px solid #e2e8f0;
}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] label {
    color: #475569;
}

/* Header */
.hero-title {
    font-size: 2.2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #0ea5e9 0%, #4f46e5 50%, #9333ea 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -0.5px;
    margin-bottom: 0;
    line-height: 1.2;
}
.hero-sub {
    color: #64748b;
    font-size: 0.95rem;
    margin-top: 2px;
}

/* Welcome card */
.welcome-card {
    background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
    border: 1px solid #e2e8f0;
    border-radius: 16px;
    padding: 32px;
    margin: 20px 0;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
}
.welcome-card h3 {
    color: #1e293b;
    font-size: 1.15rem;
    margin-bottom: 12px;
}
.welcome-card p {
    color: #475569;
    font-size: 0.9rem;
    line-height: 1.6;
}

/* Match prep header */
.match-header {
    background: linear-gradient(135deg, #eff6ff 0%, #e0e7ff 50%, #eff6ff 100%);
    border: 1px solid #c7d2fe;
    border-radius: 16px;
    padding: 28px 32px;
    margin: 16px 0 24px;
    text-align: center;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
}
.match-header h2 {
    color: #1e293b;
    font-size: 1.8rem;
    font-weight: 800;
    margin: 0;
}
.match-header .match-meta {
    color: #4338ca;
    font-size: 0.95rem;
    margin-top: 8px;
}

/* Team column header */
.team-col-hdr {
    background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 14px 20px;
    margin-bottom: 12px;
    text-align: center;
    box-shadow: 0 2px 4px -1px rgba(0, 0, 0, 0.05);
}
.team-col-hdr h3 {
    color: #1e293b;
    margin: 0;
    font-size: 1.15rem;
}

/* Palomo section */
.palomo-section {
    background: linear-gradient(135deg, #f5f3ff 0%, #ede9fe 100%);
    border: 1px solid #c4b5fd;
    border-radius: 16px;
    padding: 28px 32px;
    margin: 24px 0;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
}
.palomo-section h3 {
    color: #6d28d9;
    font-size: 1.3rem;
    margin-bottom: 16px;
}

/* Footer */
.app-footer {
    text-align: center;
    color: #94a3b8;
    font-size: 0.72rem;
    padding: 24px 0 12px;
    border-top: 1px solid #e2e8f0;
    margin-top: 32px;
}

/* Chat tweaks */
[data-testid="stChatMessage"] {
    background: transparent !important;
}

/* Hide Streamlit chrome */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header [data-testid="stDecoration"] {display: none;}

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #f1f5f9; }
::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #94a3b8; }
</style>
"""
