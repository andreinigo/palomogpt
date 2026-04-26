# ⚽ PalomoFacts (v4)

Football intelligence platform powered by **Google Gemini** with real-time web search and verified stats. Designed for high-end broadcast preparation, rigorous sports journalism, and unparalleled football knowledge.

## 🌟 Core Features

- **Zero Hallucination Policy:** Strict prompt engineering ensures the model refuses to invent stats or goalscorers. Omission of uncertain facts is prioritized over AI hallucination.
- **Supabase Persistence:** All research and chat histories are automatically synced to the cloud. Pick up where you left off from the left sidebar via seamless Programmatic Tab navigation.
- **On-Demand PDF Reports:** Generate clean, print-ready PDF dossiers instantly from any completed research package.

## 🧭 Application Modes

### 🎙️ PalomoGPT
Conversational football assistant with the personality and style of legendary ESPN narrator Fernando Palomo. Auto-detects whether you need a quick exact stat or a deep colorful narrative.
- Any league, any era, any player.
- Conversational memory powered by Supabase.

### 🏠 Investigar Club
Structured, broadcast-ready research exclusively for Club Football. Features three programmatic tabs:
1. **⚽ Partido:** Full match preparation including Head-to-Head histories, roster breakdowns, and Palomo narration phrases.
2. **🔬 Equipo:** Deep dive into a specific club's recent history, including tournament paths, final standings, and top scorers.
3. **🧑 Jugador:** Individual dossiers covering personal connections to rivals, family background, tactical evolution, and current form. 

### 🌍 Investigar Selección
The international football counterpart to the Club mode:
1. **⚽ Partido:** Head-to-Head analysis for national fixtures (World Cup, Qualifiers, Friendlies).
2. **🔬 Selección:** Historical records, World Cup runs, Confederation titles, and cultural/heroic summaries of any national team.
3. **🧑 Convocado:** Focuses intensely on a player's international cap history, debut, and performance in major tournaments compared to their club form.

## 🚀 Setup & Installation

### Requirements
- Python 3.10+
- A Google Gemini API Key
- A Supabase Project (URL & Service Key)
- Streamlit 1.50+ for hidden `/dashboard` routing with `st.Page` and `st.navigation`

### Installation
```bash
pip install -r requirements.txt
```

### Environment Variables
Locally, create a `.streamlit/secrets.toml` file (this is safely gitignored):
```toml
GEMINI_API_KEY = "your-gemini-key"
SUPABASE_URL = "https://your-project.supabase.co"
SUPABASE_KEY = "your-supabase-anon-or-service-key"
APP_ACCESS_KEY = "your-app-password"
DASHBOARD_ACCESS_KEY = "your-admin-dashboard-password"
```

### Run the App
```bash
streamlit run chatbot_app_v4.py
```

### Usage Dashboard
- Apply the existing setup SQL files plus `setup_usage_dashboard.sql` in Supabase.
- Open the normal app at `/`.
- If `APP_ACCESS_KEY` is configured, the whole app at `/` is protected by a simple password wall.
- Open the hidden admin dashboard by going directly to `/dashboard` on the same Streamlit app URL.
- The dashboard is not linked anywhere in the UI and requires `DASHBOARD_ACCESS_KEY`.
- Historical structured workflow usage can be backfilled from the dashboard; older PalomoGPT and legacy PDF exports from before this feature cannot be reconstructed.

## 🛠️ Architecture & Tech Stack

- **Streamlit** — Reactive UI framework with `st.pills` for programmatic state navigation.
- **Google Gemini (google-genai)** — LLM engine specifically prompted for strict mathematical fact-checking.
- **Supabase** — PostgreSQL backend for persistent historic state (JSON dicts).
- **Usage Ledger** — `usage_runs` stores per-run token, search, and raw-cost telemetry for admin analytics.
- **FPDF** — PDF generation engine for physical broadcaster dossiers.
- **ThreadPoolExecutor** — Parallel API calls for heavily nested tasks (e.g. batching 4 player dossiers simultaneously).

## ✅ Recent Changes

- **Blank/Stuck Protection:** Empty Gemini responses now trigger automatic retry (3 attempts with exponential backoff + Claude fallback). Investigations no longer get stuck or produce blank sections.
- **Segmented Player Dossiers:** Player research in match prep now produces a reusable base profile and a separate opponent-specific connections analysis (⚡ Conexiones con el rival). Base profiles can be reused across different match preparations.
- **Team Data Auto-Save & Reuse:** After a successful match prep, each team's history, coach, and roster are auto-saved to `team_researches`. On subsequent matches, team history is pre-loaded (with a 30-day staleness check), while roster and coach are always re-researched to ensure freshness across seasons/transfer windows.
- **Femenino Switch:** All match and team research modes now include a toggle for women's football, appending the appropriate context to prompts and searches.
- **Coach Research:** Coach dossiers are now included as part of team investigation in match preparations.
- **Player Expander Navigation:** Each player dossier now has a "⬆️ Volver al listado" button at the bottom that collapses all expanders and scrolls back to the top of the roster section.

## 📋 TODO

- (none currently)
