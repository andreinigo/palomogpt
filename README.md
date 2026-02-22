# ⚽ PalomoFacts

Football intelligence platform powered by **Perplexity Sonar Pro** with real-time web search. Two modes designed for broadcast preparation and football knowledge.

## Modes

### 🎙️ PalomoGPT
Conversational football assistant with the personality and style of legendary ESPN narrator Fernando Palomo. Auto-detects whether you need a quick stat or a deep analysis and adapts accordingly.

- Any league, any era, any player
- Real-time verified data with source citations
- Conversational memory across the session

### ⚽ Preparación de Partidos
Structured match preparation reports for broadcast. Generates a comprehensive dossier in three phases:

1. **Team Histories** — Last 2 seasons per team with full competition breakdowns, European match-by-match results, and key stats
2. **Player-by-Player Dossiers** — Individual deep research for every squad member including:
   - Full biography and career trajectory
   - Connections with the opposing team (ex-players, rejected transfers, shared national teammates, memorable performances against them)
   - Personal life, fun facts, and family football ties
   - Season stats, tactical profile, and current form
   - Contract situation and transfer context
3. **Fernando Palomo Phrases** — Broadcast-ready narration lines with verified facts, obscure connections, and dramatic flair

## Setup

### Requirements

- Python 3.10+
- A [Perplexity AI](https://www.perplexity.ai/) API key with access to `sonar-pro`

### Installation

```bash
pip install -r requirements.txt
```

### Environment Variables

Secrets are managed via [Streamlit secrets management](https://docs.streamlit.io/develop/concepts/connections/secrets-management).

**Locally**, create `.streamlit/secrets.toml` (already gitignored):

```toml
PERPLEXITY_API_KEY = "pplx-your-key-here"
```

**On Streamlit Cloud**, add `PERPLEXITY_API_KEY` in the app's Secrets settings.

### Run

```bash
streamlit run chatbot_app_v4.py
```

## Tech Stack

- **Streamlit** — UI framework
- **Perplexity Sonar Pro** — LLM with real-time web search
- **ThreadPoolExecutor** — Parallel API calls for faster research (batches of 4 for player dossiers)

## Architecture

```
User Input
    │
    ├─ PalomoGPT Mode ──► Single Perplexity call with chat history
    │
    └─ Match Prep Mode
         │
         ├─ Phase 1: Team histories (2 parallel calls)
         │
         ├─ Phase 2: Player rosters
         │    ├─ Fetch player list (1 lightweight call per team)
         │    └─ Per-player dossier (parallel batches of 4)
         │
         └─ Phase 3: Palomo narration phrases (1 call with history context)
```
