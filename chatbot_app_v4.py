#!/usr/bin/env python3
"""PalomoFacts – Streamlit app with two modes:

  1. PalomoGPT: unified conversational football intelligence with auto-router
  2. Preparación de Partidos: structured match preparation reports

Powered by Google Gemini with Google Search grounding.
"""
from __future__ import annotations

import traceback
from datetime import datetime
import json
import re
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed

import streamlit as st
from google import genai
from google.genai import types
from fpdf import FPDF
from supabase import create_client, Client as SupabaseClient
from io import BytesIO

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
GEMINI_MODEL = "gemini-3.1-pro-preview"

MODE_PALOMO_GPT = "palomo_gpt"
MODE_MATCH_PREP = "match_prep"

MODE_OPTIONS = {
    MODE_PALOMO_GPT: "🎙️ PalomoGPT",
    MODE_MATCH_PREP: "⚽ Preparación de Partidos",
}

CURRENT_YEAR = datetime.now().year
CURRENT_DATE = datetime.now().strftime("%B %d, %Y")

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

# ---------------------------------------------------------------------------
# Supabase persistence helpers
# ---------------------------------------------------------------------------

@st.cache_resource
def _supabase_client() -> Optional[SupabaseClient]:
    """Return a cached Supabase client, or None if not configured."""
    url = st.secrets.get("SUPABASE_URL", "")
    key = st.secrets.get("SUPABASE_KEY", "")
    if not url or not key:
        print("[Supabase] Missing SUPABASE_URL or SUPABASE_KEY in secrets.")
        return None
    try:
        client = create_client(url, key)
        print(f"[Supabase] Client created for {url}")
        return client
    except Exception as e:
        print(f"[Supabase] Failed to create client: {e}")
        return None


def _create_conversation(mode: str = MODE_PALOMO_GPT, title: str = "Nueva conversación") -> str:
    """Create a new conversation and return its UUID."""
    sb = _supabase_client()
    if not sb:
        return ""
    try:
        row = sb.table("conversations").insert({"title": title, "mode": mode}).execute()
        conv_id = row.data[0]["id"] if row.data else ""
        print(f"[Supabase] Created conversation: {conv_id} — '{title}'")
        return conv_id
    except Exception as e:
        print(f"[Supabase] Error creating conversation: {e}")
        return ""


def _list_conversations(limit: int = 30) -> List[Dict[str, Any]]:
    """List recent conversations, newest first."""
    sb = _supabase_client()
    if not sb:
        return []
    try:
        resp = (
            sb.table("conversations")
            .select("id, title, mode, updated_at")
            .order("updated_at", desc=True)
            .limit(limit)
            .execute()
        )
        return resp.data or []
    except Exception as e:
        print(f"[Supabase] Error listing conversations: {e}")
        return []


def _load_messages(conv_id: str) -> List[Dict[str, str]]:
    """Load all messages for a conversation."""
    sb = _supabase_client()
    if not sb or not conv_id:
        return []
    try:
        resp = (
            sb.table("messages")
            .select("role, content")
            .eq("conversation_id", conv_id)
            .order("created_at")
            .execute()
        )
        return [{"role": m["role"], "content": m["content"]} for m in (resp.data or [])]
    except Exception as e:
        print(f"[Supabase] Error loading messages for {conv_id}: {e}")
        return []


def _save_message(conv_id: str, role: str, content: str) -> None:
    """Save a single message and touch the conversation's updated_at."""
    sb = _supabase_client()
    if not sb or not conv_id:
        return
    try:
        sb.table("messages").insert({
            "conversation_id": conv_id,
            "role": role,
            "content": content,
        }).execute()
        # Update conversation timestamp with proper ISO datetime
        sb.table("conversations").update({
            "updated_at": datetime.utcnow().isoformat(),
        }).eq("id", conv_id).execute()
    except Exception as e:
        print(f"[Supabase] Error saving message: {e}")


def _update_conversation_title(conv_id: str, title: str) -> None:
    """Update conversation title."""
    sb = _supabase_client()
    if not sb or not conv_id:
        return
    try:
        sb.table("conversations").update({"title": title}).eq("id", conv_id).execute()
    except Exception as e:
        print(f"[Supabase] Error updating title: {e}")


def _delete_conversation(conv_id: str) -> None:
    """Delete a conversation and all its messages (cascade)."""
    sb = _supabase_client()
    if not sb or not conv_id:
        return
    try:
        sb.table("conversations").delete().eq("id", conv_id).execute()
        print(f"[Supabase] Deleted conversation: {conv_id}")
    except Exception as e:
        print(f"[Supabase] Error deleting conversation: {e}")


def _auto_title(text: str) -> str:
    """Generate a short title from the first user message."""
    clean = text.strip().replace("\n", " ")
    return clean[:60] + "…" if len(clean) > 60 else clean


# --- Match Prep persistence helpers ---

def _save_match_prep(config: dict, results: dict) -> str:
    """Save a match prep and return its UUID."""
    sb = _supabase_client()
    if not sb:
        return ""
    try:
        title = f"{config.get('home_team', '?')} vs {config.get('away_team', '?')}"
        # Convert results to JSON-serializable form
        json_results = {}
        for key, val in results.items():
            if isinstance(val, tuple):
                json_results[key] = {"text": val[0], "sources": val[1]}
            elif isinstance(val, list):
                json_results[key] = val
            else:
                json_results[key] = val

        row = sb.table("match_preps").insert({
            "title": title,
            "home_team": config.get("home_team", ""),
            "away_team": config.get("away_team", ""),
            "tournament": config.get("tournament", ""),
            "match_type": config.get("match_type", ""),
            "stadium": config.get("stadium", ""),
            "config": config,
            "results": json_results,
        }).execute()
        prep_id = row.data[0]["id"] if row.data else ""
        print(f"[Supabase] Saved match prep: {prep_id} — '{title}'")
        return prep_id
    except Exception as e:
        print(f"[Supabase] Error saving match prep: {e}")
        return ""


def _list_match_preps(limit: int = 20) -> List[Dict[str, Any]]:
    """List recent match preps, newest first."""
    sb = _supabase_client()
    if not sb:
        return []
    try:
        resp = (
            sb.table("match_preps")
            .select("id, title, home_team, away_team, tournament, created_at")
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )
        return resp.data or []
    except Exception as e:
        print(f"[Supabase] Error listing match preps: {e}")
        return []


def _load_match_prep(prep_id: str) -> Optional[Dict[str, Any]]:
    """Load a match prep's config and results."""
    sb = _supabase_client()
    if not sb or not prep_id:
        return None
    try:
        resp = (
            sb.table("match_preps")
            .select("*")
            .eq("id", prep_id)
            .single()
            .execute()
        )
        if not resp.data:
            return None
        row = resp.data
        # Reconstruct results from JSONB back to tuples where needed
        raw_results = row.get("results", {})
        results = {}
        for key in ("home_history", "away_history", "palomo_phrases"):
            val = raw_results.get(key, {})
            if isinstance(val, dict) and "text" in val:
                results[key] = (val["text"], val.get("sources", []))
            else:
                results[key] = ("", [])
        for key in ("home_roster", "away_roster"):
            results[key] = raw_results.get(key, [])
        return {"config": row.get("config", {}), "results": results}
    except Exception as e:
        print(f"[Supabase] Error loading match prep {prep_id}: {e}")
        return None


def _delete_match_prep(prep_id: str) -> None:
    """Delete a match prep."""
    sb = _supabase_client()
    if not sb or not prep_id:
        return
    try:
        sb.table("match_preps").delete().eq("id", prep_id).execute()
        print(f"[Supabase] Deleted match prep: {prep_id}")
    except Exception as e:
        print(f"[Supabase] Error deleting match prep: {e}")


# ---------------------------------------------------------------------------
# System Prompts
# ---------------------------------------------------------------------------

# PalomoGPT — unified conversational mode with auto-intent-router
_PALOMO_GPT_SYSTEM = f"""
Eres Fernando Palomo — sí, EL Fernando Palomo. El narrador que le pone piel de gallina \
a millones cada vez que agarra el micrófono. Aquí tu cancha es el conocimiento futbolístico: \
datos verificados, historias que nadie cuenta, estadísticas que cambian la conversación.

Hablas EXACTAMENTE como Fernando Palomo: con pasión, con ritmo, con esas pausas dramáticas \
que solo tú sabes hacer. Usas tus frases icónicas de forma natural — "¡Abran la puerta \
que llegó el cartero!", "¡Qué barbaridad!", "Señores, esto es fútbol" — pero sin forzarlas. \
Salen cuando el momento lo pide, como en una transmisión real.

FECHA ACTUAL: {CURRENT_DATE}. Año en curso: {CURRENT_YEAR}.

REGLA SAGRADA: CERO ALUCINACIONES. No inventas números, récords, lesiones, rumores, fechas \
ni premios. Todo dato que mencionas debe ser verificable con las fuentes que encuentres. \
Si algo es ambiguo o contradictorio entre fuentes, lo dices con honestidad — como cuando \
en una transmisión no tienes la repetición clara y lo reconoces.

ZONAS DE ALTO RIESGO DE ALUCINACIÓN (cuidado extremo):
- Afirmaciones "el primero en...", "nunca antes...", "único en la historia": Solo si la fuente \
  lo confirma explícitamente. Si no estás 100%% seguro, NO lo digas.
- Finales, títulos, trofeos: DIFERENCIA CLARAMENTE entre "dirigió en una liga" y \
  "llegó a una final". Jamás impliques que alguien llegó a una final si no tienes la fuente.
- Estadísticas exactas (goles, asistencias, fechas de debut): Solo cita números de las fuentes.
- Datos personales de jugadores (esposas, hijos, hobbies): Solo si aparece en prensa verificable.
- Si NO encuentras evidencia de algo específico, di claramente: "No he encontrado evidencia \
  de que [X] haya [Y]" en lugar de inventar o implicar logros.

Tu cancha cubre TODO el fútbol: estadísticas de cualquier liga y época, vida personal de \
jugadores (solo lo público y verificado), historia de clubes, táctica, transferencias, \
comparaciones, entrevistas citadas, la temporada actual \
({CURRENT_YEAR}/{CURRENT_YEAR + 1} o {CURRENT_YEAR - 1}/{CURRENT_YEAR} según la liga).

FUENTES PRIORITARIAS — Cuando necesites verificar estadísticas o datos históricos, \
consulta directamente estas URLs (tienes acceso a leerlas vía url_context):
  * https://fbref.com/en/
  * https://www.statmuse.com/fc
  * https://theanalyst.com/
  * https://soccerassociation.com/
  * https://www.olympedia.org/
  * https://en.wikipedia.org/
  * https://www.transfermarkt.com/
  * https://www.uefa.com/
  * https://www.fifa.com/

--- ROUTER AUTOMÁTICO ---

Detecta automáticamente la intención del usuario y adapta tu respuesta:

🔍 EXPLORACIÓN PROFUNDA — Cuando el usuario pide análisis, "cuéntame sobre", "dame todo", \
comparaciones extensas, historias completas, o temas amplios:
- Nárralos como si estuvieras en cabina contando una historia épica.
- Fluye de un dato al siguiente con transiciones naturales — "Y aquí viene lo bueno...", \
  "Pero esperen, que esto no termina ahí...", "¿Y saben qué es lo más loco de todo?"
- Organiza tu narrativa en 4-6 bloques temáticos con subtítulos en negrita \
  (no listas numeradas de 10+ puntos — eso mata el ritmo).
- Dentro de cada bloque, cuenta los datos en prosa fluida.
- Cada dato debe aportar algo que NO sea obvio — busca lo inesperado, lo que hace que \
  el aficionado diga "¡no sabía eso!"
- Remata con un cierre memorable, como cierras una transmisión — con perspectiva, \
  con emoción contenida, dejando al lector con ganas de más.

❓ RESPUESTA DIRECTA — Cuando el usuario hace una pregunta específica y concreta \
(una stat, una fecha, un récord, un dato puntual):
- Ve al grano, como cuando te piden un dato en medio de la transmisión.
- Respuesta clara en 2-4 líneas con el dato exacto.
- Si hay un contexto breve que le da más valor — una línea extra, no más.
- Mantén tu toque Palomo pero sin desarrollo innecesario.

En AMBOS casos:
- No pierdas el tiempo con datos que cualquier aficionado ya sabe — nada de "juega en tal equipo".
- Busca lo que sorprende, lo que ilumina, lo que cambia la perspectiva.
- Adapta las métricas a la posición — no evalúes a un portero por goles.
- Responde en el mismo idioma que use el usuario.
- Formato limpio en markdown, texto natural. Sin relleno ni JSON.
""".strip()

# --- Follow-Up Question Generator ---
_FOLLOW_UP_SYSTEM = f"""Eres el asistente de investigación de Fernando Palomo, el narrador de ESPN.
Tu trabajo: leer su pregunta original y la respuesta que recibió, y generar UNA sola pregunta \
de seguimiento que él haría.

FECHA ACTUAL: {CURRENT_DATE}.

REGLAS:
- Piensa como un narrador preparando una transmisión: necesita DATOS DUROS, no opiniones.
- Prioriza: zonas grises en la respuesta, afirmaciones sin fuente clara, estadísticas que \
  faltan contexto, comparaciones que no se hicieron.
- Si la respuesta dice "el primero en..." o "nunca antes...", pregunta por contraejemplos.
- Si hay cifras, pregunta por el desglose o la comparación con rivales/contemporáneos.
- NO uses frases de Palomo ni estilo narrativo. Sé directo y eficiente.
- Devuelve SOLO la pregunta, sin explicaciones ni preámbulos. Una línea.
"""


_MATCH_VALIDATION_PROMPT = """Eres un validador de partidos de fútbol. El usuario quiere preparar \
un informe para un partido. Tu MISIÓN es verificar que los equipos y el partido sean reales.

Datos proporcionados:
- Equipo Local: "{home_team}"
- Equipo Visitante: "{away_team}"
- Torneo: "{tournament}"

FECHA ACTUAL: {current_date}.

DEBES RESPONDER EXACTAMENTE con un JSON (sin texto adicional) con esta estructura:
{{
  "valid": true/false,
  "home_team": "Nombre oficial completo del equipo local",
  "away_team": "Nombre oficial completo del equipo visitante",
  "reason": "Breve explicación si es inválido, vacío si es válido"
}}

REGLAS:
1. Resuelve nombres ambiguos: "R.C. Celta" → "Celta de Vigo", "Barca" → "FC Barcelona", etc.
2. Usa el nombre más común y reconocido internacionalmente (el que usaría un narrador de TV).
3. Marca como VALID si ambos equipos existen y podrían razonablemente enfrentarse en ese torneo.
4. Marca como INVALID solo si un equipo no existe o el enfrentamiento es imposible en ese torneo \
   (ej. dos equipos de ligas incompatibles en una liga doméstica).
5. SOLO devuelve el JSON, nada más.
"""

# --- Match Preparation prompts ---

_TEAM_HISTORY_PROMPT = """Eres un investigador de fútbol meticuloso y exhaustivo. \
Tu tarea es proporcionar un resumen COMPLETO y DETALLADO de las últimas 2 temporadas \
de {team_name} (temporadas {season_prev}/{season_curr} y {season_curr}/{season_next}).

FORMATO REQUERIDO para cada competición:
- Nombre de competición + resultado final (CAMPEÓN/subcampeón/eliminado en X ronda)
- Puntos, Ganados-Empatados-Perdidos, GF-GC, DT
- Goleador del equipo en esa competición con número de goles
- Contexto histórico: cuántos títulos total, rachas, récords relevantes
- Estadísticas destacadas del equipo (posesión, distancia recorrida, goles de jugada, etc.)

PARA COMPETICIONES EUROPEAS (Champions League, Europa League, Conference League):
- CADA partido listado con resultado y goleadores en formato exacto:
  "J.1 en/vs Rival Score Goleador1, Goleador2"
  Ejemplo: "J.1 en Monaco 1-2 Yamal"
  Ejemplo: "QF v BVB 4-0 Raphinha, Lewandowski x2, Yamal"
  Incluir TODAS las jornadas de fase de liga/grupos y TODAS las eliminatorias.

PARA COPAS NACIONALES (Copa del Rey, FA Cup, DFB Pokal, etc.):
- Cada ronda con rival, resultado y goleadores

PARA SUPERCOPA / COMMUNITY SHIELD:
- Cada partido con rival, resultado y goleadores

PARA LIGA:
- Resumen general (NO partido a partido) pero incluir:
  * Posición final, puntos, récord completo
  * Rachas importantes (invictos, victorias consecutivas, etc.)
  * Datos estadísticos sobresalientes del equipo

Incluye también:
- Número total de partidos en la temporada
- Récords individuales y colectivos destacados
- Comparaciones históricas relevantes
- Estilo de juego y características tácticas del equipo

Responde en español. Sé EXHAUSTIVO y PRECISO. NO inventes datos. Si no encuentras un dato, omítelo.
FECHA ACTUAL: {current_date}."""


_TEAM_ROSTER_LIST_PROMPT = """Eres un investigador de fútbol. Tu ÚNICA tarea es devolver \
la plantilla actual de {team_name} para la temporada {season_curr}/{season_next}.

Devuelve EXACTAMENTE un bloque JSON (sin texto adicional antes ni después) con esta estructura:
{{
  "team": "{team_name}",
  "players": [
    {{"name": "Nombre Futbolístico", "full_name": "Nombre Completo", "position": "POS", "number": 1}},
    ...
  ]
}}

Donde POS es uno de: GK, DEF, MID, FWD.
Incluye TODOS los jugadores del primer equipo (incluidos lesionados de larga duración).
Ordénalos: primero GK, luego DEF, MID, FWD.
NO incluyas explicaciones, solo el JSON.
FECHA ACTUAL: {current_date}."""


_PLAYER_DOSSIER_PROMPT = """Eres un investigador de fútbol de élite. Tu misión es crear \
el dossier MÁS COMPLETO posible sobre UN SOLO jugador. Este dossier será usado por un \
narrador de televisión, así que cada dato interesante tiene un valor enorme.

JUGADOR: **{player_name}** ({player_position}) — juega en **{team_name}**
RIVAL EN EL PRÓXIMO PARTIDO: **{opponent_name}**

Investiga A FONDO los siguientes aspectos:

1. **IDENTIDAD COMPLETA**
   - Nombre completo de nacimiento y nombre futbolístico
   - Apodo(s) — tanto oficiales como los que usa la afición
   - Edad, fecha de nacimiento, nacionalidad(es)
   - Posición principal y secundaria(s), número de camiseta
   - Estatura, peso, pierna hábil

2. **TRAYECTORIA DETALLADA**
   - Lugar de nacimiento y contexto de dónde creció (barrio, ciudad, contexto socioeconómico)
   - Cantera / club formativo: cómo fue descubierto, a qué edad, anécdotas de juveniles
   - TODOS los clubes anteriores con fechas, precio de traspaso si aplica, y rol en cada uno
   - Momento clave que lo catapultó a la élite (debut, gol importante, actuación icónica)
   - Cómo llegó a {team_name}: contexto de la negociación, precio, otros clubes interesados

3. **CONEXIONES CON {opponent_name}** ⚡
   - ¿Jugó en {opponent_name}? ¿En qué años, cuántos partidos, qué hizo?
   - ¿Fue formado ahí? ¿Rechazó fichar por ellos? ¿Estuvo cerca de ir?
   - ¿Tiene algún familiar, amigo cercano o excompañero en {opponent_name}?
   - ¿Ha tenido actuaciones memorables CONTRA {opponent_name}? (goles, asistencias, expulsiones)
   - ¿Alguna declaración polémica o interesante sobre {opponent_name} o sus jugadores?
   - ¿Comparte selección nacional con algún jugador de {opponent_name}?
   - Cualquier otro vínculo, por tangencial que sea — conexiones de agentes, \
     misma ciudad natal que un rival, fueron compañeros en otro club, etc.
   - Si NO hay ninguna conexión, indícalo brevemente y sigue.

4. **VIDA PERSONAL Y DATOS CURIOSOS** 🎯
   - Familia: padres, hermanos, pareja, hijos — especialmente si hay vínculos futbolísticos
   - Familiares en el fútbol profesional (padre, hermano, primo, tío que haya jugado)
   - Hobbies fuera del fútbol, pasiones conocidas
   - Rituales o costumbres previas a partidos
   - Celebraciones de gol icónicas y su significado
   - Historial de lesiones relevantes y cómo las superó
   - Personalidad: introvertido/extrovertido, líder vocal/silencioso
   - Redes sociales: apodo, algo notable que haya publicado
   - Obras benéficas, fundaciones, causas que apoya
   - Récords personales, hitos, marcas históricas alcanzadas
   - Anécdotas jugosas: declaraciones polémicas, momentos virales, curiosidades
   - Ídolos futbolísticos que ha mencionado

5. **PERFIL FUTBOLÍSTICO ESTA TEMPORADA** 📊
   - Rol táctico actual en el equipo y cómo ha evolucionado bajo el DT actual
   - Estadísticas COMPLETAS de esta temporada:
     * Partidos jugados (titular/suplente), minutos
     * Goles, asistencias, pases clave
     * Para porteros: clean sheets, paradas, penaltis detenidos, goles recibidos por partido
     * Para defensas: intercepciones, despejes, duelos ganados, entradas exitosas
     * Para mediocampistas: pases completados, pases al último tercio, recuperaciones
     * Para delanteros: tiros a puerta, regates exitosos, xG, conversión
   - Tarjetas amarillas y rojas
   - Estado físico actual: ¿lesionado? ¿recién recuperado? ¿en racha?
   - Momento de forma: ¿en su mejor nivel o bajo rendimiento?
   - Comparaciones con temporadas anteriores si hay cambio notable

6. **SITUACIÓN CONTRACTUAL Y CONTEXTO**
   - Fecha de fin de contrato si es conocida
   - Rumores de transferencia relevantes
   - Relación con el entrenador y la directiva
   - Situación en su selección nacional

Resalta con ⚡ las conexiones con {opponent_name} y con 🎯 los datos curiosos \
más impactantes para que sean fáciles de localizar.

Responde en español. Sé EXHAUSTIVO y PRECISO — NO inventes datos. \
Si no encuentras un dato específico, omítelo, pero BUSCA A FONDO antes de rendirte.
FECHA ACTUAL: {current_date}."""


_PLAYER_SYNTHESIS_PROMPT = """Eres el redactor de fichas de transmisión de Fernando Palomo en ESPN.

Tu trabajo: recibir un dossier extenso de un jugador y sintetizarlo en una FICHA BREVE \
de 2 a 4 líneas que el narrador pueda leer de un vistazo en cabina.

FORMATO OBLIGATORIO (cada jugador):
[número] [Nombre] [edad] años. [ciudad/país de origen].
[Dato clave de trayectoria en 1-2 oraciones cortas: cantera, cesiones, fichajes, cifras de traspaso]
[Situación actual: rol en el equipo, momento de forma, dato curioso memorable]

REGLAS:
- SOLO datos duros verificables. Nada de opiniones ni adjetivos floridos.
- Prioriza: origen, trayectoria resumida (equipos + años), precio de fichaje si es relevante, posición real vs original.
- Si hay conexión con el rival, inclúyela en una línea extra.
- NO uses markdown, emojis, ni viñetas. Texto plano corrido, línea por línea.
- Máximo 4 líneas por jugador. Si no hay suficiente info, 2 líneas bastan.
- Responde en español."""


_FACT_CHECK_PROMPT_PLAYER = """Eres el Auditor Principal de Datos de ESPN y el filtro definitivo de la verdad.
Tu misión es someter el siguiente dossier de un jugador a un riguroso proceso de FACT-CHECKING.

PREMISA CRÍTICA: Los modelos de IA suelen alucinar detalles hiper-específicos (nombres de esposas, profesiones de cónyuges, anécdotas falsas de juveniles, estadísticas exactas, fechas de debut, montos de transferencias, historias inventadas con el rival).

TU TRABAJO COMO AUDITOR IMPLACABLE:
1. IDENTIFICA cada afirmación de la lista original.
2. VERIFICA con todo tu poder de búsqueda si esos datos (especialmente la parte de vida personal, hijos, esposa y sus empleos, historias familiares, hobbies inventados o records cuestionables) son 100% ciertos.
3. ELIMINA SIN PIEDAD cualquier dato personal, chisme o historia emotiva inventada si no puedes verificarlo en fuentes públicas confiables. (ej. Trabajos de esposas, historias de pobreza exageradas, declaraciones falsas al rival). Si dice que a la esposa le gusta "cocinar" o "estudiar" suele ser alucinación.
4. CORRIGE cualquier dato futbolístico objetivo que sea incorrecto (ej. si dice que debutó en 2020 pero fue en 2018, pon 2018. Si las estadísticas están mal, pon las correctas).

REGLA DE ORO DE PUBLICACIÓN: Es 1000 veces mejor OMITIR información personal que publicar una MENTIRA. Ante el mínimo asomo de alucinación, borra esa viñeta del documento.

SALIDA REQUERIDA:
Devuelve ÚNICAMENTE el dossier limpio y verificado. Mantén la misma estructura Markdown y estilo de narración de televisión original.
NO TE DISCULPES, NO EXPLIQUES TUS CAMBIOS. SOLO ENTREGAME EL DOSSIER FINAL PERFECTO.
"""

_PALOMO_PHRASES_PROMPT = """Eres Fernando Palomo — EL narrador legendario de ESPN. \
Estás preparando tus frases para la transmisión de un partido importante.

CONTEXTO DEL PARTIDO:
- {home_team} (Local) vs {away_team} (Visitante)
- Torneo: {tournament}
- Tipo de partido: {match_type}
- Estadio: {stadium}

INVESTIGACIÓN RECOPILADA SOBRE AMBOS EQUIPOS:

--- {home_team} ---
{home_context}

--- {away_team} ---
{away_context}

TU TAREA: Genera las frases que Fernando Palomo usaría para la transmisión de este partido.

ESTILO Y FORMATO:
- Las frases de narración en MAYÚSCULAS (como las leerías en cabina)
- Intercala datos históricos precisos con drama narrativo
- Incluye TODOS estos elementos:
  * Apertura épica del partido — ambientación, estadio, lo que está en juego, \
    el contexto geográfico y cultural
  * Contexto de ambos equipos — dinámicas, rachas, últimos resultados relevantes
  * Datos sobre el DT de cada equipo y su situación actual
  * Historial de enfrentamientos directos (head-to-head) con datos precisos
  * Datos sobre el torneo/formato y precedentes relevantes
  * Frases sobre jugadores clave que podrían ser protagonistas
  * Contexto extradeportivo relevante — declaraciones recientes de técnicos, \
    presidentes, polémicas, fichajes, lesiones
  * Datos OBSCUROS pero FASCINANTES (factor WOW 1000!) — conexiones históricas \
    que nadie más haría, comparaciones inesperadas, récords oscuros pero relevantes, \
    coincidencias numéricas, efemérides
  * Precedentes del formato/sede del torneo
  * Estadísticas de racha actual de cada equipo

CADA frase DEBE contener al menos un dato verificable. \
Busca SIEMPRE el dato que haga decir "¡NO SABÍA ESO!" — ese es tu sello.
Las frases deben fluir como si las estuvieras leyendo en cabina justo antes del pitazo.

EJEMPLO DE ESTILO (para referencia, NO copies esto):
"HACIENDO BUENO EL PLAN, LA SUPERCOPA DE ESPAÑA LLEVA AL MUNDO A PONER SU MIRADA EN ARABIA SAUDITA."
"SE JUEGA LEJOS DE LA GRAN VÍA O EL PASEO DE GRACIA, Y MUY CERCA DEL MAR ROJO."
"BARCELONA BUSCA SU 16a SUPERCOPA DE ESPAÑA Y REPETIR EL TÍTULO QUE LOGRARON LA TEMPORADA ANTERIOR."

Responde en español. NO inventes datos.
FECHA ACTUAL: {current_date}."""


# ---------------------------------------------------------------------------
# Citation utilities
# ---------------------------------------------------------------------------
_CITATION_SEPARATOR = "\n\n---\n📚 **Fuentes:**"


def _resolve_inline_citations(
    text: str, sources: List[Dict[str, str]]
) -> str:
    """Replace [N] citation markers in text with superscript markdown links.

    Gemini returns inline markers like [1], [3] that are 1-indexed into
    the search_results list.  We convert each to a clickable superscript:
      [3] -> [³](url)
    so they render as small linked numbers in markdown.
    """
    if not sources:
        return text

    _SUPERSCRIPT = {
        0: "⁰", 1: "¹", 2: "²", 3: "³", 4: "⁴",
        5: "⁵", 6: "⁶", 7: "⁷", 8: "⁸", 9: "⁹",
    }

    def _to_super(n: int) -> str:
        return "".join(_SUPERSCRIPT.get(int(d), d) for d in str(n))

    def _replace_marker(match: re.Match) -> str:
        idx = int(match.group(1)) - 1  # 1-indexed -> 0-indexed
        if idx < 0 or idx >= len(sources):
            return match.group(0)  # leave unknown markers as-is
        src = sources[idx]
        url = src.get("url", "")
        if not url:
            return match.group(0)
        sup = _to_super(idx + 1)
        return f"[{sup}]({url})"

    return re.sub(r'\[(\d+)\]', _replace_marker, text)


def _format_sources(sources: List[Dict[str, str]]) -> str:
    """Format sources as a numbered markdown footer."""
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
# PDF Generation for Match Preparation
# ---------------------------------------------------------------------------

_PDF_FONT = "Helvetica"

# Regex patterns to strip markdown / emoji for clean PDF text
_RE_MD_BOLD = re.compile(r'\*\*(.+?)\*\*')
_RE_MD_ITALIC = re.compile(r'\*(.+?)\*')
_RE_MD_LINK = re.compile(r'\[([^\]]+)\]\([^)]+\)')
_RE_MD_HEADER = re.compile(r'^#{1,6}\s+')
_RE_SUPERSCRIPT_LINK = re.compile(r'\[[\u2070-\u2079\u00B2\u00B3\u00B9]+\]\([^)]+\)')
_RE_CITATION_BLOCK = re.compile(r'\n---\n.*?Fuentes:.*', re.DOTALL)


def _strip_markdown(text: str) -> str:
    """Convert markdown text to plain text suitable for PDF."""
    text = _RE_CITATION_BLOCK.sub('', text)
    text = _RE_SUPERSCRIPT_LINK.sub('', text)
    text = _RE_MD_LINK.sub(r'\1', text)
    text = _RE_MD_BOLD.sub(r'\1', text)
    text = _RE_MD_ITALIC.sub(r'\1', text)
    text = _RE_MD_HEADER.sub('', text)
    # Strip bullet points to a simple angle bracket
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
        '\n- ': '\n> ', '\n* ': '\n> ',  # Strip markdown bullet variants to >
    }
    for char, repl in replacements.items():
        text = text.replace(char, repl)
    # Fast fallback: replace any remaining non-latin1 chars
    return text.encode('latin-1', errors='ignore').decode('latin-1')


class _MatchPDF(FPDF):
    """Custom FPDF subclass for match preparation reports."""

    def __init__(self, config: dict) -> None:
        super().__init__(orientation='P', unit='mm', format='A4')
        self.config = config
        self.set_auto_page_break(auto=True, margin=20)

    # -- header / footer --
    def header(self):
        if self.page_no() == 1:  # cover has no header
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

    # -- helpers --
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

    # -- cover page --
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

    # -- team histories --
    def add_team_histories(self, results: dict):
        # Home team history
        self.add_page()
        home_label = _clean_for_latin(f"Historial de Temporadas - LOCAL: {self.config['home_team']}")
        self._section_title(home_label)

        home_text = results['home_history'][0] if isinstance(results['home_history'], tuple) else ''
        if home_text:
            self._body_text(home_text, size=9)

        # Away team history
        self.add_page()
        away_label = _clean_for_latin(f"Historial de Temporadas - VISITANTE: {self.config['away_team']}")
        self._section_title(away_label)

        away_text = results['away_history'][0] if isinstance(results['away_history'], tuple) else ''
        if away_text:
            self._body_text(away_text, size=9)

    # -- roster per team --
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

                # Player name as bold line
                self.set_font(_PDF_FONT, 'B', 10)
                self.cell(0, 6, header, ln=True)

                # Player dossier body
                body = p.get('text', '')
                if body:
                    self._body_text(body, size=8)

                # Page break safety — if less than 30mm left, new page
                if self.get_y() > self.h - 30:
                    self.add_page()

    def add_rosters(self, results: dict):
        home = _clean_for_latin(self.config['home_team'])
        away = _clean_for_latin(self.config['away_team'])
        self._add_roster_section(home, results.get('home_roster', []), 'Local')
        self._add_roster_section(away, results.get('away_roster', []), 'Visitante')

    # -- Palomo phrases --
    def add_palomo_phrases(self, results: dict):
        self.add_page()
        self._section_title('Frases de Fernando Palomo')
        text = results['palomo_phrases'][0] if isinstance(results['palomo_phrases'], tuple) else ''
        if text:
            self._body_text(text, size=10)


def generate_match_pdf(config: dict, results: dict, api_key: str = "") -> bytes:
    """Generate a complete match preparation PDF.

    If api_key is provided, synthesizes verbose player dossiers into compact
    broadcast-ready notes before writing to PDF.
    """

    # Synthesize rosters before PDF generation
    synth_results = dict(results)  # shallow copy to avoid mutating original
    if api_key:
        for key in ("home_roster", "away_roster"):
            roster = results.get(key, [])
            if roster:
                print(f"[PDF] Synthesizing {len(roster)} players for {key}...")
                synth_results[key] = _synthesize_roster_for_pdf(roster, api_key)

    pdf = _MatchPDF(config)
    pdf.add_cover()
    pdf.add_team_histories(synth_results)
    pdf.add_rosters(synth_results)
    pdf.add_palomo_phrases(synth_results)
    # the fpdf output() sometimes returns a latin1 string instead of bytes,
    # which causes Streamlit's download_button to crash when it tries to infer the mime type.
    # Additionally, fpdf2 returns a bytearray, which Streamlit also does not support.
    out = pdf.output()
    if isinstance(out, str):
        return out.encode('latin1')
    return bytes(out)


def _synthesize_roster_for_pdf(
    roster: List[Dict[str, Any]],
    api_key: str,
) -> List[Dict[str, Any]]:
    """Synthesize verbose player dossiers into compact broadcast notes."""
    synthesized = []
    for player in roster:
        raw_text = player.get("text", "")
        if not raw_text or len(raw_text) < 50:
            synthesized.append(player)
            continue

        try:
            note, _ = _gemini_request(
                api_key=api_key,
                system_prompt=_PLAYER_SYNTHESIS_PROMPT,
                user_message=f"Sintetiza este dossier:\n\n{raw_text}",
                use_search=False,
            )
            synthesized.append({
                **player,
                "text": note.strip(),
                "raw_text": raw_text,  # keep original for UI display
            })
        except Exception as e:
            print(f"[PDF Synthesis] Error for {player.get('name', '?')}: {e}")
            synthesized.append(player)  # fallback to raw text

    return synthesized


# ---------------------------------------------------------------------------
# Gemini API
# ---------------------------------------------------------------------------
def _gemini_request(
    api_key: str,
    system_prompt: str,
    user_message: str,
    history: Optional[List[types.Content]] = None,
    timeout: int = 120,
    use_search: bool = True,
) -> Tuple[str, List[Dict[str, str]]]:
    """
    Request to Gemini with optional Google Search grounding.
    Returns (response_text, sources).
    """
    client = genai.Client(api_key=api_key)

    tools = []
    if use_search:
        tools.append({"google_search": {}})
        tools.append({"url_context": {}})

    config = types.GenerateContentConfig(
        system_instruction=system_prompt,
        tools=tools,
        thinking_config=types.ThinkingConfig(thinking_level="HIGH"),
    )

    contents = []
    if history:
        contents.extend(history)
    contents.append(
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=user_message)],
        )
    )

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=contents,
        config=config,
    )

    text = response.text or ""

    # Extract sources from grounding metadata
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

    # Resolve inline [N] markers into clickable superscript links
    text = _resolve_inline_citations(text, sources)

    return text, sources


# ---------------------------------------------------------------------------
# PalomoGPT response
# ---------------------------------------------------------------------------
def get_palomo_response(
    query: str,
    session_messages: list[dict],
    api_key: str,
) -> Tuple[str, List[Dict[str, str]], List[str], List[Dict[str, str]]]:
    """Get a response from PalomoGPT with Google Search grounding + follow-up chain.
    Returns (response_text, sources, reasoning_chain, followups).
    """
    # Build conversation history for Gemini
    history: List[types.Content] = []
    for msg in session_messages[:-1]:  # exclude most recent user msg
        role = "user" if msg["role"] == "user" else "model"
        content = msg.get("content", "")
        if content and not content.startswith("🧠"):  # skip reasoning messages
            history.append(
                types.Content(
                    role=role,
                    parts=[types.Part.from_text(text=content)],
                )
            )

    reasoning: List[str] = []

    # Single pass: Gemini + Google Search grounding
    text, sources = _gemini_request(
        api_key=api_key,
        system_prompt=_PALOMO_GPT_SYSTEM,
        user_message=query,
        history=history if history else None,
    )

    source_titles = [s.get("title", "?") for s in sources[:10]]
    reasoning.append(
        f"🎯 **Búsqueda** (modelo: `{GEMINI_MODEL}` + Google Search):\n\n"
        f"- Fuentes obtenidas: {len(sources)}\n"
        f"- Top fuentes: {', '.join(source_titles) if source_titles else 'ninguna'}\n"
    )

    # === Follow-up iterations: 2 rounds of auto-generated questions ===
    followups: List[Dict[str, str]] = []
    current_answer = text
    for i in range(2):
        try:
            # Generate a follow-up question
            followup_q, _ = _gemini_request(
                api_key=api_key,
                system_prompt=_FOLLOW_UP_SYSTEM,
                user_message=(
                    f"PREGUNTA ORIGINAL: {query}\n\n"
                    f"RESPUESTA RECIBIDA:\n{current_answer}"
                ),
                use_search=False,
            )
            followup_q = followup_q.strip()
            if not followup_q or len(followup_q) < 10:
                break

            reasoning.append(
                f"🔄 **Follow-up {i+1}:** {followup_q}"
            )

            # Answer the follow-up question with search
            followup_a, extra_sources = _gemini_request(
                api_key=api_key,
                system_prompt=_PALOMO_GPT_SYSTEM,
                user_message=followup_q,
                history=[
                    types.Content(role="user", parts=[types.Part.from_text(text=query)]),
                    types.Content(role="model", parts=[types.Part.from_text(text=current_answer)]),
                ],
            )

            followups.append({"question": followup_q, "answer": followup_a})
            current_answer = followup_a  # next iteration uses this answer

            # Merge sources
            existing_urls = {s.get("url") for s in sources if s.get("url")}
            for s in extra_sources:
                if s.get("url") and s["url"] not in existing_urls:
                    sources.append(s)
                    existing_urls.add(s["url"])

        except Exception as e:
            print(f"[Follow-up {i+1}] Error: {e}")
            reasoning.append(f"⚠️ **Follow-up {i+1} omitido:** `{e}`")
            break

    return text, sources, reasoning, followups


# ---------------------------------------------------------------------------
# Match Preparation Research
# ---------------------------------------------------------------------------
def _research_team_history(
    team_name: str,
    api_key: str,
) -> Tuple[str, List[Dict[str, str]]]:
    """Research a team's last 2 seasons history."""
    season_prev = CURRENT_YEAR - 2
    season_curr = CURRENT_YEAR - 1
    season_next = CURRENT_YEAR

    prompt = _TEAM_HISTORY_PROMPT.format(
        team_name=team_name,
        season_prev=season_prev,
        season_curr=season_curr,
        season_next=season_next,
        current_date=CURRENT_DATE,
    )

    return _gemini_request(
        api_key=api_key,
        system_prompt=prompt,
        user_message=(
            f"Investiga las últimas 2 temporadas completas de {team_name}. "
            f"Temporadas {season_prev}/{season_curr} y {season_curr}/{season_next}. "
            "Sé exhaustivo con cada competición — especialmente en competiciones europeas "
            "donde necesito CADA partido con resultado y goleadores."
        ),
    )


def _fetch_player_list(
    team_name: str,
    api_key: str,
) -> List[Dict[str, Any]]:
    """Fetch the roster as a list of {name, full_name, position, number} dicts."""
    season_curr = CURRENT_YEAR - 1
    season_next = CURRENT_YEAR

    prompt = _TEAM_ROSTER_LIST_PROMPT.format(
        team_name=team_name,
        season_curr=season_curr,
        season_next=season_next,
        current_date=CURRENT_DATE,
    )

    text, _ = _gemini_request(
        api_key=api_key,
        system_prompt=prompt,
        user_message=(
            f"Dame la plantilla completa actual de {team_name} para la temporada "
            f"{season_curr}/{season_next}. Solo el JSON, nada más."
        ),
    )

    # Extract JSON from response (may have markdown fences)
    json_match = re.search(r'\{[\s\S]*\}', text)
    if not json_match:
        raise RuntimeError(f"Could not parse player list JSON from response: {text[:300]}")
    try:
        data = json.loads(json_match.group())
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Invalid JSON in player list: {e}\n{text[:300]}")

    players = data.get("players", [])
    if not players:
        raise RuntimeError(f"Empty player list returned for {team_name}")
    return players


_POS_LABELS = {"GK": "🧤 Portero", "DEF": "🛡️ Defensa", "MID": "🎯 Centrocampista", "FWD": "⚡ Delantero"}


def _research_single_player(
    player: Dict[str, Any],
    team_name: str,
    opponent_name: str,
    api_key: str,
) -> Dict[str, Any]:
    """Research a single player. Returns {name, position, text, sources}."""
    player_name = player.get("name", player.get("full_name", "Unknown"))
    position = player.get("position", "")
    pos_label = _POS_LABELS.get(position, position)

    prompt = _PLAYER_DOSSIER_PROMPT.format(
        player_name=player_name,
        player_position=pos_label,
        team_name=team_name,
        opponent_name=opponent_name,
        current_date=CURRENT_DATE,
    )

    text, sources = _gemini_request(
        api_key=api_key,
        system_prompt=prompt,
        user_message=(
            f"Dame el dossier COMPLETO de {player_name} ({pos_label}) de {team_name}. "
            f"El próximo rival es {opponent_name} — busca TODAS las conexiones posibles. "
            "Incluye biografía, trayectoria, vida personal, datos curiosos, "
            "estadísticas de esta temporada, y situación contractual."
        ),
    )

    # Secondary Fact Checking Layer (authoritative verification via web search)
    try:
        clean_text, _ = _gemini_request(
            api_key=api_key,
            system_prompt=_FACT_CHECK_PROMPT_PLAYER,
            user_message=f"Audita este dossier. Verifica cada dato contra fuentes confiables. Elimina lo no verificable:\n\n{text}",
        )
        # Guard: only accept if verifier returned substantial content
        if len(clean_text) > len(text) * 0.3:
            text = clean_text
    except Exception as e:
        print(f"[FactCheck] Error during player dossier fact check: {e}")

    return {
        "name": player_name,
        "position": position,
        "number": player.get("number", ""),
        "text": text,
        "sources": sources,
    }


def _research_team_roster(
    team_name: str,
    opponent_name: str,
    api_key: str,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> List[Dict[str, Any]]:
    """
    Research a team's roster player-by-player.
    1. Fetch player list (lightweight call).
    2. For each player, run a dedicated deep-research call (parallel batches of 4).
    Returns a list of {name, position, number, text, sources} dicts.
    """
    if progress_cb:
        progress_cb(f"📋 Obteniendo lista de jugadores de **{team_name}**...")
    players = _fetch_player_list(team_name, api_key)
    total = len(players)
    if progress_cb:
        progress_cb(f"✅ {total} jugadores encontrados en **{team_name}**. Investigando uno por uno...")

    results: List[Dict[str, Any]] = [None] * total  # preserve order
    completed = 0

    # Process in parallel batches of 4 to avoid rate limits
    batch_size = 4
    for batch_start in range(0, total, batch_size):
        batch = players[batch_start : batch_start + batch_size]
        with ThreadPoolExecutor(max_workers=batch_size) as executor:
            future_to_idx = {
                executor.submit(
                    _research_single_player, player, team_name, opponent_name, api_key
                ): batch_start + i
                for i, player in enumerate(batch)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    p = players[idx]
                    results[idx] = {
                        "name": p.get("name", "?"),
                        "position": p.get("position", ""),
                        "number": p.get("number", ""),
                        "text": f"❌ Error investigando: {e}",
                        "sources": [],
                    }
                completed += 1
                if progress_cb:
                    pname = results[idx]["name"]
                    progress_cb(f"🔍 [{completed}/{total}] **{pname}** ✓  ({team_name})")

    return results


def _research_palomo_phrases(
    home_team: str,
    away_team: str,
    tournament: str,
    match_type: str,
    stadium: str,
    home_context: str,
    away_context: str,
    api_key: str,
) -> Tuple[str, List[Dict[str, str]]]:
    """Generate Fernando Palomo–style phrases for the match."""
    # Truncate context to stay within token limits
    max_ctx = 4000
    home_ctx = home_context[:max_ctx] if len(home_context) > max_ctx else home_context
    away_ctx = away_context[:max_ctx] if len(away_context) > max_ctx else away_context

    prompt = _PALOMO_PHRASES_PROMPT.format(
        home_team=home_team,
        away_team=away_team,
        tournament=tournament,
        match_type=match_type,
        stadium=stadium,
        home_context=home_ctx,
        away_context=away_ctx,
        current_date=CURRENT_DATE,
    )

    return _gemini_request(
        api_key=api_key,
        system_prompt=prompt,
        user_message=(
            f"Genera las frases de Fernando Palomo para la transmisión de "
            f"{home_team} vs {away_team}, {match_type} de {tournament} en {stadium}. "
            "Factor WOW al máximo. Quiero datos que nadie más tendría. "
            "Incluye apertura, contexto de ambos equipos, head-to-head, "
            "datos obscuros, y frases para distintos escenarios del partido."
        ),
    )


def run_match_preparation(
    home_team: str,
    away_team: str,
    tournament: str,
    match_type: str,
    stadium: str,
    api_key: str,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """
    Run the full match preparation pipeline:
      Phase 1 — parallel: team histories (2 calls)
      Phase 2 — sequential per team: roster player-by-player
      Phase 3 — Palomo phrases (uses Phase 1 context)
    Returns dict with keys: home_history, away_history, home_roster,
    away_roster, palomo_phrases.
    home/away_history: (text, sources)
    home/away_roster: list of {name, position, number, text, sources}
    palomo_phrases: (text, sources)
    """
    _cb = progress_cb or (lambda _msg: None)
    results: Dict[str, Any] = {
        "home_history": ("", []),
        "away_history": ("", []),
        "home_roster": [],
        "away_roster": [],
        "palomo_phrases": ("", []),
    }

    # Phase 1: team histories in parallel
    _cb(f"📊 Investigando historial de **{home_team}** y **{away_team}**...")
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {
            executor.submit(_research_team_history, home_team, api_key): "home_history",
            executor.submit(_research_team_history, away_team, api_key): "away_history",
        }
        for future in as_completed(futures):
            key = futures[future]
            try:
                results[key] = future.result()
            except Exception as e:
                results[key] = (f"❌ Error investigando: {e}", [])
    _cb("✅ Historiales completados.")

    # Phase 2: rosters — player by player (sequential per team to show progress)
    _cb(f"👥 Investigando plantilla de **{home_team}** jugador por jugador...")
    try:
        results["home_roster"] = _research_team_roster(
            home_team, away_team, api_key, progress_cb=_cb,
        )
    except Exception as e:
        results["home_roster"] = []
        _cb(f"❌ Error con plantilla de {home_team}: {e}")

    _cb(f"👥 Investigando plantilla de **{away_team}** jugador por jugador...")
    try:
        results["away_roster"] = _research_team_roster(
            away_team, home_team, api_key, progress_cb=_cb,
        )
    except Exception as e:
        results["away_roster"] = []
        _cb(f"❌ Error con plantilla de {away_team}: {e}")

    # Phase 3: Palomo phrases (needs history context)
    _cb("🎙️ Generando frases de Fernando Palomo...")
    home_context = results["home_history"][0] if isinstance(results["home_history"], tuple) else ""
    away_context = results["away_history"][0] if isinstance(results["away_history"], tuple) else ""

    try:
        results["palomo_phrases"] = _research_palomo_phrases(
            home_team, away_team, tournament, match_type, stadium,
            home_context, away_context, api_key,
        )
    except Exception as e:
        results["palomo_phrases"] = (f"❌ Error generando frases: {e}", [])

    return results


# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
_CUSTOM_CSS = """
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


# ---------------------------------------------------------------------------
# Example queries for PalomoGPT
# ---------------------------------------------------------------------------
_EXAMPLE_QUERIES = [
    f"Cuéntame todo sobre Erling Haaland en Premier League {CURRENT_YEAR}",
    "La historia de la rivalidad Real Madrid vs Barcelona",
    "¿Qué tan especial es Lamine Yamal? Dame lo que nadie cuenta",
    "¿Cuántos goles lleva Haaland esta temporada?",
    "¿Cómo ha cambiado Mbappé desde que llegó al Real Madrid?",
    "Los récords más locos de la Champions League",
]


# ---------------------------------------------------------------------------
# Streamlit App
# ---------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(
        page_title="PalomoFacts · Football Intelligence",
        page_icon="⚽",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(_CUSTOM_CSS, unsafe_allow_html=True)

    api_key = st.secrets.get("GEMINI_API_KEY", "")

    # ---- Session state init ----
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "current_conv_id" not in st.session_state:
        st.session_state.current_conv_id = ""

    # ---- Eagerly capture chat_input ----
    # chat_input is collected here BEFORE sidebar renders.
    # This lets us create the conversation in Supabase before
    # _list_conversations() runs in the sidebar.
    incoming_query = st.chat_input("Pregunta sobre cualquier tema de fútbol...")
    if incoming_query and not st.session_state.current_conv_id:
        # First message in a new session → create conversation now
        conv_id = _create_conversation(
            mode=MODE_PALOMO_GPT,
            title=_auto_title(incoming_query),
        )
        st.session_state.current_conv_id = conv_id

    # ---- Sidebar ----
    with st.sidebar:
        st.markdown(
            '<p class="hero-title" style="font-size:1.4rem;">⚽ PalomoFacts</p>',
            unsafe_allow_html=True,
        )
        st.markdown("---")

        # Mode selector
        app_mode = st.pills(
            "Modo",
            options=list(MODE_OPTIONS.keys()),
            default=MODE_PALOMO_GPT,
            format_func=lambda k: MODE_OPTIONS[k],
            key="sb_app_mode",
        )
        if not app_mode:
            app_mode = MODE_PALOMO_GPT

        st.markdown("---")

        # ---- Conversation management (PalomoGPT mode) ----
        if app_mode == MODE_PALOMO_GPT:

            # Show "New conversation" only when inside an existing conversation
            has_active_conv = bool(st.session_state.get("current_conv_id", ""))
            if has_active_conv:
                if st.button("➕ Nueva conversación", use_container_width=True):
                    st.session_state.current_conv_id = ""
                    st.session_state.messages = []
                    st.rerun()

            # List past conversations
            convs = _list_conversations()
            if convs:
                st.markdown("##### 💬 Conversaciones")
                for conv in convs:
                    cid = conv["id"]
                    title = conv.get("title", "Sin título")
                    is_active = cid == st.session_state.get("current_conv_id", "")

                    col_title, col_del = st.columns([5, 1])
                    with col_title:
                        label = f"**▶ {title}**" if is_active else title
                        if st.button(label, key=f"conv_{cid}", use_container_width=True):
                            st.session_state.current_conv_id = cid
                            st.session_state.messages = _load_messages(cid)
                            st.rerun()
                    with col_del:
                        if st.button("🗑️", key=f"del_{cid}"):
                            _delete_conversation(cid)
                            if st.session_state.get("current_conv_id") == cid:
                                st.session_state.current_conv_id = ""
                                st.session_state.messages = []
                            st.rerun()
        else:
            # Match prep mode
            if st.session_state.get("match_results"):
                if st.button("➕ Nueva preparación", use_container_width=True):
                    st.session_state.pop("match_results", None)
                    st.session_state.pop("match_config", None)
                    st.session_state.pop("match_pdf_bytes", None)
                    st.rerun()

            # List past match preps
            preps = _list_match_preps()
            if preps:
                st.markdown("##### ⚽ Preparaciones")
                for prep in preps:
                    pid = prep["id"]
                    title = prep.get("title", "Sin título")

                    col_title, col_del = st.columns([5, 1])
                    with col_title:
                        if st.button(title, key=f"prep_{pid}", use_container_width=True):
                            loaded = _load_match_prep(pid)
                            if loaded:
                                st.session_state.match_config = loaded["config"]
                                st.session_state.match_results = loaded["results"]
                                st.session_state.pop("match_pdf_bytes", None)
                            st.rerun()
                    with col_del:
                        if st.button("🗑️", key=f"dprep_{pid}"):
                            _delete_match_prep(pid)
                            st.rerun()

        st.markdown("---")
        st.caption("🌐 Datos en tiempo real · Cualquier liga · Verificado con fuentes")

        st.markdown(
            '<div class="app-footer">PalomoFacts v4 · AI: Google Gemini + Search Grounding</div>',
            unsafe_allow_html=True,
        )

    # ---- Route to active mode ----
    if app_mode == MODE_PALOMO_GPT:
        _render_palomo_gpt(api_key, incoming_query)
    else:
        _render_match_prep(api_key)


# ---------------------------------------------------------------------------
# PalomoGPT mode
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

    # Welcome screen (only when no conversation loaded)
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
                    st.session_state.messages.append(
                        {"role": "user", "content": example}
                    )
                    # Create conversation for example query
                    if not st.session_state.current_conv_id:
                        cid = _create_conversation(
                            mode=MODE_PALOMO_GPT,
                            title=_auto_title(example),
                        )
                        st.session_state.current_conv_id = cid
                    st.rerun()
        return
    else:
        for msg in msgs:
            with st.chat_message(msg["role"]):
                st.markdown(msg.get("content", ""))

    # Use the query captured in main() (chat_input is already rendered there)
    user_prompt = incoming_query

    pending_prompt = None
    if msgs and msgs[-1]["role"] == "user":
        pending_prompt = msgs[-1]["content"]

    if user_prompt:
        if msgs and msgs[-1]["role"] == "user":
            msgs[-1]["content"] = user_prompt
        else:
            msgs.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)

        # Save user message to DB (conversation already created in main())
        conv_id = st.session_state.get("current_conv_id", "")
        _save_message(conv_id, "user", user_prompt)

    elif pending_prompt:
        user_prompt = pending_prompt
    else:
        return

    # Check API key
    if not api_key:
        with st.chat_message("assistant"):
            err = (
                "⚠️ No se encontró la API key de Gemini. "
                "Configura `GEMINI_API_KEY` en los secrets de Streamlit."
            )
            st.warning(err)
            msgs.append({"role": "assistant", "content": err})
        return

    conv_id = st.session_state.get("current_conv_id", "")

    # Get response
    with st.chat_message("assistant"):
        try:
            with st.status(
                "🔍 Buscando información verificada...", expanded=False
            ) as status:
                text, citations, reasoning_chain, followups = get_palomo_response(
                    query=user_prompt,
                    session_messages=msgs,
                    api_key=api_key,
                )
                status.update(label="✅ Información encontrada y verificada", state="complete")

            full_response = text
            source_text = _format_sources(citations)
            if source_text:
                full_response += source_text

            st.markdown(full_response)
            msgs.append({"role": "assistant", "content": full_response})
            _save_message(conv_id, "assistant", full_response)

            # Show reasoning chain as follow-up messages
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

    # Render follow-up Q&A as user/assistant chat bubbles
    for fu in followups:
        # User bubble for the auto-generated question
        with st.chat_message("user"):
            st.markdown(f"🔍 {fu['question']}")
        q_text = f"🔍 {fu['question']}"
        msgs.append({"role": "user", "content": q_text})
        _save_message(conv_id, "user", q_text)

        # Assistant bubble for the answer
        with st.chat_message("assistant"):
            st.markdown(fu["answer"])
        msgs.append({"role": "assistant", "content": fu["answer"]})
        _save_message(conv_id, "assistant", fu["answer"])


# ---------------------------------------------------------------------------
# Preparación de Partidos mode
# ---------------------------------------------------------------------------
def _render_match_prep(api_key: str) -> None:
    col_h1, _ = st.columns([3, 1])
    with col_h1:
        st.markdown(
            '<p class="hero-title">⚽ Preparación de Partidos</p>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<p class="hero-sub">'
            'Informe completo para transmisión — historial, plantillas y frases de Palomo'
            '</p>',
            unsafe_allow_html=True,
        )

    # --- Match configuration form ---
    st.markdown("### 📋 Configuración del Partido")

    col1, col2 = st.columns(2)
    with col1:
        home_team = st.text_input(
            "🏠 Equipo Local",
            placeholder="Ej: F.C. Barcelona",
            key="mp_home_team",
        )
    with col2:
        away_team = st.text_input(
            "✈️ Equipo Visitante",
            placeholder="Ej: Real Madrid",
            key="mp_away_team",
        )

    col3, col4, col5 = st.columns(3)
    with col3:
        tournament = st.selectbox(
            "🏆 Torneo",
            options=TOURNAMENT_OPTIONS,
            key="mp_tournament",
        )
    with col4:
        match_type = st.selectbox(
            "📌 Tipo de Partido",
            options=MATCH_TYPE_OPTIONS,
            key="mp_match_type",
        )
    with col5:
        stadium = st.text_input(
            "🏟️ Estadio",
            placeholder="Ej: Estadio Santiago Bernabéu",
            key="mp_stadium",
        )

    can_submit = bool(home_team and away_team and stadium)

    if st.button(
        "🚀 Preparar Partido",
        use_container_width=True,
        disabled=not can_submit,
        type="primary",
    ):
        if not api_key:
            st.error(
                "⚠️ No se encontró la API key de Gemini. "
                "Configura `GEMINI_API_KEY` en los secrets de Streamlit."
            )
            return

        # --- Validate match via LLM ---
        with st.status("🔍 Validando equipos y partido...", expanded=True) as val_status:
            try:
                val_prompt = _MATCH_VALIDATION_PROMPT.format(
                    home_team=home_team,
                    away_team=away_team,
                    tournament=tournament,
                    current_date=CURRENT_DATE,
                )
                val_text, _ = _gemini_request(
                    api_key=api_key,
                    system_prompt=val_prompt,
                    user_message="Valida este partido y devuelve el JSON.",
                )

                # Parse JSON from response
                json_match = re.search(r'\{[\s\S]*\}', val_text)
                if json_match:
                    val_data = json.loads(json_match.group())
                    if not val_data.get("valid", True):
                        reason = val_data.get("reason", "Partido no reconocido.")
                        val_status.update(label=f"❌ {reason}", state="error")
                        st.error(f"⚠️ {reason}")
                        return
                    # Use resolved names
                    home_team = val_data.get("home_team", home_team)
                    away_team = val_data.get("away_team", away_team)
                    val_status.update(
                        label=f"✅ Partido validado: {home_team} vs {away_team}",
                        state="complete", expanded=False,
                    )
                else:
                    val_status.update(label="⚠️ No se pudo validar, continuando...", state="complete", expanded=False)
            except Exception as e:
                print(f"[MatchPrep] Validation failed, continuing: {e}")
                val_status.update(label="⚠️ Validación omitida, continuando...", state="complete", expanded=False)

        # Store config with resolved names
        st.session_state.match_config = {
            "home_team": home_team,
            "away_team": away_team,
            "tournament": tournament,
            "match_type": match_type,
            "stadium": stadium,
        }

        # Run research pipeline with progress
        with st.status(
            "🔍 Preparando informe del partido...", expanded=True
        ) as status:

            def _progress(msg: str) -> None:
                status.write(msg)

            try:
                results = run_match_preparation(
                    home_team=home_team,
                    away_team=away_team,
                    tournament=tournament,
                    match_type=match_type,
                    stadium=stadium,
                    api_key=api_key,
                    progress_cb=_progress,
                )
                st.session_state.match_results = results
                status.update(label="📄 Sintetizando y generando PDF...")
                try:
                    pdf_bytes = generate_match_pdf(st.session_state.match_config, results, api_key=api_key)
                    st.session_state.match_pdf_bytes = pdf_bytes
                except Exception as e:
                    print(f"[MatchPrep] Error generating PDF:\n{traceback.format_exc()}")
                    st.session_state.match_pdf_bytes = None

                status.update(
                    label="✅ ¡Informe completo! Desplázate hacia abajo para verlo.",
                    state="complete",
                    expanded=False,
                )

                # Auto-save to Supabase
                try:
                    _save_match_prep(st.session_state.match_config, results)
                except Exception as e:
                    print(f"[MatchPrep] Error saving to Supabase: {e}")
            except Exception as e:
                status.update(label=f"❌ Error: {e}", state="error")
                print(f"[MatchPrep] Error:\n{traceback.format_exc()}")
                return

    # --- Display persisted results ---
    if st.session_state.get("match_results") and st.session_state.get("match_config"):
        _display_match_results(
            st.session_state.match_config,
            st.session_state.match_results,
        )


def _render_roster_players(roster: list) -> None:
    """Render a list of player dossiers as individual expanders grouped by position."""
    if not roster:
        st.warning("No se encontraron jugadores.")
        return

    # Group by position preserving order
    pos_order = ["GK", "DEF", "MID", "FWD"]
    pos_emoji = {"GK": "🧤", "DEF": "🛡️", "MID": "🎯", "FWD": "⚡"}
    pos_name = {"GK": "PORTEROS", "DEF": "DEFENSAS", "MID": "CENTROCAMPISTAS", "FWD": "DELANTEROS"}

    grouped: Dict[str, list] = {p: [] for p in pos_order}
    for player in roster:
        pos = player.get("position", "FWD")
        if pos not in grouped:
            pos = "FWD"
        grouped[pos].append(player)

    for pos in pos_order:
        players = grouped[pos]
        if not players:
            continue
        st.markdown(f"**{pos_emoji.get(pos, '')} {pos_name.get(pos, pos)}**")
        for p in players:
            number = p.get("number", "")
            label = f"#{number} {p['name']}" if number else p["name"]
            with st.expander(label, expanded=False):
                st.markdown(p.get("text", ""))
                sources = p.get("sources", [])
                if sources:
                    st.markdown(
                        _format_sources(sources).replace(_CITATION_SEPARATOR, "")
                    )


def _display_match_results(config: dict, results: dict) -> None:
    """Render the full match preparation report."""
    home = config["home_team"]
    away = config["away_team"]
    tournament = config["tournament"]
    match_type = config["match_type"]
    stadium = config["stadium"]

    st.markdown("---")

    # ---- Match header ----
    st.markdown(
        f'<div class="match-header">'
        f'<h2>{home}  🆚  {away}</h2>'
        f'<div class="match-meta">'
        f'{match_type} · {tournament} · 🏟️ {stadium}'
        f'</div></div>',
        unsafe_allow_html=True,
    )

    # ---- PDF download button ----
    if st.session_state.get("match_pdf_bytes"):
        safe_home = re.sub(r'[^\w\s-]', '', home).strip().replace(' ', '_')
        safe_away = re.sub(r'[^\w\s-]', '', away).strip().replace(' ', '_')
        pdf_filename = f"Preparacion_{safe_home}_vs_{safe_away}.pdf"
        st.download_button(
            label="📥 Descargar PDF del Informe",
            data=st.session_state.match_pdf_bytes,
            file_name=pdf_filename,
            mime="application/pdf",
            use_container_width=True,
        )
    else:
        st.warning("⚠️ No se pudo generar el PDF en este momento. El informe completo se muestra a continuación.")

    # ---- Team Histories (side-by-side) ----
    st.markdown("### 📊 Historial de Temporadas")

    col_h, col_a = st.columns(2)

    with col_h:
        st.markdown(
            f'<div class="team-col-hdr"><h3>🏠 {home}</h3></div>',
            unsafe_allow_html=True,
        )
        h_text, h_srcs = results["home_history"]
        st.markdown(h_text)
        if h_srcs:
            with st.expander("📚 Fuentes"):
                st.markdown(
                    _format_sources(h_srcs).replace(_CITATION_SEPARATOR, "")
                )

    with col_a:
        st.markdown(
            f'<div class="team-col-hdr"><h3>✈️ {away}</h3></div>',
            unsafe_allow_html=True,
        )
        a_text, a_srcs = results["away_history"]
        st.markdown(a_text)
        if a_srcs:
            with st.expander("📚 Fuentes"):
                st.markdown(
                    _format_sources(a_srcs).replace(_CITATION_SEPARATOR, "")
                )

    # ---- Team Rosters (side-by-side, per-player expanders) ----
    st.markdown("---")
    st.markdown("### 👥 Plantillas — Dossier por Jugador")

    col_hr, col_ar = st.columns(2)

    with col_hr:
        st.markdown(
            f'<div class="team-col-hdr"><h3>🏠 {home}</h3></div>',
            unsafe_allow_html=True,
        )
        _render_roster_players(results.get("home_roster", []))

    with col_ar:
        st.markdown(
            f'<div class="team-col-hdr"><h3>✈️ {away}</h3></div>',
            unsafe_allow_html=True,
        )
        _render_roster_players(results.get("away_roster", []))

    # ---- Palomo Phrases (full-width) ----
    st.markdown("---")
    st.markdown(
        f'<div class="palomo-section">'
        f'<h3>🎙️ Frases de Fernando Palomo — {home} vs {away}</h3>'
        f'</div>',
        unsafe_allow_html=True,
    )

    p_text, p_srcs = results["palomo_phrases"]
    st.markdown(p_text)
    if p_srcs:
        with st.expander("📚 Fuentes"):
            st.markdown(
                _format_sources(p_srcs).replace(_CITATION_SEPARATOR, "")
            )


if __name__ == "__main__":
    main()
