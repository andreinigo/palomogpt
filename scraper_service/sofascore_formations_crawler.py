from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime, date
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable, Optional
from urllib.parse import urljoin

from playwright.sync_api import Browser, BrowserContext, Locator, Page, TimeoutError as PlaywrightTimeoutError, sync_playwright

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False

BASE_URL = "https://www.sofascore.com"
TEAM_LINK_RX = re.compile(r"/football/team/")
MATCH_LINK_RX = re.compile(r"/football/match/")
FORMATION_RX = re.compile(r"^\d(?:-\d+)+$")
DATE_RX = re.compile(r"\b(\d{2}/\d{2}/\d{4})\b")


@dataclass
class MatchLineup:
    index: int
    match_url: str
    match_date: Optional[str]
    home_team: str
    away_team: str
    target_team: str
    opponent: str
    target_side: str
    formation: Optional[str]
    players: list[str]
    image_path: str
    raw_image_path: str


class SofascoreCrawlerError(RuntimeError):
    pass


def normalize_text(value: Optional[str]) -> str:
    if not value:
        return ""
    return re.sub(r"\s+", " ", value).strip()


def slugify(value: str) -> str:
    value = normalize_text(value).lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    value = re.sub(r"-+", "-", value).strip("-")
    return value or "item"


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, normalize_text(a).lower(), normalize_text(b).lower()).ratio()


def maybe_click(locator: Locator, timeout: int = 1200) -> bool:
    try:
        locator.first.click(timeout=timeout)
        return True
    except Exception:
        return False


def dismiss_overlays(page: Page) -> None:
    candidates = [
        re.compile(r"accept", re.I),
        re.compile(r"agree", re.I),
        re.compile(r"allow", re.I),
        re.compile(r"consent", re.I),
        re.compile(r"got it", re.I),
        re.compile(r"close", re.I),
        re.compile(r"ok", re.I),
    ]
    for _ in range(3):
        clicked_any = False
        for rx in candidates:
            for getter in (
                lambda p, r=rx: p.get_by_role("button", name=r),
                lambda p, r=rx: p.get_by_text(r),
            ):
                try:
                    loc = getter(page)
                    if loc.count() and loc.first.is_visible():
                        loc.first.click(timeout=700)
                        page.wait_for_timeout(300)
                        clicked_any = True
                except Exception:
                    pass
        if not clicked_any:
            break


def wait_short(page: Page, ms: int = 900) -> None:
    page.wait_for_timeout(ms)


def choose_best_team_result(page: Page, team_query: str) -> str:
    search_input = None
    candidate_selectors = [
        "input[placeholder*='Search']",
        "input[aria-label*='Search']",
        "input[type='text']",
        "input",
    ]
    for sel in candidate_selectors:
        loc = page.locator(sel)
        try:
            if loc.count() and loc.first.is_visible():
                search_input = loc.first
                break
        except Exception:
            continue

    if search_input is None:
        raise SofascoreCrawlerError("No encontré el buscador de Sofascore.")

    search_input.click()
    search_input.fill(team_query)
    wait_short(page, 1200)

    links = page.locator("a[href*='/football/team/']")
    try:
        links.first.wait_for(timeout=10000)
    except PlaywrightTimeoutError as exc:
        raise SofascoreCrawlerError(f"No aparecieron resultados para '{team_query}'.") from exc

    candidates: list[tuple[float, str, str]] = []
    count = min(links.count(), 20)
    for i in range(count):
        try:
            link = links.nth(i)
            href = link.get_attribute("href") or ""
            text = normalize_text(link.inner_text())
            if not href or not TEAM_LINK_RX.search(href):
                continue
            if not text:
                text = normalize_text(link.text_content())
            score = max(similarity(text, team_query), similarity(href, team_query))
            candidates.append((score, text, urljoin(BASE_URL, href)))
        except Exception:
            continue

    if not candidates:
        raise SofascoreCrawlerError(f"No pude resolver el equipo '{team_query}' a una URL de equipo.")

    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][2]


def _resolve_team_url_via_google(page: Page, team_query: str) -> Optional[str]:
    """Use Google to find the Sofascore team URL, bypassing Sofascore's own search."""
    import re as _re
    try:
        q = f"site:sofascore.com/team/football {team_query}"
        page.goto(
            f"https://www.google.com/search?q={q}&num=5",
            wait_until="domcontentloaded",
            timeout=15000,
        )
        page.wait_for_timeout(2000)

        # Extract all sofascore team URLs from Google results
        links = page.locator("a[href*='sofascore.com/team/football/']")
        candidates: list[tuple[float, str]] = []
        for i in range(min(links.count(), 10)):
            try:
                href = links.nth(i).get_attribute("href") or ""
                # Clean up Google redirect URLs
                if "url?q=" in href:
                    href = href.split("url?q=")[1].split("&")[0]
                if "/team/football/" in href:
                    score = similarity(href, team_query)
                    candidates.append((score, href))
            except Exception:
                continue

        if candidates:
            candidates.sort(key=lambda x: x[0], reverse=True)
            url = candidates[0][1].split("?")[0].split("#")[0]
            if not url.startswith("http"):
                url = "https://www.sofascore.com" + url
            return url
    except Exception as exc:
        print(f"[google-resolve] {exc}")
    return None


def resolve_team_url(page: Page, team_query: str, team_url: Optional[str] = None) -> str:
    if team_url:
        return team_url
    # Try Google first (works from datacenters where Sofascore search fails)
    google_url = _resolve_team_url_via_google(page, team_query)
    if google_url:
        return google_url
    # Fall back to Sofascore's own search
    page.goto(BASE_URL, wait_until="domcontentloaded")
    dismiss_overlays(page)
    return choose_best_team_result(page, team_query)


def click_tab_if_present(page: Page, label: str) -> bool:
    options = [
        page.get_by_role("tab", name=re.compile(fr"^{re.escape(label)}$", re.I)),
        page.get_by_role("link", name=re.compile(fr"^{re.escape(label)}$", re.I)),
        page.get_by_text(re.compile(fr"^{re.escape(label)}$", re.I)),
    ]
    for loc in options:
        try:
            if loc.count() and loc.first.is_visible():
                loc.first.click(timeout=3000)
                wait_short(page, 700)
                return True
        except Exception:
            pass
    return False


def collect_match_urls(page: Page, needed: int, max_scan: int = 40) -> list[str]:
    # Ir al tab de partidos/resultados si existe.
    click_tab_if_present(page, "Matches")
    click_tab_if_present(page, "Results")

    urls: list[str] = []
    seen: set[str] = set()
    stagnant = 0

    while len(urls) < max_scan and stagnant < 8:
        before = len(urls)
        try:
            hrefs = page.locator("a[href*='/football/match/']").evaluate_all(
                "els => els.map(e => e.href || e.getAttribute('href')).filter(Boolean)"
            )
        except Exception:
            hrefs = []

        for href in hrefs:
            full = urljoin(BASE_URL, href.split("?")[0])
            if MATCH_LINK_RX.search(full) and full not in seen:
                seen.add(full)
                urls.append(full)

        if len(urls) >= max_scan:
            break

        clicked = False
        for rx in (
            re.compile(r"show more", re.I),
            re.compile(r"more", re.I),
            re.compile(r"next", re.I),
        ):
            try:
                btn = page.get_by_role("button", name=rx)
                if btn.count() and btn.first.is_visible():
                    btn.first.click(timeout=2000)
                    clicked = True
                    break
            except Exception:
                pass

        if not clicked:
            try:
                page.mouse.wheel(0, 5000)
            except Exception:
                pass

        wait_short(page, 1100)
        stagnant = stagnant + 1 if len(urls) == before else 0

    return urls[: max(needed * 3, needed)]


def parse_match_date_from_page(page: Page) -> Optional[date]:
    try:
        body = normalize_text(page.locator("body").inner_text(timeout=3000))
    except Exception:
        return None
    match = DATE_RX.search(body)
    if not match:
        return None
    try:
        return datetime.strptime(match.group(1), "%d/%m/%Y").date()
    except Exception:
        return None


def get_lineup_card(page: Page) -> Locator:
    card = page.locator("div:has(.FootballTerrainHalf__root--side_left):has(.FootballTerrainHalf__root--side_right)").first
    card.wait_for(timeout=12000)
    return card


JS_EXTRACT_CARD_META = r'''
(card) => {
  const clean = (s) => (s || '').replace(/\s+/g, ' ').trim();
  const isWordy = (s) => /[A-Za-zÀ-ÿ]/.test(s);
  const teamImgs = Array.from(card.querySelectorAll("img[src*='/api/v1/team/']"));
  const teamNames = teamImgs.map((img) => {
    const sibling = img.nextElementSibling;
    const texts = sibling
      ? Array.from(sibling.querySelectorAll('span')).map((x) => clean(x.textContent)).filter(Boolean)
      : [];
    return texts.find((t) => isWordy(t) && !/^\d/.test(t) && !/^\d(?:-\d+)+$/.test(t) && !/^\d+(?:\.\d+)?$/.test(t)) || null;
  }).filter(Boolean);

  const formations = Array.from(card.querySelectorAll('span'))
    .map((x) => clean(x.textContent))
    .filter((t) => /^\d(?:-\d+)+$/.test(t));

  return { team_names: teamNames, formations };
}
'''

JS_EXTRACT_PLAYERS = r'''
(half) => {
  const clean = (s) => (s || '').replace(/\s+/g, ' ').trim();
  const normalizePlayer = (t) => clean(t).replace(/^(\d{1,2})([A-Za-zÀ-ÿ(])/, '$1 $2');
  const raw = Array.from(half.querySelectorAll('span')).map((x) => normalizePlayer(x.textContent)).filter(Boolean);
  const out = [];
  for (const text of raw) {
    if (!/^\d{1,2}(?:\s+|\s*\(c\)\s*)[A-Za-zÀ-ÿ].+/.test(text)) continue;
    if (/^\d+(?:\.\d+)?$/.test(text)) continue;
    if (/^\d(?:-\d+)+$/.test(text)) continue;
    if (!out.includes(text)) out.push(text);
  }
  return out;
}
'''


def extract_card_meta(card: Locator) -> tuple[list[str], list[str]]:
    meta = card.evaluate(JS_EXTRACT_CARD_META)
    team_names = [normalize_text(x) for x in meta.get("team_names", []) if normalize_text(x)]
    formations = [normalize_text(x) for x in meta.get("formations", []) if normalize_text(x)]
    return team_names, formations


def extract_team_names_from_page(page: Page) -> list[str]:
    """Fallback: extract home/away team names from page-level team links."""
    JS = r'''
    () => {
        const clean = (s) => (s || '').replace(/\s+/g, ' ').trim();
        const links = Array.from(document.querySelectorAll("a[href*='/football/team/']"));
        const seen = new Set();
        const names = [];
        for (const link of links) {
            const text = clean(link.textContent);
            if (!text || text.length > 50 || /compare|score|schedule|results/i.test(text)) continue;
            if (seen.has(text)) continue;
            seen.add(text);
            names.push(text);
            if (names.length >= 2) break;
        }
        return names;
    }
    '''
    raw = page.evaluate(JS)
    return [normalize_text(x) for x in raw if normalize_text(x)]


def pick_target_side(team_names: list[str], team_query: str) -> tuple[str, str, str]:
    if len(team_names) < 2:
        raise SofascoreCrawlerError("No pude detectar home/away en la tarjeta de lineups.")

    left_name, right_name = team_names[0], team_names[1]
    left_score = similarity(left_name, team_query)
    right_score = similarity(right_name, team_query)

    if left_score >= right_score:
        return "left", left_name, right_name
    return "right", right_name, left_name


def get_target_half(card: Locator, side: str) -> Locator:
    half = card.locator(f".FootballTerrainHalf__root--side_{side}").first
    half.wait_for(timeout=8000)
    return half


def extract_players(half: Locator) -> list[str]:
    try:
        players = half.evaluate(JS_EXTRACT_PLAYERS)
    except Exception:
        players = []
    return [normalize_text(x) for x in players if normalize_text(x)]


def load_font(size: int):
    if not PIL_AVAILABLE:
        return None
    for candidate in (
        "DejaVuSans-Bold.ttf",
        "DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        try:
            return ImageFont.truetype(candidate, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


def compose_header_image(raw_path: Path, final_path: Path, title: str, subtitle: str) -> None:
    if not PIL_AVAILABLE:
        shutil.move(str(raw_path), str(final_path))
        return

    image = Image.open(raw_path).convert("RGB")
    pad = 14
    title_font = load_font(22)
    subtitle_font = load_font(15)

    probe = Image.new("RGB", (10, 10))
    draw_probe = ImageDraw.Draw(probe)
    title_box = draw_probe.textbbox((0, 0), title, font=title_font)
    subtitle_box = draw_probe.textbbox((0, 0), subtitle, font=subtitle_font)
    title_h = title_box[3] - title_box[1]
    subtitle_h = subtitle_box[3] - subtitle_box[1]
    header_h = pad * 3 + title_h + subtitle_h

    canvas = Image.new("RGB", (image.width, image.height + header_h), (20, 23, 28))
    draw = ImageDraw.Draw(canvas)
    draw.text((pad, pad), title, font=title_font, fill=(245, 245, 245))
    draw.text((pad, pad * 2 + title_h), subtitle, font=subtitle_font, fill=(190, 198, 209))
    canvas.paste(image, (0, header_h))
    canvas.save(final_path)
    raw_path.unlink(missing_ok=True)


def ensure_lineups_tab(page: Page) -> bool:
    for label in ("Lineups", "Alineaciones", "Formations"):
        if click_tab_if_present(page, label):
            return True
    return False


def process_match(
    page: Page,
    match_url: str,
    team_query: str,
    output_dir: Path,
    index: int,
    season_start: Optional[date] = None,
) -> Optional[MatchLineup]:
    page.goto(match_url, wait_until="domcontentloaded")
    dismiss_overlays(page)
    wait_short(page, 900)

    match_date_obj = parse_match_date_from_page(page)
    if season_start and match_date_obj and match_date_obj < season_start:
        return None

    if not ensure_lineups_tab(page):
        return None

    card = get_lineup_card(page)
    team_names, formations = extract_card_meta(card)
    if len(team_names) < 2:
        team_names = extract_team_names_from_page(page)
    target_side, target_team, opponent = pick_target_side(team_names, team_query)
    half = get_target_half(card, target_side)
    players = extract_players(half)

    formation = None
    if len(formations) >= 2:
        formation = formations[0] if target_side == "left" else formations[1]
    elif formations:
        formation = formations[0]

    date_str = match_date_obj.isoformat() if match_date_obj else "unknown-date"
    raw_name = f"{index:02d}_{date_str}_{slugify(target_team)}_vs_{slugify(opponent)}_raw.png"
    raw_path = output_dir / raw_name
    half.screenshot(path=str(raw_path))

    final_name = f"{index:02d}_{date_str}_{slugify(target_team)}_vs_{slugify(opponent)}"
    if formation:
        final_name += f"_{slugify(formation)}"
    final_path = output_dir / f"{final_name}.png"

    title = f"{target_team} vs {opponent}"
    subtitle = f"{date_str} • {formation or 'formation N/A'}"
    compose_header_image(raw_path, final_path, title, subtitle)

    result = MatchLineup(
        index=index,
        match_url=match_url,
        match_date=match_date_obj.isoformat() if match_date_obj else None,
        home_team=team_names[0] if len(team_names) > 0 else "",
        away_team=team_names[1] if len(team_names) > 1 else "",
        target_team=target_team,
        opponent=opponent,
        target_side=target_side,
        formation=formation,
        players=players,
        image_path=str(final_path),
        raw_image_path=str(raw_path),
    )

    sidecar_path = output_dir / f"{final_name}.json"
    sidecar_path.write_text(json.dumps(asdict(result), ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def build_context(browser: Browser, headless_locale: str = "en-US") -> BrowserContext:
    import random
    # Rotate user agents to reduce fingerprinting
    _UA_POOL = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36",
    ]
    return browser.new_context(
        locale=headless_locale,
        viewport={"width": 1920, "height": 1080},
        user_agent=random.choice(_UA_POOL),
        color_scheme="light",
        timezone_id="Europe/Madrid",
        extra_http_headers={
            "Accept-Language": "en-US,en;q=0.9,es;q=0.8",
        },
    )


def _new_stealth_page(context: BrowserContext) -> Page:
    """Create a new page with stealth patches applied (if available)."""
    page = context.new_page()
    try:
        from playwright_stealth import stealth_sync
        stealth_sync(page)
    except ImportError:
        pass
    return page


def crawl_team_lineups(
    team_query: str,
    limit: int,
    output_dir: Path,
    team_url: Optional[str] = None,
    season_start: Optional[date] = None,
    headless: bool = True,
) -> list[MatchLineup]:
    output_dir.mkdir(parents=True, exist_ok=True)
    import time as _time
    t0 = _time.time()

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        context = build_context(browser)
        page = _new_stealth_page(context)
        page.set_default_timeout(12000)
        page.set_default_navigation_timeout(20000)

        print(f"[crawler] [{_time.time()-t0:.1f}s] resolving team URL...", flush=True)
        team_page_url = resolve_team_url(page, team_query=team_query, team_url=team_url)
        print(f"[crawler] [{_time.time()-t0:.1f}s] team URL: {team_page_url}", flush=True)
        page.goto(team_page_url, wait_until="domcontentloaded")
        dismiss_overlays(page)
        wait_short(page, 1200)

        print(f"[crawler] [{_time.time()-t0:.1f}s] collecting match URLs...", flush=True)
        match_urls = collect_match_urls(page, needed=limit)
        print(f"[crawler] [{_time.time()-t0:.1f}s] found {len(match_urls)} match URLs", flush=True)
        if not match_urls:
            raise SofascoreCrawlerError("No pude extraer URLs de partidos desde la pestaña de resultados.")

        saved: list[MatchLineup] = []
        for idx, url in enumerate(match_urls):
            if len(saved) >= limit:
                break
            print(f"[crawler] [{_time.time()-t0:.1f}s] processing match {idx+1}/{len(match_urls)}: {url}", flush=True)
            try:
                result = process_match(
                    page=page,
                    match_url=url,
                    team_query=team_query,
                    output_dir=output_dir,
                    index=len(saved) + 1,
                    season_start=season_start,
                )
                if result is None:
                    # Si ya salimos de temporada, como la lista va de más reciente a más antigua,
                    # se puede parar cuando la fecha cruza el corte.
                    current_date = parse_match_date_from_page(page)
                    if season_start and current_date and current_date < season_start:
                        break
                    continue
                saved.append(result)
            except Exception as exc:
                print(f"[WARN] No pude procesar {url}: {exc}", file=sys.stderr)
                continue

        context.close()
        browser.close()

    return saved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extrae las últimas formaciones de un equipo en Sofascore y guarda el parado táctico como PNG."
    )
    parser.add_argument("team", help="Nombre del equipo, tal como lo buscarías en Sofascore.")
    parser.add_argument("-n", "--limit", type=int, default=10, help="Cantidad de partidos a extraer. Default: 10")
    parser.add_argument("--out", default="output_sofascore", help="Directorio de salida.")
    parser.add_argument(
        "--team-url",
        default=None,
        help="URL directa del equipo en Sofascore. Si la pasas, se salta el buscador.",
    )
    parser.add_argument(
        "--season-start",
        default=None,
        help="Fecha de inicio de temporada en formato YYYY-MM-DD. Si se pasa, no baja de esa fecha.",
    )
    parser.add_argument(
        "--headed",
        action="store_true",
        help="Muestra el navegador para debug. Por defecto corre headless.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    season_start = None
    if args.season_start:
        season_start = datetime.strptime(args.season_start, "%Y-%m-%d").date()

    output_dir = Path(args.out)

    results = crawl_team_lineups(
        team_query=args.team,
        limit=args.limit,
        output_dir=output_dir,
        team_url=args.team_url,
        season_start=season_start,
        headless=not args.headed,
    )

    index_path = output_dir / "index.json"
    index_path.write_text(
        json.dumps([asdict(x) for x in results], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps({
        "team": args.team,
        "saved": len(results),
        "output_dir": str(output_dir.resolve()),
        "index_json": str(index_path.resolve()),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
