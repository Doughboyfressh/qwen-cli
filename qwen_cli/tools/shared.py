"""Shared tool implementations (web search, URL fetch, diff, backup).

Importers must set the config vars below before calling any function that
needs them (search keys, BACKUPS_DIR, etc.).  The simplest pattern:

    from qwen_cli.tools import shared as _qt
    _qt.GOOGLE_API_KEY = GOOGLE_API_KEY
    _qt.BACKUPS_DIR    = DATA_DIR / "backups"
    from qwen_cli.tools.shared import do_web_search, do_fetch_url, ...
"""

from __future__ import annotations

import contextlib
import gzip as _gzip
import json
import logging
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import zlib as _zlib
from datetime import datetime
from pathlib import Path

_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config vars — set by importers after import
# ---------------------------------------------------------------------------
GOOGLE_API_KEY: str = os.environ.get("GOOGLE_API_KEY", "")
GOOGLE_CSE_ID: str = os.environ.get("GOOGLE_CSE_ID", "")
BRAVE_API_KEY: str = os.environ.get("BRAVE_API_KEY", "")

_DATA_DIR_DEFAULT = Path.home() / ".qwen-cli"
BACKUPS_DIR: Path = _DATA_DIR_DEFAULT / "backups"

# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------


def _resolve(path: str) -> Path:
    """Resolve a relative or absolute path string to a Path object."""
    p = Path(path).expanduser()
    return p.resolve(strict=False) if p.is_absolute() else (Path.cwd() / p).resolve(strict=False)


def _find_hunk_location(result: list[str], old_part: list[str], hint: int) -> int | None:
    """Search for old_part's content in result, starting near hint.

    Returns the index of the first matching line, or None if no match exists
    anywhere in the file. Used as a fallback when a hunk's stated line number
    no longer lines up with the file on disk — e.g. the model's belief about
    line numbers drifted after a context-compaction summary, or an earlier
    edit in the same patch shifted things unexpectedly. Without this, a single
    stale line number fails the whole hunk with no recovery path, which is
    what pushed patch_file's exact-line matching to be abandoned in favor of
    ad-hoc line-number archaeology via run_script.

    Searches an expanding window around hint first (cheap, and right most of
    the time for small drift), then falls back to a full-file scan.
    """
    if not old_part:
        return None
    norm_old = [ln.rstrip() for ln in old_part]
    n = len(old_part)
    last_start = len(result) - n
    if last_start < 0:
        return None

    def matches_at(i: int) -> bool:
        return [ln.rstrip() for ln in result[i : i + n]] == norm_old

    for radius in (25, 100, 500):
        lo = max(0, hint - radius)
        hi = min(last_start, hint + radius)
        for i in range(lo, hi + 1):
            if matches_at(i):
                return i

    for i in range(last_start + 1):
        if matches_at(i):
            return i
    return None


def _apply_diff(original: str, diff: str) -> str:
    """Apply a unified diff to original text. Raises ValueError on mismatch."""
    if not diff.strip():
        return original

    orig_lines = original.splitlines(keepends=True)
    if orig_lines and not orig_lines[-1].endswith("\n"):
        orig_lines[-1] += "\n"

    result = list(orig_lines)
    diff_lines = diff.splitlines(keepends=True)
    hunk_re = re.compile(r"^@@ -(\d+)(?:,\d+)? \+\d+(?:,\d+)? @@")

    i = 0
    while i < len(diff_lines) and not hunk_re.match(diff_lines[i]):
        i += 1

    offset = 0

    while i < len(diff_lines):
        m = hunk_re.match(diff_lines[i])
        if not m:
            i += 1
            continue

        src_start = int(m.group(1)) - 1
        i += 1
        old_part: list[str] = []
        new_part: list[str] = []

        while i < len(diff_lines) and not hunk_re.match(diff_lines[i]):
            dl = diff_lines[i]
            i += 1
            if dl.startswith("\\"):
                continue
            prefix = dl[0] if dl else " "
            body = dl[1:] if len(dl) > 1 else "\n"
            if not body.endswith("\n"):
                body += "\n"
            if prefix == " ":
                old_part.append(body)
                new_part.append(body)
            elif prefix == "-":
                old_part.append(body)
            elif prefix == "+":
                new_part.append(body)

        target = src_start + offset
        actual = result[target : target + len(old_part)]

        def _norm(ls: list[str]) -> list[str]:
            """Strip trailing whitespace from lines for diff comparison."""
            return [ln.rstrip() for ln in ls]

        if _norm(actual) != _norm(old_part):
            found = _find_hunk_location(result, old_part, target)
            if found is None:
                msg = (
                    f"Hunk at original line {src_start + 1} does not match, and no matching "
                    f"content was found anywhere else in the file either.\n"
                    f"Expected:\n{''.join(old_part)}"
                    f"Got at line {target + 1}:\n{''.join(actual)}"
                )
                raise ValueError(
                    msg,
                )
            offset += found - target
            target = found

        result[target : target + len(old_part)] = new_part
        offset += len(new_part) - len(old_part)

    return "".join(result)


def _cleanup_backups(keep: int = 50) -> None:
    """Keep only the most recent N backup files, deleting the rest."""
    if not BACKUPS_DIR.exists():
        return
    files = sorted(BACKUPS_DIR.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
    for old in files[keep:]:
        with contextlib.suppress(Exception):
            old.unlink()


def _backup_file(p: Path) -> None:
    """Create a timestamped backup of a file before editing it."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = BACKUPS_DIR / f"{p.name}.{stamp}.bak"
    BACKUPS_DIR.mkdir(parents=True, exist_ok=True)
    backup.write_text(p.read_text(encoding="utf-8", errors="replace"), encoding="utf-8")
    _cleanup_backups()


# ---------------------------------------------------------------------------
# Fetch cache — short-lived in-memory cache to avoid re-fetching the same URL
# ---------------------------------------------------------------------------

_FETCH_CACHE: dict[str, tuple[float, str]] = {}
_FETCH_CACHE_TTL = 300  # 5 minutes


def _fetch_cache_get(url: str) -> str | None:
    """Retrieve a cached URL fetch result, if it exists and is not expired."""
    entry = _FETCH_CACHE.get(url)
    if entry and (time.time() - entry[0]) < _FETCH_CACHE_TTL:
        return entry[1]
    return None


def _fetch_cache_set(url: str, content: str) -> None:
    """Cache a URL fetch result with an expiration time."""
    _FETCH_CACHE[url] = (time.time(), content)
    if len(_FETCH_CACHE) > 60:
        oldest = min(_FETCH_CACHE, key=lambda k: _FETCH_CACHE[k][0])
        del _FETCH_CACHE[oldest]


# ---------------------------------------------------------------------------
# Search cache — short-lived, avoids re-hitting rate-limited engines when the
# same query is searched again shortly after (auto-presearch immediately
# followed by an explicit web_search call, a model retry, etc.)
# ---------------------------------------------------------------------------

_SEARCH_CACHE: dict[tuple[str, int, str], tuple[float, str]] = {}
_SEARCH_CACHE_TTL = 180  # 3 minutes — shorter than fetch cache since freshness matters more


def _search_cache_get(key: tuple[str, int, str]) -> str | None:
    """Retrieve a cached search result, if it exists and is not expired."""
    entry = _SEARCH_CACHE.get(key)
    if entry and (time.time() - entry[0]) < _SEARCH_CACHE_TTL:
        return entry[1]
    return None


def _search_cache_set(key: tuple[str, int, str], content: str) -> None:
    """Cache a search result with an expiration time."""
    _SEARCH_CACHE[key] = (time.time(), content)
    if len(_SEARCH_CACHE) > 40:
        oldest = min(_SEARCH_CACHE, key=lambda k: _SEARCH_CACHE[k][0])
        del _SEARCH_CACHE[oldest]


# ---------------------------------------------------------------------------
# Web search — all engines in parallel, merged via Reciprocal Rank Fusion
# ---------------------------------------------------------------------------

_BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)


def _format_search_results(query: str, results: list, source: str = "") -> str:
    """Format raw search engine results into a readable Markdown string."""
    tag = f" (via {source})" if source else ""
    lines = [f'Web search results{tag} for: "{query}"\n']
    for i, r in enumerate(results, 1):
        title = r.get("title") or r.get("name") or f"Result {i}"
        url = r.get("href") or r.get("url") or ""
        body = r.get("body") or r.get("snippet") or r.get("description") or ""
        date = r.get("date") or r.get("published") or ""
        lines.append(f"{i}. {title}")
        if url:
            lines.append(f"   URL: {url}")
        if date:
            lines.append(f"   Date: {date}")
        if body:
            lines.append(f"   {body[:300]}")
        lines.append("")
    return "\n".join(lines)


def _merge_results(
    all_results: list[tuple[str, list[dict]]],
    max_results: int,
) -> tuple[str, list[dict]]:
    """Merge multi-engine results via Reciprocal Rank Fusion (RRF, k=60)."""
    scores: dict[str, float] = {}
    url_to_result: dict[str, dict] = {}
    sources_used: list[str] = []

    for source_name, results in all_results:
        if not results:
            continue
        sources_used.append(source_name)
        seen: set[str] = set()
        for rank, r in enumerate(results):
            raw_url = r.get("href") or r.get("url") or ""
            # Normalize for deduplication (strip query string + trailing slash)
            norm = raw_url.split("?")[0].rstrip("/")
            if not norm or norm in seen:
                continue
            seen.add(norm)
            scores[norm] = scores.get(norm, 0.0) + 1.0 / (rank + 60)
            if norm not in url_to_result:
                url_to_result[norm] = r

    sorted_urls = sorted(scores, key=lambda u: -scores[u])[:max_results]
    return "+".join(sources_used), [url_to_result[u] for u in sorted_urls]


def _search_google(query: str, max_results: int) -> list[dict]:
    """Perform a web search using the Google Custom Search API."""
    if not GOOGLE_API_KEY or not GOOGLE_CSE_ID:
        return []
    q = urllib.parse.quote_plus(query)
    url = f"https://www.googleapis.com/customsearch/v1?cx={GOOGLE_CSE_ID}&q={q}&num={min(max_results, 10)}"
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "qwen-tools/1.0",
            "X-goog-api-key": GOOGLE_API_KEY,
        },
    )
    with urllib.request.urlopen(req, timeout=10) as r:
        data = json.loads(r.read())
    return [
        {"title": it.get("title", ""), "href": it.get("link", ""), "body": it.get("snippet", "")}
        for it in (data.get("items") or [])
    ]


def _search_brave(query: str, max_results: int) -> list[dict]:
    """Perform a web search using the Brave Search API."""
    if not BRAVE_API_KEY:
        return []
    q = urllib.parse.quote_plus(query)
    url = f"https://api.search.brave.com/res/v1/web/search?q={q}&count={min(max_results, 20)}"
    req = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": BRAVE_API_KEY,
        },
    )
    with urllib.request.urlopen(req, timeout=10) as r:
        raw = r.read()
        enc = r.headers.get("Content-Encoding") or ""
        data = json.loads(_gzip.decompress(raw) if "gzip" in enc else raw)
    return [
        {"title": it.get("title", ""), "href": it.get("url", ""), "body": it.get("description", "")}
        for it in ((data.get("web") or {}).get("results") or [])
    ]


def _search_brave_news(query: str, max_results: int) -> list[dict]:
    """Perform a news-specific search using the Brave Search API."""
    if not BRAVE_API_KEY:
        return []
    q = urllib.parse.quote_plus(query)
    url = f"https://api.search.brave.com/res/v1/news/search?q={q}&count={min(max_results, 20)}"
    req = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": BRAVE_API_KEY,
        },
    )
    with urllib.request.urlopen(req, timeout=10) as r:
        raw = r.read()
        enc = r.headers.get("Content-Encoding") or ""
        data = json.loads(_gzip.decompress(raw) if "gzip" in enc else raw)
    return [
        {
            "title": it.get("title", ""),
            "href": it.get("url", ""),
            "body": it.get("description", ""),
            "date": it.get("age", ""),
        }
        for it in (data.get("results") or [])
    ]


def _search_ddg(query: str, max_results: int) -> list[dict]:
    """Perform a web search via DuckDuckGo HTML scraping."""
    try:
        from ddgs import DDGS
    except ImportError:
        from duckduckgo_search import DDGS  # type: ignore
    with DDGS(timeout=15) as ddgs:
        # backend="auto" fans out to every registered engine, including grokipedia
        # (~37% 502) and the unofficial brave scrape (~84% 429) -- pin to the
        # engines that actually succeed to cut wasted round trips.
        return list(ddgs.text(query, max_results=max_results, backend="duckduckgo,yahoo,yandex,mojeek,wikipedia"))


def _search_ddg_news(query: str, max_results: int) -> list[dict]:
    """Perform a news search via DuckDuckGo HTML scraping."""
    try:
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS  # type: ignore
        with DDGS(timeout=15) as ddgs:
            raw = list(ddgs.news(query, max_results=max_results))
        return [
            {
                "title": r.get("title", ""),
                "href": r.get("url", "") or r.get("link", ""),
                "body": r.get("body", "") or r.get("excerpt", ""),
                "date": r.get("date", ""),
            }
            for r in raw
        ]
    except Exception:
        return []


def _search_bing_scrape(query: str, max_results: int) -> list[dict]:
    """Perform a web search via Bing HTML scraping."""
    url = f"https://www.bing.com/search?q={urllib.parse.quote_plus(query)}&count={max_results}"
    req = urllib.request.Request(url, headers={"User-Agent": _BROWSER_UA})
    with urllib.request.urlopen(req, timeout=12) as resp:
        html_body = resp.read().decode("utf-8", errors="replace")
    import html as html_mod

    results: list[dict] = []
    for pattern in [
        r'<h2[^>]*><a[^>]+href="(https?://[^"]+)"[^>]*>(.*?)</a></h2>.*?<p[^>]*>(.*?)</p>',
        r'<a[^>]+href="(https?://[^"]+)"[^>]*>(.*?)</a>.*?<span[^>]*>(.*?)</span>',
    ]:
        if results:
            break
        for block in re.findall(pattern, html_body, re.DOTALL)[:max_results]:
            href = block[0]
            title = html_mod.unescape(re.sub(r"<[^>]+>", "", block[1] if len(block) > 1 else "")).strip()
            snip = html_mod.unescape(re.sub(r"<[^>]+>", "", block[2] if len(block) > 2 else "")).strip()
            if href and "bing.com" not in href:
                results.append({"href": href, "title": title, "body": snip})
    return results


def do_web_search(query: str, max_results: int = 6, type: str = "web") -> str:
    """Search the web across all engines, merge results via Reciprocal Rank Fusion."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    cache_key = (query, max_results, type)
    cached = _search_cache_get(cache_key)
    if cached is not None:
        return cached

    if type == "news":
        engines = [("DDG-news", _search_ddg_news)]
        if BRAVE_API_KEY:
            engines.append(("Brave-news", _search_brave_news))
        if GOOGLE_API_KEY and GOOGLE_CSE_ID:
            engines.append(("Google", _search_google))
        if BRAVE_API_KEY:
            engines.append(("Brave", _search_brave))
    else:
        engines = []
        if GOOGLE_API_KEY and GOOGLE_CSE_ID:
            engines.append(("Google", _search_google))
        if BRAVE_API_KEY:
            engines.append(("Brave", _search_brave))
        engines.append(("DDG", _search_ddg))
        engines.append(("Bing", _search_bing_scrape))

    all_results: list[tuple[str, list[dict]]] = []

    ex = ThreadPoolExecutor(max_workers=4)
    try:
        futures = {ex.submit(fn, query, max_results): name for name, fn in engines}
        try:
            for fut in as_completed(futures, timeout=20):
                name = futures[fut]
                try:
                    items = fut.result()
                    if items:
                        all_results.append((name, items))
                except Exception:
                    _logger.debug("Search engine '%s' failed", name)
        except TimeoutError:
            _logger.debug("Web search timed out after 20s")
    finally:
        ex.shutdown(wait=False)

    if not all_results:
        return f"[all search engines failed for: '{query}']"

    source_label, merged = _merge_results(all_results, max_results)
    formatted = _format_search_results(query, merged, source=source_label)
    _search_cache_set(cache_key, formatted)
    return formatted


def do_search_news(query: str, max_results: int = 8) -> str:
    """Search specifically for recent news articles about the query."""
    year = datetime.now().year
    # Ensure the query includes the current year if it doesn't already reference a year
    if not re.search(r"\b20\d{2}\b", query):
        query = f"{query} {year}"
    return do_web_search(query, max_results=max_results, type="news")


# ---------------------------------------------------------------------------
# Auto-presearch decision — shared trigger logic for CLI and web
# ---------------------------------------------------------------------------

_CONVERSATIONAL_RE = re.compile(
    r"^\s*(yes|no|ok|okay|sure|thanks|thank you|got it|cool|great|nice|nope|yep|nah|"
    r"sounds good|perfect|understood|makes sense|agreed|i see|alright|right|correct|"
    r"good|awesome|np|please|go ahead|do it|continue|proceed|keep going|stop)\s*[.!?]?\s*$",
    re.IGNORECASE,
)

_SKIP_PRESEARCH_RE = re.compile(
    r"^\s*(/|```|"
    r"\bwrite\s+(this|the\s+file)|"
    r"\bcreate\s+(the\s+file|a\s+file|this\s+file)|"
    r"\bedit\s+(the\s+file|this\s+file)|"
    r"\brun\s+this|"
    r"\bwhat\s+(are|is)\s+(you|your)|"
    r"\btell\s+me\s+about\s+(your|you)\b|"
    r"\bshow\s+me\s+(your|the)\s+(code|files?|source)|"
    r"\bhow\s+(can|do|should|would)\s+(we|i|you)\s+(improve|fix|update|change|use)\s+"
    r"(your|the)\s+(gui|cli|ui|interface|app|tool)|"
    r"\b(translate|rewrite|rephrase|reword|paraphrase|proofread|"
    r"fix\s+(the\s+)?grammar|format\s+this|summarize\s+this\s+(text|paragraph))\b)",
    re.IGNORECASE,
)

_FACTUAL_RE = re.compile(
    r"\?"
    r"|^\s*(is|are|was|were|do|does|did|can|could|will|would|should|has|have|had)\b"
    r"|\b(what|what'?s|how|why|when|where|who|who'?s|which|whose|whom|"
    r"latest|current|currently|recent|recently|today|tonight|now|nowadays|"
    r"news|price|prices|cost|stock|market|weather|forecast|score|"
    r"release|released|version|update|updated|changelog|deadline|"
    r"tell\s+me\s+about|explain|describe|find|search|look\s+up|research|"
    r"compare|versus|best|top|review|reviews|recommend|recommendation|"
    r"docs?|documentation|tutorial|guide|example|"
    r"(19|20)\d{2})\b",
    re.IGNORECASE,
)

_TIME_SENSITIVE_RE = re.compile(
    r"\b(latest|current|recent|today|now|new|newest|best|top|trending|"
    r"update|updated|release|released|version|changelog|announce|announced)\b",
    re.IGNORECASE,
)


def _presearch_query(text: str) -> str:
    """Pick a concise search query and add year context when the query is time-sensitive."""
    t = " ".join(text.strip().split())
    if len(t) > 180:
        parts = re.split(r"(?<=[.?!])\s+", t)
        chosen = next((p for p in parts if "?" in p), parts[0] if parts else t)
        t = chosen.strip()
    # Append the current year to time-sensitive queries that don't already name a year
    if _TIME_SENSITIVE_RE.search(t) and not re.search(r"\b20\d{2}\b", t):
        t = f"{t} {datetime.now().year}"
    return t[:200]


def presearch_decision(text: str, mode: str = "aggressive") -> tuple[bool, str]:
    """Decide whether to auto-run a web search before the model's first reply.

        mode = "off"        -> never auto-search
               "smart"      -> search only when the message shows factual intent
               "aggressive" -> search anything that isn't chit-chat, a code block,
                               or a text-transform / meta task (the default)

    Returns (should_search, query).
    """
    t = (text or "").strip()
    mode = (mode or "aggressive").lower()
    if mode not in ("off", "smart", "aggressive"):
        mode = "aggressive"
    if mode == "off" or not t:
        return (False, "")
    if len(t.split()) < 2:
        return (False, "")
    if _CONVERSATIONAL_RE.match(t):
        return (False, "")
    if _SKIP_PRESEARCH_RE.match(t):
        return (False, "")
    if "```" in t:
        return (False, "")
    if mode == "smart" and not _FACTUAL_RE.search(t):
        return (False, "")
    return (True, _presearch_query(t))


# ---------------------------------------------------------------------------
# HTML extraction — readability → trafilatura → regex fallback
# ---------------------------------------------------------------------------


def _html_to_text(html: str, url: str = "") -> str:
    """Extract clean readable text from HTML using the best available library."""
    # readability-lxml: strips boilerplate, extracts article body
    try:
        from readability import Document

        doc = Document(html)
        body = doc.summary(html_partial=True)
        text = re.sub(r"<[^>]+>", " ", body)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        if len(text) > 200:
            return text
    except ImportError:
        _logger.debug("readability not installed, skipping")
    except Exception:
        _logger.debug("readability extraction failed for %s", url or "<unknown>")

    # trafilatura: good general-purpose extraction
    try:
        import trafilatura

        extracted = trafilatura.extract(
            html,
            url=url or None,
            include_tables=True,
            include_links=False,
            favor_recall=True,
        )
        if extracted and len(extracted.strip()) > 100:
            return extracted
    except ImportError:
        _logger.debug("trafilatura not installed, skipping")
    except Exception:
        _logger.debug("trafilatura extraction failed for %s", url or "<unknown>")

    # regex fallback
    text = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


# ---------------------------------------------------------------------------
# URL fetch (plain HTTP) — with retry, decompression, caching, JSON/XML handling
# ---------------------------------------------------------------------------

_FETCH_HEADERS = {
    "User-Agent": _BROWSER_UA,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate",
    "DNT": "1",
}


def _decompress(raw: bytes, encoding: str) -> bytes:
    """Decompress gzip or bz2 encoded response content."""
    enc = encoding.lower()
    if "gzip" in enc:
        try:
            return _gzip.decompress(raw)
        except Exception:
            _logger.debug("gzip decompression failed, trying next method")
    if "deflate" in enc:
        try:
            return _zlib.decompress(raw)
        except Exception:
            try:
                return _zlib.decompress(raw, -15)
            except Exception:
                _logger.debug("deflate decompression failed")
    return raw


def _smart_truncate(text: str, limit: int) -> str:
    """Truncate at a sentence boundary near limit, falling back to hard cut."""
    if len(text) <= limit:
        return text
    window = text[max(0, limit - 600) : limit]
    last_break = max(window.rfind(". "), window.rfind(".\n"), window.rfind("\n\n"))
    cut = max(0, limit - 600) + last_break + 1 if last_break > 0 else limit
    return text[:cut] + f"\n\n... [truncated at {limit:,} chars]"


def _pdf_to_text(data: bytes, max_chars: int) -> str:
    """Extract text content from a PDF file using PyPDF2."""
    try:
        import io

        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        pages = [p.extract_text() or "" for p in reader.pages]
        text = "\n\n".join(p.strip() for p in pages if p.strip())
        return text[:max_chars] if len(text) > max_chars else text
    except ImportError:
        return "[PDF support requires: pip install pypdf]"
    except Exception as e:
        return f"[PDF extraction error: {e}]"


_JS_SHELL_MARKERS = (
    'id="root"',
    'id="app"',
    "ng-version",
    "data-reactroot",
    "__next",
    "you need to enable javascript",
)


def _looks_like_js_shell(html: str, text: str) -> bool:
    """Heuristic: does this response look like an empty client-rendered SPA shell?

    Plain HTTP fetches can't execute JS, so React/Vue/Angular apps often come back
    as a near-empty <div id="root"> with all real content injected client-side.
    Flag it so the caller knows to escalate to fetch_rendered instead of silently
    treating a near-empty page as "this page just doesn't have much content."
    """
    stripped = text.strip()
    if len(stripped) >= 250:
        return False
    if len(html) < 1500:
        return False
    lowered = html.lower()
    return any(marker in lowered for marker in _JS_SHELL_MARKERS) or len(stripped) < 40


_ANTIBOT_MARKERS = (
    "checking your browser",
    "cf-browser-verification",
    "cf-challenge",
    "just a moment",
    "captcha",
    "hcaptcha",
    "recaptcha",
    "cloudflare turnstile",
    "ddos protection by",
    "access denied",
    "request blocked",
    "perimeterx",
    "datadome",
    "please verify you are a human",
    "please enable cookies",
)


def _looks_like_antibot_block(html: str) -> bool:
    """Heuristic: is this an anti-bot/CAPTCHA interstitial rather than real content?

    Plain HTTP sends no cookies, no JS execution, and no browser fingerprint —
    it's the fetch method most likely to trip these walls. Flag it so the
    caller escalates instead of treating "Just a moment..." as the real page.
    """
    lowered = html.lower()
    return any(marker in lowered for marker in _ANTIBOT_MARKERS)


def do_fetch_url(url: str, max_chars: int = 20000, **kwargs) -> str:
    """Fetch the raw text content of a URL.

    Handles redirects, compression, PDF extraction, and HTML-to-text conversion.
    Plain HTTP can't run JavaScript, hold cookies, or pass a fingerprint check —
    if the response looks like an empty client-rendered SPA shell or an
    anti-bot/CAPTCHA interstitial, a hint is appended telling the caller to
    retry with fetch_rendered or browser_action instead.
    """
    cached = _fetch_cache_get(url)
    if cached:
        return f"[cached] {cached}"

    last_err: Exception | None = None
    for attempt in range(3):
        if attempt:
            time.sleep(1.5 * attempt)
        try:
            req = urllib.request.Request(url, headers=dict(_FETCH_HEADERS))
            with urllib.request.urlopen(req, timeout=15) as resp:
                raw = resp.read()
                ctype = (resp.headers.get("Content-Type") or "").lower()
                encoding = resp.headers.get("Content-Encoding") or ""

            raw = _decompress(raw, encoding)

            # PDF
            if "pdf" in ctype or url.lower().endswith(".pdf"):
                text = _pdf_to_text(raw, max_chars)
                result = f"URL: {url}\n\n{text}"
                _fetch_cache_set(url, result)
                return result

            text = raw.decode("utf-8", errors="replace")

            # JSON
            stripped = text.lstrip()
            if "json" in ctype or (stripped and stripped[0] in ("{", "[")):
                try:
                    parsed = json.loads(text)
                    pretty = json.dumps(parsed, indent=2, ensure_ascii=False)
                    result = f"URL: {url}\n[JSON response]\n\n{_smart_truncate(pretty, max_chars)}"
                    _fetch_cache_set(url, result)
                    return result
                except Exception:
                    _logger.debug("JSON parse failed for %s, trying XML", url)

            # XML
            if "xml" in ctype or stripped[:5].lower() in ("<?xml", "<rss ", "<feed"):
                result = f"URL: {url}\n[XML/RSS]\n\n{_smart_truncate(text, max_chars)}"
                _fetch_cache_set(url, result)
                return result

            # HTML extraction
            escalation_hint = ""
            if "html" in ctype or stripped[:9].lower() in ("<!doctype", "<html"):
                extracted = _html_to_text(text, url=url)
                if _looks_like_antibot_block(text):
                    escalation_hint = (
                        "\n\n[NOTE: this response looks like an anti-bot/CAPTCHA challenge page, "
                        "not real content — plain HTTP sends no cookies or browser fingerprint to "
                        "pass it. Retry with fetch_rendered (may pass simple JS checks) or "
                        "browser_action (handles CAPTCHA) instead.]"
                    )
                elif _looks_like_js_shell(text, extracted):
                    escalation_hint = (
                        "\n\n[NOTE: this page returned little to no text over plain HTTP — "
                        "it looks like a JavaScript-rendered app (SPA shell). Retry with "
                        "fetch_rendered to get the actual content.]"
                    )
                text = extracted

            result = f"URL: {url}\n\n{_smart_truncate(text, max_chars)}{escalation_hint}"
            _fetch_cache_set(url, result)
            return result

        except urllib.error.HTTPError as e:
            if e.code in (429, 503, 502) and attempt < 2:
                last_err = e
                continue
            if e.code in (401, 403, 429):
                return (
                    f"[HTTP {e.code}: {e.reason} — {url}]\n"
                    "Tip: this looks like an anti-bot or access-control block. Try fetch_rendered "
                    "or browser_action instead of plain HTTP."
                )
            return f"[HTTP {e.code}: {e.reason} — {url}]"
        except Exception as e:
            last_err = e
            if attempt < 2:
                continue
            break

    return f"[fetch error after {attempt + 1} attempts: {last_err}]"


# ---------------------------------------------------------------------------
# Vision / image description
# ---------------------------------------------------------------------------


def do_describe_image(url: str, llm_client=None, llm_model: str = "") -> str:
    """Download an image and describe it via vision API (if client provided),
    with PIL metadata as fallback.
    """
    import base64
    import io

    try:
        req = urllib.request.Request(url, headers={"User-Agent": _BROWSER_UA})
        with urllib.request.urlopen(req, timeout=15) as r:
            data = r.read()
            ctype = r.headers.get_content_type() or "image/jpeg"
    except Exception as e:
        return f"[image fetch error: {e}]"

    lines = [f"Image URL: {url}"]

    try:
        from PIL import Image

        with Image.open(io.BytesIO(data)) as img:
            lines.append(f"Format: {img.format}  Size: {img.size[0]}x{img.size[1]}  Mode: {img.mode}")
            exif = img._getexif() if hasattr(img, "_getexif") else None
            if exif:
                from PIL.ExifTags import TAGS

                for tag_id, val in exif.items():
                    tag = TAGS.get(tag_id, tag_id)
                    if tag in ("Make", "Model", "DateTime", "GPSInfo", "ImageDescription"):
                        lines.append(f"EXIF {tag}: {val}")
    except Exception:
        _logger.debug("PIL/EXIF metadata extraction failed for %s", url)

    if llm_client:
        try:
            b64 = base64.b64encode(data).decode()
            resp = llm_client.chat.completions.create(
                model=llm_model or "Qwen3.6-27B",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Describe this image in detail."},
                            {"type": "image_url", "image_url": {"url": f"data:{ctype};base64,{b64}"}},
                        ],
                    }
                ],
                max_tokens=600,
                stream=False,
            )
            description = (resp.choices[0].message.content or "").strip()
            if description:
                lines.append(f"\nDescription:\n{description}")
                return "\n".join(lines)
        except Exception:
            lines.append("[Vision API not supported by the current model — metadata only]")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# YouTube / video transcript
# ---------------------------------------------------------------------------


def do_get_video_transcript(url: str, lang: str = "en") -> str:
    """Get transcript of a YouTube video, trying multiple language fallbacks."""
    yt_id_match = re.search(
        r"(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/shorts/)([A-Za-z0-9_-]{11})",
        url,
    )
    if yt_id_match:
        video_id = yt_id_match.group(1)
        try:
            from youtube_transcript_api import YouTubeTranscriptApi

            api = YouTubeTranscriptApi()
            tl = None
            for attempt in ([lang, "en"], ["en"], None):
                try:
                    tl = api.fetch(video_id, languages=attempt) if attempt else next(iter(api.list(video_id))).fetch()
                    break
                except Exception:
                    _logger.debug("YouTube transcript fetch failed for %s (attempt %s)", video_id, attempt)
                    continue
            if tl:
                text = " ".join(s.get("text", "") for s in tl)
                if len(text) > 15_000:
                    text = text[:15_000] + "\n...[truncated]"
                return f"YouTube transcript for: {url}\n\n{text}"
        except Exception:
            _logger.debug("YouTubeTranscriptApi failed for %s", url)
    return f"[No transcript available — page content follows]\n{do_fetch_url(url, max_chars=8_000)}"
