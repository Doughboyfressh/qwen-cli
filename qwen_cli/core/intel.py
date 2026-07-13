"""Live Intelligence — background web crawlers + feed + memory training.

Extracted from main.py as part of the monolith split. This module owns all
intel state (topics, queue, feed, crawler threads, enable/stop flags); other
modules reach it through main's re-exports (`_main._intel_enabled`, ...).

Cross-subsystem calls (aux LLM, persistent memory, web search) go through a
lazy `import qwen_cli.main as _main` at call time — the project convention for
split-out modules (see core/repl.py). That keeps main.py the namespace of
record while those subsystems still live there: tests monkeypatch
`qwen_cli.main._bg_llm` etc. and the patch takes effect here too. When a
subsystem moves out of main, only the `_main.` references here change.
"""

import json
import logging
import threading
import time
from datetime import datetime

from qwen_cli.core.config import (
    _INTEL_INJECT_N,
    _INTEL_INTERVAL,
    AUX_LLM_TIMEOUT,
    INTEL_DIR,
    INTEL_FEED,
    INTEL_MODE,
    INTEL_QUEUE,
    INTEL_TOPICS,
)

_logger = logging.getLogger(__name__)

_INTEL_CRAWLERS = 3  # number of parallel background browser threads

_intel_stop = threading.Event()
_intel_lock = threading.Lock()
_intel_enabled = threading.Event()  # thread-safe flag for intel crawlers
_intel_memory_written: dict[str, str] = {}  # topic_name → date; prevents duplicate entries
_intel_threads_started = False  # crawler threads spawn at most once per process


def _intel_default_topics() -> list[dict]:
    """Return the default set of topics for the Live Intelligence background crawlers."""
    year = datetime.now().year
    return [
        {"name": "AI & LLM news", "query": "latest AI LLM model releases news today", "last_checked": 0},
        {"name": "Python ecosystem", "query": f"Python new libraries tools releases {year}", "last_checked": 0},
        {
            "name": "Security vulnerabilities",
            "query": "critical security vulnerabilities CVE this week",
            "last_checked": 0,
        },
        {"name": "Tech industry news", "query": "technology industry news today", "last_checked": 0},
        {"name": "Open source trending", "query": "trending open source projects GitHub today", "last_checked": 0},
        {"name": "Developer APIs", "query": f"new developer APIs web services released {year}", "last_checked": 0},
    ]


_INTEL_DEFAULT_TOPICS: list[dict] = _intel_default_topics()


def _intel_load_topics() -> list[dict]:
    """Internal helper: intel load topics."""
    if INTEL_TOPICS.exists():
        try:
            return json.loads(INTEL_TOPICS.read_text(encoding="utf-8"))
        except Exception:
            _logger.debug("Failed to load intel topics from %s", INTEL_TOPICS)
    return [dict(t) for t in _INTEL_DEFAULT_TOPICS]


def _intel_save_topics(topics: list[dict]) -> None:
    """Internal helper: intel save topics."""
    INTEL_DIR.mkdir(exist_ok=True)
    INTEL_TOPICS.write_text(json.dumps(topics, indent=2, ensure_ascii=False), encoding="utf-8")


def _intel_enqueue(topic_name: str, query: str, raw: str) -> None:
    """Internal helper: intel enqueue."""
    with _intel_lock:
        INTEL_DIR.mkdir(exist_ok=True)
        try:
            items = json.loads(INTEL_QUEUE.read_text(encoding="utf-8")) if INTEL_QUEUE.exists() else []
        except Exception:
            items = []
        items.append({"topic": topic_name, "query": query, "raw": raw[:3000], "ts": time.time()})
        INTEL_QUEUE.write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")


def _intel_dequeue_all() -> list[dict]:
    """Internal helper: intel dequeue all."""
    with _intel_lock:
        if not INTEL_QUEUE.exists():
            return []
        try:
            items = json.loads(INTEL_QUEUE.read_text(encoding="utf-8"))
            INTEL_QUEUE.write_text("[]", encoding="utf-8")
            return items
        except Exception:
            return []


def _intel_load_feed() -> str:
    """Internal helper: intel load feed."""
    if INTEL_FEED.exists():
        try:
            return INTEL_FEED.read_text(encoding="utf-8").strip()
        except Exception:
            _logger.debug("Failed to load intel feed from %s", INTEL_FEED)
    return ""


def _intel_prepend_entry(topic_name: str, summary: str) -> None:
    """Internal helper: intel prepend entry."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    entry = f"<!-- {ts} | {topic_name} -->\n{summary.strip()}"
    with _intel_lock:
        existing = _intel_load_feed()
        chunks = [c.strip() for c in existing.split("\n\n") if c.strip()] if existing else []
        chunks.insert(0, entry)
        chunks = chunks[:40]  # cap at 40 entries
        INTEL_FEED.write_text("\n\n".join(chunks), encoding="utf-8")


def intel_get_recent(n: int = _INTEL_INJECT_N) -> str:
    """Return the N most recent intel entries for injection into system prompt."""
    feed = _intel_load_feed()
    if not feed:
        return ""
    chunks = [c.strip() for c in feed.split("\n\n") if c.strip()]
    return "\n\n".join(chunks[:n])


def _intel_crawl_once() -> None:
    """Pick the least-recently-crawled topic, do a web search, enqueue the raw result."""
    import qwen_cli.main as _main

    if not _intel_enabled.is_set():
        return
    topic: dict | None = None
    try:
        # _INTEL_CRAWLERS runs multiple copies of this function concurrently.
        # Claim the topic (mark last_checked, save) under the lock *before*
        # doing the slow web search, so two threads never pick the same
        # least-recently-crawled topic and _intel_extract_topics()'s topic
        # additions can't be lost to an overlapping read-modify-write.
        with _intel_lock:
            topics = _intel_load_topics()
            if not topics:
                return
            topic = min(topics, key=lambda t: t.get("last_checked", 0))
            for t in topics:
                if t["name"] == topic["name"]:
                    t["last_checked"] = time.time()
                    break
            _intel_save_topics(topics)
        raw = _main.do_web_search(topic["query"], max_results=5)
        if raw and "error" not in raw.lower()[:40]:
            _intel_enqueue(topic["name"], topic["query"], raw)
    except Exception:
        _logger.debug("Intel background crawl failed for topic '%s'", topic.get("name", "?") if topic else "?")


def _intel_process_queue(client) -> None:
    """Post-turn: LLM-summarize queued raw results, update feed, train memory."""
    import qwen_cli.main as _main

    items = _intel_dequeue_all()
    if not items:
        return
    with _main._main_llm_busy_lock:
        if _main._main_llm_busy and _main._aux_client is None:
            return
    for item in items:
        try:
            prompt = [
                {
                    "role": "system",
                    "content": (
                        "Summarize these web search results into 3-5 concise bullet points "
                        "(max 90 chars each). Focus on concrete facts, releases, and updates. "
                        "No preamble, just the bullets."
                    ),
                },
                {"role": "user", "content": f"Topic: {item['topic']}\n\n{item['raw'][:3000]}"},
            ]
            bg_client, bg_model = _main._bg_llm(client)
            resp = bg_client.chat.completions.create(
                model=bg_model,
                messages=prompt,
                stream=False,
                max_tokens=250,
                timeout=AUX_LLM_TIMEOUT,
            )
            summary = (resp.choices[0].message.content or "").strip()
            if not summary or len(summary) < 20:
                continue
            _intel_prepend_entry(item["topic"], summary)
            _intel_train_memory(client, item["topic"], summary)
        except Exception:
            _logger.debug("Intel queue item '%s' failed", item.get("topic", "?"))


def _intel_train_memory(client: object, topic_name: str, summary: str) -> None:
    """If the intel summary contains durable facts, add them to persistent memory."""
    import qwen_cli.main as _main

    today = datetime.now().strftime("%Y-%m-%d")
    if _intel_memory_written.get(topic_name) == today:
        return  # already wrote facts for this topic today
    try:
        prompt = [
            {
                "role": "system",
                "content": (
                    "Decide if any of these facts should be saved to a persistent memory file "
                    "(version numbers, critical releases, security alerts, key API changes). "
                    "If yes, output 1-2 short bullet lines starting with '- '. "
                    "If nothing is worth persisting, reply exactly: SKIP"
                ),
            },
            {"role": "user", "content": f"Topic: {topic_name}\n{summary}"},
        ]
        bg_client, bg_model = _main._bg_llm(client)
        resp = bg_client.chat.completions.create(
            model=bg_model,
            messages=prompt,
            stream=False,
            max_tokens=120,
            timeout=AUX_LLM_TIMEOUT,
        )
        facts = _main._clean_memory_facts((resp.choices[0].message.content or "").strip(), drop_negations=True)
        if facts and facts.startswith("-"):
            with _main._memory_lock:
                mem = _main.load_memory()
                tag = f"\n\n<!-- intel {today} -->\n{facts}"
                _main.save_memory((mem + tag).strip())
            _intel_memory_written[topic_name] = today
    except Exception:
        _logger.debug("Intel memory training failed for '%s'", topic_name)


def _intel_extract_topics(client, user_msg: str, reply: str) -> None:
    """Post-turn: extract new search-worthy topics from this exchange and track them."""
    import qwen_cli.main as _main

    with _main._main_llm_busy_lock:
        if _main._main_llm_busy and _main._aux_client is None:
            return
    try:
        prompt = [
            {
                "role": "system",
                "content": (
                    "Extract up to 2 web-searchable topics from this conversation worth monitoring "
                    "(new technologies, frameworks, tools, or domains the user cares about). "
                    "Reply one per line as: NAME|search query\n"
                    "Example: FastAPI 1.0|FastAPI 1.0 release features changelog\n"
                    "If nothing new to track, reply: NONE"
                ),
            },
            {"role": "user", "content": f"User: {user_msg[:400]}\nAssistant: {reply[:400]}"},
        ]
        bg_client, bg_model = _main._bg_llm(client)
        resp = bg_client.chat.completions.create(
            model=bg_model,
            messages=prompt,
            stream=False,
            max_tokens=80,
            timeout=AUX_LLM_TIMEOUT,
        )
        text = (resp.choices[0].message.content or "").strip()
        if not text or text.upper() == "NONE":
            return
        # See _intel_crawl_once(): topics.json is shared with the background
        # crawler threads, so the read-modify-write must hold _intel_lock too
        # or a concurrent crawl can silently overwrite these additions.
        with _intel_lock:
            topics = _intel_load_topics()
            existing = {t["name"].lower() for t in topics}
            for line in text.splitlines():
                if "|" not in line:
                    continue
                name, query = line.split("|", 1)
                name, query = name.strip(), query.strip()
                if name and query and name.lower() not in existing:
                    topics.append({"name": name, "query": query, "last_checked": 0})
                    existing.add(name.lower())
            if len(topics) > 25:
                topics = sorted(topics, key=lambda t: t.get("last_checked", 0), reverse=True)[:25]
            _intel_save_topics(topics)
    except Exception:
        _logger.debug("Intel topic extraction failed")


def _intel_crawler_thread(delay_s: int) -> None:
    """Background daemon: crawl one topic every _INTEL_INTERVAL seconds."""
    INTEL_DIR.mkdir(exist_ok=True)
    _intel_stop.wait(timeout=delay_s)  # stagger startup
    while not _intel_stop.is_set():
        try:
            _intel_crawl_once()
        except Exception:
            # _intel_crawl_once() already wraps its own body in try/except;
            # this outer guard exists so a bug there can never silently kill
            # this thread forever (as `_intel_enabled = False` used to, by
            # replacing the Event with a plain bool — see _cmd_intel).
            _logger.exception("Intel crawler thread tick failed")
        _intel_stop.wait(timeout=_INTEL_INTERVAL)


def start_intel_crawlers(force: bool = False) -> None:
    """Start _INTEL_CRAWLERS background crawler threads, staggered.

    Opt-in: crawlers cost 3 background browser threads plus system-prompt
    tokens for the injected feed. They start at launch only with config
    intel="on" (or QWEN_INTEL=on); /intel on passes force=True to start
    them mid-session. Idempotent — a second call just re-enables crawling
    if it was paused with /intel off.
    """
    global _intel_threads_started
    if not force and INTEL_MODE != "on":
        return
    if not INTEL_TOPICS.exists():
        _intel_save_topics([dict(t) for t in _INTEL_DEFAULT_TOPICS])
    # threading.Event() starts unset; without this, _intel_crawl_once()'s
    # `if not _intel_enabled.is_set(): return` guard is true forever and no
    # crawler thread ever does real work.
    _intel_enabled.set()
    if _intel_threads_started:
        return
    _intel_threads_started = True
    stagger = max(15, _INTEL_INTERVAL // _INTEL_CRAWLERS)
    for i in range(_INTEL_CRAWLERS):
        t = threading.Thread(
            target=_intel_crawler_thread,
            args=(15 + i * stagger,),
            daemon=True,
            name=f"intel-crawler-{i}",
        )
        t.start()
