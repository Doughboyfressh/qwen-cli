#!/usr/bin/env python3
"""Browser automation module — Playwright-based stealth browser for qwen-cli."""

import contextlib
import json
import random as _r
import time
import urllib.parse
from collections.abc import Callable
from pathlib import Path
from typing import Any

from rich.console import Console

from qwen_cli.core.config import DATA_DIR, _load_config
from qwen_cli.tools.shared import _html_to_text

console = Console(force_terminal=True, legacy_windows=False)
_CFG = _load_config()


def _clear_loitering_event_loop() -> None:
    """Close a leftover "running" asyncio loop on this thread before starting Playwright.

    Playwright's sync API refuses to start if it detects a running asyncio
    event loop on the current thread ("Sync API inside asyncio loop"). LSP
    handshakes and other background work can leave the main thread's asyncio
    state marked as running without cleanly returning control, which is
    exactly what core.repl._close_loitering_event_loop() already guards
    against before every input prompt for the same reason. sync_playwright()
    calls here need the identical guard, or a stale loop from earlier
    background work makes browser_action/fetch_rendered fail outright.
    """
    import asyncio

    try:
        loop = asyncio.get_running_loop()
        with contextlib.suppress(RuntimeError):
            loop.close()
    except RuntimeError:
        pass
    with contextlib.suppress(Exception):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())

# ---------------------------------------------------------------------------
# Browser automation (Playwright) — stealth, smart waits, CAPTCHA pause, cookies
# ---------------------------------------------------------------------------

_browser_state: dict = {}
_render_state: dict = {}  # dedicated headless page for fetch_rendered (separate from browser_action)
COOKIE_FILE = DATA_DIR / "browser_cookies.json"
# Persistent on-disk Chrome profiles (real cookies/localStorage/IndexedDB/cache,
# not a fresh throwaway context every run). Session/device history is itself a
# signal heavier anti-bot systems (DataDome, Akamai, PerimeterX) weight -- a
# browser with zero history on every request looks bot-like no matter how good
# the JS stealth patches are. Separate dirs because two Chrome processes can't
# share one user-data-dir, and browser_action/fetch_rendered can run concurrently.
BROWSER_PROFILE_DIR = DATA_DIR / "browser_profile"
RENDER_PROFILE_DIR = DATA_DIR / "browser_profile_render"

_STEALTH_JS = """// === Comprehensive Anti-Detection ===

// 0. Block Playwright-specific Function.toString leak
(function() {
  const _origToString = Function.prototype.toString;
  Function.prototype.toString = function() {
    if (typeof this === 'function' && this.name) {
      return 'function ' + this.name + '() { [native code] }';
    }
    return _origToString.apply(this, arguments);
  };
  Function.prototype.toString.toString = () => 'function toString() { [native code] }';
})();

// 1. Hide webdriver flag — multiple detection vectors
Object.defineProperty(navigator, 'webdriver', {
  get: () => undefined,
  configurable: true,
  enumerable: true,
});

// 1b. hasOwnProperty('webdriver') detection bypass
(function() {
  const _origHas = Object.prototype.hasOwnProperty;
  Object.prototype.hasOwnProperty = function(prop) {
    if (this === navigator && prop === 'webdriver') return false;
    return _origHas.apply(this, arguments);
  };
})();

// 2. Full chrome object (with missing app, support, runtime)
window.chrome = {
  runtime: {
    onMessage: { addListener: () => {}, removeListener: () => {}, hasListener: () => false },
    connect: () => ({ onMessage: { addListener: () => {}, removeListener: () => {} } }),
    sendMessage: () => {},
    executionContext: 1,
  },
  app: {
    isInstalled: false,
    InstallState: { disabled: 'disabled', installed: 'installed', not_installed: 'not_installed' },
    RunningState: { running: 'running', not_running: 'not_running' },
    getDetails: () => ({ id: '' }),
  },
  loadTimes: function() {},
  csi: function() {},
  support: { createScript: () => {}, removeScript: () => {} },
};

// 3. Plugin and MimeType arrays
const _plugins = [
  {name:'Chrome PDF Plugin', filename:'internal-pdf-viewer', description:'Portable Document Format', length:1},
  {name:'Chrome PDF Viewer', filename:'mhjfbmdgcfjbbpaeojofohoefgiehjai', description:'Chrome PDF Plugin', length:1},
  {name:'Widevine Content Decryption Module', filename:'widevinecdm.dll', description:'Widevine', length:1},
];
Object.defineProperty(navigator, 'plugins', {
  get: () => {
    const arr = Object.values(_plugins).map(p => Object.assign({}, p, { enabled: true }));
    arr.length = _plugins.length;
    return arr;
  },
  configurable: true,
});

const _mimeTypes = [
  {type:'application/pdf', suffixes:'pdf', description:'Portable Document Format'},
  {type:'application/x-google-chrome-pdf', suffixes:'pdf', description:'Portable Document Format'},
];
Object.defineProperty(navigator, 'mimeTypes', {
  get: () => {
    const arr = Object.values(_mimeTypes).map(m => Object.assign({}, m));
    arr.length = _mimeTypes.length;
    return arr;
  },
  configurable: true,
});

// 4. Standard navigator properties
Object.defineProperty(navigator, 'languages', { get: () => ['en-US', 'en'], configurable: true });
Object.defineProperty(navigator, 'hardwareConcurrency', { get: () => 4, configurable: true });
Object.defineProperty(navigator, 'maxTouchPoints', { get: () => 1, configurable: true });
Object.defineProperty(navigator, 'deviceMemory', { get: () => 8, configurable: true });
Object.defineProperty(navigator, 'vendor', { get: () => 'Google Inc.', configurable: true });
Object.defineProperty(navigator, 'platform', { get: () => 'Win32', configurable: true });
Object.defineProperty(navigator, 'pdfViewer', { get: () => true, configurable: true });

// 4b. WebRTC IP leak prevention
try {
  const _origRTCPeer = window.RTCPeerConnection;
  if (_origRTCPeer) {
    window.RTCPeerConnection = function(config) {
      if (config && config.iceServers) config.iceServers = [];
      return new _origRTCPeer(config);
    };
    window.RTCPeerConnection.prototype = _origRTCPeer.prototype;
  }
} catch(e) {}

// 5. Screen properties and outer dimensions (headless browsers leak outerWidth=0)
// Snapshot the real values BEFORE redefining window.screen -- a getter that
// reads the bare identifier `screen` (as the old code did via
// Object.assign({}, screen, ...)) resolves to window.screen, i.e. the very
// property being redefined, recursing into itself on every access until the
// stack overflows. Any real page reading screen.width/height (extremely
// common) would hit an uncaught RangeError -- a far worse tell than not
// spoofing screen at all. A static snapshot avoids re-deriving anything live.
const _realScreen = window.screen;
const _screenSnapshot = {
  width: _realScreen.width,
  height: _realScreen.height,
  availWidth: _realScreen.availWidth,
  availHeight: _realScreen.availHeight,
  availLeft: _realScreen.availLeft,
  availTop: _realScreen.availTop,
  colorDepth: 24,
  pixelDepth: 24,
  orientation: { type: 'landscape-primary', angle: 0, onchange: null },
};
Object.defineProperty(window, 'screen', {
  get: () => _screenSnapshot,
  configurable: true,
  enumerable: true,
});
if (window.outerWidth === 0 || window.outerHeight === 0) {
  Object.defineProperty(window, 'outerWidth', { get: () => window.innerWidth + 16, configurable: true });
  Object.defineProperty(window, 'outerHeight', { get: () => window.innerHeight + 78, configurable: true });
}

// 5b. Screen position offsets (real browsers are rarely exactly 0,0)
if (window.screenLeft === 0 && window.screenTop === 0) {
  Object.defineProperty(window, 'screenLeft', { value: 20, configurable: true });
  Object.defineProperty(window, 'screenTop', { value: 20, configurable: true });
  Object.defineProperty(window, 'screenX', { value: 20, configurable: true });
  Object.defineProperty(window, 'screenY', { value: 20, configurable: true });
}

// 6. navigator.permissions override
const origQuery = window.navigator.permissions.query;
window.navigator.permissions.query = (params) =>
    params.name === 'notifications'
        ? Promise.resolve({ state: Notification.permission })
        : origQuery(params);

// 6b. TrustedTypes — present in real Chrome, absent in headless
try {
  if (!window.trustedTypes) {
    window.trustedTypes = {
      createPolicy: (name, config) => ({
        createScript: (s) => s,
        createScriptUrl: (s) => s,
        createScriptElement: (s) => null,
        createStyle: (s) => s,
        createURL: (s) => s,
      }),
      isHTML: () => false,
      isScriptURL: () => false,
    };
  }
} catch(e) {}

// 7. WebGL 1.0 and 2.0 vendor/renderer spoofing
const glParam = WebGLRenderingContext.prototype.getParameter;
WebGLRenderingContext.prototype.getParameter = function(param) {
  if (param === 37445) return 'Google Inc. (NVIDIA)';
  if (param === 37446) return 'ANGLE (NVIDIA, NVIDIA GeForce, Direct3D)';
  return glParam.apply(this, arguments);
};
try {
  const gl2Param = WebGL2RenderingContext.prototype.getParameter;
  WebGL2RenderingContext.prototype.getParameter = function(param) {
    if (param === 37445) return 'Google Inc. (NVIDIA)';
    if (param === 37446) return 'ANGLE (NVIDIA, NVIDIA GeForce, Direct3D12)';
    return gl2Param.apply(this, arguments);
  };
} catch(e) {}

// 8. Canvas fingerprint noise — perturb sub-pixel rendering
const _origGetContext = HTMLCanvasElement.prototype.getContext;
HTMLCanvasElement.prototype.getContext = function(type) {
  const ctx = _origGetContext.apply(this, arguments);
  if (!ctx || type !== '2d') return ctx;
  const _fillText = ctx.fillText.bind(ctx);
  ctx.fillText = function(...args) {
    ctx.imageSmoothingEnabled = false;
    return _fillText(...args);
  };
  return ctx;
};

// 9. AudioContext fingerprint noise
try {
  const _origGetFloat = AnalyserNode.prototype.getFloatFrequencyData;
  AnalyserNode.prototype.getFloatFrequencyData = function(freqData) {
    _origGetFloat.call(this, freqData);
    for (let i = 0; i < freqData.length; i++) {
      if (!isNaN(freqData[i])) {
        freqData[i] += (Math.random() - 0.5) * 1e-30;
      }
    }
  };
} catch(e) {}

// 10. navigator.connection
Object.defineProperty(navigator, 'connection', {
  get: () => ({ downlink: 10, effectiveType: '4g', rtt: 50, saveData: false }),
  configurable: true,
});

// 11. Remove iframe sandbox detection
try { delete HTMLIFrameElement.prototype.sandbox; } catch(e) {}

// 12. Performance entries — filter out devtools/CDP URLs
const _origPerfEntries = performance.getEntriesByType.bind(performance);
performance.getEntriesByType = function(type) {
  if (type === 'resource') {
    return _origPerfEntries(type).filter(e =>
      !e.name.includes('/devtools/') && !e.name.includes('chrome-devtools')
    );
  }
  return _origPerfEntries(type);
};

// 13. Deep toString override for patched objects
[
  [window.navigator, ['plugins', 'mimeTypes', 'languages', 'permissions', 'connection', 'hardwareConcurrency', 'deviceMemory']],
  [window, ['chrome', 'screen']],
].forEach(([obj, keys]) => {
  keys.forEach(key => {
    try {
      const desc = Object.getOwnPropertyDescriptor(obj, key);
      if (desc && desc.get) {
        Object.defineProperty(desc.get, 'toString', {
          value: () => `function ${key}() { [native code] }`,
          configurable: true,
        });
      }
    } catch(e) {}
  });
});

// 14. Constructor toString override
[window.navigator, window].forEach(obj => {
  Object.getOwnPropertyNames(obj.constructor).forEach(key => {
    try {
      Object.defineProperty(obj.constructor[key], 'toString', {
        value: () => `function ${key}() { [native code] }`,
        configurable: true,
      });
    } catch(e) {}
  });
});

// 15. navigator.serviceWorker — suppress errors gracefully
if (navigator.serviceWorker) {
  const _swReg = navigator.serviceWorker.register.bind(navigator.serviceWorker);
  navigator.serviceWorker.register = function(...args) {
    return _swReg(...args).catch(() => {});
  };
}

// 16. IntersectionObserver timing — headless fires immediately, real browsers defer
try {
  const _origIO = window.IntersectionObserver;
  window.IntersectionObserver = function(callback, options) {
    const observer = new _origIO(callback, options);
    const _origObserve = observer.observe.bind(observer);
    observer.observe = (el) => setTimeout(() => _origObserve(el), 50);
    return observer;
  };
  window.IntersectionObserver.prototype = _origIO.prototype;
} catch(e) {}
"""

_CAPTCHA_SIGNALS = [
    "captcha",
    "verify you are human",
    "i am not a robot",
    "hcaptcha",
    "recaptcha",
    "cf-challenge",
    "cloudflare",
    "bot verification",
    "security check",
    # PerimeterX/HUMAN's specific press-and-hold widget copy -- confirmed live
    # against a real PerimeterX-protected site (zillow.com in headless mode),
    # where the block page's exact text is "Press & Hold to confirm you are
    # a human (and not a bot)". Neither this nor the generic patterns below
    # matched it, so this response was silently returned as if it were real
    # page content instead of being flagged.
    "press & hold",
    "press and hold",
    "confirm you are a human",
]


_ANTIBOT_RESPONSE_PATTERNS = [
    "challenge",
    "blocked",
    "forbidden",
    "access denied",
    "rate limit",
    "too many requests",
    "please try again later",
    "suspicious activity",
    "cf-turnstile",
    "cloudflare turnstile",
    "checking your browser",
    "under attack mode",
    "ddos protection",
    "waf block",
    "perimeter x",
    "datadome",
    "imperva",
    "shape security",
    "you need to enable javascript",
    "blocked by firewall",
    # PerimeterX's actual block-page title phrasing doesn't contain the
    # contiguous substring "access denied" ("Access to this page has been
    # denied") -- confirmed live against zillow.com, where this exact page
    # went undetected by every existing pattern.
    "page has been denied",
]

# Anti-bot challenge widgets are commonly served from a cross-origin iframe
# (DataDome's slider, hCaptcha, Cloudflare Turnstile, Arkose/FunCaptcha,
# PerimeterX). Cross-origin iframes are a separate document tree, so
# page.title() / page.inner_text("body") can't see into them at all -- the
# outer page can be fully blocked by a loaded DataDome widget while its own
# visible text says nothing about it. Frame URLs are visible regardless of
# origin, so checking them catches challenges the text-based checks miss.
_ANTIBOT_IFRAME_DOMAINS = [
    "captcha-delivery.com",  # DataDome (covers geo.captcha-delivery.com too)
    "hcaptcha.com",
    "recaptcha.net",
    "google.com/recaptcha",
    "challenges.cloudflare.com",  # Cloudflare Turnstile
    "arkoselabs.com",  # Arkose Labs / FunCaptcha
    "perimeterx.net",
    "px-cdn.net",
]


def _browser_detect_challenge_iframe(page) -> str:
    """Return the matched domain if a known anti-bot challenge iframe is present."""
    try:
        for frame in page.frames:
            furl = frame.url or ""
            for dom in _ANTIBOT_IFRAME_DOMAINS:
                if dom in furl:
                    return dom
    except Exception:
        pass
    return ""


def _browser_detect_antibot(page) -> str:
    """Check if the page is showing an anti-bot block page. Returns hint string or empty."""
    iframe_hit = _browser_detect_challenge_iframe(page)
    if iframe_hit:
        return iframe_hit
    try:
        title = page.title().lower()
        body = ""
        with contextlib.suppress(Exception):
            body = page.inner_text("body", timeout=3000).lower()
        content = title + " " + body
        for pat in _ANTIBOT_RESPONSE_PATTERNS:
            if pat in content:
                return pat
    except Exception:
        pass
    return ""


def _browser_wait_out_antibot(page, max_wait_s: float = 8.0) -> str:
    """If an anti-bot signal is showing, poll for it to clear on its own.

    Simple Cloudflare-style "Checking your browser..." JS challenges
    typically auto-resolve in a few seconds once real JS actually executes,
    which this stealth session supports -- giving up after a token ~0.5s
    delay (the old behavior) reports "blocked" on plenty of sites that would
    have passed with a bit more patience. Returns the still-present signal
    string, or "" if it cleared (or was never there).
    """
    pat = _browser_detect_antibot(page)
    if not pat:
        return ""
    deadline = time.time() + max_wait_s
    while time.time() < deadline:
        with contextlib.suppress(Exception):
            page.wait_for_load_state("networkidle", timeout=1500)
        time.sleep(0.5)
        pat = _browser_detect_antibot(page)
        if not pat:
            return ""
    return pat


def _browser_settle_after_nav(page, mouse_state: dict | None = None) -> None:
    """Perform a small ambient scroll (+ mouse nudge, if tracking state is given)
    right after a page loads.

    A session that goes straight from "loaded" to "acting on a specific
    selector" with zero ambient activity is itself a signal -- and some
    behavioral anti-bot checks gate real content behind seeing at least one
    scroll/mousemove event first. mouse_state is _browser_state for the headed
    browser_action page (so later clicks continue the tracked path from here);
    fetch_rendered's headless page has nothing to click afterward, so it's
    passed None and only gets the scroll.
    """
    with contextlib.suppress(Exception):
        page.evaluate("(y) => window.scrollBy(0, y)", _r.randint(80, 260))
    if mouse_state is None:
        return
    with contextlib.suppress(Exception):
        tx, ty = _r.uniform(200, 900), _r.uniform(150, 500)
        page.mouse.move(tx, ty, steps=_r.randint(3, 6))
        mouse_state["mouse_pos"] = (tx, ty)


def _browser_random_viewport() -> dict:
    """Return a randomized viewport to avoid fingerprint matching."""
    widths = [1280, 1366, 1440, 1536, 1600, 1920]
    heights = [720, 768, 800, 900, 1024, 1080, 1200]
    return {"width": _r.choice(widths), "height": _r.choice(heights)}


def _browser_random_ua() -> str:
    """Return a randomized but realistic user-agent string."""
    major = _r.choice(["131", "130", "129", "128", "127"])
    minor = _r.randint(0, 999)
    return (
        f"Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        f"AppleWebKit/537.36 (KHTML, like Gecko) "
        f"Chrome/{major}.{minor}.0.0 Safari/537.36"
    )


def _browser_random_delay(min_ms: float = 100, max_ms: float = 600) -> None:
    """Introduce a small random delay to simulate human timing."""
    time.sleep(_r.uniform(min_ms / 1000, max_ms / 1000))


def _browser_human_mouse_move(page, target_x: float, target_y: float) -> None:
    """Walk the mouse toward (target_x, target_y) via a short, slightly wobbly path.

    page.mouse.move(x, y, ...) takes *absolute* viewport coordinates, not
    relative deltas. The previous implementation called it with small
    (-30..30, -10..10) values on every fill/click, which meant the cursor
    teleported to near the top-left corner before every single interaction --
    a far stronger bot tell to behavioral anti-bot checks (DataDome, etc.)
    than not moving the mouse at all. This tracks the last known position in
    _browser_state and interpolates toward the real target instead.
    """
    sx, sy = _browser_state.get("mouse_pos", (_r.uniform(200, 800), _r.uniform(150, 500)))
    steps = _r.randint(3, 6)
    for i in range(1, steps + 1):
        t = i / steps
        wobble = (1 - t) * _r.uniform(-15, 15)
        ix = sx + (target_x - sx) * t + wobble
        iy = sy + (target_y - sy) * t + wobble
        page.mouse.move(ix, iy, steps=_r.randint(2, 5))
    page.mouse.move(target_x, target_y, steps=_r.randint(2, 4))
    _browser_state["mouse_pos"] = (target_x, target_y)


def _browser_move_toward_element(page, loc) -> None:
    """Move the mouse to a random point inside loc's bounding box before acting on it.

    Playwright's own click()/hover() warp the cursor to the target instantly;
    walking there first leaves a mouse trail that looks like a real user's,
    which matters to behavioral checks that track movement, not just clicks.
    """
    try:
        box = loc.bounding_box()
    except Exception:
        box = None
    if not box:
        return
    tx = box["x"] + box["width"] * _r.uniform(0.3, 0.7)
    ty = box["y"] + box["height"] * _r.uniform(0.3, 0.7)
    _browser_human_mouse_move(page, tx, ty)


def _browser_has_captcha(page) -> bool:
    """Internal helper: browser has captcha.

    Checks known challenge-iframe domains first (see _browser_detect_challenge_iframe)
    since a DataDome/hCaptcha/Turnstile widget can be fully loaded in a
    cross-origin iframe with the outer page's visible text still saying nothing.
    """
    if _browser_detect_challenge_iframe(page):
        return True
    try:
        content = (page.title() + " " + page.inner_text("body", timeout=3000)).lower()
        return any(kw in content for kw in _CAPTCHA_SIGNALS)
    except Exception:
        return False


def _browser_save_cookies(page) -> None:
    """Save cookies + localStorage for the current context via storage_state()."""
    with contextlib.suppress(Exception):
        page.context.storage_state(path=str(COOKIE_FILE))


def _parse_proxy_config(proxy_url: str) -> dict | None:
    """Parse a 'http://user:pass@host:port' style URL into the dict shape
    Playwright's launch(proxy=...) expects. Returns None if empty/unparseable.

    A bare 'host:port' (no scheme) is treated as http:// -- urlsplit()
    without "://" present would otherwise misparse the host itself as the
    scheme (urlsplit("host:8080").hostname is None), silently dropping a
    proxy a user typed without the scheme prefix.
    """
    proxy_url = (proxy_url or "").strip()
    if not proxy_url:
        return None
    if "://" not in proxy_url:
        proxy_url = f"http://{proxy_url}"
    parsed = urllib.parse.urlsplit(proxy_url)
    if not parsed.hostname:
        return None
    server = f"{parsed.scheme}://{parsed.hostname}"
    if parsed.port:
        server += f":{parsed.port}"
    proxy: dict = {"server": server}
    if parsed.username:
        proxy["username"] = urllib.parse.unquote(parsed.username)
    if parsed.password:
        proxy["password"] = urllib.parse.unquote(parsed.password)
    return proxy


def _resolve_sync_playwright():
    """Return a sync_playwright() factory, preferring patchright over playwright.

    patchright (https://github.com/Kaliiiiiiiiii-Vinyzu/patchright-python) is a
    patched Playwright build that closes CDP-protocol-level detection vectors
    (e.g. the Runtime.enable leak) which no amount of JS-level stealth (the
    _STEALTH_JS init script, real Chrome channel, etc.) can hide, because
    they're visible below the page's own JS. It's API-compatible with
    playwright.sync_api, so this is a drop-in swap when installed. Not a hard
    dependency -- falls back to plain playwright if patchright isn't present.
    """
    try:
        from patchright.sync_api import sync_playwright

        return sync_playwright
    except ImportError:
        pass
    try:
        from playwright.sync_api import sync_playwright

        return sync_playwright
    except ImportError as e:
        msg = (
            "no browser automation backend installed — run: pip install playwright && "
            "playwright install chrome (or, for stronger anti-bot resistance: "
            "pip install patchright && patchright install chrome)"
        )
        raise RuntimeError(msg) from e


def _migrate_legacy_cookie_file(ctx) -> None:
    """One-time import of pre-persistent-profile cookies into a fresh profile.

    Before persistent profiles, cookies/localStorage lived only in COOKIE_FILE
    (a Playwright storage_state JSON) because every session started from a
    throwaway context. A brand-new profile directory has none of that, so
    without this a user upgrading to this version would silently lose whatever
    session they'd already saved. Cookies only (not localStorage) -- there's no
    context-level API to inject localStorage across arbitrary origins.
    """
    if not COOKIE_FILE.exists():
        return
    try:
        data = json.loads(COOKIE_FILE.read_text())
        cookies = data.get("cookies") if isinstance(data, dict) else None
        if cookies:
            ctx.add_cookies(cookies)
    except Exception:
        pass


def _launch_persistent_chromium(
    pw,
    *,
    user_data_dir: str,
    headless: bool,
    proxy_config: dict | None,
    args: list[str],
    base_context_kwargs: dict,
    fallback_extra_kwargs: dict,
) -> tuple[Any, bool]:
    """Launch a persistent Chromium context, preferring the real installed Chrome channel.

    launch_persistent_context() merges browser launch and context creation into
    one call (unlike launch() + new_context()), so which kwargs apply isn't
    known until we find out whether the "chrome" channel actually succeeded --
    hence base_context_kwargs (always applied) vs. fallback_extra_kwargs
    (user_agent/sec-ch-ua overrides, only applied when real Chrome isn't
    available and bundled Chromium's own UA needs masking). Spoofing UA/Client
    Hints on top of a genuine Chrome install would mismatch navigator.userAgentData
    against the real installed version -- a worse tell than leaving it alone.

    Returns (context, used_real_chrome).
    """
    try:
        ctx = pw.chromium.launch_persistent_context(
            user_data_dir,
            headless=headless,
            channel="chrome",
            proxy=proxy_config,
            args=args,
            ignore_default_args=["--enable-automation"],
            **base_context_kwargs,
        )
        return ctx, True
    except Exception:
        ctx = pw.chromium.launch_persistent_context(
            user_data_dir,
            headless=headless,
            proxy=proxy_config,
            args=args,
            ignore_default_args=["--enable-automation"],
            **{**base_context_kwargs, **fallback_extra_kwargs},
        )
        return ctx, False


def _get_page() -> Any:
    """Internal helper: get page."""
    if "page" not in _browser_state or _browser_state.get("closed"):
        _clear_loitering_event_loop()
        sync_playwright = _resolve_sync_playwright()
        pw = sync_playwright().start()
        _ua = _browser_random_ua()
        _major = _ua.split("Chrome/")[1].split(".")[0] if "Chrome/" in _ua else "131"
        _proxy_url = _CFG.get("browser_proxy", "")
        _proxy_config = _parse_proxy_config(_proxy_url)
        if _proxy_url and not _proxy_config:
            console.print(f"[yellow]  [browser] browser_proxy={_proxy_url!r} could not be parsed — ignoring[/yellow]")
        elif _proxy_config:
            console.print(f"[dim]  [browser] using proxy: {_proxy_config['server']}[/dim]")
        BROWSER_PROFILE_DIR.mkdir(exist_ok=True)
        _is_fresh_profile = not any(BROWSER_PROFILE_DIR.iterdir())
        _base_kwargs: dict[str, Any] = {
            "viewport": _browser_random_viewport(),
            "locale": "en-US",
            "timezone_id": "America/New_York",
            "geolocation": {"latitude": 40.7128, "longitude": -74.0060},
            "permissions": ["geolocation"],
            "color_scheme": "light",
        }
        _fallback_extra = {
            "user_agent": _ua,
            "extra_http_headers": {
                "sec-ch-ua": f'"Chromium";v="{_major}", "Google Chrome";v="{_major}"',
                "sec-ch-ua-mobile": "?0",
                "sec-ch-ua-platform": '"Windows"',
                "sec-fetch-dest": "document",
                "sec-fetch-mode": "navigate",
                "sec-fetch-site": "none",
                "sec-fetch-user": "?1",
                "accept-language": "en-US,en;q=0.9",
            },
        }
        ctx, _is_real_chrome = _launch_persistent_chromium(
            pw,
            user_data_dir=str(BROWSER_PROFILE_DIR),
            headless=False,  # headed mode is harder to detect
            proxy_config=_proxy_config,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox",
                "--disable-features=IsolateOrigins,site-per-process",
                "--disable-site-isolation-trials",
                "--disable-infobars",
                "--disable-extensions",
                "--disable-background-timer-throttling",
                "--disable-backgrounding-occluded-windows",
                "--disable-renderer-backgrounding",
                "--disable-accelerated-2d-canvas",
                "--disable-dev-shm-usage",
                "--lang=en-US",
            ],
            base_context_kwargs=_base_kwargs,
            fallback_extra_kwargs=_fallback_extra,
        )
        ctx.add_init_script(_STEALTH_JS)
        if _is_fresh_profile:
            _migrate_legacy_cookie_file(ctx)
        page = ctx.pages[0] if ctx.pages else ctx.new_page()
        # Links with target="_blank", OAuth popups, and window.open() all spawn a
        # new Playwright Page in this context. Without this, _browser_state["page"]
        # would keep pointing at the now-background original tab, and every
        # subsequent action would silently act on stale content.
        ctx.on("page", lambda new_page: _browser_state.update(page=new_page))
        _browser_state.update(playwright=pw, context=ctx, page=page, closed=False)
    return _browser_state["page"]


def _browser_resolve_selector(page, selector: str) -> Any:
    """Internal helper: browser resolve selector."""
    if selector.startswith("label:"):
        return page.get_by_label(selector[6:])
    if selector.startswith("button:"):
        return page.get_by_role("button", name=selector[7:])
    if selector.startswith("link:"):
        return page.get_by_role("link", name=selector[5:])
    if selector.startswith("text:"):
        return page.get_by_text(selector[5:])
    return page.locator(selector)


def _browser_smart_wait(page, timeout: int = 8000) -> None:
    """Wait for the page to settle — networkidle with domcontentloaded fallback."""
    try:
        page.wait_for_load_state("networkidle", timeout=timeout)
    except Exception:
        with contextlib.suppress(Exception):
            page.wait_for_load_state("domcontentloaded", timeout=timeout)


def _browser_check_captcha_pause(page) -> str:
    """If a CAPTCHA is detected, pause and let the user solve it. Returns status string.

    Only for the headed (visible) browser_action session — the user needs an
    actual window to solve a CAPTCHA in. For headless contexts, use
    _browser_captcha_hint instead. Covers iframe-embedded widgets (DataDome
    slider, hCaptcha, Turnstile) via _browser_has_captcha's iframe check, not
    just text-visible CAPTCHAs on the outer page.
    """
    if not _browser_has_captcha(page):
        return ""
    console.print("[bold yellow]  [browser] CAPTCHA / human-verification detected.[/bold yellow]")
    console.print("[bold yellow]  Solve it in the browser window, then press Enter to continue.[/bold yellow]")
    with contextlib.suppress(KeyboardInterrupt, EOFError):
        console.input("")
    _browser_save_cookies(page)
    return "[captcha-paused: user resolved]"


def _browser_captcha_hint(page) -> str:
    """Non-blocking CAPTCHA check for headless pages (fetch_rendered).

    There's no visible window here for a human to solve anything in, unlike
    browser_action's headed session — blocking on console.input() would just
    hang waiting for a CAPTCHA nobody can see. Flag it and point at
    browser_action instead.
    """
    if not _browser_has_captcha(page):
        return ""
    return (
        "[CAPTCHA/human-verification detected — this is a headless page with no visible "
        "window, so it can't be solved here. Use browser_action (navigate) instead, which "
        "opens a real visible browser window.]"
    )


def _browser_do_close(url="", selector="", value="", screenshot_path=""):
    # launch_persistent_context() has no separate Browser object -- closing
    # the context closes the underlying browser process too. Session state
    # itself now lives in the on-disk profile (BROWSER_PROFILE_DIR); the
    # storage_state export below is just a portable backup/inspection copy.
    if "context" in _browser_state:
        _browser_save_cookies(_browser_state["page"])
        _browser_state["context"].close()
    if "playwright" in _browser_state:
        _browser_state["playwright"].stop()
    _browser_state.clear()
    _browser_state["closed"] = True
    return "[browser closed — cookies saved]"


def _browser_do_navigate(page, url, selector="", value="", screenshot_path=""):
    if not url:
        return "[navigate requires a url]"
    console.print(f"[bold cyan]  [browser][/bold cyan] navigate → {url}")
    last_nav_err = None
    for _attempt in range(3):
        if _attempt > 0:
            import time as _time

            _time.sleep(1.5 * _attempt)
            console.print(f"[dim]  [browser] retry {_attempt}/2...[/dim]")
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=30000)
            last_nav_err = None
            break
        except Exception as _nav_err:
            last_nav_err = _nav_err
            _e = str(_nav_err)
            if "net::" not in _e and "ERR_" not in _e and "timeout" not in _e.lower():
                break
    if last_nav_err:
        return f"[browser navigation failed: {last_nav_err}]"
    _browser_smart_wait(page, timeout=12000)
    antibot_pat = _browser_wait_out_antibot(page)
    if antibot_pat:
        console.print(f"[bold yellow]  [browser] anti-bot signal detected: {antibot_pat}[/bold yellow]")
    _browser_settle_after_nav(page, _browser_state)
    captcha_note = _browser_check_captcha_pause(page)
    _browser_save_cookies(page)
    title = page.title()
    result = f"[navigated to: {url}]\nPage title: {title}"
    if antibot_pat:
        result += f"\n[anti-bot signal: {antibot_pat} — proceeding with caution]"
    if captcha_note:
        result += f"\n{captcha_note}"
    return result


def _browser_do_fill(page, url="", selector="", value="", screenshot_path=""):
    if not selector or value is None:
        return "[fill requires selector and value]"
    console.print(f"[bold cyan]  [browser][/bold cyan] fill {selector!r} = {value!r}")
    loc = _browser_resolve_selector(page, selector)
    loc.first.wait_for(state="visible", timeout=8000)
    _browser_random_delay(200, 500)
    _browser_move_toward_element(page, loc.first)
    loc.first.fill(value)
    return f"[filled {selector!r} with {value!r}]"


def _browser_do_type(page, url="", selector="", value="", screenshot_path=""):
    if not selector or value is None:
        return "[type requires selector and value]"
    console.print(f"[bold cyan]  [browser][/bold cyan] type {selector!r} = {value!r}")
    loc = _browser_resolve_selector(page, selector)
    loc.first.wait_for(state="visible", timeout=8000)
    _browser_move_toward_element(page, loc.first)
    loc.first.click()
    # A constant per-keystroke delay is itself a rhythm no real typist has --
    # type one character at a time with a randomized delay instead of a single
    # fixed-delay call across the whole string.
    for ch in value:
        loc.first.type(ch, delay=_r.uniform(40, 180))
    return f"[typed into {selector!r}]"


def _browser_do_click(page, url="", selector="", value="", screenshot_path=""):
    if not selector:
        return "[click requires a selector]"
    console.print(f"[bold cyan]  [browser][/bold cyan] click {selector!r}")
    loc = _browser_resolve_selector(page, selector)
    loc.first.wait_for(state="visible", timeout=8000)
    _browser_random_delay(150, 400)
    _browser_move_toward_element(page, loc.first)
    loc.first.click()
    _browser_smart_wait(page)
    captcha_note = _browser_check_captcha_pause(page)
    _browser_save_cookies(page)
    result = f"[clicked {selector!r} — now at: {page.url}]"
    if captcha_note:
        result += f"\n{captcha_note}"
    return result


def _browser_do_select(page, url="", selector="", value="", screenshot_path=""):
    if not selector or not value:
        return "[select requires selector and value]"
    console.print(f"[bold cyan]  [browser][/bold cyan] select {selector!r} = {value!r}")
    loc = _browser_resolve_selector(page, selector)
    loc.first.wait_for(state="visible", timeout=8000)
    loc.first.select_option(label=value)
    return f"[selected {value!r} in {selector!r}]"


def _browser_do_submit(page, url="", selector="", value="", screenshot_path=""):
    target = selector or "form"
    console.print(f"[bold cyan]  [browser][/bold cyan] submit {target!r}")
    if selector:
        _browser_resolve_selector(page, selector).first.press("Enter")
    else:
        page.locator("form").first.evaluate("f => f.submit()")
    _browser_smart_wait(page, timeout=15000)
    captcha_note = _browser_check_captcha_pause(page)
    _browser_save_cookies(page)
    result = f"[form submitted — now at: {page.url}]"
    if captcha_note:
        result += f"\n{captcha_note}"
    return result


def _browser_do_wait_for(page, url="", selector="", value="", screenshot_path=""):
    if not selector:
        return "[wait_for requires a selector]"
    console.print(f"[bold cyan]  [browser][/bold cyan] wait_for {selector!r}")
    _browser_resolve_selector(page, selector).first.wait_for(state="visible", timeout=15000)
    return f"[element visible: {selector!r}]"


def _browser_do_scroll(page, url="", selector="", value="", screenshot_path=""):
    if selector:
        console.print(f"[bold cyan]  [browser][/bold cyan] scroll to {selector!r}")
        _browser_resolve_selector(page, selector).first.scroll_into_view_if_needed()
        return f"[scrolled to {selector!r}]"
    pixels = int(value) if value and value.lstrip("-").isdigit() else 0
    if pixels:
        console.print(f"[bold cyan]  [browser][/bold cyan] scroll {pixels}px")
        page.evaluate("window.scrollBy(0, {})", pixels)
        return f"[scrolled by {pixels}px]"
    console.print("[bold cyan]  [browser][/bold cyan] scroll down")
    page.evaluate("window.scrollBy(0, window.innerHeight)")
    return "[scrolled down one page]"


def _browser_do_hover(page, url="", selector="", value="", screenshot_path=""):
    if not selector:
        return "[hover requires a selector]"
    console.print(f"[bold cyan]  [browser][/bold cyan] hover {selector!r}")
    loc = _browser_resolve_selector(page, selector)
    loc.first.wait_for(state="visible", timeout=8000)
    _browser_random_delay()
    _browser_move_toward_element(page, loc.first)
    loc.first.hover()
    return f"[hovered over {selector!r}]"


def _browser_do_press_key(page, url="", selector="", value="", screenshot_path=""):
    if not value:
        return "[press_key requires a value (e.g. 'Enter', 'Tab', 'Escape', 'Control+A')]"
    console.print(f"[bold cyan]  [browser][/bold cyan] press_key {value!r}")
    if selector:
        _browser_resolve_selector(page, selector).first.press(value)
    else:
        page.keyboard.press(value)
    return f"[pressed {value!r}]"


def _browser_do_get_url(page, url="", selector="", value="", screenshot_path=""):
    return f"[current URL: {page.url}]"


def _browser_do_get_links(page, url="", selector="", value="", screenshot_path=""):
    console.print("[bold cyan]  [browser][/bold cyan] get_links")
    links = page.evaluate(
        "() => Array.from(document.querySelectorAll('a[href]'))"
        ".map(a => ({text: a.innerText.trim().slice(0,100), href: a.href}))"
        ".filter(l => l.href.startsWith('http')).slice(0, 50)",
    )
    if not links:
        return "[no links found on page]"
    lines = [f"[links on {page.url}]"]
    for lnk in links:
        lines.append(f"  {lnk['text'][:40]!r:42} → {lnk['href']}")
    return "\n".join(lines)


def _browser_do_screenshot(page, url="", selector="", value="", screenshot_path=""):
    path = screenshot_path or str(Path.home() / "screenshot.png")
    console.print(f"[bold cyan]  [browser][/bold cyan] screenshot → {path}")
    page.screenshot(path=path, full_page=True)
    return f"[screenshot saved: {path}]\nURL: {page.url}\nTitle: {page.title()}"


def _browser_do_go_back(page, url="", selector="", value="", screenshot_path=""):
    console.print("[bold cyan]  [browser][/bold cyan] go_back")
    resp = page.go_back(wait_until="domcontentloaded", timeout=15000)
    if resp is None:
        return "[go_back: no previous page in history]"
    _browser_smart_wait(page, timeout=8000)
    return f"[navigated back — now at: {page.url}]\nPage title: {page.title()}"


def _browser_do_go_forward(page, url="", selector="", value="", screenshot_path=""):
    console.print("[bold cyan]  [browser][/bold cyan] go_forward")
    resp = page.go_forward(wait_until="domcontentloaded", timeout=15000)
    if resp is None:
        return "[go_forward: no next page in history]"
    _browser_smart_wait(page, timeout=8000)
    return f"[navigated forward — now at: {page.url}]\nPage title: {page.title()}"


def _browser_do_evaluate(page, url="", selector="", value="", screenshot_path=""):
    if not value:
        return "[evaluate requires a JS expression in 'value', e.g. 'document.title' or '[...document.querySelectorAll(\"h2\")].map(e => e.innerText)']"
    console.print(f"[bold cyan]  [browser][/bold cyan] evaluate: {value[:80]!r}")
    try:
        result = page.evaluate(value)
    except Exception as e:
        return f"[evaluate error: {e}]"
    try:
        text = json.dumps(result, indent=2, ensure_ascii=False, default=str)
    except Exception:
        text = str(result)
    if len(text) > 8000:
        text = text[:8000] + "\n...[truncated]"
    return f"[evaluate result]\n{text}"


def _browser_do_upload_file(page, url="", selector="", value="", screenshot_path=""):
    if not selector or not value:
        return "[upload_file requires selector (file input) and value (file path)]"
    file_path = Path(value).expanduser()
    if not file_path.is_file():
        return f"[upload_file error: file not found: {value}]"
    console.print(f"[bold cyan]  [browser][/bold cyan] upload_file {selector!r} = {value!r}")
    loc = _browser_resolve_selector(page, selector)
    loc.first.set_input_files(str(file_path))
    return f"[uploaded {value!r} to {selector!r}]"


def _browser_do_get_text(page, url="", selector="", value="", screenshot_path=""):
    console.print("[bold cyan]  [browser][/bold cyan] get_text")
    if selector:
        text = _browser_resolve_selector(page, selector).first.inner_text()
    else:
        text = page.inner_text("body")
    if len(text) > 16000:
        text = text[:16000] + "\n...[truncated]"
    return f"[page text — {page.url}]\n{text}"


_BROWSER_HANDLERS: dict[str, Callable] = {
    "close": lambda _page, url="", selector="", value="", screenshot_path="": _browser_do_close(
        url, selector, value, screenshot_path
    ),  # noqa: ARG005
    "navigate": lambda page, url="", selector="", value="", screenshot_path="": _browser_do_navigate(
        page, url, selector, value, screenshot_path
    ),
    "fill": _browser_do_fill,
    "type": _browser_do_type,
    "click": _browser_do_click,
    "select": _browser_do_select,
    "submit": _browser_do_submit,
    "wait_for": _browser_do_wait_for,
    "scroll": _browser_do_scroll,
    "hover": _browser_do_hover,
    "press_key": _browser_do_press_key,
    "get_url": _browser_do_get_url,
    "get_links": _browser_do_get_links,
    "screenshot": _browser_do_screenshot,
    "get_text": _browser_do_get_text,
    "go_back": _browser_do_go_back,
    "go_forward": _browser_do_go_forward,
    "evaluate": _browser_do_evaluate,
    "upload_file": _browser_do_upload_file,
}


# Actions that plausibly trigger a real, trusted-gesture navigation (a target=
# "_blank" link, an OAuth "Sign in with..." button) get an active, bounded wait
# for a popup. Playwright's sync API only delivers context.on("page", ...)
# events on the next blocking Playwright call — a plain click() returning does
# NOT mean a same-tick popup would already be visible to us, so without this
# wait the switch would only be noticed one tool call later (or never, if the
# model doesn't happen to make another call). expect_page() actively pumps
# for the event instead of relying on it turning up incidentally.
_POPUP_PRONE_ACTIONS = frozenset({"click", "submit", "press_key"})
_POPUP_WAIT_MS = 900


def do_browser_action(
    action: str, url: str = "", selector: str = "", value: str = "", screenshot_path: str = ""
) -> str:
    """Control a real Chromium browser via playwright automation.

    Actions: navigate, fill, type, click, select, submit, wait_for, scroll,
    hover, press_key, screenshot, get_text, get_url, get_links, go_back,
    go_forward, evaluate, upload_file, close.
    Returns output text or status messages. If click/submit/press_key opens a
    new tab (target="_blank" link, OAuth popup), that tab becomes the active
    page and the result notes the switch. Other actions pick up the switch on
    the next call if one happens to trigger a popup too.
    """
    try:
        page = _get_page()
        handler = _BROWSER_HANDLERS.get(action)
        if handler is None:
            return f"[unknown browser action: {action}]"

        if action in _POPUP_PRONE_ACTIONS:
            # handler() runs first, outside any popup-wait scope, so a genuine
            # error it raises (e.g. selector not found) propagates as-is —
            # Playwright's "no popup arrived" and "your click/wait timed out"
            # cases both raise the exact same TimeoutError class, so they must
            # not share a try/except or a real failure could be silently
            # misreported as "no popup, click succeeded".
            result = handler(page, url, selector, value, screenshot_path)
            from playwright.sync_api import TimeoutError as _PwTimeoutError

            try:
                popup_page = page.context.wait_for_event("page", timeout=_POPUP_WAIT_MS)
            except _PwTimeoutError:
                popup_page = None
            if popup_page is not None:
                _browser_state["page"] = popup_page
                with contextlib.suppress(Exception):
                    popup_page.wait_for_load_state("domcontentloaded", timeout=8000)
        else:
            result = handler(page, url, selector, value, screenshot_path)

        new_page = _browser_state.get("page")
        if new_page is not None and new_page is not page and action != "close":
            with contextlib.suppress(Exception):
                result += f"\n[note: a new tab/popup opened and is now the active page — url: {new_page.url}]"
        return result

    except Exception as e:
        _emsg = str(e)
        if "net::" in _emsg or "ERR_" in _emsg:
            return (
                f"[browser network error: {e}]\n"
                "Tip: Check the URL or your connection. Try a different URL or use fetch_url instead."
            )
        if "timeout" in _emsg.lower():
            return (
                f"[browser timeout: {e}]\n"
                "Tip: The page may be JS-heavy or slow — try increasing wait time or use get_text after waiting."
            )
        if "closed" in _emsg.lower() or "Target page" in _emsg:
            _browser_state.clear()
            _browser_state["closed"] = True
            return f"[browser closed unexpectedly: {e}]\nTip: Use navigate action to reopen the browser."
        if "captcha" in _emsg.lower() or "challenge" in _emsg.lower():
            return (
                f"[browser blocked (anti-bot): {e}]\n"
                "Tip: The site may require manual CAPTCHA solving — navigate there first."
            )
        return f"[browser error: {e}]"


def _get_render_page() -> Any:
    """Get (or create) the dedicated headless Playwright page used by fetch_rendered."""
    if "page" not in _render_state or _render_state.get("closed"):
        _clear_loitering_event_loop()
        sync_playwright = _resolve_sync_playwright()
        pw = sync_playwright().start()
        _ua = _browser_random_ua()
        _major = _ua.split("Chrome/")[1].split(".")[0] if "Chrome/" in _ua else "131"
        RENDER_PROFILE_DIR.mkdir(exist_ok=True)
        _base_kwargs: dict[str, Any] = {
            "viewport": _browser_random_viewport(),
            "locale": "en-US",
            "timezone_id": "America/New_York",
            "color_scheme": "light",
        }
        _fallback_extra = {
            "user_agent": _ua,
            "extra_http_headers": {
                "sec-ch-ua": f'"Chromium";v="{_major}", "Google Chrome";v="{_major}"',
                "sec-ch-ua-mobile": "?0",
                "sec-ch-ua-platform": '"Windows"',
                "sec-fetch-dest": "document",
                "sec-fetch-mode": "navigate",
                "sec-fetch-site": "none",
                "sec-fetch-user": "?1",
                "accept-language": "en-US,en;q=0.9",
            },
        }
        ctx, _is_real_chrome = _launch_persistent_chromium(
            pw,
            user_data_dir=str(RENDER_PROFILE_DIR),
            headless=True,
            proxy_config=_parse_proxy_config(_CFG.get("browser_proxy", "")),
            args=[
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox",
                "--disable-features=IsolateOrigins,site-per-process",
                "--disable-site-isolation-trials",
                "--disable-background-timer-throttling",
                "--disable-backgrounding-occluded-windows",
                "--disable-renderer-backgrounding",
                "--disable-accelerated-2d-canvas",
                "--disable-dev-shm-usage",
                "--lang=en-US",
            ],
            base_context_kwargs=_base_kwargs,
            fallback_extra_kwargs=_fallback_extra,
        )
        ctx.add_init_script(_STEALTH_JS)
        page = ctx.pages[0] if ctx.pages else ctx.new_page()
        ctx.on("page", lambda new_page: _render_state.update(page=new_page))
        _render_state.update(playwright=pw, context=ctx, page=page, closed=False)
    return _render_state["page"]


def do_fetch_rendered(url: str, max_chars: int = 15000) -> str:
    """Fetch a URL with full JS rendering via a dedicated headless Playwright instance.
    Separate from browser_action so it never shares state or interferes.
    Uses trafilatura/readability on the rendered HTML for clean text extraction.
    """
    try:
        page = _get_render_page()
        console.print(f"[dim cyan]  fetch_rendered: {url}[/dim cyan]")

        # Navigate with retry for transient network errors
        last_err = None
        for _attempt in range(3):
            if _attempt:
                import time as _t

                _t.sleep(1.5 * _attempt)
                console.print(f"[dim]  [fetch_rendered] retry {_attempt}/2...[/dim]")
            try:
                page.goto(url, wait_until="domcontentloaded", timeout=30000)
                last_err = None
                break
            except Exception as _nav_err:
                last_err = _nav_err
                _e = str(_nav_err)
                if "net::" not in _e and "ERR_" not in _e and "timeout" not in _e.lower():
                    break
        if last_err:
            return f"[fetch_rendered navigation failed: {last_err}]"

        # Wait for JS content to settle
        _browser_smart_wait(page, timeout=10000)

        # Simple Cloudflare-style JS challenges typically auto-resolve in a
        # few seconds once real JS executes -- wait it out rather than
        # reporting the challenge page as if it were the real content.
        antibot_pat = _browser_wait_out_antibot(page)
        _browser_settle_after_nav(page)

        # Check for CAPTCHA (headless triggers it less, but still possible).
        # Non-blocking: this page has no visible window, so there's no one who
        # could solve it here even if we paused.
        captcha_note = _browser_captcha_hint(page)

        # Extract text: prefer trafilatura/readability on full rendered HTML
        try:
            html = page.content()
            text = _html_to_text(html, url=url)
        except Exception:
            text = page.inner_text("body")

        if len(text) > max_chars:
            # Smart truncate at sentence boundary
            window = text[max(0, max_chars - 500) : max_chars]
            last_break = max(window.rfind(". "), window.rfind("\n\n"))
            cut = (max(0, max_chars - 500) + last_break + 1) if last_break > 0 else max_chars
            text = text[:cut] + "\n...[truncated]"

        title = page.title()
        result = f"[Rendered: {url}]\nTitle: {title}\n\n{text}"
        if antibot_pat:
            result += f"\n[anti-bot signal: {antibot_pat} — this may be a challenge page, not real content]"
        if captcha_note:
            result += f"\n{captcha_note}"
        return result

    except Exception as e:
        _emsg = str(e)
        if "closed" in _emsg.lower() or "Target page" in _emsg:
            _render_state.clear()
            _render_state["closed"] = True
        return f"[fetch_rendered error: {e}]"
