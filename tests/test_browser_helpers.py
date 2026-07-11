"""Unit tests for the pure-Python helpers in qwen_cli.tools.browser.

These run without Playwright installed — fake page objects stand in for real ones.
"""

import re

import qwen_cli.tools.browser as br


class FakeMouse:
    def __init__(self):
        self.moves = []

    def move(self, x, y, steps=1):
        self.moves.append((x, y, steps))


class FakePage:
    def __init__(self, title="", body=""):
        self._title = title
        self._body = body
        self.mouse = FakeMouse()

    def title(self):
        return self._title

    def inner_text(self, selector, timeout=0):
        return self._body


class TestRandomUA:
    def test_format_is_realistic_chrome(self):
        ua = br._browser_random_ua()
        assert re.fullmatch(
            r"Mozilla/5\.0 \(Windows NT 10\.0; Win64; x64\) "
            r"AppleWebKit/537\.36 \(KHTML, like Gecko\) "
            r"Chrome/\d{3}\.\d{1,3}\.0\.0 Safari/537\.36",
            ua,
        ), ua

    def test_major_version_in_supported_range(self):
        for _ in range(20):
            major = int(br._browser_random_ua().split("Chrome/")[1].split(".")[0])
            assert 127 <= major <= 131


class TestRandomViewport:
    def test_returns_plausible_dimensions(self):
        for _ in range(20):
            vp = br._browser_random_viewport()
            assert vp["width"] in (1280, 1366, 1440, 1536, 1600, 1920)
            assert vp["height"] in (720, 768, 800, 900, 1024, 1080, 1200)


class TestJitterMouse:
    def test_moves_mouse_within_bounds(self):
        page = FakePage()
        br._browser_jitter_mouse(page)
        assert 2 <= len(page.mouse.moves) <= 5
        for dx, dy, steps in page.mouse.moves:
            assert -30 <= dx <= 30
            assert -10 <= dy <= 10
            assert 3 <= steps <= 8


class TestCaptchaHint:
    def test_clean_page_returns_empty(self):
        page = FakePage(title="Example Domain", body="This domain is for examples.")
        assert br._browser_captcha_hint(page) == ""

    def test_captcha_page_returns_non_blocking_hint(self):
        # Must not call console.input() -- this is the headless fetch_rendered
        # path, where there's no visible window for a human to solve anything in.
        page = FakePage(title="Verify you are human", body="Please complete the captcha below")
        hint = br._browser_captcha_hint(page)
        assert "headless" in hint
        assert "browser_action" in hint

    def test_page_error_returns_empty(self):
        class ExplodingPage:
            def title(self):
                raise RuntimeError("page gone")

        assert br._browser_captcha_hint(ExplodingPage()) == ""


class TestDetectAntibot:
    def test_clean_page_returns_empty(self):
        page = FakePage(title="Example Domain", body="This domain is for examples.")
        assert br._browser_detect_antibot(page) == ""

    def test_cloudflare_challenge_detected(self):
        page = FakePage(title="Just a moment...", body="Checking your browser before accessing")
        assert br._browser_detect_antibot(page) == "checking your browser"

    def test_rate_limit_detected_in_body(self):
        page = FakePage(title="Error", body="429 — too many requests, slow down")
        assert br._browser_detect_antibot(page) != ""

    def test_page_error_returns_empty(self):
        class ExplodingPage:
            def title(self):
                raise RuntimeError("page gone")

        assert br._browser_detect_antibot(ExplodingPage()) == ""


class TestResolveStorageState:
    def test_missing_file_returns_none(self, tmp_path, monkeypatch):
        monkeypatch.setattr(br, "COOKIE_FILE", tmp_path / "nope.json")
        assert br._resolve_storage_state() is None

    def test_new_format_dict_with_cookies_returns_path(self, tmp_path, monkeypatch):
        cookie_file = tmp_path / "cookies.json"
        cookie_file.write_text('{"cookies": [], "origins": []}')
        monkeypatch.setattr(br, "COOKIE_FILE", cookie_file)
        assert br._resolve_storage_state() == str(cookie_file)

    def test_legacy_bare_list_format_is_ignored(self, tmp_path, monkeypatch):
        # Old versions of this file stored context.cookies() directly -- a
        # bare list, not the {"cookies": ..., "origins": ...} shape
        # new_context(storage_state=...) requires.
        cookie_file = tmp_path / "cookies.json"
        cookie_file.write_text("[]")
        monkeypatch.setattr(br, "COOKIE_FILE", cookie_file)
        assert br._resolve_storage_state() is None

    def test_corrupt_json_returns_none(self, tmp_path, monkeypatch):
        cookie_file = tmp_path / "cookies.json"
        cookie_file.write_text("{not valid json")
        monkeypatch.setattr(br, "COOKIE_FILE", cookie_file)
        assert br._resolve_storage_state() is None
