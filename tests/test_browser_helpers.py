"""Unit tests for the pure-Python helpers in qwen_cli.tools.browser.

These run without Playwright installed — fake page objects stand in for real ones.
"""

import re
import time

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


class TestParseProxyConfig:
    def test_empty_returns_none(self):
        assert br._parse_proxy_config("") is None

    def test_plain_host_port(self):
        assert br._parse_proxy_config("http://proxy.example.com:8080") == {
            "server": "http://proxy.example.com:8080"
        }

    def test_with_credentials(self):
        result = br._parse_proxy_config("http://alice:s3cret@proxy.example.com:8080")
        assert result == {
            "server": "http://proxy.example.com:8080",
            "username": "alice",
            "password": "s3cret",
        }

    def test_url_encoded_credentials_are_decoded(self):
        result = br._parse_proxy_config("http://ali%40ce:p%40ss@proxy.example.com:8080")
        assert result["username"] == "ali@ce"
        assert result["password"] == "p@ss"

    def test_defaults_to_http_scheme_if_missing(self):
        result = br._parse_proxy_config("proxy.example.com:8080")
        assert result["server"] == "http://proxy.example.com:8080"

    def test_unparseable_returns_none(self):
        assert br._parse_proxy_config("   ") is None


class TestWaitOutAntibot:
    def test_clean_page_returns_immediately(self):
        page = FakePage(title="Example Domain", body="clean content")
        start = time.time()
        result = br._browser_wait_out_antibot(page, max_wait_s=5)
        assert result == ""
        assert time.time() - start < 1  # no polling loop entered at all

    def test_challenge_that_clears_returns_empty(self):
        class ClearingPage:
            def __init__(self):
                self.calls = 0

            def title(self):
                self.calls += 1
                return "Just a moment..." if self.calls < 2 else "Welcome"

            def inner_text(self, selector, timeout=0):
                return "checking your browser" if self.calls < 2 else "real content"

            def wait_for_load_state(self, state, timeout=None):
                pass

        result = br._browser_wait_out_antibot(ClearingPage(), max_wait_s=5)
        assert result == ""

    def test_persistent_challenge_returns_signal_after_timeout(self):
        page = FakePage(title="Just a moment...", body="checking your browser")
        start = time.time()
        result = br._browser_wait_out_antibot(page, max_wait_s=0.6)
        assert result == "checking your browser"
        assert time.time() - start >= 0.6


class TestLaunchChromium:
    def test_prefers_real_chrome_channel(self):
        calls = []

        class FakePW:
            class chromium:
                @staticmethod
                def launch(**kwargs):
                    calls.append(kwargs)
                    return "BROWSER"

        browser, is_real = br._launch_chromium(FakePW(), headless=False, proxy_config=None, args=[])
        assert browser == "BROWSER"
        assert is_real is True
        assert calls[0].get("channel") == "chrome"
        assert len(calls) == 1

    def test_falls_back_when_chrome_channel_unavailable(self):
        calls = []

        class FakePW:
            class chromium:
                @staticmethod
                def launch(**kwargs):
                    calls.append(kwargs)
                    if kwargs.get("channel") == "chrome":
                        raise RuntimeError("chrome not found")
                    return "BROWSER"

        browser, is_real = br._launch_chromium(FakePW(), headless=False, proxy_config=None, args=[])
        assert browser == "BROWSER"
        assert is_real is False
        assert len(calls) == 2

    def test_ignore_default_args_disables_automation_flag(self):
        calls = []

        class FakePW:
            class chromium:
                @staticmethod
                def launch(**kwargs):
                    calls.append(kwargs)
                    return "BROWSER"

        br._launch_chromium(FakePW(), headless=True, proxy_config=None, args=["--foo"])
        assert calls[0]["ignore_default_args"] == ["--enable-automation"]
        assert calls[0]["args"] == ["--foo"]
        assert calls[0]["headless"] is True
