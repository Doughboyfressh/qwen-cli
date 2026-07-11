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


class FakeFrame:
    def __init__(self, url):
        self.url = url


class FakeFramePage(FakePage):
    def __init__(self, title="", body="", frame_urls=None):
        super().__init__(title, body)
        self.frames = [FakeFrame(u) for u in (frame_urls or [])]


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


class TestHumanMouseMove:
    def setup_method(self):
        br._browser_state.clear()

    def test_ends_at_target_and_tracks_position(self):
        page = FakePage()
        br._browser_human_mouse_move(page, 500, 300)
        assert 4 <= len(page.mouse.moves) <= 7
        last_x, last_y, _steps = page.mouse.moves[-1]
        assert (last_x, last_y) == (500, 300)
        assert br._browser_state["mouse_pos"] == (500, 300)

    def test_second_move_to_same_target_settles_there(self):
        # Wobble is independent noise, not distance-scaled, so intermediate
        # points still curve a bit even when start == target -- only the
        # final settle point (the last move call) is guaranteed exact.
        page = FakePage()
        br._browser_human_mouse_move(page, 500, 300)
        page.mouse.moves.clear()
        br._browser_human_mouse_move(page, 500, 300)
        last_x, last_y, _steps = page.mouse.moves[-1]
        assert (last_x, last_y) == (500, 300)


class TestMoveTowardElement:
    def setup_method(self):
        br._browser_state.clear()

    def test_moves_into_bounding_box(self):
        class FakeLoc:
            def bounding_box(self):
                return {"x": 100, "y": 200, "width": 40, "height": 20}

        page = FakePage()
        br._browser_move_toward_element(page, FakeLoc())
        assert page.mouse.moves
        last_x, last_y, _steps = page.mouse.moves[-1]
        assert 100 <= last_x <= 140
        assert 200 <= last_y <= 220

    def test_missing_bounding_box_is_noop(self):
        class FakeLoc:
            def bounding_box(self):
                return None

        page = FakePage()
        br._browser_move_toward_element(page, FakeLoc())
        assert page.mouse.moves == []


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

    def test_iframe_challenge_with_clean_outer_text_returns_hint(self):
        # DataDome/hCaptcha/Turnstile widgets load in a cross-origin iframe --
        # the outer page's title/body can say nothing while it's blocking.
        page = FakeFramePage(title="Loading...", body="", frame_urls=["https://geo.captcha-delivery.com/captcha/"])
        hint = br._browser_captcha_hint(page)
        assert "browser_action" in hint

    def test_page_error_returns_empty(self):
        class ExplodingPage:
            def title(self):
                raise RuntimeError("page gone")

        assert br._browser_captcha_hint(ExplodingPage()) == ""


class TestDetectChallengeIframe:
    def test_no_matching_frames_returns_empty(self):
        page = FakeFramePage(frame_urls=["https://example.com/widget"])
        assert br._browser_detect_challenge_iframe(page) == ""

    def test_datadome_iframe_detected(self):
        page = FakeFramePage(frame_urls=["https://geo.captcha-delivery.com/captcha/"])
        assert br._browser_detect_challenge_iframe(page) == "captcha-delivery.com"

    def test_hcaptcha_iframe_detected(self):
        page = FakeFramePage(frame_urls=["https://newassets.hcaptcha.com/captcha/v1/frame"])
        assert br._browser_detect_challenge_iframe(page) == "hcaptcha.com"

    def test_page_without_frames_attr_returns_empty(self):
        assert br._browser_detect_challenge_iframe(FakePage()) == ""


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

    def test_challenge_iframe_detected_despite_clean_text(self):
        page = FakeFramePage(title="Loading...", body="", frame_urls=["https://geo.captcha-delivery.com/x"])
        assert br._browser_detect_antibot(page) == "captcha-delivery.com"


class FakeCtx:
    def __init__(self):
        self.added_cookies = None

    def add_cookies(self, cookies):
        self.added_cookies = cookies


class TestMigrateLegacyCookieFile:
    def test_missing_file_is_noop(self, tmp_path, monkeypatch):
        monkeypatch.setattr(br, "COOKIE_FILE", tmp_path / "nope.json")
        ctx = FakeCtx()
        br._migrate_legacy_cookie_file(ctx)
        assert ctx.added_cookies is None

    def test_cookies_are_imported(self, tmp_path, monkeypatch):
        cookie_file = tmp_path / "cookies.json"
        cookie_file.write_text('{"cookies": [{"name": "sid", "value": "1"}], "origins": []}')
        monkeypatch.setattr(br, "COOKIE_FILE", cookie_file)
        ctx = FakeCtx()
        br._migrate_legacy_cookie_file(ctx)
        assert ctx.added_cookies == [{"name": "sid", "value": "1"}]

    def test_legacy_bare_list_format_is_ignored(self, tmp_path, monkeypatch):
        # Old versions of this file stored context.cookies() directly -- a
        # bare list, not the {"cookies": ..., "origins": ...} shape.
        cookie_file = tmp_path / "cookies.json"
        cookie_file.write_text("[]")
        monkeypatch.setattr(br, "COOKIE_FILE", cookie_file)
        ctx = FakeCtx()
        br._migrate_legacy_cookie_file(ctx)
        assert ctx.added_cookies is None

    def test_corrupt_json_is_noop(self, tmp_path, monkeypatch):
        cookie_file = tmp_path / "cookies.json"
        cookie_file.write_text("{not valid json")
        monkeypatch.setattr(br, "COOKIE_FILE", cookie_file)
        ctx = FakeCtx()
        br._migrate_legacy_cookie_file(ctx)
        assert ctx.added_cookies is None


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


class TestLaunchPersistentChromium:
    def test_prefers_real_chrome_channel(self):
        calls = []

        class FakePW:
            class chromium:
                @staticmethod
                def launch_persistent_context(user_data_dir, **kwargs):
                    calls.append((user_data_dir, kwargs))
                    return "CONTEXT"

        ctx, is_real = br._launch_persistent_chromium(
            FakePW(),
            user_data_dir="profile-dir",
            headless=False,
            proxy_config=None,
            args=[],
            base_context_kwargs={"locale": "en-US"},
            fallback_extra_kwargs={"user_agent": "should-not-be-used"},
        )
        assert ctx == "CONTEXT"
        assert is_real is True
        assert calls[0][0] == "profile-dir"
        assert calls[0][1].get("channel") == "chrome"
        assert "user_agent" not in calls[0][1]
        assert len(calls) == 1

    def test_falls_back_and_applies_extra_kwargs_when_chrome_unavailable(self):
        calls = []

        class FakePW:
            class chromium:
                @staticmethod
                def launch_persistent_context(user_data_dir, **kwargs):
                    calls.append(kwargs)
                    if kwargs.get("channel") == "chrome":
                        raise RuntimeError("chrome not found")
                    return "CONTEXT"

        ctx, is_real = br._launch_persistent_chromium(
            FakePW(),
            user_data_dir="profile-dir",
            headless=False,
            proxy_config=None,
            args=[],
            base_context_kwargs={"locale": "en-US"},
            fallback_extra_kwargs={"user_agent": "fallback-ua"},
        )
        assert ctx == "CONTEXT"
        assert is_real is False
        assert len(calls) == 2
        assert calls[1]["user_agent"] == "fallback-ua"
        assert calls[1]["locale"] == "en-US"

    def test_ignore_default_args_disables_automation_flag(self):
        calls = []

        class FakePW:
            class chromium:
                @staticmethod
                def launch_persistent_context(user_data_dir, **kwargs):
                    calls.append(kwargs)
                    return "CONTEXT"

        br._launch_persistent_chromium(
            FakePW(),
            user_data_dir="profile-dir",
            headless=True,
            proxy_config=None,
            args=["--foo"],
            base_context_kwargs={},
            fallback_extra_kwargs={},
        )
        assert calls[0]["ignore_default_args"] == ["--enable-automation"]
        assert calls[0]["args"] == ["--foo"]
        assert calls[0]["headless"] is True


class TestResolveSyncPlaywright:
    def test_falls_back_to_playwright_when_patchright_absent(self, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "patchright.sync_api" or name.startswith("patchright"):
                raise ImportError("no patchright")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        # playwright itself is a real dependency in this project's venv, so this
        # should resolve to playwright.sync_api.sync_playwright without raising.
        factory = br._resolve_sync_playwright()
        assert callable(factory)

    def test_raises_helpful_error_when_neither_installed(self, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name in ("patchright.sync_api", "playwright.sync_api") or name.startswith(
                ("patchright", "playwright")
            ):
                raise ImportError(f"no {name}")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        try:
            br._resolve_sync_playwright()
            raise AssertionError("expected RuntimeError")
        except RuntimeError as e:
            assert "pip install" in str(e)


class TestSettleAfterNav:
    def test_scrolls_and_skips_mouse_without_state(self):
        class FakePageWithEval(FakePage):
            def __init__(self):
                super().__init__()
                self.evaluated = []

            def evaluate(self, expr, arg=None):
                self.evaluated.append((expr, arg))

        page = FakePageWithEval()
        br._browser_settle_after_nav(page, mouse_state=None)
        assert len(page.evaluated) == 1
        assert page.mouse.moves == []

    def test_moves_mouse_and_tracks_position_when_state_given(self):
        class FakePageWithEval(FakePage):
            def evaluate(self, expr, arg=None):
                pass

        page = FakePageWithEval()
        state: dict = {}
        br._browser_settle_after_nav(page, mouse_state=state)
        assert len(page.mouse.moves) == 1
        assert state["mouse_pos"] == page.mouse.moves[0][:2]

    def test_page_without_evaluate_is_noop_not_error(self):
        # FakePage has no .evaluate -- must not raise.
        br._browser_settle_after_nav(FakePage(), mouse_state=None)
