"""Unit tests for do_fetch_url's JS-shell and anti-bot detection in qwen_cli.tools.shared."""

import urllib.error
from unittest.mock import patch

import qwen_cli.tools.shared as shared


class TestLooksLikeJsShell:
    def test_normal_article_page_is_not_a_shell(self):
        html = "<html><body>" + ("<p>Some real article text.</p>" * 20) + "</body></html>"
        text = "Some real article text. " * 20
        assert shared._looks_like_js_shell(html, text) is False

    def test_react_root_with_no_text_is_a_shell(self):
        html = '<html><body><div id="root"></div>' + ("<script>x</script>" * 100) + "</body></html>"
        text = ""
        assert shared._looks_like_js_shell(html, text) is True

    def test_short_real_page_is_not_flagged(self):
        # Small HTML overall (below the size floor) shouldn't be flagged just
        # because it also has little text -- e.g. a short real confirmation page.
        html = "<html><body><p>Thanks!</p></body></html>"
        text = "Thanks!"
        assert shared._looks_like_js_shell(html, text) is False

    def test_enable_javascript_message_is_a_shell(self):
        html = "<html><body>" + ("<!-- padding -->" * 200) + "You need to enable JavaScript to run this app.</body></html>"
        text = "You need to enable JavaScript to run this app."
        assert shared._looks_like_js_shell(html, text) is True


class TestLooksLikeAntibotBlock:
    def test_normal_page_is_not_flagged(self):
        html = "<html><body>" + ("<p>Some real article text.</p>" * 20) + "</body></html>"
        assert shared._looks_like_antibot_block(html) is False

    def test_cloudflare_challenge_is_flagged(self):
        html = "<html><title>Just a moment...</title><body>Checking your browser before accessing</body></html>"
        assert shared._looks_like_antibot_block(html) is True

    def test_captcha_page_is_flagged(self):
        html = "<html><body>Please complete the CAPTCHA to continue</body></html>"
        assert shared._looks_like_antibot_block(html) is True


class FakeResponse:
    def __init__(self, body: bytes, content_type: str = "text/html"):
        self._body = body
        self._headers = {"Content-Type": content_type}

    def read(self):
        return self._body

    @property
    def headers(self):
        return self

    def get(self, key, default=None):
        return self._headers.get(key, default)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class TestDoFetchUrlJsShellHint:
    def setup_method(self):
        shared._FETCH_CACHE.clear()

    def test_spa_shell_response_gets_escalation_hint(self):
        html = '<html><body><div id="root"></div>' + ("<script>x</script>" * 100) + "</body></html>"
        with patch("urllib.request.urlopen", return_value=FakeResponse(html.encode())):
            result = shared.do_fetch_url("http://example.com/spa-page")
        assert "fetch_rendered" in result
        assert "NOTE" in result

    def test_normal_page_response_has_no_hint(self):
        html = "<html><body>" + ("<p>Some real article text with plenty of content.</p>" * 20) + "</body></html>"
        with patch("urllib.request.urlopen", return_value=FakeResponse(html.encode())):
            result = shared.do_fetch_url("http://example.com/normal-page")
        assert "fetch_rendered" not in result

    def test_antibot_challenge_response_gets_escalation_hint(self):
        html = "<html><title>Just a moment...</title><body>Checking your browser before accessing example.com</body></html>"
        with patch("urllib.request.urlopen", return_value=FakeResponse(html.encode())):
            result = shared.do_fetch_url("http://example.com/blocked-page")
        assert "anti-bot" in result
        assert "browser_action" in result


class TestDoFetchUrlHttpErrorHints:
    def setup_method(self):
        shared._FETCH_CACHE.clear()

    def test_403_gets_antibot_hint(self):
        err = urllib.error.HTTPError("http://example.com/x", 403, "Forbidden", {}, None)
        with patch("urllib.request.urlopen", side_effect=err):
            result = shared.do_fetch_url("http://example.com/x")
        assert "HTTP 403" in result
        assert "anti-bot" in result

    def test_404_has_no_antibot_hint(self):
        err = urllib.error.HTTPError("http://example.com/x", 404, "Not Found", {}, None)
        with patch("urllib.request.urlopen", side_effect=err):
            result = shared.do_fetch_url("http://example.com/x")
        assert "HTTP 404" in result
        assert "anti-bot" not in result
