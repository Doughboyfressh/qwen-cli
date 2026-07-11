"""Unit tests for do_fetch_url's JS-shell detection in qwen_cli.tools.shared."""

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
