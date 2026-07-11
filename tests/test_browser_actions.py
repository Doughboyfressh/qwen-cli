"""Unit tests for newer browser_action verbs (go_back/go_forward, evaluate,
upload_file) and new-tab/popup auto-tracking in qwen_cli.tools.browser.

Fake page/context objects stand in for real Playwright objects -- no browser
process is launched. The new-tab-detection tests for click/submit/press_key
still need the `playwright` package importable (for TimeoutError), since
do_browser_action imports it inline for those actions; no browser is started.
"""

from playwright.sync_api import TimeoutError as PwTimeoutError

import qwen_cli.tools.browser as br


class FakePage:
    def __init__(self, url="http://example.com", title="Example"):
        self.url = url
        self._title = title
        self.back_result = "RESP"
        self.forward_result = "RESP"
        self.evaluate_result = None
        self.evaluate_error = None

    def title(self):
        return self._title

    def go_back(self, wait_until=None, timeout=None):
        return self.back_result

    def go_forward(self, wait_until=None, timeout=None):
        return self.forward_result

    def evaluate(self, expr):
        if self.evaluate_error:
            raise self.evaluate_error
        return self.evaluate_result

    def wait_for_load_state(self, state, timeout=None):
        pass


class TestGoBackForward:
    def test_go_back_with_history(self):
        assert "navigated back" in br._browser_do_go_back(FakePage())

    def test_go_back_no_history(self):
        page = FakePage()
        page.back_result = None
        assert "no previous page" in br._browser_do_go_back(page)

    def test_go_forward_with_history(self):
        assert "navigated forward" in br._browser_do_go_forward(FakePage())

    def test_go_forward_no_history(self):
        page = FakePage()
        page.forward_result = None
        assert "no next page" in br._browser_do_go_forward(page)


class TestEvaluate:
    def test_requires_value(self):
        assert "requires a JS expression" in br._browser_do_evaluate(FakePage(), value="")

    def test_returns_json_result(self):
        page = FakePage()
        page.evaluate_result = {"a": 1, "b": [1, 2, 3]}
        result = br._browser_do_evaluate(page, value="({a:1,b:[1,2,3]})")
        assert '"a": 1' in result

    def test_non_serializable_result_does_not_crash(self):
        page = FakePage()
        page.evaluate_result = object()
        result = br._browser_do_evaluate(page, value="something")
        assert "evaluate result" in result

    def test_js_error_is_caught(self):
        page = FakePage()
        page.evaluate_error = RuntimeError("boom")
        assert "evaluate error" in br._browser_do_evaluate(page, value="throw")


class TestUploadFile:
    def test_requires_selector_and_value(self):
        assert "requires selector" in br._browser_do_upload_file(FakePage(), selector="", value="")

    def test_missing_file_returns_error(self, tmp_path):
        missing = tmp_path / "nope.txt"
        result = br._browser_do_upload_file(FakePage(), selector="input[type=file]", value=str(missing))
        assert "file not found" in result


class FakeContext:
    """Mimics BrowserContext.wait_for_event("page", ...) -- popup_page=None
    means no popup opens, matching the real TimeoutError-on-no-event behavior.
    """

    def __init__(self, popup_page=None):
        self._popup_page = popup_page

    def wait_for_event(self, event, timeout=None):
        if self._popup_page is None:
            raise PwTimeoutError("no popup")
        return self._popup_page


class TestNewTabDetection:
    def setup_method(self):
        br._browser_state.clear()

    def test_click_that_opens_new_tab_is_noted(self, monkeypatch):
        page2 = FakePage(url="http://b.com")
        page1 = FakePage(url="http://a.com")
        page1.context = FakeContext(popup_page=page2)
        monkeypatch.setattr(br, "_get_page", lambda: page1)
        br._browser_state["page"] = page1

        monkeypatch.setitem(
            br._BROWSER_HANDLERS,
            "click",
            lambda page, url="", selector="", value="", screenshot_path="": "[clicked 'link:Open']",
        )
        result = br.do_browser_action(action="click", selector="link:Open")
        assert "new tab/popup opened" in result
        assert "http://b.com" in result
        assert br._browser_state["page"] is page2

    def test_click_with_no_popup_is_not_noted(self, monkeypatch):
        page1 = FakePage(url="http://a.com")
        page1.context = FakeContext(popup_page=None)
        monkeypatch.setattr(br, "_get_page", lambda: page1)
        br._browser_state["page"] = page1

        monkeypatch.setitem(
            br._BROWSER_HANDLERS,
            "click",
            lambda page, url="", selector="", value="", screenshot_path="": "[clicked 'button:Submit']",
        )
        result = br.do_browser_action(action="click", selector="button:Submit")
        assert "new tab/popup" not in result
        assert br._browser_state["page"] is page1

    def test_no_new_tab_no_note_for_non_popup_prone_action(self, monkeypatch):
        page1 = FakePage(url="http://a.com")
        monkeypatch.setattr(br, "_get_page", lambda: page1)
        br._browser_state["page"] = page1

        monkeypatch.setitem(br._BROWSER_HANDLERS, "get_text", lambda *a, **kw: "[page text]")
        result = br.do_browser_action(action="get_text")
        assert "new tab/popup" not in result

    def test_real_handler_error_is_not_swallowed_as_no_popup(self, monkeypatch):
        """A genuine failure inside the handler (e.g. selector never became
        visible) raises the same TimeoutError class Playwright uses for "no
        popup arrived". It must still surface as a real error, not get
        silently treated as a successful click with no popup.
        """
        page1 = FakePage(url="http://a.com")
        page1.context = FakeContext(popup_page=None)
        monkeypatch.setattr(br, "_get_page", lambda: page1)
        br._browser_state["page"] = page1

        def failing_click(page, url="", selector="", value="", screenshot_path=""):
            raise PwTimeoutError("element not found")

        monkeypatch.setitem(br._BROWSER_HANDLERS, "click", failing_click)
        result = br.do_browser_action(action="click", selector="button:Missing")
        assert "element not found" in result
        assert "clicked" not in result.lower()
        assert "new tab/popup" not in result
