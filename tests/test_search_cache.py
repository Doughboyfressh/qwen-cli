"""Unit tests for do_web_search's short-lived result cache in qwen_cli.tools.shared."""

import qwen_cli.tools.shared as shared


def _stub_engines(monkeypatch, ddg_fn):
    monkeypatch.setattr(shared, "_search_ddg", ddg_fn)
    monkeypatch.setattr(shared, "_search_google", lambda q, m: [])
    monkeypatch.setattr(shared, "_search_brave", lambda q, m: [])
    monkeypatch.setattr(shared, "_search_bing_scrape", lambda q, m: [])


class TestSearchCache:
    def setup_method(self):
        shared._SEARCH_CACHE.clear()

    def test_repeated_query_is_served_from_cache(self, monkeypatch):
        calls = {"n": 0}

        def fake_ddg(query, max_results):
            calls["n"] += 1
            return [{"title": "T", "href": "http://x.com", "body": "B"}]

        _stub_engines(monkeypatch, fake_ddg)

        r1 = shared.do_web_search("hello world", max_results=3)
        r2 = shared.do_web_search("hello world", max_results=3)
        assert r1 == r2
        assert calls["n"] == 1

    def test_different_max_results_bypasses_cache(self, monkeypatch):
        calls = {"n": 0}

        def fake_ddg(query, max_results):
            calls["n"] += 1
            return [{"title": "T", "href": "http://x.com", "body": "B"}]

        _stub_engines(monkeypatch, fake_ddg)

        shared.do_web_search("foo", max_results=3)
        shared.do_web_search("foo", max_results=5)
        assert calls["n"] == 2

    def test_failed_search_is_not_cached(self, monkeypatch):
        def failing_ddg(query, max_results):
            raise RuntimeError("engine down")

        _stub_engines(monkeypatch, failing_ddg)

        result = shared.do_web_search("bar", max_results=3)
        assert "all search engines failed" in result
        assert shared._search_cache_get(("bar", 3, "web")) is None
