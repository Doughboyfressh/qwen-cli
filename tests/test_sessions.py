"""Tests for session persistence: save_session, load_session, _session_meta,
_silent_autosave, and the _fuzzy_find_session helper."""
import json
from pathlib import Path

import pytest


@pytest.fixture()
def sessions_dir(qwen_cli, tmp_path, monkeypatch):
    """Redirect SESSIONS_DIR to a temp location for each test."""
    d = tmp_path / "sessions"
    d.mkdir()
    monkeypatch.setattr(qwen_cli, "SESSIONS_DIR", d)
    monkeypatch.setattr(qwen_cli, "AUTOSAVE_FILE", d / "autosave.json")
    return d


# ---------------------------------------------------------------------------
# save_session / load_session round-trip
# ---------------------------------------------------------------------------

class TestSaveLoad:
    def _history(self):
        return [
            {"role": "user",      "content": "what is 2+2?"},
            {"role": "assistant", "content": "4"},
        ]

    def test_roundtrip_preserves_history(self, qwen_cli, sessions_dir):
        history = self._history()
        qwen_cli.save_session(history, "SYSTEM", name="test_session")
        loaded, prompt = qwen_cli.load_session("test_session")
        assert loaded == history
        assert prompt == "SYSTEM"

    def test_save_creates_json_file(self, qwen_cli, sessions_dir):
        qwen_cli.save_session(self._history(), "SYS", name="mysession")
        assert (sessions_dir / "mysession.json").exists()

    def test_load_nonexistent_returns_none(self, qwen_cli, sessions_dir):
        history, prompt = qwen_cli.load_session("ghost")
        assert history is None
        assert prompt is None

    def test_meta_topic_from_first_user_message(self, qwen_cli, sessions_dir):
        qwen_cli.save_session(self._history(), "SYS", name="meta_test")
        meta = qwen_cli._session_meta(sessions_dir / "meta_test.json")
        assert "what is 2+2" in meta["topic"]

    def test_meta_turns_counts_assistant_messages(self, qwen_cli, sessions_dir):
        history = [
            {"role": "user", "content": "q1"}, {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "q2"}, {"role": "assistant", "content": "a2"},
        ]
        qwen_cli.save_session(history, "SYS", name="turns_test")
        meta = qwen_cli._session_meta(sessions_dir / "turns_test.json")
        assert meta["turns"] == 2

    def test_meta_on_corrupt_file_returns_defaults(self, qwen_cli, sessions_dir):
        bad = sessions_dir / "bad.json"
        bad.write_text("not json", encoding="utf-8")
        meta = qwen_cli._session_meta(bad)
        assert meta["turns"] == 0
        assert meta["topic"] == ""


# ---------------------------------------------------------------------------
# _silent_autosave
# ---------------------------------------------------------------------------

class TestAutosave:
    def test_writes_autosave_file(self, qwen_cli, sessions_dir):
        history = [{"role": "user", "content": "auto"}, {"role": "assistant", "content": "saved"}]
        qwen_cli._silent_autosave(history, "SYSTEM")
        assert (sessions_dir / "autosave.json").exists()
        data = json.loads((sessions_dir / "autosave.json").read_text(encoding="utf-8"))
        assert data["history"] == history

    def test_never_raises_on_bad_path(self, qwen_cli, tmp_path, monkeypatch):
        monkeypatch.setattr(qwen_cli, "AUTOSAVE_FILE", tmp_path / "no_dir" / "autosave.json")
        qwen_cli._silent_autosave([{"role": "user", "content": "x"}], "SYS")


# ---------------------------------------------------------------------------
# _fuzzy_find_session
# ---------------------------------------------------------------------------

class TestFuzzyFind:
    def test_finds_by_stem(self, qwen_cli, sessions_dir):
        qwen_cli.save_session([{"role": "user", "content": "refactor auth module"}], "SYS", name="refactor_auth")
        matches = qwen_cli._fuzzy_find_session("refactor")
        stems = [m.stem for m in matches]
        assert "refactor_auth" in stems

    def test_finds_by_topic(self, qwen_cli, sessions_dir):
        qwen_cli.save_session([{"role": "user", "content": "debug the login flow"}], "SYS", name="20260101_120000")
        # "debug the login" is a substring of the stored topic — fuzzy search uses `in`
        matches = qwen_cli._fuzzy_find_session("debug the login")
        assert any("20260101" in m.stem for m in matches)

    def test_no_match_returns_empty(self, qwen_cli, sessions_dir):
        qwen_cli.save_session([{"role": "user", "content": "something else"}], "SYS", name="unrelated")
        matches = qwen_cli._fuzzy_find_session("xyzzy_not_here")
        assert matches == []
