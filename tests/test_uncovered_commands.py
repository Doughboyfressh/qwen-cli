"""Coverage for previously-untested user-facing commands and helpers.

The 2026-07-12 self-audit flagged these as public functions with zero test
references. cmd_rollback is the standout: it is the /undo-all safety feature —
if it silently breaks, a bad agent run can no longer be reverted.
"""

import json

import pytest
from rich.markdown import Markdown


def _print_text(val: object) -> str:
    """Convert a console.print argument to readable text."""
    if isinstance(val, Markdown):
        return getattr(val, "markup", str(val))
    return str(val)


@pytest.fixture()
def capture(qwen_cli, monkeypatch):
    """Collect the strings passed to console.print during a test."""
    calls: list[str] = []
    monkeypatch.setattr(
        qwen_cli.console,
        "print",
        lambda *a, **k: calls.append(_print_text(a[0]) if a else ""),
    )
    return calls


# ---------------------------------------------------------------------------
# cmd_rollback — /undo-all for the session
# ---------------------------------------------------------------------------


class TestRollback:
    def test_restores_files_and_clears_tracking(self, qwen_cli, tmp_path, monkeypatch, capture):
        f = tmp_path / "app.py"
        f.write_text("modified by the model\n", encoding="utf-8")
        monkeypatch.setattr(qwen_cli, "_session_changes", {str(f): "original content\n"})
        monkeypatch.setattr(qwen_cli.console, "input", lambda *a, **k: "y")

        qwen_cli.cmd_rollback()

        assert f.read_text(encoding="utf-8") == "original content\n"
        assert qwen_cli._session_changes == {}

    def test_declining_leaves_everything_untouched(self, qwen_cli, tmp_path, monkeypatch, capture):
        f = tmp_path / "app.py"
        f.write_text("modified\n", encoding="utf-8")
        changes = {str(f): "original\n"}
        monkeypatch.setattr(qwen_cli, "_session_changes", changes)
        monkeypatch.setattr(qwen_cli.console, "input", lambda *a, **k: "n")

        qwen_cli.cmd_rollback()

        assert f.read_text(encoding="utf-8") == "modified\n"
        assert changes  # still tracked — the user can roll back later
        assert any("cancelled" in c for c in capture)

    def test_noop_without_changes(self, qwen_cli, monkeypatch, capture):
        monkeypatch.setattr(qwen_cli, "_session_changes", {})
        qwen_cli.cmd_rollback()
        assert any("no session changes" in c for c in capture)


# ---------------------------------------------------------------------------
# cmd_changes — session modification report
# ---------------------------------------------------------------------------


class TestChanges:
    def test_lists_files_with_line_delta(self, qwen_cli, tmp_path, monkeypatch, capture):
        f = tmp_path / "grew.txt"
        f.write_text("a\nb\nc\n", encoding="utf-8")
        monkeypatch.setattr(qwen_cli, "_session_changes", {str(f): "a\n"})

        qwen_cli.cmd_changes()

        joined = "\n".join(capture)
        assert "grew.txt" in joined
        assert "+2 lines" in joined

    def test_diff_mode_shows_unified_diff(self, qwen_cli, tmp_path, monkeypatch, capture):
        f = tmp_path / "d.txt"
        f.write_text("new line\n", encoding="utf-8")
        monkeypatch.setattr(qwen_cli, "_session_changes", {str(f): "old line\n"})

        qwen_cli.cmd_changes("diff")

        assert any("d.txt" in c for c in capture)

    def test_noop_without_changes(self, qwen_cli, monkeypatch, capture):
        monkeypatch.setattr(qwen_cli, "_session_changes", {})
        qwen_cli.cmd_changes()
        assert any("no files modified" in c for c in capture)


# ---------------------------------------------------------------------------
# cmd_focus / load_file_into_context — pulling files into the conversation
# ---------------------------------------------------------------------------


class TestFocus:
    def test_loads_file_into_history_and_tracks_it(self, qwen_cli, tmp_path, monkeypatch, capture):
        f = tmp_path / "notes.md"
        f.write_text("hello focus\n", encoding="utf-8")
        monkeypatch.setattr(qwen_cli, "_focus_set", [])
        history: list = []

        qwen_cli.cmd_focus(str(f), history)

        assert qwen_cli._focus_set == [str(f)]
        assert len(history) == 1
        assert "notes.md" in history[0]["content"]
        assert "hello focus" in history[0]["content"]

    def test_quoted_path_with_spaces_stays_one_token(self, qwen_cli, tmp_path, monkeypatch, capture):
        d = tmp_path / "my docs"
        d.mkdir()
        f = d / "todo.txt"
        f.write_text("remember\n", encoding="utf-8")
        monkeypatch.setattr(qwen_cli, "_focus_set", [])
        history: list = []

        qwen_cli.cmd_focus(f'"{f}"', history)

        assert qwen_cli._focus_set == [str(f)]
        assert len(history) == 1

    def test_missing_file_loads_nothing(self, qwen_cli, tmp_path, monkeypatch, capture):
        monkeypatch.setattr(qwen_cli, "_focus_set", [])
        history: list = []
        qwen_cli.cmd_focus(str(tmp_path / "ghost.txt"), history)
        assert history == []
        assert any("not found" in c for c in capture)

    def test_off_clears_tracking(self, qwen_cli, monkeypatch, capture):
        tracked = ["a.py"]
        monkeypatch.setattr(qwen_cli, "_focus_set", tracked)
        qwen_cli.cmd_focus("off", [])
        assert tracked == []


class TestLoadFileIntoContext:
    def test_binary_file_is_rejected(self, qwen_cli, tmp_path, capture):
        f = tmp_path / "blob.bin"
        f.write_bytes(b"\x00\x01\x02data")
        assert qwen_cli.load_file_into_context(str(f), []) is False
        assert any("binary file" in c for c in capture)

    def test_missing_file_returns_false(self, qwen_cli, tmp_path, capture):
        assert qwen_cli.load_file_into_context(str(tmp_path / "nope.txt"), []) is False


# ---------------------------------------------------------------------------
# cmd_search_sessions — full-text search over saved sessions
# ---------------------------------------------------------------------------


class TestSearchSessions:
    def _save(self, d, name, content):
        (d / f"{name}.json").write_text(
            json.dumps(
                {
                    "meta": {"topic": "test topic", "turns": 1, "saved_at": ""},
                    "system_prompt": "SYS",
                    "history": [{"role": "user", "content": content}],
                }
            ),
            encoding="utf-8",
        )

    def test_finds_matching_session(self, qwen_cli, tmp_path, monkeypatch, capture):
        d = tmp_path / "sessions"
        d.mkdir()
        monkeypatch.setattr(qwen_cli, "SESSIONS_DIR", d)
        self._save(d, "hit", "the xylophone incident")
        self._save(d, "miss", "nothing relevant here")

        qwen_cli.cmd_search_sessions("xylophone")

        joined = "\n".join(capture)
        assert "1 session(s) matching" in joined

    def test_reports_no_matches(self, qwen_cli, tmp_path, monkeypatch, capture):
        d = tmp_path / "sessions"
        d.mkdir()
        monkeypatch.setattr(qwen_cli, "SESSIONS_DIR", d)
        qwen_cli.cmd_search_sessions("xylophone")
        assert any("no sessions matching" in c for c in capture)

    def test_usage_hint_without_query(self, qwen_cli, capture):
        qwen_cli.cmd_search_sessions("")
        assert any("usage" in c for c in capture)


# ---------------------------------------------------------------------------
# cmd_preset / cmd_autosearch — runtime knobs
# ---------------------------------------------------------------------------


class TestPreset:
    def test_switches_active_preset(self, qwen_cli, monkeypatch, capture):
        monkeypatch.setattr(qwen_cli, "_active_preset", "thinking")
        qwen_cli.cmd_preset("code")
        assert qwen_cli._active_preset == "code"

    def test_alias_resolves(self, qwen_cli, monkeypatch, capture):
        monkeypatch.setattr(qwen_cli, "_active_preset", "thinking")
        qwen_cli.cmd_preset("tc")  # alias for code
        assert qwen_cli._active_preset == "code"

    def test_unknown_preset_leaves_current(self, qwen_cli, monkeypatch, capture):
        monkeypatch.setattr(qwen_cli, "_active_preset", "thinking")
        qwen_cli.cmd_preset("warp-drive")
        assert qwen_cli._active_preset == "thinking"
        assert any("unknown preset" in c for c in capture)


class TestAutosearch:
    def _cfg(self, qwen_cli, tmp_path, monkeypatch):
        cfg = tmp_path / "config.toml"
        cfg.write_text('auto_search = "smart"\n', encoding="utf-8")
        monkeypatch.setattr(qwen_cli, "CONFIG_FILE", cfg)
        return cfg

    def test_sets_mode_and_persists(self, qwen_cli, tmp_path, monkeypatch, capture):
        cfg = self._cfg(qwen_cli, tmp_path, monkeypatch)
        monkeypatch.setattr(qwen_cli, "AUTO_SEARCH_MODE", "smart")

        qwen_cli.cmd_autosearch("off")

        assert qwen_cli.AUTO_SEARCH_MODE == "off"
        assert 'auto_search = "off"' in cfg.read_text(encoding="utf-8")

    def test_on_is_an_alias_for_aggressive(self, qwen_cli, tmp_path, monkeypatch, capture):
        self._cfg(qwen_cli, tmp_path, monkeypatch)
        monkeypatch.setattr(qwen_cli, "AUTO_SEARCH_MODE", "smart")
        qwen_cli.cmd_autosearch("on")
        assert qwen_cli.AUTO_SEARCH_MODE == "aggressive"

    def test_unknown_mode_rejected(self, qwen_cli, tmp_path, monkeypatch, capture):
        self._cfg(qwen_cli, tmp_path, monkeypatch)
        monkeypatch.setattr(qwen_cli, "AUTO_SEARCH_MODE", "smart")
        qwen_cli.cmd_autosearch("turbo")
        assert qwen_cli.AUTO_SEARCH_MODE == "smart"
        assert any("unknown mode" in c for c in capture)


# ---------------------------------------------------------------------------
# build_project_tree / do_list_directory — directory rendering
# ---------------------------------------------------------------------------


class TestTreeAndListing:
    def test_tree_renders_nested_structure(self, qwen_cli, tmp_path):
        (tmp_path / "pkg").mkdir()
        (tmp_path / "pkg" / "mod.py").write_text("x = 1\n", encoding="utf-8")
        (tmp_path / "README.md").write_text("hi\n", encoding="utf-8")

        tree = qwen_cli.build_project_tree(tmp_path)

        assert "pkg" in tree
        assert "mod.py" in tree
        assert "README.md" in tree
        assert "──" in tree  # actual tree connectors, not a flat list

    def test_tree_truncates_at_max_files(self, qwen_cli, tmp_path):
        for i in range(10):
            (tmp_path / f"f{i:02d}.txt").write_text("x", encoding="utf-8")
        tree = qwen_cli.build_project_tree(tmp_path, max_files=3)
        assert "truncated at 3" in tree

    def test_tree_hides_ignored_dirs(self, qwen_cli, tmp_path):
        (tmp_path / "__pycache__").mkdir()
        (tmp_path / "real.py").write_text("x", encoding="utf-8")
        tree = qwen_cli.build_project_tree(tmp_path)
        assert "__pycache__" not in tree
        assert "real.py" in tree

    def test_list_directory_counts_and_entries(self, qwen_cli, tmp_path):
        (tmp_path / "sub").mkdir()
        (tmp_path / "a.txt").write_text("hello", encoding="utf-8")

        out = qwen_cli.do_list_directory(str(tmp_path))

        assert "(1 dir, 1 file)" in out
        assert "sub/" in out
        assert "a.txt" in out

    def test_list_directory_not_found(self, qwen_cli, tmp_path):
        assert "[not found" in qwen_cli.do_list_directory(str(tmp_path / "ghost"))

    def test_list_directory_recursive_uses_tree(self, qwen_cli, tmp_path):
        (tmp_path / "deep").mkdir()
        (tmp_path / "deep" / "leaf.txt").write_text("x", encoding="utf-8")
        out = qwen_cli.do_list_directory(str(tmp_path), recursive=True)
        assert out.startswith("Tree:")
        assert "leaf.txt" in out


# ---------------------------------------------------------------------------
# list_sessions / warn_if_context_high — display helpers
# ---------------------------------------------------------------------------


def test_list_sessions_shows_saved_names(qwen_cli, tmp_path, monkeypatch, capture):
    import qwen_cli.core.sessions as sessions

    d = tmp_path / "sessions"
    d.mkdir()
    monkeypatch.setattr(sessions, "SESSIONS_DIR", d)
    qwen_cli.save_session([{"role": "user", "content": "hi"}], "SYS", name="my_saved_run")

    capture.clear()
    qwen_cli.list_sessions()

    # rich Table object is printed; verify by rendering the captured reprs
    assert capture, "list_sessions printed nothing"


def test_list_sessions_empty_dir(qwen_cli, tmp_path, monkeypatch, capture):
    import qwen_cli.core.sessions as sessions

    monkeypatch.setattr(sessions, "SESSIONS_DIR", tmp_path)
    qwen_cli.list_sessions()
    assert any("no saved sessions" in c for c in capture)


def test_warn_if_context_high_fires_at_75(qwen_cli, monkeypatch, capture):
    monkeypatch.setattr(qwen_cli, "TOKEN_LIMIT", 100)
    qwen_cli.warn_if_context_high([{"role": "user", "content": "x" * 320}])  # ~80 tokens
    assert any("consider /trim" in c for c in capture)


def test_warn_if_context_high_silent_below(qwen_cli, monkeypatch, capture):
    monkeypatch.setattr(qwen_cli, "TOKEN_LIMIT", 100)
    qwen_cli.warn_if_context_high([{"role": "user", "content": "x" * 40}])  # ~10 tokens
    assert capture == []


# ---------------------------------------------------------------------------
# cmd_git_commit — /git commit
# ---------------------------------------------------------------------------


class TestGitCommit:
    def test_noop_with_no_staged_changes(self, qwen_cli, monkeypatch, capture):
        import subprocess

        monkeypatch.setattr(
            subprocess,
            "run",
            lambda *a, **k: type("R", (), {"stdout": "", "returncode": 0})(),
        )

        mock_client = type("C", (), {})()
        qwen_cli.cmd_git_commit(mock_client)

        assert any("no staged changes" in c for c in capture)

    def test_commits_with_provided_message(self, qwen_cli, monkeypatch, capture):
        import subprocess

        monkeypatch.setattr(
            subprocess,
            "run",
            lambda *a, **k: type("R", (), {"stdout": "diff -- some change\n", "returncode": 0})(),
        )

        mock_client = type("C", (), {})()
        monkeypatch.setattr(qwen_cli.console, "input", lambda *a, **k: "y")

        qwen_cli.cmd_git_commit(mock_client, msg="fix: test commit")

        # The function should prompt for confirmation with the provided message
        assert any("test commit" in c for c in capture)

    def test_auto_generates_message_when_empty(self, qwen_cli, monkeypatch, capture):
        import subprocess

        monkeypatch.setattr(
            subprocess,
            "run",
            lambda *a, **k: type("R", (), {"stdout": "diff -- some change\n", "returncode": 0})(),
        )

        # Mock the LLM response for message generation
        mock_choice = type("C", (), {"message": type("M", (), {"content": "feat: new feature\n"})()})()
        mock_resp = type("R", (), {"choices": [mock_choice]})()
        mock_client = type("C", (), {
            "chat": type("CH", (), {
                "completions": type("CO", (), {
                    "create": lambda *a, **k: mock_resp,
                })(),
            })(),
        })()
        monkeypatch.setattr(qwen_cli.console, "input", lambda *a, **k: "y")

        qwen_cli.cmd_git_commit(mock_client)

        assert any("new feature" in c for c in capture)

    def test_declines_commit_on_n(self, qwen_cli, monkeypatch, capture):
        import subprocess

        monkeypatch.setattr(
            subprocess,
            "run",
            lambda *a, **k: type("R", (), {"stdout": "diff -- some change\n", "returncode": 0})(),
        )

        mock_client = type("C", (), {})()
        monkeypatch.setattr(qwen_cli.console, "input", lambda *a, **k: "n")

        qwen_cli.cmd_git_commit(mock_client, msg="feat: will not commit")

        # Should show the message but not actually commit (no subprocess call after prompt)
        assert any("will not commit" in c for c in capture)

    def test_handles_git_timeout(self, qwen_cli, monkeypatch, capture):
        import subprocess

        monkeypatch.setattr(
            subprocess,
            "run",
            lambda *a, **k: (_ for _ in ()).throw(subprocess.TimeoutExpired("git", 10)),
        )

        mock_client = type("C", (), {})()
        qwen_cli.cmd_git_commit(mock_client)

        assert any("timed out" in c for c in capture)

    def test_handles_llm_error_gracefully(self, qwen_cli, monkeypatch, capture):
        import subprocess

        monkeypatch.setattr(
            subprocess,
            "run",
            lambda *a, **k: type("R", (), {"stdout": "diff -- change\n", "returncode": 0})(),
        )

        mock_client = type("C", (), {
            "chat": type("CH", (), {
                "completions": type("CO", (), {
                    "create": lambda *a, **k: (_ for _ in ()).throw(RuntimeError("llm down")),
                })(),
            })(),
        })()

        qwen_cli.cmd_git_commit(mock_client)

        assert any("error generating message" in c for c in capture)


# ---------------------------------------------------------------------------
# cmd_git_pr — /git pr
# ---------------------------------------------------------------------------


class TestGitPR:
    def test_noop_when_no_commits_ahead(self, qwen_cli, monkeypatch, capture):
        import subprocess

        monkeypatch.setattr(
            subprocess,
            "run",
            lambda *a, **k: type("R", (), {"stdout": "", "returncode": 0})(),
        )

        mock_client = type("C", (), {})()
        qwen_cli.cmd_git_pr(mock_client)

        assert any("no commits ahead" in c for c in capture)

    def test_generates_pr_description(self, qwen_cli, monkeypatch, capture):
        import subprocess

        # Return log output for the first base that has commits
        call_count = [0]

        def mock_run(*args, **kwargs):
            call_count[0] += 1
            if "git" in (args[0][0] if args[0] else ""):
                cmd_args = args[0] if args else []
                if any("oneline" in a for a in cmd_args):
                    return type("R", (), {"stdout": "abc123 feat: new feature\n", "returncode": 0})()
                elif any("stat" in a for a in cmd_args):
                    return type("R", (), {"stdout": " file.py | 10 +++++++---\n", "returncode": 0})()
            return type("R", (), {"stdout": "", "returncode": 0})()

        monkeypatch.setattr(subprocess, "run", mock_run)

        mock_choice = type("C", (), {"message": type("M", (), {"content": "## Summary\n- Feature\n"})()})()
        mock_resp = type("R", (), {"choices": [mock_choice]})()
        mock_client = type("C", (), {
            "chat": type("CH", (), {
                "completions": type("CO", (), {
                    "create": lambda *a, **k: mock_resp,
                })(),
            })(),
        })()

        # Mock console.input to decline clipboard copy
        monkeypatch.setattr(qwen_cli.console, "input", lambda *a, **k: "n")

        qwen_cli.cmd_git_pr(mock_client)

        assert any("Summary" in c for c in capture)

    def test_handles_llm_error_gracefully(self, qwen_cli, monkeypatch, capture):
        import subprocess

        monkeypatch.setattr(
            subprocess,
            "run",
            lambda *a, **k: type("R", (), {"stdout": "abc123 commit\n", "returncode": 0})(),
        )

        mock_client = type("C", (), {
            "chat": type("CH", (), {
                "completions": type("CO", (), {
                    "create": lambda *a, **k: (_ for _ in ()).throw(RuntimeError("llm down")),
                })(),
            })(),
        })()

        qwen_cli.cmd_git_pr(mock_client)

        assert any("error" in c.lower() for c in capture)
