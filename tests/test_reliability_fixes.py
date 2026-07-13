"""Tests for the reliability batch: LSP idle-tracking global fix, spawned-agent
auto-approve, read_input event-loop fallback, memory sanitation, and test
data-dir isolation."""

import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


# ---------------------------------------------------------------------------
# Test isolation — the suite must never touch the real ~/.qwen-cli data
# ---------------------------------------------------------------------------


class TestDataDirIsolation:
    def test_data_dir_is_redirected(self):
        from qwen_cli.core.config import DATA_DIR

        assert "qwen-cli-test-data-" in str(DATA_DIR)
        assert DATA_DIR != Path.home() / ".qwen-cli"

    def test_memory_file_under_redirected_dir(self, qwen_cli):
        from qwen_cli.core.config import DATA_DIR, MEMORY_FILE

        assert str(MEMORY_FILE).startswith(str(DATA_DIR))


# ---------------------------------------------------------------------------
# _clean_memory_facts — memory pipeline sanitation
# ---------------------------------------------------------------------------


class TestCleanMemoryFacts:
    def test_bare_skip_rejected(self, qwen_cli):
        assert qwen_cli._clean_memory_facts("SKIP") == ""
        assert qwen_cli._clean_memory_facts("- SKIP") == ""
        assert qwen_cli._clean_memory_facts("* **SKIP**") == ""
        assert qwen_cli._clean_memory_facts("- No") == ""
        assert qwen_cli._clean_memory_facts("none.") == ""

    def test_tool_call_syntax_rejected_wholesale(self, qwen_cli):
        junk = (
            "It looks like the browser is loading.\n\n<tool_call>\n"
            "<function=browser_action>\n<parameter=action>\nscreenshot\n"
            "</parameter>\n</function>\n</tool_call>"
        )
        assert qwen_cli._clean_memory_facts(junk) == ""
        assert qwen_cli._clean_memory_facts("- fact\n<think>hmm</think>") == ""

    def test_negations_dropped_only_when_requested(self, qwen_cli):
        text = "- No critical CVEs detected this week.\n- CVE-2026-1234: RCE in libfoo 2.1"
        cleaned = qwen_cli._clean_memory_facts(text, drop_negations=True)
        assert "CVE-2026-1234" in cleaned
        assert "No critical" not in cleaned
        # Without the flag, negation lines survive (legit for user facts).
        kept = qwen_cli._clean_memory_facts(text)
        assert "No critical" in kept

    def test_mixed_junk_and_facts_keeps_facts(self, qwen_cli):
        text = "- SKIP\n- User prefers tabs over spaces\n- none"
        assert qwen_cli._clean_memory_facts(text) == "- User prefers tabs over spaces"

    def test_legit_facts_pass_through(self, qwen_cli):
        text = "- Project uses Python 3.13\n- Server runs on port 8080"
        assert qwen_cli._clean_memory_facts(text) == text

    def test_intel_train_skips_bulleted_skip(self, qwen_cli, monkeypatch, tmp_path):
        # The old check only caught bare 'SKIP'; '- SKIP' was written to memory.
        import qwen_cli.core.intel as intel

        fake_resp = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="- SKIP"))]
        )
        fake_client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **k: fake_resp))
        )
        # _bg_llm and save_memory still live in main and intel reaches them
        # lazily via _main — patching main must keep working.
        monkeypatch.setattr(qwen_cli, "_bg_llm", lambda c: (fake_client, "fake-model"))
        monkeypatch.setattr(intel, "_intel_memory_written", {})
        saved = []
        monkeypatch.setattr(qwen_cli, "save_memory", lambda text: saved.append(text))
        qwen_cli._intel_train_memory(None, "some topic", "irrelevant summary")
        assert saved == []


# ---------------------------------------------------------------------------
# read_input — loitering-event-loop fallback instead of fatal crash
# ---------------------------------------------------------------------------


class TestReadInputFallback:
    def test_running_loop_error_falls_back_to_thread(self, qwen_cli, monkeypatch):
        import qwen_cli.core.repl as repl_mod

        def _boom():
            raise RuntimeError("asyncio.run() cannot be called from a running event loop")

        monkeypatch.setattr(repl_mod, "_read_input_inline", _boom)
        monkeypatch.setattr(repl_mod, "_read_input_in_thread", lambda: "recovered input")
        assert repl_mod.read_input() == "recovered input"

    def test_other_runtime_errors_still_raise(self, qwen_cli, monkeypatch):
        import pytest

        import qwen_cli.core.repl as repl_mod

        def _boom():
            raise RuntimeError("something else entirely")

        monkeypatch.setattr(repl_mod, "_read_input_inline", _boom)
        monkeypatch.setattr(repl_mod, "_read_input_in_thread", lambda: "unused")
        with pytest.raises(RuntimeError, match="something else"):
            repl_mod.read_input()


# ---------------------------------------------------------------------------
# LSP idle tracking — the missing-global bug
# ---------------------------------------------------------------------------


class TestLspIdleTracking:
    def test_ensure_server_refreshes_last_access_global(self, lsp_client, monkeypatch):
        sentinel = object()
        monkeypatch.setattr(lsp_client, "_LSP_SERVER", sentinel)
        monkeypatch.setattr(lsp_client, "_LSP_LANGUAGE", "python")
        monkeypatch.setattr(lsp_client, "_LSP_LAST_ACCESS", 0.0)
        result = lsp_client._ensure_server("whatever.py")
        assert result is sentinel  # reused, not recreated
        # The regression: this stayed 0.0 because the assignment was a local.
        assert lsp_client._LSP_LAST_ACCESS > time.time() - 5

    def test_recent_access_survives_idle_check(self, lsp_client, monkeypatch):
        shutdowns = []
        monkeypatch.setattr(lsp_client, "_LSP_SERVER", object())
        monkeypatch.setattr(lsp_client, "_LSP_LAST_ACCESS", time.time())
        monkeypatch.setattr(lsp_client, "_shutdown_server", lambda s: shutdowns.append(s))
        lsp_client._check_idle_shutdown()
        assert shutdowns == []  # fresh server must NOT be killed
        assert lsp_client._LSP_SERVER is not None

    def test_stale_access_triggers_shutdown(self, lsp_client, monkeypatch):
        shutdowns = []
        monkeypatch.setattr(lsp_client, "_LSP_SERVER", object())
        monkeypatch.setattr(lsp_client, "_LSP_LAST_ACCESS", time.time() - 9999)
        monkeypatch.setattr(lsp_client, "_shutdown_server", lambda s: shutdowns.append(s))
        lsp_client._check_idle_shutdown()
        assert len(shutdowns) == 1


# ---------------------------------------------------------------------------
# Spawned agents (--task) must never block on the auto-approve prompt
# ---------------------------------------------------------------------------


class TestSpawnedAgentAutoApprove:
    def test_task_mode_enables_auto_approve(self, qwen_cli, monkeypatch):
        seen = {}

        def fake_agent(goal, history, base_system, client):
            seen["auto"] = qwen_cli._auto_approve
            seen["goal"] = goal

        monkeypatch.setattr(qwen_cli, "_auto_approve", False)
        with (
            patch.object(qwen_cli, "make_client", return_value="fake_client"),
            patch.object(qwen_cli, "make_aux_client", return_value=None),
            patch("sys.stdin") as mock_stdin,
            patch("sys.argv", ["qwen-cli.py", "--task", "audit the code"]),
            patch.object(qwen_cli, "expand_at_refs", side_effect=lambda t: t),
            patch.object(qwen_cli, "cmd_agent", side_effect=fake_agent),
            patch("threading.Thread"),
        ):
            mock_stdin.isatty.return_value = True
            qwen_cli.main()

        assert seen["goal"] == "audit the code"
        assert seen["auto"] is True
