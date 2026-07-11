"""Tests for do_run_command's audit logging.

do_run_command previously had zero direct test coverage (test_dangerous.py
only tests the _is_dangerous() regex in isolation). The only prior trace of
what a model actually ran was chat history — no persistent, append-only
record of what ran, when, or from where. These tests exercise the real
subprocess execution path (simple echo/exit commands) with AUDIT_LOG_FILE
redirected to a tmp_path.
"""

import sys


def _patch_audit_log(monkeypatch, qwen_cli, tmp_path):
    monkeypatch.setattr(qwen_cli, "AUDIT_LOG_FILE", tmp_path / "audit.log")
    return tmp_path / "audit.log"


class TestAuditLog:
    def test_successful_command_logs_exit_code(self, qwen_cli, tmp_path, monkeypatch):
        log_path = _patch_audit_log(monkeypatch, qwen_cli, tmp_path)
        qwen_cli.do_run_command(f'{sys.executable} -c "print(1)"', timeout=10)

        assert log_path.exists()
        line = log_path.read_text(encoding="utf-8")
        assert "exit=0" in line
        assert "-c" in line

    def test_failing_command_logs_nonzero_exit_code(self, qwen_cli, tmp_path, monkeypatch):
        log_path = _patch_audit_log(monkeypatch, qwen_cli, tmp_path)
        qwen_cli.do_run_command(f'{sys.executable} -c "import sys; sys.exit(7)"', timeout=10)

        line = log_path.read_text(encoding="utf-8")
        assert "exit=7" in line

    def test_declined_dangerous_command_is_logged(self, qwen_cli, tmp_path, monkeypatch):
        log_path = _patch_audit_log(monkeypatch, qwen_cli, tmp_path)
        monkeypatch.setattr(qwen_cli.console, "input", lambda _prompt: "n")

        result = qwen_cli.do_run_command("rm -rf ./build", timeout=10)

        assert "cancelled" in result
        line = log_path.read_text(encoding="utf-8")
        assert "declined_by_user" in line
        assert "rm -rf ./build" in line

    def test_timeout_is_logged(self, qwen_cli, tmp_path, monkeypatch):
        log_path = _patch_audit_log(monkeypatch, qwen_cli, tmp_path)
        cmd = f'{sys.executable} -c "import time; time.sleep(5)"'
        result = qwen_cli.do_run_command(cmd, timeout=0.2)

        assert "timed out" in result
        line = log_path.read_text(encoding="utf-8")
        assert "timed_out" in line

    def test_log_entry_includes_timestamp_and_cwd(self, qwen_cli, tmp_path, monkeypatch):
        log_path = _patch_audit_log(monkeypatch, qwen_cli, tmp_path)
        qwen_cli.do_run_command(f'{sys.executable} -c "print(1)"', cwd=str(tmp_path), timeout=10)

        line = log_path.read_text(encoding="utf-8")
        assert str(tmp_path) in line
        assert "T" in line  # ISO timestamp

    def test_multiple_commands_append_rather_than_overwrite(self, qwen_cli, tmp_path, monkeypatch):
        log_path = _patch_audit_log(monkeypatch, qwen_cli, tmp_path)
        qwen_cli.do_run_command(f'{sys.executable} -c "print(1)"', timeout=10)
        qwen_cli.do_run_command(f'{sys.executable} -c "print(2)"', timeout=10)

        lines = log_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2

    def test_logging_failure_does_not_break_the_command(self, qwen_cli, tmp_path, monkeypatch):
        # AUDIT_LOG_FILE points at a path whose parent doesn't exist — _audit_log
        # must swallow the write failure rather than propagate it.
        monkeypatch.setattr(qwen_cli, "AUDIT_LOG_FILE", tmp_path / "no_such_dir" / "audit.log")
        result = qwen_cli.do_run_command(f'{sys.executable} -c "print(1)"', timeout=10)

        assert "exit code: 0" in result
