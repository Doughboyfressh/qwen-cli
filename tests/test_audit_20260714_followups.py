"""Regression tests for the lower-severity half of the 2026-07-14 audit.

1. A dangerous command in an unattended session (piped input, or an agent spawned
   by team_spawn_agent) used to sit on a y/N prompt no one would ever answer —
   the spawned agent hung forever. It is now denied outright: fail closed, fail
   fast, and tell the model to hand the command back to the user.
2. /long stashed the current budget in _TOKEN_LIMIT_BASE — the same var /model
   restores from — so `/model aux` → `/long on` → `/model main` came back to the
   aux-capped 28000 instead of the configured limit for the rest of the session.
3. config.py defaulted token_limit to 32000, leaving ~768 tokens of headroom
   under the -c 49152 / 16384-out split instead of the documented ~4.7k.
"""

from unittest.mock import patch

import pytest

from qwen_cli.tools.files import do_run_command, do_run_script


# ---------------------------------------------------------------------------
# 1. Unattended sessions deny dangerous commands instead of blocking on a prompt
# ---------------------------------------------------------------------------


@pytest.fixture
def unattended(qwen_cli):
    with patch.object(qwen_cli, "_unattended", True):
        yield


def _explode_on_prompt(*a, **k):
    raise AssertionError("prompted for confirmation with no user attached — this is the hang")


def test_unattended_dangerous_command_is_denied_not_prompted(qwen_cli, unattended, tmp_path):
    with patch.object(qwen_cli.console, "input", side_effect=_explode_on_prompt):
        out = do_run_command("rm -rf /", cwd=str(tmp_path))
    assert "blocked" in out.lower()
    assert "not run" in out.lower()


def test_unattended_dangerous_script_is_denied_not_prompted(qwen_cli, unattended, tmp_path):
    with patch.object(qwen_cli.console, "input", side_effect=_explode_on_prompt):
        out = do_run_script("python", "import shutil; shutil.rmtree('/')", cwd=str(tmp_path))
    assert "blocked" in out.lower()
    assert "not run" in out.lower()


def test_unattended_denial_is_audited(qwen_cli, unattended, tmp_path, monkeypatch):
    audit = tmp_path / "audit.log"
    monkeypatch.setattr(qwen_cli, "AUDIT_LOG_FILE", audit)
    with patch.object(qwen_cli.console, "input", side_effect=_explode_on_prompt):
        do_run_command("git push --force", cwd=str(tmp_path))
    assert "denied_unattended" in audit.read_text(encoding="utf-8")


def test_unattended_still_runs_safe_commands(qwen_cli, unattended, tmp_path):
    """The denial is scoped to dangerous commands — ordinary work still proceeds."""
    out = do_run_command("echo hello", cwd=str(tmp_path))
    assert "hello" in out
    assert "blocked" not in out.lower()


def test_attended_session_still_prompts(qwen_cli, tmp_path):
    """With a user attached, the gate is a prompt — not an automatic denial."""
    with (
        patch.object(qwen_cli, "_unattended", False),
        patch.object(qwen_cli.console, "input", return_value="n") as prompt,
    ):
        out = do_run_command("rm -rf /", cwd=str(tmp_path))
    assert prompt.called, "an attended session must still get the y/N decision"
    assert "cancelled by user" in out


def test_attended_prompt_survives_eof(qwen_cli, tmp_path):
    """A closed stdin mid-prompt must decline cleanly, not raise out of the tool."""
    with (
        patch.object(qwen_cli, "_unattended", False),
        patch.object(qwen_cli.console, "input", side_effect=EOFError),
    ):
        out = do_run_command("rm -rf /", cwd=str(tmp_path))
    assert "cancelled by user" in out


def test_run_piped_does_not_latch_unattended_on(qwen_cli, monkeypatch):
    """run_piped sets _unattended; it must put it back.

    Left latched, every later dangerous-command prompt in the same process turns
    into a silent denial — which is exactly how this leak first showed up, as an
    unrelated audit-log test failing only when the suite ran in one process.
    """
    monkeypatch.setattr(qwen_cli, "_unattended", False)
    monkeypatch.setattr(qwen_cli.sys, "stdin", __import__("io").StringIO("hi"))
    monkeypatch.setattr(qwen_cli.sys, "argv", ["qwen"])
    monkeypatch.setattr(qwen_cli, "build_system_prompt", lambda _b: "sys")

    class _Boom:
        class chat:  # noqa: N801
            class completions:  # noqa: N801
                @staticmethod
                def create(**_k):
                    raise RuntimeError("no server")

    with pytest.raises(SystemExit):
        qwen_cli.run_piped(_Boom())

    assert qwen_cli._unattended is False, "run_piped leaked _unattended=True into the process"


# ---------------------------------------------------------------------------
# 2. /long no longer clobbers the budget /model restores from
# ---------------------------------------------------------------------------


def test_long_does_not_clobber_the_model_restore_baseline(qwen_cli, monkeypatch):
    """`/model aux` → `/long on` → `/long off` must return to the aux budget..."""
    monkeypatch.setattr(qwen_cli, "_TOKEN_LIMIT_BASE", 32000)
    monkeypatch.setattr(qwen_cli, "TOKEN_LIMIT", 28000)  # as /model aux leaves it
    monkeypatch.setattr(qwen_cli, "_long_mode", False)
    monkeypatch.setattr(qwen_cli, "_pre_long_token_limit", 0)
    monkeypatch.setattr(qwen_cli, "_model_params", {})

    qwen_cli.cmd_long("on")
    assert qwen_cli._TOKEN_LIMIT_BASE == 32000, "/long must not overwrite the config baseline"

    qwen_cli.cmd_long("off")
    assert qwen_cli.TOKEN_LIMIT == 28000  # back to the aux budget, not the config one
    assert qwen_cli._TOKEN_LIMIT_BASE == 32000  # ...and /model can still restore 32000


def test_long_round_trip_restores_the_configured_budget(qwen_cli, monkeypatch):
    monkeypatch.setattr(qwen_cli, "_TOKEN_LIMIT_BASE", 28000)
    monkeypatch.setattr(qwen_cli, "TOKEN_LIMIT", 28000)
    monkeypatch.setattr(qwen_cli, "_long_mode", False)
    monkeypatch.setattr(qwen_cli, "_pre_long_token_limit", 0)
    monkeypatch.setattr(qwen_cli, "_model_params", {})

    qwen_cli.cmd_long("on")
    assert qwen_cli._model_params["max_tokens"] == qwen_cli._LONG_OUTPUT
    assert qwen_cli.TOKEN_LIMIT < 28000  # input budget shrank to fund the output

    qwen_cli.cmd_long("off")
    assert qwen_cli.TOKEN_LIMIT == 28000
    assert "max_tokens" not in qwen_cli._model_params


# ---------------------------------------------------------------------------
# 3. The default input budget fits the server's -c
# ---------------------------------------------------------------------------


def test_default_token_limit_leaves_real_headroom():
    """The fallback budget (no config.toml, no env) must fit the server's -c.

    Read out of the source rather than by reloading the module: reloading
    core.config re-runs it against the live environment, and with QWEN_DATA_DIR
    gone DATA_DIR would swing back to the user's REAL ~/.qwen-cli — which is the
    exact pollution conftest sets QWEN_DATA_DIR to prevent.
    """
    import re
    from pathlib import Path

    import qwen_cli.core.config as cfg

    source = Path(cfg.__file__).read_text(encoding="utf-8")
    # _pcfg since provider profiles landed: profile value, then top-level, then env.
    m = re.search(r'TOKEN_LIMIT = int\(_p?cfg\("token_limit", "QWEN_TOKEN_LIMIT", "(\d+)"\)\)', source)
    assert m, "token_limit default moved — update this test"
    default = int(m.group(1))

    ctx_window, reserved_output = 49152, 16384  # start-qwen.bat -c / preset max_tokens
    headroom = ctx_window - reserved_output - default
    assert default == 28000
    assert headroom > 4000, f"only {headroom} tokens for tokenizer drift + tool schemas"
