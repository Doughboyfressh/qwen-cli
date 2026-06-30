"""Tests for the per-cwd git-context cache.

build_system_prompt() runs every turn and used to spawn up to 4 git subprocesses
each time. get_git_context() now caches per-cwd with a short TTL so back-to-back
turns reuse the result; _invalidate_git_cache() forces a refresh.
"""


def test_git_context_caches_within_ttl(qwen_cli, monkeypatch):
    calls = {"n": 0}

    def fake_compute():
        calls["n"] += 1
        return f"Branch: main (compute #{calls['n']})"

    qwen_cli._invalidate_git_cache()
    monkeypatch.setattr(qwen_cli, "_compute_git_context", fake_compute)

    first = qwen_cli.get_git_context()
    second = qwen_cli.get_git_context()
    third = qwen_cli.get_git_context()

    # Only computed once despite three calls — the subprocesses are not respawned.
    assert calls["n"] == 1
    assert first == second == third


def test_invalidate_forces_recompute(qwen_cli, monkeypatch):
    calls = {"n": 0}

    def fake_compute():
        calls["n"] += 1
        return f"state-{calls['n']}"

    qwen_cli._invalidate_git_cache()
    monkeypatch.setattr(qwen_cli, "_compute_git_context", fake_compute)

    assert qwen_cli.get_git_context() == "state-1"
    qwen_cli._invalidate_git_cache()           # e.g. just committed
    assert qwen_cli.get_git_context() == "state-2"
    assert calls["n"] == 2


def test_ttl_expiry_recomputes(qwen_cli, monkeypatch):
    calls = {"n": 0}
    clock = {"t": 1000.0}

    monkeypatch.setattr(qwen_cli, "_compute_git_context",
                        lambda: f"v{(calls.__setitem__('n', calls['n'] + 1)) or calls['n']}")
    monkeypatch.setattr(qwen_cli.time, "monotonic", lambda: clock["t"])

    qwen_cli._invalidate_git_cache()
    qwen_cli.get_git_context()                 # computes at t=1000
    clock["t"] += qwen_cli._GIT_CTX_TTL + 1    # advance past the TTL
    qwen_cli.get_git_context()                 # recomputes
    assert calls["n"] == 2
