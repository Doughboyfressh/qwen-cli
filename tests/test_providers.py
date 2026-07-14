"""Any OpenAI-compatible LLM, not just Qwen.

The client was always an OpenAI client, so llama.cpp, Ollama, vLLM, OpenAI,
Anthropic's OpenAI-compatible endpoint, OpenRouter, DeepSeek and Groq all speak
the same protocol. Three things stopped you from actually using them:

1. stream.py did `from ...config import MODEL`, which binds the value at IMPORT.
   Every request carried the configured model no matter what. /model was
   cosmetic, and the documented OpenAI fallback was broken — make_client set
   MODEL = "gpt-4o-mini" while the request still said "Qwen3.6-27B". It only
   looked fine locally because llama.cpp ignores the model field and serves
   whatever is loaded.
2. The primary client was hardcoded to api_key="no-key", so an authenticated
   provider could never be the main backend, only an emergency fallback.
3. top_k / min_p / repeat_penalty / chat_template_kwargs are llama.cpp
   extensions. Cloud APIs reject unknown fields outright.
"""

from unittest.mock import patch

import pytest

from qwen_cli.core.config import _is_local
from qwen_cli.core.stream import _build_stream_kwargs

MSGS = [{"role": "user", "content": "hi"}]


# ---------------------------------------------------------------------------
# The model actually sent is the live one
# ---------------------------------------------------------------------------


def test_request_carries_the_live_model_not_the_import_time_constant(qwen_cli):
    """The bug that made /model cosmetic and broke the cloud fallback."""
    with patch.object(qwen_cli, "MODEL", "claude-sonnet-4-5"):
        kwargs = _build_stream_kwargs(MSGS, use_tools=False)
    assert kwargs["model"] == "claude-sonnet-4-5"


def test_switching_model_changes_the_next_request(qwen_cli):
    with patch.object(qwen_cli, "MODEL", "gpt-4o"):
        assert _build_stream_kwargs(MSGS, use_tools=False)["model"] == "gpt-4o"
    with patch.object(qwen_cli, "MODEL", "Qwen3.6-27B"):
        assert _build_stream_kwargs(MSGS, use_tools=False)["model"] == "Qwen3.6-27B"


# ---------------------------------------------------------------------------
# llama.cpp-only samplers are not sent to providers that reject them
# ---------------------------------------------------------------------------


def test_local_backend_gets_the_llama_cpp_samplers(qwen_cli):
    with patch.object(qwen_cli, "SAMPLER_EXTRAS", True):
        kwargs = _build_stream_kwargs(MSGS, use_tools=False)
    assert "top_k" in kwargs["extra_body"]


def test_cloud_backend_gets_no_extra_body(qwen_cli):
    """OpenAI/Anthropic 400 on unknown fields — top_k, min_p, repeat_penalty,
    chat_template_kwargs are llama.cpp extensions and must not be sent."""
    with patch.object(qwen_cli, "SAMPLER_EXTRAS", False):
        kwargs = _build_stream_kwargs(MSGS, use_tools=False)
    assert "extra_body" not in kwargs
    # the standard fields still go, since every provider understands them
    for k in ("temperature", "top_p", "max_tokens", "presence_penalty"):
        assert k in kwargs


def test_presets_still_define_the_samplers(qwen_cli):
    """Gating is about what we SEND. The presets themselves are unchanged, so
    switching back to a local backend restores them."""
    from qwen_cli.core.config import SAMPLING_PRESETS

    assert "top_k" in SAMPLING_PRESETS["thinking"]["extra_body"]


# ---------------------------------------------------------------------------
# Provider profiles
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("url", "local"),
    [
        ("http://localhost:8080/v1", True),
        ("http://127.0.0.1:8080/v1", True),
        ("https://api.anthropic.com/v1/", False),
        ("https://api.openai.com/v1", False),
        ("https://openrouter.ai/api/v1", False),
    ],
)
def test_is_local_decides_the_sampler_default(url, local):
    assert _is_local(url) is local


def test_profile_overrides_top_level_config(qwen_cli, monkeypatch):
    import qwen_cli.core.config as config

    monkeypatch.setattr(
        config,
        "_CFG",
        {
            "base_url": "http://localhost:8080/v1",
            "model": "Qwen3.6-27B",
            "providers": {"claude": {"base_url": "https://api.anthropic.com/v1/", "model": "claude-sonnet-4-5"}},
        },
    )
    monkeypatch.setattr(config, "_P", config._provider_cfg("claude"))

    assert config._pcfg("model", "QWEN_MODEL", "x") == "claude-sonnet-4-5"
    assert config._pcfg("base_url", "QWEN_BASE_URL", "x") == "https://api.anthropic.com/v1/"
    # a key the profile does not set still falls through to the top level
    assert config._pcfg("editor", "EDITOR", "nano") == "nano"


def test_no_profile_falls_back_to_top_level(qwen_cli, monkeypatch):
    """Existing config.toml files, which have no [providers] table, keep working."""
    import qwen_cli.core.config as config

    monkeypatch.setattr(config, "_CFG", {"model": "Qwen3.6-27B"})
    monkeypatch.setattr(config, "_P", {})
    assert config._pcfg("model", "QWEN_MODEL", "default") == "Qwen3.6-27B"


# ---------------------------------------------------------------------------
# /provider switches model, budget and samplers together
# ---------------------------------------------------------------------------


def test_provider_switch_moves_everything_as_a_set(qwen_cli, monkeypatch):
    """They must move together: a request carrying one backend's model name and
    another's sampler fields is exactly the broken-fallback bug, rebuilt."""
    from qwen_cli.core.commands import _cmd_provider
    from qwen_cli.core.repl import _ReplContext
    import qwen_cli.core.config as config

    monkeypatch.setattr(
        config,
        "_CFG",
        {
            "providers": {
                "claude": {
                    "base_url": "https://api.anthropic.com/v1/",
                    "api_key": "sk-ant-test",
                    "model": "claude-sonnet-4-5",
                    "token_limit": 180000,
                }
            }
        },
    )
    ctx = _ReplContext([], "sys", object())

    with (
        patch.object(qwen_cli, "make_client", return_value="claude_client"),
        patch.object(qwen_cli, "MODEL", "Qwen3.6-27B"),
        patch.object(qwen_cli, "BASE_URL", "http://localhost:8080/v1"),
        patch.object(qwen_cli, "API_KEY", "no-key"),
        patch.object(qwen_cli, "SAMPLER_EXTRAS", True),
        patch.object(qwen_cli, "TOKEN_LIMIT", 28000),
        patch.object(qwen_cli, "ACTIVE_PROVIDER", ""),
        patch.object(qwen_cli, "_model_params", {"max_tokens": 16384}),
    ):
        _cmd_provider(ctx, "claude")

        assert qwen_cli.MODEL == "claude-sonnet-4-5"
        assert qwen_cli.BASE_URL == "https://api.anthropic.com/v1/"
        assert qwen_cli.API_KEY == "sk-ant-test"
        assert qwen_cli.TOKEN_LIMIT == 180000
        assert qwen_cli.SAMPLER_EXTRAS is False, "llama.cpp samplers must not go to a cloud provider"
        assert "max_tokens" not in qwen_cli._model_params, "a local output cap is wrong for a 200k model"
        assert ctx.client == "claude_client"


def test_unknown_provider_is_rejected(qwen_cli, monkeypatch):
    from qwen_cli.core.commands import _cmd_provider
    from qwen_cli.core.repl import _ReplContext
    import qwen_cli.core.config as config

    monkeypatch.setattr(config, "_CFG", {"providers": {}})
    ctx = _ReplContext([], "sys", object())
    before = qwen_cli.MODEL

    with patch.object(qwen_cli, "make_client") as mk:
        _cmd_provider(ctx, "nope")

    assert not mk.called, "an unknown provider must not tear down the working client"
    assert qwen_cli.MODEL == before


# ---------------------------------------------------------------------------
# A remote provider that fails must NOT silently become OpenAI
# ---------------------------------------------------------------------------


def test_remote_provider_failure_does_not_fall_back_to_openai(qwen_cli):
    """Falling back would bill the wrong account and answer as the wrong model.
    A bad key or a typo'd URL is a real failure and must be reported, not papered
    over. The local-server fallback is different: that one is expected and cheap."""
    broken = type("C", (), {"models": type("M", (), {"list": staticmethod(lambda: (_ for _ in ()).throw(OSError()))})})

    with (
        patch.object(qwen_cli, "BASE_URL", "https://api.anthropic.com/v1/"),
        patch.object(qwen_cli, "OPENAI_API_KEY", "sk-openai-key"),
        patch.object(qwen_cli, "MODEL", "claude-sonnet-4-5"),
        patch.object(qwen_cli, "_get_openai", return_value=lambda **_kw: broken()),
    ):
        qwen_cli.make_client()
        assert qwen_cli.MODEL == "claude-sonnet-4-5", "must not silently switch to the OpenAI fallback model"


def test_max_tool_depth_is_per_provider(qwen_cli, monkeypatch):
    """Rounds only pay off if the findings survive them. On a 28k window more
    rounds mean more mid-run compaction — which is what shredded a live audit's
    measurements. On a 200k model they are nearly free. So it belongs in the
    profile, not in one global default."""
    from qwen_cli.core.commands import _cmd_provider
    from qwen_cli.core.repl import _ReplContext
    import qwen_cli.core.config as config

    monkeypatch.setattr(
        config,
        "_CFG",
        {"providers": {"big": {"base_url": "https://api.anthropic.com/v1/", "model": "x", "max_tool_depth": 40}}},
    )
    ctx = _ReplContext([], "sys", object())

    with (
        patch.object(qwen_cli, "make_client", return_value="c"),
        patch.object(qwen_cli, "MAX_TOOL_DEPTH", 20),
    ):
        _cmd_provider(ctx, "big")
        assert qwen_cli.MAX_TOOL_DEPTH == 40
