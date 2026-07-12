"""Tests for /long high-output mode (Qwen3's 81,920-token recommendation for hard problems)."""


def test_long_mode_toggle(qwen_cli):
    base = qwen_cli.TOKEN_LIMIT
    default_out = qwen_cli.SAMPLING_PRESETS[qwen_cli._active_preset]["max_tokens"]
    expected_input = max(8192, base + default_out - qwen_cli._LONG_OUTPUT)
    try:
        # ON: output reserved at 81,920; input budget shrinks to match.
        qwen_cli.cmd_long("on")
        assert qwen_cli._long_mode is True
        assert qwen_cli._model_params.get("max_tokens") == qwen_cli._LONG_OUTPUT
        assert qwen_cli.TOKEN_LIMIT == expected_input

        # OFF: override removed, input budget restored.
        qwen_cli.cmd_long("off")
        assert qwen_cli._long_mode is False
        assert "max_tokens" not in qwen_cli._model_params
        assert qwen_cli.TOKEN_LIMIT == base

        # Bare /long toggles on from off.
        qwen_cli.cmd_long("")
        assert qwen_cli._long_mode is True
    finally:
        # Leave the shared module state clean for other tests.
        qwen_cli.cmd_long("off")
        qwen_cli._model_params.clear()


def test_long_output_is_qwen_recommendation(qwen_cli):
    # 81,920 is the documented ceiling for competition-style math/coding.
    assert qwen_cli._LONG_OUTPUT == 81920
    # Default output reservation pairs with the server's -c 49152
    # (49152 - 16384 = 32768 input ceiling); see core/config.py.
    assert all(p["max_tokens"] == 16384 for p in qwen_cli.SAMPLING_PRESETS.values())
