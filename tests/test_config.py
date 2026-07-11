"""Tests for qwen_cli.core.config — TOML loading, validation warnings, and
config > env > default precedence. This module is load-bearing infrastructure
(every constant in the app flows through it) but had no dedicated coverage."""

import qwen_cli.core.config as config


# ==============================================================================
# _validate_config — warnings go to stderr, never raise
# ==============================================================================


class TestValidateConfig:
    def test_valid_config_is_silent(self, capsys):
        config._validate_config(
            {
                "base_url": "http://localhost:8080/v1",
                "token_limit": 28000,
                "auto_search": "smart",
                "preset": "code",
            }
        )
        assert capsys.readouterr().err == ""

    def test_unknown_key_warns_with_suggestions(self, capsys):
        config._validate_config({"token_limt": 28000})  # typo
        err = capsys.readouterr().err
        assert "unknown key 'token_limt'" in err
        assert "token_limit" in err  # closest known key is suggested

    def test_wrong_type_warns(self, capsys):
        config._validate_config({"token_limit": "lots"})
        assert "token_limit='lots' is invalid" in capsys.readouterr().err

    def test_non_positive_token_limit_warns(self, capsys):
        config._validate_config({"token_limit": 0})
        assert "token_limit=0 is invalid" in capsys.readouterr().err

    def test_negative_max_auto_continue_warns(self, capsys):
        config._validate_config({"max_auto_continue": -1})
        assert "max_auto_continue=-1 is invalid" in capsys.readouterr().err

    def test_zero_max_auto_continue_is_valid(self, capsys):
        config._validate_config({"max_auto_continue": 0})
        assert capsys.readouterr().err == ""

    def test_bad_auto_search_value_warns(self, capsys):
        config._validate_config({"auto_search": "always"})
        err = capsys.readouterr().err
        assert "auto_search='always' is invalid" in err
        assert "off, smart, or aggressive" in err

    def test_bad_preset_value_warns(self, capsys):
        config._validate_config({"preset": "turbo"})
        err = capsys.readouterr().err
        assert "preset='turbo' is invalid" in err
        assert "thinking, code, or instruct" in err

    def test_never_raises_on_garbage(self):
        config._validate_config({"token_limit": None, "preset": 42, "auto_search": [1]})


# ==============================================================================
# _cfg — precedence: config.toml > env var > default
# ==============================================================================


class TestCfgPrecedence:
    def test_config_value_wins_over_env(self, monkeypatch):
        monkeypatch.setattr(config, "_CFG", {"model": "from-toml"})
        monkeypatch.setenv("QWEN_TEST_MODEL", "from-env")
        assert config._cfg("model", "QWEN_TEST_MODEL", "from-default") == "from-toml"

    def test_env_value_wins_over_default(self, monkeypatch):
        monkeypatch.setattr(config, "_CFG", {})
        monkeypatch.setenv("QWEN_TEST_MODEL", "from-env")
        assert config._cfg("model", "QWEN_TEST_MODEL", "from-default") == "from-env"

    def test_default_used_when_nothing_set(self, monkeypatch):
        monkeypatch.setattr(config, "_CFG", {})
        monkeypatch.delenv("QWEN_TEST_MODEL", raising=False)
        assert config._cfg("model", "QWEN_TEST_MODEL", "from-default") == "from-default"

    def test_result_is_always_str(self, monkeypatch):
        monkeypatch.setattr(config, "_CFG", {"token_limit": 28000})
        assert config._cfg("token_limit", "QWEN_TEST_TL", "32000") == "28000"


# ==============================================================================
# _load_config — must never raise, whatever is on disk
# ==============================================================================


class TestLoadConfig:
    def test_missing_file_returns_empty(self, monkeypatch, tmp_path):
        monkeypatch.setattr(config, "CONFIG_FILE", tmp_path / "nope.toml")
        assert config._load_config() == {}

    def test_malformed_toml_returns_empty(self, monkeypatch, tmp_path):
        bad = tmp_path / "config.toml"
        bad.write_text("this is [not valid toml", encoding="utf-8")
        monkeypatch.setattr(config, "CONFIG_FILE", bad)
        assert config._load_config() == {}

    def test_valid_toml_is_parsed(self, monkeypatch, tmp_path):
        good = tmp_path / "config.toml"
        good.write_text('model = "test-model"\ntoken_limit = 12345\n', encoding="utf-8")
        monkeypatch.setattr(config, "CONFIG_FILE", good)
        loaded = config._load_config()
        assert loaded == {"model": "test-model", "token_limit": 12345}


# ==============================================================================
# SAMPLING_PRESETS — shape contract relied on by /preset and run_turn
# ==============================================================================


class TestSamplingPresets:
    def test_all_documented_presets_exist(self):
        assert set(config.SAMPLING_PRESETS) == {"thinking", "code", "instruct"}

    def test_presets_have_required_request_params(self):
        for name, p in config.SAMPLING_PRESETS.items():
            for key in ("temperature", "top_p", "presence_penalty", "max_tokens", "extra_body"):
                assert key in p, f"preset {name!r} missing {key!r}"
            assert "top_k" in p["extra_body"], f"preset {name!r} missing extra_body.top_k"

    def test_thinking_presets_preserve_reasoning(self):
        # thinking + code run with preserve_thinking so the server keeps
        # reasoning across turns; instruct must NOT send it.
        for name in ("thinking", "code"):
            kwargs = config.SAMPLING_PRESETS[name]["extra_body"].get("chat_template_kwargs", {})
            assert kwargs.get("preserve_thinking") is True, name
        assert "chat_template_kwargs" not in config.SAMPLING_PRESETS["instruct"]["extra_body"]

    def test_auto_search_mode_is_normalized(self):
        assert config.AUTO_SEARCH_MODE in ("off", "smart", "aggressive")
