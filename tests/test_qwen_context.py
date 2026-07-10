"""Tests for qwen_context — importance tagging, snapshots, compaction, growth tracking."""

import json
import os
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# ImportanceClassifier
# ---------------------------------------------------------------------------


class TestImportanceClassifier:
    def test_system_critical(self, qwen_context):
        msg = {"role": "system", "content": "CURRENT TASK: implement login"}
        assert qwen_context.ImportanceClassifier().classify(msg) == "critical"

    def test_system_default_important(self, qwen_context):
        msg = {"role": "system", "content": "You are a helpful assistant."}
        assert qwen_context.ImportanceClassifier().classify(msg) == "important"

    def test_user_critical(self, qwen_context):
        msg = {"role": "user", "content": "requirement: must support HTTPS"}
        assert qwen_context.ImportanceClassifier().classify(msg) == "critical"

    def test_user_substantial_is_important(self, qwen_context):
        msg = {"role": "user", "content": "Can you help me debug this issue?"}
        assert qwen_context.ImportanceClassifier().classify(msg) == "important"

    def test_user_short_is_normal(self, qwen_context):
        msg = {"role": "user", "content": "yes"}
        assert qwen_context.ImportanceClassifier().classify(msg) == "normal"

    def test_assistant_critical(self, qwen_context):
        msg = {"role": "assistant", "content": "[patched: main.py (2 lines changed)]"}
        assert qwen_context.ImportanceClassifier().classify(msg) == "critical"

    def test_assistant_important(self, qwen_context):
        msg = {"role": "assistant", "content": "web_search: results follow..."}
        assert qwen_context.ImportanceClassifier().classify(msg) == "important"

    def test_assistant_disposable(self, qwen_context):
        msg = {"role": "assistant", "content": "thanks"}
        assert qwen_context.ImportanceClassifier().classify(msg) == "disposable"

    def test_assistant_normal(self, qwen_context):
        msg = {"role": "assistant", "content": "Let me think about that."}
        assert qwen_context.ImportanceClassifier().classify(msg) == "normal"

    def test_tool_with_error_is_important(self, qwen_context):
        msg = {"role": "tool", "content": "operation returned error code 42"}
        assert qwen_context.ImportanceClassifier().classify(msg) == "important"

    def test_tool_critical(self, qwen_context):
        msg = {"role": "tool", "content": "error: task failed completely"}
        assert qwen_context.ImportanceClassifier().classify(msg) == "critical"

    def test_unknown_role_is_normal(self, qwen_context):
        msg = {"role": "unknown", "content": "something"}
        assert qwen_context.ImportanceClassifier().classify(msg) == "normal"

    def test_empty_content(self, qwen_context):
        msg = {"role": "user", "content": ""}
        assert qwen_context.ImportanceClassifier().classify(msg) == "normal"

    def test_none_content(self, qwen_context):
        msg = {"role": "assistant"}
        assert qwen_context.ImportanceClassifier().classify(msg) == "normal"

    def test_tag_history_adds_importance_key(self, qwen_context):
        history = [
            {"role": "system", "content": "be helpful"},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "ok"},
        ]
        tagged = qwen_context.ImportanceClassifier().tag_history(history)
        assert len(tagged) == 3
        for msg in tagged:
            assert "_importance" in msg


# ---------------------------------------------------------------------------
# ContextSnapshot
# ---------------------------------------------------------------------------


class TestContextSnapshot:
    def test_create_returns_snapshot(self, qwen_context, tmp_path):
        classifier = qwen_context.ImportanceClassifier()
        history = [
            {"role": "system", "content": "CURRENT TASK: build feature"},
            {"role": "user", "content": "hello"},
        ]
        snap = qwen_context.ContextSnapshot.create(history, classifier, session_id="test")
        assert snap["timestamp"]
        assert snap["session_id"] == "test"
        assert snap["message_count"] == 2
        assert snap["current_task"] == "build feature"

    def test_create_extracts_critical_and_decisions(self, qwen_context, tmp_path):
        classifier = qwen_context.ImportanceClassifier()
        history = [
            {"role": "user", "content": "decision: use FastAPI"},
            {"role": "assistant", "content": "error: test failed"},
        ]
        snap = qwen_context.ContextSnapshot.create(history, classifier)
        assert len(snap["critical_messages"]) > 0
        assert any("decision" in d.lower() for d in snap["decisions"])
        assert any("error" in e.lower() for e in snap["errors"])

    def test_create_respects_cap_lengths(self, qwen_context, tmp_path):
        classifier = qwen_context.ImportanceClassifier()
        long = "X" * 2000
        history = [{"role": "user", "content": f"requirement: {long}"}]
        snap = qwen_context.ContextSnapshot.create(history, classifier)
        assert len(snap["critical_messages"][0]["content"]) <= 1000

    def test_load_recent_returns_empty_when_no_snapshots(self, qwen_context, monkeypatch):
        monkeypatch.setattr(qwen_context, "_SNAPSHOT_DIR", Path(os.devnull))
        assert qwen_context.ContextSnapshot.load_recent() == []

    def test_build_restore_prompt_empty(self, qwen_context):
        assert qwen_context.ContextSnapshot.build_restore_prompt([]) == ""

    def test_build_restore_prompt_with_snapshot(self, qwen_context):
        snap = {
            "timestamp": "2026-01-01T00:00:00",
            "current_task": "test task",
            "decisions": ["use pytest"],
            "file_changes": ["edited main.py"],
            "errors": ["fix bug"],
        }
        prompt = qwen_context.ContextSnapshot.build_restore_prompt([snap])
        assert "test task" in prompt
        assert "use pytest" in prompt
        assert "edited main.py" in prompt
        assert "fix bug" in prompt


# ---------------------------------------------------------------------------
# importance_aware_truncate
# ---------------------------------------------------------------------------


class TestImportanceAwareTruncate:
    def test_noop_when_history_small(self, qwen_context):
        history = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
        out, stats = qwen_context.importance_aware_truncate(history, keep_first=2, keep_last=2)
        assert stats["dropped"] == 0

    def test_drops_disposable_first(self, qwen_context):
        history = [{"role": "user", "content": "hi"} for _ in range(10)]
        history += [{"role": "assistant", "content": "ok got it"} for _ in range(10)]
        history += [{"role": "assistant", "content": "important result"} for _ in range(5)]
        out, stats = qwen_context.importance_aware_truncate(history, target_reduction=0.5, keep_first=2, keep_last=2)
        assert stats["dropped"] > 0

    def test_preserves_critical(self, qwen_context):
        history = [{"role": "system", "content": "CURRENT TASK: build"}]
        history += [{"role": "user", "content": "x"} for _ in range(50)]
        out, stats = qwen_context.importance_aware_truncate(history, target_reduction=0.5, keep_first=2, keep_last=2)
        assert any("CURRENT TASK" in m.get("content", "") for m in out)

    def test_returns_stats_dict(self, qwen_context):
        history = [{"role": "user", "content": "x"} for _ in range(20)]
        out, stats = qwen_context.importance_aware_truncate(history, target_reduction=0.3, keep_first=3, keep_last=3)
        assert "dropped" in stats
        assert "truncated" in stats
        assert "dropped_by_importance" in stats
        assert "preserved_critical" in stats

    def test_no_importance_tags_leak(self, qwen_context):
        history = [{"role": "user", "content": "x"} for _ in range(20)]
        out, _ = qwen_context.importance_aware_truncate(history, target_reduction=0.3, keep_first=3, keep_last=3)
        for msg in out:
            assert "_importance" not in msg


# ---------------------------------------------------------------------------
# GrowthTracker
# ---------------------------------------------------------------------------


class TestGrowthTracker:
    def test_record_and_get_history(self, qwen_context):
        gt = qwen_context.GrowthTracker()
        gt.record(1000)
        gt.record(2000)
        assert gt.get_history() == [1000, 2000]

    def test_get_trend_insufficient_data(self, qwen_context):
        gt = qwen_context.GrowthTracker()
        trend = gt.get_trend()
        assert trend["is_stable"] is True
        assert trend["growth_rate"] == 0

    def test_get_trend_with_data(self, qwen_context):
        gt = qwen_context.GrowthTracker()
        for v in [1000, 2000, 3000, 4000, 5000]:
            gt.record(v)
        trend = gt.get_trend()
        assert trend["growth_rate"] > 0
        assert trend["growth_rate_recent"] > 0
        assert trend["current_tokens"] == 5000

    def test_get_trend_caps_at_history_size(self, qwen_context):
        gt = qwen_context.GrowthTracker(history_size=3)
        for v in range(10):
            gt.record(v * 1000)
        assert len(gt.get_history()) == 3

    def test_linear_regression(self, qwen_context):
        gt = qwen_context.GrowthTracker()
        for v in [0, 1000, 2000, 3000, 4000]:
            gt.record(v)
        slope, intercept = gt._linear_regression()
        assert slope == pytest.approx(1000, abs=10)
        assert intercept == pytest.approx(0, abs=10)

    def test_predict_at_turn(self, qwen_context):
        gt = qwen_context.GrowthTracker()
        for v in [0, 1000, 2000]:
            gt.record(v)
        pred = gt.predict_at_turn(3)
        assert pred > 0


# ---------------------------------------------------------------------------
# format_compaction_report
# ---------------------------------------------------------------------------


class TestFormatCompactionReport:
    def test_basic_report(self, qwen_context):
        stats = {
            "dropped": 5,
            "dropped_by_importance": {"disposable": 3, "normal": 2},
            "truncated": 1,
            "preserved_critical": 2,
        }
        report = qwen_context.format_compaction_report(stats, 10_000, 7_000)
        assert "10,000" in report
        assert "7,000" in report
        assert "3,000" in report
        assert "30%" in report
        assert "5" in report or "5 messages" in report
        assert "disposable" in report
        assert "2 messages" in report

    def test_no_dropped_by_importance(self, qwen_context):
        stats = {"dropped": 0}
        report = qwen_context.format_compaction_report(stats, 1000, 900)
        assert "Dropped by importance" not in report


# ---------------------------------------------------------------------------
# Integration helpers
# ---------------------------------------------------------------------------


class TestIntegrationHelpers:
    def test_get_growth_tracker_singleton(self, qwen_context):
        gt1 = qwen_context.get_growth_tracker()
        gt2 = qwen_context.get_growth_tracker()
        assert gt1 is gt2

    def test_track_and_get_trend(self, qwen_context):
        qwen_context.track_growth(1000)
        trend = qwen_context.get_growth_trend()
        assert "growth_rate" in trend

    def test_predict_turns_to_threshold_no_data(self, qwen_context):
        result = qwen_context.predict_turns_to_threshold(80)
        assert result == -1 or result >= 0

    def test_should_warn_fast_growth(self, qwen_context):
        trend = {"turns_to_80": 2, "turns_to_90": 5}
        warn, msg = qwen_context.should_warn(50, trend)
        assert warn is True

    def test_should_warn_high_pct(self, qwen_context):
        trend = {"turns_to_80": 20}
        warn, msg = qwen_context.should_warn(65, trend)
        assert warn is True

    def test_should_warn_no_warning(self, qwen_context):
        trend = {"turns_to_80": 20}
        warn, msg = qwen_context.should_warn(30, trend)
        assert warn is False

    def test_clean_old_snapshots_empty(self, qwen_context, tmp_path, monkeypatch):
        monkeypatch.setattr(qwen_context, "_SNAPSHOT_DIR", tmp_path / "no_snaps")
        assert qwen_context.clean_old_snapshots(keep=5) == 0
