"""Tests for the visible plan/progress tracking (update_plan tool, /plan command)."""

import pytest

import qwen_cli.main as qmain
from qwen_cli.core.stream import TOOLS


@pytest.fixture(autouse=True)
def _reset_plan():
    """_current_plan is shared module state — isolate each test."""
    qmain._current_plan.clear()
    yield
    qmain._current_plan.clear()


class TestUpdatePlanTool:
    def test_registered_in_tools_schema(self):
        names = [t["function"]["name"] for t in TOOLS]
        assert "update_plan" in names

    def test_schema_requires_steps(self):
        spec = next(t for t in TOOLS if t["function"]["name"] == "update_plan")
        assert spec["function"]["parameters"]["required"] == ["steps"]

    def test_basic_update_stores_plan(self):
        result = qmain.do_update_plan(
            [{"text": "Read the file", "status": "pending"}, {"text": "Patch it", "status": "pending"}]
        )
        assert qmain._current_plan == [
            {"text": "Read the file", "status": "pending"},
            {"text": "Patch it", "status": "pending"},
        ]
        assert "0/2" in result

    def test_status_transitions_and_completion_count(self):
        qmain.do_update_plan([{"text": "Step 1", "status": "in_progress"}])
        result = qmain.do_update_plan([{"text": "Step 1", "status": "completed"}])
        assert qmain._current_plan[0]["status"] == "completed"
        assert "1/1" in result

    def test_empty_list_rejected(self):
        result = qmain.do_update_plan([])
        assert "error" in result
        assert qmain._current_plan == []

    def test_non_list_rejected(self):
        result = qmain.do_update_plan("not a list")
        assert "error" in result

    def test_invalid_status_defaults_to_pending(self):
        qmain.do_update_plan([{"text": "Step 1", "status": "bogus"}])
        assert qmain._current_plan[0]["status"] == "pending"

    def test_missing_text_step_is_dropped(self):
        qmain.do_update_plan([{"text": "  ", "status": "pending"}, {"text": "Real step", "status": "pending"}])
        assert qmain._current_plan == [{"text": "Real step", "status": "pending"}]

    def test_all_steps_missing_text_rejected(self):
        result = qmain.do_update_plan([{"text": "", "status": "pending"}])
        assert "error" in result
        assert qmain._current_plan == []

    def test_replaces_rather_than_appends(self):
        qmain.do_update_plan([{"text": "A", "status": "pending"}, {"text": "B", "status": "pending"}])
        qmain.do_update_plan([{"text": "Only this one", "status": "pending"}])
        assert len(qmain._current_plan) == 1

    def test_dispatched_through_tool_handler(self):
        qmain._current_plan.clear()
        handler = qmain._TOOL_HANDLERS_SAFE["update_plan"]
        result = handler({"steps": [{"text": "Do the thing", "status": "pending"}]})
        assert "0/1" in result
        assert qmain._current_plan[0]["text"] == "Do the thing"

    def test_update_plan_is_parallel_safe(self):
        assert "update_plan" in qmain._PARALLEL_TOOLS


class TestRenderPlanPanel:
    def test_noop_with_no_plan(self):
        qmain._render_plan_panel()  # must not raise

    def test_renders_without_raising(self):
        qmain.do_update_plan(
            [
                {"text": "Done step", "status": "completed"},
                {"text": "Active step", "status": "in_progress"},
                {"text": "Future step", "status": "pending"},
            ]
        )
        qmain._render_plan_panel()  # must not raise
