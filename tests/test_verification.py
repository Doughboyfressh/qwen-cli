"""Tests for enforced agent verification and session-change memory logging."""

from pathlib import Path

import pytest

import qwen_cli.main as qmain


class TestVerificationPending:
    def test_no_tools_stays_clear(self):
        assert qmain._verification_pending([]) is False

    def test_mutation_without_verify_is_pending(self):
        assert qmain._verification_pending(["write_file"]) is True
        assert qmain._verification_pending(["patch_file"]) is True
        assert qmain._verification_pending(["delete_file"]) is True
        assert qmain._verification_pending(["move_file"]) is True

    def test_mutation_then_verify_clears(self):
        assert qmain._verification_pending(["write_file", "read_file"]) is False
        assert qmain._verification_pending(["patch_file", "run_command"]) is False
        assert qmain._verification_pending(["patch_file", "run_script"]) is False
        assert qmain._verification_pending(["write_file", "lsp_query"]) is False

    def test_verify_before_mutation_does_not_count(self):
        # read happened BEFORE the write — the write is unverified
        assert qmain._verification_pending(["read_file", "write_file"]) is True

    def test_last_mutation_wins(self):
        assert qmain._verification_pending(["write_file", "run_command", "patch_file"]) is True

    def test_pending_carries_across_turns(self):
        # previous turn left an unverified mutation; this turn only searches the web
        assert qmain._verification_pending(["web_search"], pending=True) is True
        # this turn runs the tests — cleared
        assert qmain._verification_pending(["run_command"], pending=True) is False

    def test_non_tool_noise_ignored(self):
        assert qmain._verification_pending(["web_search", "fetch_url"]) is False


class TestRecordSessionChangesMemory:
    @pytest.fixture(autouse=True)
    def _isolate(self, tmp_path, monkeypatch):
        self.mem_file = tmp_path / "memory.md"
        monkeypatch.setattr(qmain, "MEMORY_FILE", self.mem_file)
        monkeypatch.setattr(qmain, "_session_changes", {})
        self.tmp_path = tmp_path

    def test_noop_when_no_changes(self):
        qmain.record_session_changes_memory(client=None)
        assert not self.mem_file.exists()

    def test_logs_modified_files_without_client(self):
        f = self.tmp_path / "example.py"
        f.write_text("print('new')\n", encoding="utf-8")
        qmain._session_changes[str(f)] = "print('old')\n"

        qmain.record_session_changes_memory(client=None)

        mem = self.mem_file.read_text(encoding="utf-8")
        assert "# Recent Changes" in mem
        assert "example.py" in mem
        assert "modified" in mem

    def test_inserts_under_existing_recent_changes_header(self):
        self.mem_file.write_text(
            "# Identity\n\n- I am a test.\n\n# Recent Changes\n\n- 2020-01-01: old entry\n",
            encoding="utf-8",
        )
        f = self.tmp_path / "mod.py"
        f.write_text("x = 2\n", encoding="utf-8")
        qmain._session_changes[str(f)] = "x = 1\n"

        qmain.record_session_changes_memory(client=None)

        mem = self.mem_file.read_text(encoding="utf-8")
        assert mem.count("# Recent Changes") == 1
        # new entry lands before the old one (most recent first)
        assert mem.index("mod.py") < mem.index("old entry")

    def test_survives_client_failure(self):
        class BoomClient:
            class chat:
                class completions:
                    @staticmethod
                    def create(**kwargs):
                        raise RuntimeError("model down")

        f = self.tmp_path / "robust.py"
        f.write_text("b\n", encoding="utf-8")
        qmain._session_changes[str(f)] = "a\n"

        qmain.record_session_changes_memory(client=BoomClient())

        mem = self.mem_file.read_text(encoding="utf-8")
        assert "robust.py" in mem  # deterministic fallback still wrote the entry

    def test_backup_per_file_cap(self, tmp_path, monkeypatch):
        import time

        monkeypatch.setattr(qmain, "BACKUPS_DIR", tmp_path)
        # 15 backups of "hot.txt" + 2 of "rare.txt", hot.txt newest
        for i in range(2):
            p = tmp_path / f"rare.txt.2026010{i}_000000.bak"
            p.write_text("r", encoding="utf-8")
        time.sleep(0.01)
        for i in range(15):
            p = tmp_path / f"hot.txt.202607{i:02d}_000000.bak"
            p.write_text("h", encoding="utf-8")

        qmain._cleanup_backups(keep=50, keep_per_file=10)

        remaining = [f.name for f in tmp_path.iterdir()]
        assert sum(1 for n in remaining if n.startswith("hot.txt")) == 10  # capped
        assert sum(1 for n in remaining if n.startswith("rare.txt")) == 2  # untouched

    def test_dedup_same_day_same_files(self):
        f = self.tmp_path / "dup.py"
        f.write_text("y\n", encoding="utf-8")
        qmain._session_changes[str(f)] = "x\n"

        qmain.record_session_changes_memory(client=None)
        qmain.record_session_changes_memory(client=None)
        qmain.record_session_changes_memory(client=None)

        mem = self.mem_file.read_text(encoding="utf-8")
        assert mem.count("dup.py") == 1  # logged exactly once
