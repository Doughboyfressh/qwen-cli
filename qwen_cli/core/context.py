"""Smart Context Management — importance tagging, snapshots, and predictive compaction.

This module replaces the blunt truncate-middle approach with a layered system that:
1. Tags messages by importance (critical/important/normal/disposable)
2. Preserves critical content during compaction
3. Saves persistent snapshots before trimming so context survives
4. Tracks context growth with trend analysis for early warnings
5. Reports what was kept vs. dropped after compaction
"""

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

from qwen_cli.core.config import CONTEXT_SNAPSHOTS_DIR as _SNAPSHOT_DIR

UTC = timezone.utc  # noqa: UP017 — datetime.UTC exists only on 3.11+; alias keeps 3.10 compat

_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Importance levels and their meaning
# ---------------------------------------------------------------------------

IMPORTANCE_CRITICAL = "critical"  # Never drop: user intent, task definition, key decisions
IMPORTANCE_IMPORTANT = "important"  # Truncate but don't drop: search results, code, tool output
IMPORTANCE_NORMAL = "normal"  # Safe to drop: general conversation, thinking
IMPORTANCE_DISPOSABLE = "disposable"  # Drop first: status, progress, repeated info

# ---------------------------------------------------------------------------
# Snapshot storage
#
# _SNAPSHOT_DIR lives in ~/.qwen-cli/ (like memory.md, sessions/, backups/)
# rather than inside the package source tree — otherwise a reinstall/upgrade
# that replaces the qwen_cli package directory silently loses every snapshot.
# ---------------------------------------------------------------------------


def _ensure_snapshot_dir() -> Path:
    try:
        _SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass  # non-directory path or permission issue — callers handle absence
    return _SNAPSHOT_DIR


# ---------------------------------------------------------------------------
# Message Importance Classifier
# ---------------------------------------------------------------------------

# Patterns that indicate critical content
_CRITICAL_PATTERNS = [
    r"(?i)current\s*task\s*:",  # Explicit current task markers
    r"(?i)TODO(?::|s)?\s",  # TODO items
    r"(?i)must\s+(not\s+)?(do|fix|implement|avoid|remember)",  # Constraints
    r"(?i)requirement[s]?",  # Requirements
    r"(?i)decision[:s]?\s",  # Decisions made
    r"(?i)agreed\s+(to|on|upon)",  # Agreements
    r"(?i)user\s*(instruction|request|says|wants|needs)",  # User intent
    r"(?i)file\s*(modified|changed|created|patched|deleted)",  # File changes
    r"(?i)\[patched:|\[created:|\[updated:|\[write_file]|\[patch_file]",  # Work markers
    r"(?i)error[:s]?\s+.*?(failed|crash|exception|broken)",  # Error reports
    r"(?i)security\s+(vulnerabilit|issue|concern)",  # Security issues
    r"(?i)test\s+(fail|pass|result)",  # Test outcomes
]

# Patterns that indicate disposable content
_DISPOSABLE_PATTERNS = [
    r"(?i)^\[dim\]",  # UI formatting
    r"(?i)^\[no\s+(answer|tasks|messages|changes)",  # Empty responses
    r"(?i)^\[cancelled\]",  # Cancelled operations
    r"(?i)^ok\b\s*$|^got\s*(it|it\.|that|that\.|it!|that!)",  # Bare acknowledgments
    r"(?i)^yes\b\s*$|^no\b\s*$",  # Single-word answers
    r"(?i)progress\s*(update|report|status)",  # Status updates
    r"(?i)^thanks?\b",  # Politeness
    r"(?i)^here\s*(you|we|is|are|'\s*re|'re)\s",  # Introductory phrases
    r"(?i)summar(iz|y)",  # Summaries (redundant after trimming)
    r"(?i)^\[\s*truncated",  # Truncation markers
    r"(?i)^\[\s*\.\.\.\s*{.*}\s*chars\s+condensed",  # Condensation markers
]

# Patterns that indicate important content
_IMPORTANT_PATTERNS = [
    r"(?i)(web_search|fetch_url|search_news|run_command|run_script)[\s:]",  # Tool usage
    r"(?i)(def |class |import |from .+ import)",  # Code snippets
    r"(?i)(result|output|response)[\s:]",  # Tool results
    r"(?i)(version|config|setting|parameter)",  # Configuration details
    r"(?i)(install|update|upgrade|pip|npm|apt)",  # Package operations
    r"(?i)(file|path|url|link)",  # References
    r"(?i)(test |testing |assert |expect)",  # Testing
]


class ImportanceClassifier:
    """Classify message importance based on content patterns and role."""

    def __init__(self) -> None:
        self._critical_re = [re.compile(p) for p in _CRITICAL_PATTERNS]
        self._disposable_re = [re.compile(p) for p in _DISPOSABLE_PATTERNS]
        self._important_re = [re.compile(p) for p in _IMPORTANT_PATTERNS]

    def classify(self, message: dict) -> str:
        """Classify a message's importance level.

        Args:
            message: A chat message dict with 'role' and 'content' keys.

        Returns:
            One of: 'critical', 'important', 'normal', 'disposable'

        """
        role = message.get("role", "")
        content = message.get("content", "") or ""

        # System messages with task markers are critical
        if role == "system":
            if any(r.search(content) for r in self._critical_re):
                return IMPORTANCE_CRITICAL
            # System messages are generally important
            return IMPORTANCE_IMPORTANT

        # User messages with critical patterns
        if role == "user":
            if any(r.search(content) for r in self._critical_re):
                return IMPORTANCE_CRITICAL
            # User messages are generally important (they express intent)
            if len(content.strip()) > 20:  # Substantial user input
                return IMPORTANCE_IMPORTANT
            return IMPORTANCE_NORMAL

        # Assistant messages
        if role == "assistant":
            if any(r.search(content) for r in self._critical_re):
                return IMPORTANCE_CRITICAL
            if any(r.search(content) for r in self._important_re):
                return IMPORTANCE_IMPORTANT
            if any(r.search(content) for r in self._disposable_re):
                return IMPORTANCE_DISPOSABLE
            return IMPORTANCE_NORMAL

        # Tool messages
        if role == "tool":
            # Tool errors are important
            if any(r.search(content) for r in self._critical_re):
                return IMPORTANCE_CRITICAL
            # Recent tool results are important
            if "error" in content.lower() or "fail" in content.lower():
                return IMPORTANCE_IMPORTANT
            return IMPORTANCE_NORMAL

        return IMPORTANCE_NORMAL

    def tag_history(self, history: list) -> list:
        """Tag all messages in history with importance levels.

        Returns a new list with '_importance' key added to each message.
        """
        return [{**msg, "_importance": self.classify(msg)} for msg in history]


# ---------------------------------------------------------------------------
# Context Snapshot System
# ---------------------------------------------------------------------------


class ContextSnapshot:
    """Capture and restore context state before/after compaction."""

    @staticmethod
    def create(history: list, classifier: ImportanceClassifier, session_id: str | None = None) -> dict:
        """Create a snapshot of the current context state.

        Captures:
        - Summary of critical messages
        - File modification log
        - Current task state
        - Key decisions and constraints
        - Token budget at time of snapshot
        """
        timestamp = datetime.now(UTC).isoformat()

        # Extract critical information
        critical_msgs = []
        important_msgs = []
        file_changes = []
        decisions = []
        errors = []

        for msg in history:
            content = msg.get("content", "") or ""
            role = msg.get("role", "")
            importance = classifier.classify(msg)

            if importance == IMPORTANCE_CRITICAL:
                critical_msgs.append(
                    {
                        "role": role,
                        "content": content[:1000],  # Cap length
                    }
                )

            if importance == IMPORTANCE_IMPORTANT:
                important_msgs.append(
                    {
                        "role": role,
                        "content": content[:500],  # Shorter for important
                    }
                )

            # Extract specific types of information
            if any(marker in content for marker in ["[patched:", "[created:", "[updated:"]):
                file_changes.append(content[:300])
            if "decision" in content.lower():
                decisions.append(content[:300])
            if "error" in content.lower() or "fail" in content.lower():
                errors.append(content[:300])

        # Extract current task if present
        current_task = None
        for msg in reversed(history):
            content = msg.get("content", "") or ""
            task_match = re.search(r"(?i)current\s*task\s*:\s*(.+?)(?:\n|$)", content)
            if task_match:
                current_task = task_match.group(1).strip()
                break

        snapshot = {
            "timestamp": timestamp,
            "session_id": session_id or "main",
            "message_count": len(history),
            "critical_messages": critical_msgs,
            "file_changes": file_changes[-10:],  # Last 10 file changes
            "decisions": decisions[-10:],
            "errors": errors[-10:],
            "current_task": current_task,
            "importance_summary": {
                "critical": sum(1 for m in history if classifier.classify(m) == IMPORTANCE_CRITICAL),
                "important": sum(1 for m in history if classifier.classify(m) == IMPORTANCE_IMPORTANT),
                "normal": sum(1 for m in history if classifier.classify(m) == IMPORTANCE_NORMAL),
                "disposable": sum(1 for m in history if classifier.classify(m) == IMPORTANCE_DISPOSABLE),
            },
        }

        # Save to disk
        snapshot_path = _ensure_snapshot_dir() / f"snapshot_{timestamp[:16].replace(':', '').replace('-', '')}.json"
        with snapshot_path.open("w", encoding="utf-8") as f:
            json.dump(snapshot, f, indent=2, ensure_ascii=False)

        return snapshot

    @staticmethod
    def load_recent(session_id: str | None = None, max_count: int = 3) -> list:
        """Load the most recent snapshots.

        Returns list of snapshot dicts, most recent first.
        """
        snapshot_dir = _ensure_snapshot_dir()
        if not snapshot_dir.is_dir():
            return []

        snapshots = sorted(
            snapshot_dir.glob("snapshot_*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        results = []
        for snap_path in snapshots[:max_count]:
            try:
                with snap_path.open(encoding="utf-8") as f:
                    snap = json.load(f)
                if session_id is None or snap.get("session_id") == session_id:
                    results.append(snap)
            except (json.JSONDecodeError, OSError):
                continue

        return results

    @staticmethod
    def build_restore_prompt(snapshots: list) -> str:
        """Build a system prompt from snapshots to restore context.

        Inject this into the system prompt after compaction so the model
        remembers what was lost.
        """
        if not snapshots:
            return ""

        parts = ["=== Context Snapshots (preserved from compaction) ==="]

        for snap in snapshots:
            parts.append(f"\n--- Snapshot: {snap['timestamp']} ---")

            if snap.get("current_task"):
                parts.append(f"Current task: {snap['current_task']}")

            if snap.get("decisions"):
                parts.append("Key decisions:")
                for d in snap["decisions"][-5:]:
                    parts.append(f"  - {d.strip()}")

            if snap.get("file_changes"):
                parts.append("Recent file changes:")
                for fc in snap["file_changes"][-5:]:
                    parts.append(f"  - {fc.strip()}")

            if snap.get("errors"):
                parts.append("Recent errors:")
                for e in snap["errors"][-3:]:
                    parts.append(f"  - {e.strip()}")

        parts.append("\n=== End Snapshots ===")
        return "\n".join(parts)


# ---------------------------------------------------------------------------
# Importance-Aware Truncation
# ---------------------------------------------------------------------------


def importance_aware_truncate(
    history: list, target_reduction: float = 0.3, keep_first: int = 6, keep_last: int = 20
) -> tuple[list, dict]:
    """Truncate history while respecting message importance levels.

    Strategy:
    1. Tag all messages with importance
    2. Drop disposable messages first
    3. Drop normal messages if needed
    4. Truncate (not drop) important messages
    5. Never drop critical messages

    Args:
        history: List of message dicts
        target_reduction: Fraction of context to reduce (0.3 = reduce by 30%)
        keep_first: Minimum messages to keep from the start
        keep_last: Minimum messages to keep from the end

    Returns:
        Tuple of (new_history, stats) where stats reports what was done

    """
    classifier = ImportanceClassifier()
    tagged = classifier.tag_history(history)

    sys_msgs = [m for m in tagged if m.get("role") == "system"]
    chat_msgs = [m for m in tagged if m.get("role") != "system"]

    if len(chat_msgs) <= keep_first + keep_last:
        # Clean up tags
        return [_strip_importance(m) for m in history], {"dropped": 0, "truncated": 0}

    first = chat_msgs[:keep_first]
    last = chat_msgs[-keep_last:]
    middle = chat_msgs[keep_first:-keep_last] if keep_last > 0 else chat_msgs[keep_first:]

    if not middle:
        return [_strip_importance(m) for m in history], {"dropped": 0, "truncated": 0}

    # Categorize middle messages
    disposable = [m for m in middle if m.get("_importance") == IMPORTANCE_DISPOSABLE]
    normal = [m for m in middle if m.get("_importance") == IMPORTANCE_NORMAL]
    important = [m for m in middle if m.get("_importance") == IMPORTANCE_IMPORTANT]
    critical = [m for m in middle if m.get("_importance") == IMPORTANCE_CRITICAL]

    # Start with all messages
    kept_middle = list(middle)
    dropped_disposable = len(disposable)
    dropped_normal = 0

    # Drop disposable messages entirely
    kept_middle = [m for m in kept_middle if m.get("_importance") != IMPORTANCE_DISPOSABLE]

    # If we need more reduction, drop normal messages
    total_chars = sum(len(m.get("content", "") or "") for m in kept_middle)
    target_chars = total_chars * (1 - target_reduction)

    if total_chars > target_chars:
        # Drop normal messages
        normal_to_drop = 0
        for m in reversed(normal):
            if total_chars <= target_chars:
                break
            kept_middle = [k for k in kept_middle if k is not m]
            total_chars -= len(m.get("content", "") or "")
            normal_to_drop += 1
        dropped_normal = normal_to_drop

        # If still over, truncate important messages (don't drop)
        if total_chars > target_chars:
            for _i, m in enumerate(important):
                if total_chars <= target_chars:
                    break
                content = m.get("content", "") or ""
                if len(content) > 200:
                    truncated = content[:200] + f"\n[... {len(content) - 200:,} chars condensed]"
                    m["content"] = truncated
                    total_chars -= len(content) - len(truncated)

    # Reconstruct with critical preserved
    new_middle = kept_middle + critical
    new_middle.sort(key=lambda m: middle.index(m) if m in middle else 999)

    new_history = sys_msgs + first + new_middle + last

    # Add truncation marker
    dropped_total = dropped_disposable + dropped_normal
    if dropped_total > 0:
        marker = {
            "role": "system",
            "content": (
                f"[{dropped_total} messages were removed during context management. "
                f"Disposable: {dropped_disposable}, Normal: {dropped_normal}. "
                f"Critical and important content was preserved.]"
            ),
        }
        new_history = sys_msgs + first + [marker] + new_middle + last

    # Clean up importance tags
    new_history = [_strip_importance(m) for m in new_history]

    stats = {
        "dropped": dropped_total,
        "dropped_by_importance": {
            "disposable": dropped_disposable,
            "normal": dropped_normal,
            "important": 0,  # Never dropped
            "critical": 0,  # Never dropped
        },
        "truncated": sum(1 for m in new_middle if "[... " in (m.get("content") or "")),
        "preserved_critical": len(critical),
    }

    return new_history, stats


def _strip_importance(message: dict) -> dict:
    """Remove _importance tag from a message."""
    return {k: v for k, v in message.items() if k != "_importance"}


# ---------------------------------------------------------------------------
# Growth Tracker with Trend Analysis
# ---------------------------------------------------------------------------


class GrowthTracker:
    """Track context growth and predict when thresholds will be hit."""

    def __init__(self, history_size: int = 50) -> None:
        self._history: list[tuple[float, int]] = []  # (timestamp, token_count)
        self._history_size = history_size

    def record(self, token_count: int) -> None:
        """Record a token count measurement."""
        import time

        self._history.append((time.time(), token_count))
        if len(self._history) > self._history_size:
            self._history.pop(0)

    def get_trend(self) -> dict:
        """Calculate growth trend and predictions.

        Returns dict with:
            - growth_rate: tokens per turn (average)
            - growth_rate_recent: tokens per turn (last 5 turns)
            - linear_fit: (slope, intercept) for token prediction
            - turns_to_80: estimated turns until 80% threshold
            - turns_to_90: estimated turns until 90% threshold
            - is_stable: True if growth rate is near zero
        """
        if len(self._history) < 2:
            return {
                "growth_rate": 0,
                "growth_rate_recent": 0,
                "turns_to_80": -1,
                "turns_to_90": -1,
                "is_stable": True,
            }

        # Calculate growth rates
        all_growths = []
        for i in range(1, len(self._history)):
            growth = self._history[i][1] - self._history[i - 1][1]
            if growth > 0:
                all_growths.append(growth)

        avg_growth = sum(all_growths) // len(all_growths) if all_growths else 0

        # Recent growth (last 5)
        recent = self._history[-5:] if len(self._history) >= 5 else self._history
        recent_growths = []
        for i in range(1, len(recent)):
            growth = recent[i][1] - recent[i - 1][1]
            if growth > 0:
                recent_growths.append(growth)
        recent_avg_growth = sum(recent_growths) // len(recent_growths) if recent_growths else 0

        # Linear regression for prediction
        slope, intercept = self._linear_regression()

        current_tokens = self._history[-1][1]

        # Import TOKEN_LIMIT from main module
        try:
            from qwen_cli import TOKEN_LIMIT
        except ImportError:
            token_limit = 128_000  # Default fallback
        else:
            token_limit = TOKEN_LIMIT

        turns_to_80 = -1
        turns_to_90 = -1

        if avg_growth > 0:
            threshold_80 = token_limit * 0.80
            threshold_90 = token_limit * 0.90

            remaining_80 = max(0, threshold_80 - current_tokens)
            remaining_90 = max(0, threshold_90 - current_tokens)

            turns_to_80 = max(1, remaining_80 // avg_growth) if remaining_80 > 0 else 0
            turns_to_90 = max(1, remaining_90 // avg_growth) if remaining_90 > 0 else 0

        is_stable = avg_growth < 100  # Less than 100 tokens per turn

        return {
            "growth_rate": avg_growth,
            "growth_rate_recent": recent_avg_growth,
            "current_tokens": current_tokens,
            "turns_to_80": turns_to_80,
            "turns_to_90": turns_to_90,
            "is_stable": is_stable,
            "slope": slope,
            "intercept": intercept,
        }

    def _linear_regression(self) -> tuple[float, float]:
        """Simple linear regression on token counts.

        Returns (slope, intercept) where tokens = slope * turn + intercept
        """
        n = len(self._history)
        if n < 2:
            return 0, 0

        # Use turn index (0, 1, 2, ...) as x
        xs = list(range(n))
        ys = [h[1] for h in self._history]

        sum_x = sum(xs)
        sum_y = sum(ys)
        sum_xy = sum(x * y for x, y in zip(xs, ys, strict=False))
        sum_x2 = sum(x * x for x in xs)

        denom = n * sum_x2 - sum_x * sum_x
        if denom == 0:
            return 0, sum_y / n

        slope = (n * sum_xy - sum_x * sum_y) / denom
        intercept = (sum_y - slope * sum_x) / n

        return slope, intercept

    def predict_at_turn(self, turns_ahead: int) -> int:
        """Predict token count N turns from now."""
        if not self._history:
            return 0

        slope, intercept = self._linear_regression()
        current_turn = len(self._history)
        return max(0, int(slope * (current_turn + turns_ahead) + intercept))

    def get_history(self) -> list[int]:
        """Get the raw token count history."""
        return [h[1] for h in self._history]


# ---------------------------------------------------------------------------
# Compaction Report
# ---------------------------------------------------------------------------


def format_compaction_report(stats: dict, before_tokens: int, after_tokens: int) -> str:
    """Format a human-readable report of what happened during compaction.

    Args:
        stats: Stats dict from importance_aware_truncate
        before_tokens: Token count before compaction
        after_tokens: Token count after compaction

    Returns:
        Formatted string report

    """
    freed = before_tokens - after_tokens
    pct_freed = (freed * 100 // before_tokens) if before_tokens > 0 else 0

    lines = [
        "Context compaction complete:",
        f"  Before: {before_tokens:,} tokens",
        f"  After:  {after_tokens:,} tokens",
        f"  Freed:  {freed:,} tokens ({pct_freed}%)",
        f"  Dropped: {stats.get('dropped', 0)} messages",
    ]

    if stats.get("dropped_by_importance"):
        dri = stats["dropped_by_importance"]
        dropped_parts = []
        for level in ["disposable", "normal", "important", "critical"]:
            if dri.get(level, 0) > 0:
                dropped_parts.append(f"{dri[level]} {level}")
        if dropped_parts:
            lines.append(f"    Dropped by importance: {', '.join(dropped_parts)}")

    if stats.get("preserved_critical", 0) > 0:
        lines.append(f"  Preserved critical: {stats['preserved_critical']} messages")

    if stats.get("truncated", 0) > 0:
        lines.append(f"  Truncated: {stats['truncated']} messages (content shortened, not dropped)")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Integration helpers — replace functions in qwen-cli.py
# ---------------------------------------------------------------------------

# Global growth tracker instance
_growth_tracker = GrowthTracker()


def get_growth_tracker() -> GrowthTracker:
    """Get the global growth tracker instance."""
    return _growth_tracker


def track_growth(token_count: int) -> None:
    """Record a token count for growth tracking."""
    _growth_tracker.record(token_count)


def get_growth_trend() -> dict:
    """Get the current growth trend."""
    return _growth_tracker.get_trend()


def predict_turns_to_threshold(threshold_pct: int = 80) -> int:
    """Predict turns until context reaches given percentage of TOKEN_LIMIT."""
    trend = get_growth_trend()
    if threshold_pct == 80:
        return trend.get("turns_to_80", -1)
    if threshold_pct == 90:
        return trend.get("turns_to_90", -1)
    # Calculate for arbitrary threshold
    try:
        from qwen_cli import TOKEN_LIMIT
        token_limit = TOKEN_LIMIT
    except ImportError:
        token_limit = 128_000

    current = trend.get("current_tokens", 0)
    avg_growth = trend.get("growth_rate", 0)
    if avg_growth <= 0:
        return -1

    threshold_tokens = token_limit * threshold_pct // 100
    remaining = max(0, threshold_tokens - current)
    return max(1, remaining // avg_growth)


def should_warn(context_pct: int, trend: dict) -> tuple[bool, str]:
    """Determine if an early warning should be shown based on growth trend.

    Returns (should_warn, warning_message)
    """
    turns_to_80 = trend.get("turns_to_80", -1)

    # If growth is high, warn even at lower percentages
    if turns_to_80 > 0 and turns_to_80 <= 3:
        return True, f"Context growing fast — will reach 80% in ~{turns_to_80} turns"

    # Standard warning at 60%+
    if context_pct >= 60:
        return True, f"Context at {context_pct}% — consider /trim if approaching limit"

    return False, ""


def clean_old_snapshots(keep: int = 10) -> int:
    """Remove old context snapshots, keeping the N most recent.

    Returns the number of snapshots removed.
    """
    snapshot_dir = _ensure_snapshot_dir()
    if not snapshot_dir.is_dir():
        return 0

    snapshots = sorted(
        snapshot_dir.glob("snapshot_*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    removed = 0
    for snap_path in snapshots[keep:]:
        try:
            snap_path.unlink()
            removed += 1
        except OSError:
            _logger.debug("Failed to remove old snapshot %s", snap_path)

    return removed

