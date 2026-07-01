"""Script to add missing type hints to qwen-cli.py"""
import re

filepath = r"C:\Users\Dough\.qwen-cli\qwen-cli.py"
with open(filepath, "r", encoding="utf-8") as f:
    lines = f.readlines()

# Define the fixes: (line_number_1indexed, old_line, new_line)
fixes = [
    # save_memory:1469
    (1469, "def save_memory(text: str):", "def save_memory(text: str) -> None:"),
    # save_session:1930
    (1930, "def save_session(history: list, system_prompt: str, name: str | None = None):",
     "def save_session(history: list, system_prompt: str, name: str | None = None) -> None:"),
    # _generate_handoff:1971
    (1971, "def _generate_handoff(client, history, base_system):",
     "def _generate_handoff(client, history, base_system) -> str:"),
    # _write_handoff:1996
    (1996, "def _write_handoff(summary, history):",
     "def _write_handoff(summary, history) -> None:"),
    # _save_exit_handoff:2016
    (2016, "def _save_exit_handoff(history):",
     "def _save_exit_handoff(history) -> None:"),
    # _consume_handoff:2043
    (2043, "def _consume_handoff():",
     "def _consume_handoff() -> dict | None:"),
    # walk:2731
    (2731, "    def walk(path: Path, depth: int, prefix: str):",
     "    def walk(path: Path, depth: int, prefix: str) -> None:"),
    # _bg_index:2787
    (2787, "    def _bg_index():",
     "    def _bg_index() -> None:"),
    # _get_page:3220
    (3220, "def _get_page():",
     "def _get_page() -> Any:"),
    # _browser_resolve_selector:3278
    (3278, "def _browser_resolve_selector(page, selector: str):",
     "def _browser_resolve_selector(page, selector: str) -> Any:"),
    # _get_render_page:3532
    (3532, "def _get_render_page():",
     "def _get_render_page() -> Any:"),
    # _pipe:3703
    (3703, "        def _pipe(stream, buf: list[str], style: str):",
     "        def _pipe(stream, buf: list[str], style: str) -> None:"),
    # _walk_files:4032
    (4032, "        def _walk_files(root: Path):",
     "        def _walk_files(root: Path) -> Iterator[Path]:"),
    # get_completions:4274
    (4274, "        def get_completions(self, document, complete_event):",
     "        def get_completions(self, document, complete_event) -> Iterator[_PtCompletion]:"),
    # _close_loitering_event_loop:4309
    (4309, "def _close_loitering_event_loop():",
     "def _close_loitering_event_loop() -> None:"),
    # _prompt_thread:4355
    (4355, "    def _prompt_thread():",
     "    def _prompt_thread() -> None:"),
    # _live_updater:4394
    (4394, "def _live_updater(live: Live, max_lines: int = _LIVE_PREVIEW_LINES):",
     "def _live_updater(live: Live, max_lines: int = _LIVE_PREVIEW_LINES) -> Callable[[str], None]:"),
    # update:4396
    (4396, "        def update(text: str):",
     "        def update(text: str) -> None:"),
    # _create_with_retry:4407
    (4407, "def _create_with_retry(client: OpenAI, **kwargs):",
     "def _create_with_retry(client: OpenAI, **kwargs) -> Any:"),
    # _target:5392
    (5392, "    def _target():",
     "    def _target() -> None:"),
    # __init__:6217
    (6217, "    def __init__(self, history: list[dict], base_system: str, client):",
     "    def __init__(self, history: list[dict], base_system: str, client) -> None:"),
    # _ensure_lsp:6725
    (6725, "    def _ensure_lsp():",
     "    def _ensure_lsp() -> None:"),
    # main:7235
    (7235, "def main():",
     "def main() -> None:"),
    # _cleanup_watch:7369
    (7369, "def _cleanup_watch():",
     "def _cleanup_watch() -> None:"),
]

# Apply fixes
for lineno, old, new in fixes:
    idx = lineno - 1
    if idx < len(lines):
        actual = lines[idx].rstrip("\n")
        if actual == old:
            lines[idx] = new + "\n"
            print(f"  Fixed line {lineno}: {lines[idx].strip()}")
        else:
            print(f"  WARNING: Line {lineno} mismatch!")
            print(f"    expected: {old!r}")
            print(f"    actual:   {actual!r}")
    else:
        print(f"  ERROR: Line {lineno} out of range (file has {len(lines)} lines)")

# Check if we need to add imports
content = "".join(lines)
needs_any = "def _get_page() -> Any:" in content or "def _browser_resolve_selector(page, selector: str) -> Any:" in content or "def _get_render_page() -> Any:" in content or "def _create_with_retry(client: OpenAI, **kwargs) -> Any:" in content
needs_iterator = "Iterator[Path]" in content or "Iterator[_PtCompletion]" in content
needs_callable = "Callable[[str], None]" in content

if needs_any and "from typing import" in content:
    # Check if Any is already imported
    for line in lines[:50]:
        if "from typing import" in line and "Any" in line:
            needs_any = False
            break
if needs_iterator and "from typing import" in content:
    for line in lines[:50]:
        if "from typing import" in line and "Iterator" in line:
            needs_iterator = False
            break
if needs_callable and "from typing import" in content:
    for line in lines[:50]:
        if "from typing import" in line and "Callable" in line:
            needs_callable = False
            break

# Add missing imports if needed
for line in lines[:50]:
    if "from typing import" in line:
        import_line_idx = lines.index(line)
        current_imports = line.strip()
        to_add = []
        if needs_any and "Any" not in current_imports:
            to_add.append("Any")
        if needs_iterator and "Iterator" not in current_imports:
            to_add.append("Iterator")
        if needs_callable and "Callable" not in current_imports:
            to_add.append("Callable")
        if to_add:
            # Add to existing import
            # Remove trailing ) or whitespace
            stripped = current_imports.rstrip()
            if stripped.endswith("):"):
                stripped = stripped[:-2]
                lines[import_line_idx] = stripped + ", " + ", ".join(to_add) + "):\n"
            elif stripped.endswith("("):
                lines[import_line_idx] = stripped + ", ".join(to_add) + "):\n"
            else:
                # Single line import
                if stripped.endswith(","):
                    lines[import_line_idx] = stripped + " " + ", ".join(to_add) + "\n"
                else:
                    lines[import_line_idx] = stripped + ", " + ", ".join(to_add) + "\n"
            print(f"  Added imports: {', '.join(to_add)} to line {import_line_idx+1}")
        break

# Write back
with open(filepath, "w", encoding="utf-8") as f:
    f.writelines(lines)

print("Done! Applied fixes.")
