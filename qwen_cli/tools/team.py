"""Team coordination — clawteam management, tasks, inbox, spawn."""

import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

import qwen_cli.core.config as _config

UTC = timezone.utc  # noqa: UP017 — datetime.UTC exists only on 3.11+; alias keeps 3.10 compat

_logger = logging.getLogger("qwen.team")


def _ct_team_dir(team: str) -> Path:
    d = _config.CT_DIR / "teams" / team
    d.mkdir(parents=True, exist_ok=True)
    return d


def _ct_tasks_dir(team: str) -> Path:
    d = _config.CT_DIR / "tasks" / team
    d.mkdir(parents=True, exist_ok=True)
    return d


def _ct_inbox_dir(team: str, agent: str) -> Path:
    d = _ct_team_dir(team) / "inboxes" / agent
    d.mkdir(parents=True, exist_ok=True)
    return d


def _ct_now() -> str:
    return datetime.now(UTC).isoformat()


def _ct_atomic_write(path: Path, data: str) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.stem + "-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(data)
        Path(tmp).replace(path)
    except Exception:
        Path(tmp).unlink(missing_ok=True)
        raise


def _ct_load_team(team: str) -> dict | None:
    cfg = _ct_team_dir(team) / "config.json"
    if not cfg.exists():
        return None
    try:
        return json.loads(cfg.read_text(encoding="utf-8"))
    except Exception:
        return None


def _ct_save_team(team: str, config: dict) -> None:
    cfg = _ct_team_dir(team) / "config.json"
    _ct_atomic_write(cfg, json.dumps(config, indent=2, ensure_ascii=False))


def _ct_team_create(team: str, description: str = "") -> dict:
    existing = _ct_load_team(team)
    if existing:
        return existing
    config = {
        "name": team,
        "description": description,
        "leadAgentId": "",
        "createdAt": _ct_now(),
        "members": [],
        "budgetCents": 0.0,
    }
    _ct_save_team(team, config)
    return config


def _ct_team_list() -> list[str]:
    teams_dir = _config.CT_DIR / "teams"
    if not teams_dir.exists():
        return []
    return sorted(d.name for d in teams_dir.iterdir() if d.is_dir() and (d / "config.json").exists())


def _ct_team_join(team: str, agent_name: str, agent_type: str = "general-purpose") -> dict:
    config = _ct_load_team(team) or _ct_team_create(team)
    for m in config["members"]:
        if m["name"] == agent_name:
            return m
    member = {
        "name": agent_name,
        "user": "",
        "agentId": uuid.uuid4().hex[:12],
        "agentType": agent_type,
        "joinedAt": _ct_now(),
    }
    config["members"].append(member)
    _ct_save_team(team, config)
    return member


def _ct_task_add(team: str, subject: str, owner: str = "", description: str = "", priority: str = "medium") -> dict:
    task_id = uuid.uuid4().hex[:8]
    task = {
        "id": task_id,
        "subject": subject,
        "description": description,
        "status": "pending",
        "owner": owner,
        "priority": priority,
        "blocks": [],
        "blocked_by": [],
        "metadata": {},
        "locked_by": "",
        "locked_at": "",
        "started_at": "",
        "created_at": _ct_now(),
        "updated_at": _ct_now(),
    }
    _ct_atomic_write(_ct_tasks_dir(team) / f"task-{task_id}.json", json.dumps(task, indent=2, ensure_ascii=False))
    return task


def _ct_task_list(team: str, owner: str = "", status: str = "") -> list[dict]:
    tasks = []
    for f in sorted(_ct_tasks_dir(team).glob("task-*.json")):
        try:
            t = json.loads(f.read_text(encoding="utf-8"))
            if owner and t.get("owner", "") != owner:
                continue
            if status and t.get("status", "") != status:
                continue
            tasks.append(t)
        except Exception:
            _logger.debug("Skipping unreadable task file %s", f)
            continue
    return tasks


def _ct_task_update(
    team: str, task_id: str, status: str | None = None, owner: str | None = None, note: str = ""
) -> dict | None:
    tasks_dir = _ct_tasks_dir(team)
    path = tasks_dir / f"task-{task_id}.json"
    if not path.exists():
        matches = list(tasks_dir.glob(f"task-{task_id}*.json"))
        if not matches:
            return None
        path = matches[0]
    try:
        task = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if status:
        task["status"] = status
        if status == "in_progress" and not task.get("started_at"):
            task["started_at"] = _ct_now()
    if owner is not None:
        task["owner"] = owner
    if note:
        task.setdefault("notes", []).append({"text": note, "at": _ct_now()})
    task["updated_at"] = _ct_now()
    _ct_atomic_write(path, json.dumps(task, indent=2, ensure_ascii=False))
    return task


def _ct_inbox_send(team: str, to_agent: str, message: str, from_agent: str = "user") -> str:
    msg_id = uuid.uuid4().hex[:12]
    msg = {
        "id": msg_id,
        "type": "message",
        "from": from_agent,
        "to": to_agent,
        "subject": message[:80],
        "body": message,
        "team": team,
        "sentAt": _ct_now(),
        "read": False,
    }
    _ct_atomic_write(
        _ct_inbox_dir(team, to_agent) / f"msg-{msg_id}.json", json.dumps(msg, indent=2, ensure_ascii=False)
    )
    return msg_id


def _ct_inbox_receive(team: str, agent: str, peek: bool = False) -> list[dict]:
    messages = []
    for f in sorted(_ct_inbox_dir(team, agent).glob("msg-*.json")):
        try:
            if peek:
                msg = json.loads(f.read_text(encoding="utf-8"))
                messages.append(msg)
            else:
                claimed = f.with_suffix(".consuming")
                f.rename(claimed)
                try:
                    msg = json.loads(claimed.read_text(encoding="utf-8"))
                    messages.append(msg)
                finally:
                    claimed.unlink(missing_ok=True)
        except (FileNotFoundError, FileExistsError):
            continue
        except Exception:
            _logger.debug("Skipping unreadable message file %s", f)
            continue
    return messages


def _ct_board_render(team: str) -> str:
    config = _ct_load_team(team)
    if not config:
        return f"[team not found: {team}]"
    tasks = _ct_task_list(team)
    lines = [f"Team: {config.get('name', team)}"]
    if config.get("description"):
        lines.append(f"  {config['description']}")
    members = config.get("members", [])
    lines.append(f"\nMembers ({len(members)}):")
    for m in members:
        lines.append(f"  {m.get('name', '?')}  ({m.get('agentType', 'agent')})")
    lines.append(f"\nTasks ({len(tasks)}):")
    by_status: dict[str, list] = {}
    for t in tasks:
        by_status.setdefault(t.get("status", "pending"), []).append(t)
    for st in ("in_progress", "pending", "blocked", "completed"):
        group = by_status.get(st, [])
        if not group:
            continue
        lines.append(f"\n  {st.upper()} ({len(group)}):")
        for t in group[:30]:
            owner_tag = f"  @{t['owner']}" if t.get("owner") else ""
            pri = t.get("priority", "medium")
            pri_tag = f"  [{pri}]" if pri != "medium" else ""
            nn = len(t.get("notes", []))
            notes_tag = f"  ({nn} note{'s' if nn != 1 else ''})" if nn else ""
            lines.append(f"    [{t['id'][:6]}] {t.get('subject', '?')}{owner_tag}{pri_tag}{notes_tag}")
    return "\n".join(lines)


def do_team_task_add(team: str, subject: str, owner: str = "", priority: str = "medium") -> str:
    task = _ct_task_add(team, subject, owner=owner, priority=priority)
    if not task:
        return f"[failed to add task to team '{team}']"
    owner_tag = f"  @{task['owner']}" if task.get("owner") else ""
    return f"[{task['id'][:6]}] {task['subject']}{owner_tag}  [{task['priority']}]  status: {task['status']}"


def do_team_board(team: str) -> str:
    return _ct_board_render(team)


def do_team_list() -> str:
    teams = _ct_team_list()
    if not teams:
        return "[no teams — create one with /team create <name>]"
    lines = [f"Teams ({len(teams)}):"]
    for t in teams:
        cfg = _ct_load_team(t) or {}
        members = cfg.get("members", [])
        tasks = _ct_task_list(t)
        pending = sum(1 for x in tasks if x.get("status") != "completed")
        lines.append(f"  {t}  ({len(members)} members, {pending} open tasks)")
    return "\n".join(lines)


def _ct_spawn(team: str, agent_name: str, task: str, cwd: str = "") -> str:
    qwen_bin = shutil.which("qwen") or shutil.which("qwen.bat") or str(Path(sys.executable).parent / "qwen.bat")
    if not _ct_load_team(team):
        _ct_team_create(team)
    _ct_team_join(team, agent_name)
    members = (_ct_load_team(team) or {}).get("members", [])
    leader = members[0]["name"] if members else "leader"

    new_task = _ct_task_add(team, task, owner=agent_name, priority="medium")
    task_id = new_task["id"] if new_task else "?"

    prompt = (
        f"# Agent Brief\n\n"
        f"**Name:** {agent_name}\n"
        f"**Team:** {team}\n"
        f"**Leader:** {leader}\n"
        f"**Your task ID:** {task_id[:6]}\n\n"
        f"## Task\n\n{task}\n\n"
        f"## How to operate\n\n"
        f"You are running in autonomous agent mode. Work through your task using your tools. "
        f"Verify the result of each action before moving on. "
        f"When done, send your findings/results back to the leader via team_inbox_send, then mark the task completed.\n\n"
        f"## Your tools (you have full access to all of these)\n\n"
        f"- **web_search** — search the web for information\n"
        f"- **fetch_url** — fetch raw HTTP content of a URL\n"
        f"- **fetch_rendered** — fetch JS-rendered page content via browser\n"
        f"- **browser_action** — full browser automation (navigate, click, fill, screenshot, etc.)\n"
        f"- **run_command** — run shell commands, git, pip, npm, etc.\n"
        f"- **run_script** — execute inline Python/JS/bash/PowerShell/etc.\n"
        f"- **read_file** — read a file from disk\n"
        f"- **patch_file** — apply a unified diff patch to a file (preferred for edits)\n"
        f"- **write_file** — write/create a file\n"
        f"- **move_file** — move or rename a file\n"
        f"- **delete_file** — delete a file\n"
        f"- **list_directory** — list directory contents\n"
        f"- **find_files** — find files matching a glob pattern\n"
        f"- **search_files** — search file contents with regex\n"
        f"- **ask_user** — ask the user a clarifying question\n"
        f"- **team_spawn_agent** — spawn your own sub-subagents for parallel work\n"
        f"- **team_task_update**, **team_inbox_send**, **team_inbox_receive**, **team_board** — coordination\n\n"
        f"## Coordination protocol\n\n"
        f"1. Mark your task in progress immediately: `team_task_update` team={team} task_id={task_id[:6]} status=in_progress\n"
        f"2. Check your inbox for any extra context: `team_inbox_receive` team={team} agent={agent_name}\n"
        f"3. Log major progress milestones: `team_task_update` team={team} task_id={task_id[:6]} note=...\n"
        f"4. When done: send results to leader with `team_inbox_send` team={team} to={leader} message=<your full results>\n"
        f"5. Mark complete: `team_task_update` team={team} task_id={task_id[:6]} status=completed\n\n"
        f"Team files: {_config.CT_DIR / 'teams' / team}\n"
        f"Task files: {_config.CT_DIR / 'tasks' / team}\n"
    )

    spawn_dir = _config.CT_DIR / "spawn"
    spawn_dir.mkdir(exist_ok=True)
    task_file = spawn_dir / f"{team}-{agent_name}-task.md"
    _ct_atomic_write(task_file, prompt)

    work_dir = cwd or str(Path.cwd())
    safe_name = re.sub(r"[^\w-]", "", agent_name)[:32]
    try:
        subprocess.Popen(
            ["cmd.exe", "/c", "start", f"qwen-{safe_name}", "cmd", "/k", str(qwen_bin), "--task", f"@{task_file}"],
            shell=False,
            cwd=work_dir,
        )
        return f"Spawned agent '{agent_name}' for team '{team}'.\nTask ID: {task_id[:6]}\nBrief: {task_file}"
    except Exception as e:
        return f"[spawn error: {e}]"
