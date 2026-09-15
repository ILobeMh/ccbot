"""Hook subcommand for Claude Code session tracking.

Called by Claude Code's SessionStart hook to maintain a window↔session
mapping in <CCBOT_DIR>/session_map.json. Also provides `--install` to
auto-configure the hook in ~/.claude/settings.json.

This module must NOT import config.py (which requires TELEGRAM_BOT_TOKEN),
since hooks run inside tmux panes where bot env vars are not set.
Config directory resolution uses utils.ccbot_dir() (shared with config.py).

Key functions: hook_main() (CLI entry), _install_hook().
"""

import argparse
import fcntl
import json
import logging
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# Validate session_id looks like a UUID
_UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")

_CLAUDE_SETTINGS_FILE = Path.home() / ".claude" / "settings.json"

# The hook command suffix for detection
_HOOK_COMMAND_SUFFIX = "ccbot hook"


def _find_ccbot_path() -> str:
    """Find the full path to the ccbot executable.

    Priority:
    1. shutil.which("ccbot") - if ccbot is in PATH
    2. Same directory as the Python interpreter (for venv installs)
    """
    # Try PATH first
    ccbot_path = shutil.which("ccbot")
    if ccbot_path:
        return ccbot_path

    # Fall back to the directory containing the Python interpreter
    # This handles the case where ccbot is installed in a venv
    python_dir = Path(sys.executable).parent
    ccbot_in_venv = python_dir / "ccbot"
    if ccbot_in_venv.exists():
        return str(ccbot_in_venv)

    # Last resort: assume it will be in PATH
    return "ccbot"


_NON_INTERACTIVE_FLAGS = ("-p", "--print", "--bg", "--background", "--remote-control")
_CONTINUATION_SOURCES = frozenset({"resume", "clear", "compact", "fork"})


def _process_chain(pid: int, max_depth: int = 12) -> list[str]:
    """Return command lines of ``pid``'s ancestors (nearest first).

    Uses ``ps`` so it works on both macOS and Linux without psutil.
    """
    chain: list[str] = []
    for _ in range(max_depth):
        if pid <= 1:
            break
        try:
            out = subprocess.run(
                ["ps", "-o", "ppid=,command=", "-p", str(pid)],
                capture_output=True,
                text=True,
                timeout=2,
                check=False,
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            break
        if not out:
            break
        ppid_str, _, command = out.partition(" ")
        try:
            pid = int(ppid_str)
        except ValueError:
            break
        chain.append(command.strip())
    return chain


def _is_claude_command(command: str) -> bool:
    argv0 = command.split(" ", 1)[0]
    return os.path.basename(argv0) == "claude" or argv0.endswith("/claude")


def _should_skip_nested(env: dict[str, str], chain: list[str]) -> str | None:
    """Reason to ignore this SessionStart, or None to proceed.

    A Claude Code session spawned *by* the tracked session (Bash tool
    running ``claude -p``, SDK, ``--bg`` workers) fires the same hook inside
    the same tmux pane and would overwrite the window's mapping.
    """
    entrypoint = env.get("CLAUDE_CODE_ENTRYPOINT", "")
    if entrypoint.startswith("sdk-"):
        return f"CLAUDE_CODE_ENTRYPOINT={entrypoint}"
    claude_ancestors = [c for c in chain if _is_claude_command(c)]
    if not claude_ancestors:
        return None
    nearest = claude_ancestors[0]
    args = nearest.split()[1:]
    for flag in _NON_INTERACTIVE_FLAGS:
        if flag in args:
            return f"non-interactive claude ({flag})"
    if len(claude_ancestors) >= 2:
        return "nested claude (spawned by another claude)"
    return None


def _valid_transcript_path(transcript_path: str, session_id: str) -> str:
    """Return ``transcript_path`` if it is absolute and named after the session."""
    if not transcript_path or not os.path.isabs(transcript_path):
        return ""
    if Path(transcript_path).stem != session_id:
        return ""
    return transcript_path


def _find_pane_key(
    session_id: str, cwd: str, session_map: dict
) -> tuple[str, str] | None:
    """Fallback when TMUX_PANE is missing (claude --bg-pty-host strips it).

    1. The session is already mapped → keep its window (compact/resume).
    2. Exactly one pane with a ``claude`` process runs in ``cwd`` → that one.
    Returns ``(session_window_key, window_name)`` or None.
    """
    for key, info in session_map.items():
        if isinstance(info, dict) and info.get("session_id") == session_id:
            return key, info.get("window_name", "")
    try:
        out = subprocess.run(
            [
                "tmux",
                "list-panes",
                "-a",
                "-F",
                "#{session_name}:#{window_id}\x1f#{window_name}"
                "\x1f#{pane_current_path}\x1f#{pane_current_command}",
            ],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    matches: list[tuple[str, str]] = []
    for line in out.splitlines():
        parts = line.split("\x1f")
        if len(parts) != 4:
            continue
        key, wname, pane_cwd, pane_cmd = parts
        if pane_cmd == "claude" and cwd and pane_cwd == cwd:
            matches.append((key, wname))
    if len(matches) == 1:
        return matches[0]
    logger.warning(
        "TMUX_PANE unset and %d panes run claude in %s — cannot map", len(matches), cwd
    )
    return None


def _is_hook_installed(settings: dict) -> bool:
    """Check if ccbot hook is already installed in the settings.

    Detects both 'ccbot hook' and full paths like '/path/to/ccbot hook'.
    """
    hooks = settings.get("hooks", {})
    session_start = hooks.get("SessionStart", [])

    for entry in session_start:
        if not isinstance(entry, dict):
            continue
        inner_hooks = entry.get("hooks", [])
        for h in inner_hooks:
            if not isinstance(h, dict):
                continue
            cmd = h.get("command", "")
            # Match 'ccbot hook' or paths ending with 'ccbot hook'
            if cmd == _HOOK_COMMAND_SUFFIX or cmd.endswith("/" + _HOOK_COMMAND_SUFFIX):
                return True
    return False


def _install_hook() -> int:
    """Install the ccbot hook into Claude's settings.json.

    Returns 0 on success, 1 on error.
    """
    settings_file = _CLAUDE_SETTINGS_FILE
    settings_file.parent.mkdir(parents=True, exist_ok=True)

    # Read existing settings
    settings: dict = {}
    if settings_file.exists():
        try:
            settings = json.loads(settings_file.read_text())
        except (json.JSONDecodeError, OSError) as e:
            logger.error("Error reading %s: %s", settings_file, e)
            print(f"Error reading {settings_file}: {e}", file=sys.stderr)
            return 1

    # Check if already installed
    if _is_hook_installed(settings):
        logger.info("Hook already installed in %s", settings_file)
        print(f"Hook already installed in {settings_file}")
        return 0

    # Find the full path to ccbot
    ccbot_path = _find_ccbot_path()
    hook_command = f"{ccbot_path} hook"
    hook_config = {"type": "command", "command": hook_command, "timeout": 5}
    logger.info("Installing hook command: %s", hook_command)

    # Install the hook
    if "hooks" not in settings:
        settings["hooks"] = {}
    if "SessionStart" not in settings["hooks"]:
        settings["hooks"]["SessionStart"] = []

    settings["hooks"]["SessionStart"].append({"hooks": [hook_config]})

    # Write back
    try:
        settings_file.write_text(
            json.dumps(settings, indent=2, ensure_ascii=False) + "\n"
        )
    except OSError as e:
        logger.error("Error writing %s: %s", settings_file, e)
        print(f"Error writing {settings_file}: {e}", file=sys.stderr)
        return 1

    logger.info("Hook installed successfully in %s", settings_file)
    print(f"Hook installed successfully in {settings_file}")
    return 0


def hook_main() -> None:
    """Process a Claude Code hook event from stdin, or install the hook."""
    # Configure logging for the hook subprocess (main.py logging doesn't apply here)
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        level=logging.DEBUG,
        stream=sys.stderr,
    )

    parser = argparse.ArgumentParser(
        prog="ccbot hook",
        description="Claude Code session tracking hook",
    )
    parser.add_argument(
        "--install",
        action="store_true",
        help="Install the hook into ~/.claude/settings.json",
    )
    # Parse only known args to avoid conflicts with stdin JSON
    args, _ = parser.parse_known_args(sys.argv[2:])

    if args.install:
        logger.info("Hook install requested")
        sys.exit(_install_hook())

    # Normal hook processing: read JSON from stdin
    logger.debug("Processing hook event from stdin")
    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning("Failed to parse stdin JSON: %s", e)
        return

    session_id = payload.get("session_id", "")
    cwd = payload.get("cwd", "")
    event = payload.get("hook_event_name", "")

    if not session_id or not event:
        logger.debug("Empty session_id or event, ignoring")
        return

    # Validate session_id format
    if not _UUID_RE.match(session_id):
        logger.warning("Invalid session_id format: %s", session_id)
        return

    # Validate cwd is an absolute path (if provided)
    if cwd and not os.path.isabs(cwd):
        logger.warning("cwd is not absolute: %s", cwd)
        return

    if event != "SessionStart":
        logger.debug("Ignoring non-SessionStart event: %s", event)
        return

    skip_reason = _should_skip_nested(dict(os.environ), _process_chain(os.getpid()))
    if skip_reason:
        logger.info("Ignoring SessionStart from %s", skip_reason)
        return

    source = payload.get("source", "")
    transcript_path = _valid_transcript_path(
        str(payload.get("transcript_path", "")), session_id
    )

    # Get tmux session:window key for the pane running this hook.
    # TMUX_PANE is set by tmux for every process inside a pane — except
    # when Claude Code's --bg-pty-host supervisor re-execs the hook after
    # /clear, /compact or --resume, in which case we fall back below.
    session_window_key = ""
    window_name = ""
    tmux_session_name = ""
    pane_id = os.environ.get("TMUX_PANE", "")
    if pane_id:
        result = subprocess.run(
            [
                "tmux",
                "display-message",
                "-t",
                pane_id,
                "-p",
                "#{session_name}:#{window_id}:#{window_name}",
            ],
            capture_output=True,
            text=True,
        )
        raw_output = result.stdout.strip()
        # Expected format: "session_name:@id:window_name"
        parts = raw_output.split(":", 2)
        if len(parts) < 3:
            logger.warning(
                "Failed to parse session:window_id:window_name from tmux "
                "(pane=%s, output=%s)",
                pane_id,
                raw_output,
            )
            return
        tmux_session_name, window_id, window_name = parts
        # Key uses window_id for uniqueness
        session_window_key = f"{tmux_session_name}:{window_id}"
    elif source not in _CONTINUATION_SOURCES:
        logger.warning("TMUX_PANE not set, cannot determine window")
        return

    logger.debug(
        "tmux key=%s, window_name=%s, session_id=%s, cwd=%s, source=%s",
        session_window_key or "<fallback>",
        window_name,
        session_id,
        cwd,
        source,
    )

    # Read-modify-write with file locking to prevent concurrent hook races
    from .utils import ccbot_dir

    map_file = ccbot_dir() / "session_map.json"
    map_file.parent.mkdir(parents=True, exist_ok=True)

    lock_path = map_file.with_suffix(".lock")
    try:
        with open(lock_path, "w") as lock_f:
            fcntl.flock(lock_f, fcntl.LOCK_EX)
            logger.debug("Acquired lock on %s", lock_path)
            try:
                session_map: dict[str, dict[str, str]] = {}
                if map_file.exists():
                    try:
                        session_map = json.loads(map_file.read_text())
                    except (json.JSONDecodeError, OSError):
                        logger.warning(
                            "Failed to read existing session_map, starting fresh"
                        )

                if not session_window_key:
                    found = _find_pane_key(session_id, cwd, session_map)
                    if found is None:
                        return
                    session_window_key, window_name = found
                    tmux_session_name = session_window_key.split(":", 1)[0]
                    logger.info(
                        "TMUX_PANE unset (source=%s); mapped via fallback to %s",
                        source,
                        session_window_key,
                    )

                entry: dict[str, str] = {
                    "session_id": session_id,
                    "cwd": cwd,
                    "window_name": window_name,
                }
                previous = session_map.get(session_window_key) or {}
                if not transcript_path and previous.get("session_id") == session_id:
                    # e.g. compact re-fires without transcript_path
                    transcript_path = previous.get("transcript_path", "")
                if transcript_path:
                    entry["transcript_path"] = transcript_path
                session_map[session_window_key] = entry

                # Clean up old-format key ("session:window_name") if it exists.
                # Previous versions keyed by window_name instead of window_id.
                old_key = f"{tmux_session_name}:{window_name}"
                if old_key != session_window_key and old_key in session_map:
                    del session_map[old_key]
                    logger.info("Removed old-format session_map key: %s", old_key)

                from .utils import atomic_write_json

                atomic_write_json(map_file, session_map)
                logger.info(
                    "Updated session_map: %s -> session_id=%s, cwd=%s",
                    session_window_key,
                    session_id,
                    cwd,
                )
            finally:
                fcntl.flock(lock_f, fcntl.LOCK_UN)
    except OSError as e:
        logger.error("Failed to write session_map: %s", e)
