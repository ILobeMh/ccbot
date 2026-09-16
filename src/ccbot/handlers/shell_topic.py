"""The ``shell`` special topic: a raw shell for the bot's host, no Claude Code.

Every message in the topic is run as a shell command; the reply shows the
command, its combined stdout/stderr, exit code, duration and the working
directory. ``cd`` persists between commands (the cwd is tracked by the
bot, not by a long-lived shell), long output is attached as a .txt file,
and a running command can be cancelled with the ⏹ button.

Key components:
  - ShellTopic: SpecialTopic implementation registered as "shell"
  - run_shell(): asyncio subprocess wrapper returning ShellResult
  - CCBOT_SHELL_TIMEOUT: per-command timeout in seconds (default 120)
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
import shlex
import time
from dataclasses import dataclass
from pathlib import Path

from telegram import Bot, InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import ContextTypes

from ..config import config
from . import special_topics
from .callback_data import CB_SHELL_KILL
from .message_sender import safe_edit, safe_reply, safe_send

logger = logging.getLogger(__name__)

# Marker appended by the wrapper so cwd and exit code survive in stdout
_MARK = "\x1e__CCBOT__"
INLINE_LIMIT = 3000  # chars of output shown inline before attaching a file


@dataclass
class ShellResult:
    output: str
    exit_code: int | None  # None = timed out / killed
    cwd: str
    duration: float
    timed_out: bool = False
    cancelled: bool = False


def _shell_binary() -> str:
    return os.environ.get("SHELL") or "/bin/bash"


async def run_shell(
    command: str,
    cwd: str,
    timeout: float,
    on_start: asyncio.Future[asyncio.subprocess.Process] | None = None,
) -> ShellResult:
    """Run ``command`` in ``cwd`` with a login shell, capturing everything.

    An EXIT trap prints the final cwd and exit code after a marker, so a
    ``cd`` carries over to the next command and an explicit ``exit N``
    still reports N. Output is read incrementally so a timed-out or killed
    command still returns what it printed.
    """
    wrapped = (
        f'trap \'printf "\\n{_MARK}%s\\n%s" "$PWD" "${{__rc-$?}}"\' EXIT\n'
        f"cd {shlex.quote(cwd)} 2>/dev/null || cd ~\n"
        f"{command}\n"
        f"__rc=$?\n"
    )
    start = time.monotonic()
    proc = await asyncio.create_subprocess_exec(
        _shell_binary(),
        "-lc",
        wrapped,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        env={**os.environ, "TERM": "dumb", "NO_COLOR": "1", "PAGER": "cat"},
        start_new_session=True,
    )
    if on_start is not None and not on_start.done():
        on_start.set_result(proc)
    assert proc.stdout is not None
    reader = asyncio.ensure_future(proc.stdout.read())
    timed_out = False
    try:
        raw = await asyncio.wait_for(asyncio.shield(reader), timeout=timeout)
    except asyncio.TimeoutError:
        timed_out = True
        kill_process_group(proc)
        try:
            raw = await asyncio.wait_for(reader, timeout=5)
        except asyncio.TimeoutError:
            raw = b""
    await proc.wait()
    duration = time.monotonic() - start
    text = raw.decode("utf-8", errors="replace")
    new_cwd, exit_code = cwd, None
    if _MARK in text:
        text, _, tail = text.rpartition(_MARK)
        lines = tail.split("\n", 1)
        if lines[0].strip():
            new_cwd = lines[0].strip()
        if len(lines) > 1 and lines[1].strip().lstrip("-").isdigit():
            exit_code = int(lines[1].strip())
    killed = proc.returncode is not None and proc.returncode < 0 and not timed_out
    return ShellResult(
        output=text.rstrip("\n"),
        exit_code=exit_code,
        cwd=new_cwd,
        duration=duration,
        timed_out=timed_out,
        cancelled=killed,
    )


def kill_process_group(proc: asyncio.subprocess.Process) -> None:
    """SIGKILL the command and everything it spawned (own session)."""
    try:
        os.killpg(proc.pid, 9)
    except (ProcessLookupError, PermissionError):
        pass


def _pretty_cwd(cwd: str) -> str:
    home = str(Path.home())
    return "~" + cwd[len(home) :] if cwd == home or cwd.startswith(home + "/") else cwd


def _fence(text: str) -> str:
    return "```\n" + text.replace("```", "'''") + "\n```"


class ShellTopic:
    name = "shell"
    callback_prefixes: tuple[str, ...] = (CB_SHELL_KILL,)

    def __init__(self) -> None:
        self.cwd = str(Path.home())
        self._seq = 0  # job ids for the ⏹ Kill button
        self._procs: dict[int, asyncio.subprocess.Process] = {}

    async def on_ready(self, bot: Bot, chat_id: int, thread_id: int) -> None:
        await safe_send(
            bot,
            chat_id,
            f"🐚 **shell** on `{os.uname().nodename}` — every message here runs as a "
            f"command (`{_shell_binary()} -lc`), `cd` persists, timeout "
            f"{config.shell_timeout:.0f}s, output > {INLINE_LIMIT} chars is attached "
            f"as a file.\ncwd: `{_pretty_cwd(self.cwd)}`",
            message_thread_id=thread_id,
        )

    async def handle_text(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE, text: str
    ) -> None:
        msg = update.message
        if msg is None:
            return
        command = text.strip()
        if not command:
            return
        self._seq += 1
        job_id = self._seq
        progress = await safe_reply(
            msg,
            f"⏳ `{_pretty_cwd(self.cwd)}`\n{_fence('$ ' + command)}",
            reply_markup=InlineKeyboardMarkup(
                [
                    [
                        InlineKeyboardButton(
                            "⏹ Kill", callback_data=f"{CB_SHELL_KILL}{job_id}"
                        )
                    ]
                ]
            ),
        )
        started: asyncio.Future[asyncio.subprocess.Process] = (
            asyncio.get_running_loop().create_future()
        )
        task = asyncio.create_task(
            run_shell(command, self.cwd, config.shell_timeout, on_start=started)
        )
        started.add_done_callback(lambda f: self._procs.__setitem__(job_id, f.result()))
        try:
            result = await task
        finally:
            self._procs.pop(job_id, None)

        self.cwd = result.cwd
        if result.cancelled:
            status = "⏹ killed"
        elif result.timed_out:
            status = f"⏱ timed out after {config.shell_timeout:.0f}s (killed)"
        elif result.exit_code == 0:
            status = "✅ exit 0"
        else:
            status = f"❌ exit {result.exit_code}"
        footer = f"{status} · {result.duration:.1f}s · `{_pretty_cwd(result.cwd)}`"

        output = result.output
        if not output:
            body = "_(no output)_"
        elif len(output) > INLINE_LIMIT:
            head = output[:INLINE_LIMIT].rsplit("\n", 1)[0]
            body = (
                _fence(head)
                + f"\n_… {len(output) - len(head)} more chars in the attached file_"
            )
        else:
            body = _fence(output)

        await safe_edit(progress, f"{_fence('$ ' + command)}\n{body}\n{footer}")
        if len(output) > INLINE_LIMIT:
            await msg.reply_document(
                document=io.BytesIO(output.encode("utf-8")),
                filename="output.txt",
                caption=f"{command[:200]}",
            )

    async def handle_callback(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE, data: str
    ) -> None:
        query = update.callback_query
        if query is None:
            return
        raw = data[len(CB_SHELL_KILL) :]
        proc = self._procs.get(int(raw)) if raw.isdigit() else None
        if proc is None or proc.returncode is not None:
            await query.answer("Already finished")
            return
        kill_process_group(proc)
        await query.answer("Killed")


shell_topic = ShellTopic()
special_topics.register(shell_topic)
