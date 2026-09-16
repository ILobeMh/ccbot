"""Tests for the shell special topic (real subprocesses, short commands)."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import ccbot.handlers.shell_topic as sh


class TestRunShell:
    @pytest.mark.asyncio
    async def test_output_exit_code_and_cwd(self, tmp_path):
        r = await sh.run_shell("echo hi; echo err >&2; false", str(tmp_path), 10)
        assert r.output.split() == ["hi", "err"]
        assert r.exit_code == 1
        assert r.cwd == str(tmp_path)

    @pytest.mark.asyncio
    async def test_cd_persists(self, tmp_path):
        sub = tmp_path / "sub"
        sub.mkdir()
        r = await sh.run_shell("cd sub && pwd", str(tmp_path), 10)
        assert r.exit_code == 0
        assert r.cwd == str(sub.resolve())

    @pytest.mark.asyncio
    async def test_explicit_exit_code(self, tmp_path):
        r = await sh.run_shell("exit 3", str(tmp_path), 5)
        assert r.exit_code == 3

    @pytest.mark.asyncio
    async def test_timeout_keeps_partial_output(self, tmp_path):
        r = await sh.run_shell("echo start; sleep 20; echo never", str(tmp_path), 0.8)
        assert r.timed_out
        assert r.exit_code is None
        assert r.output == "start"
        assert r.duration < 5

    @pytest.mark.asyncio
    async def test_kill_marks_cancelled(self, tmp_path):
        fut = asyncio.get_running_loop().create_future()
        task = asyncio.create_task(
            sh.run_shell("echo a; sleep 20", str(tmp_path), 30, on_start=fut)
        )
        proc = await fut
        await asyncio.sleep(0.3)
        sh.kill_process_group(proc)
        r = await task
        assert r.cancelled and not r.timed_out
        assert r.output == "a"

    @pytest.mark.asyncio
    async def test_missing_cwd_falls_back_to_home(self, tmp_path):
        r = await sh.run_shell("pwd", str(tmp_path / "gone"), 5)
        assert r.exit_code == 0
        assert r.cwd != str(tmp_path / "gone")


class TestShellTopic:
    @pytest.fixture
    def topic(self, monkeypatch, tmp_path):
        monkeypatch.setattr(sh.config, "shell_timeout", 10.0)
        t = sh.ShellTopic()
        t.cwd = str(tmp_path)
        return t

    def _update(self, text: str):
        progress = MagicMock()
        progress.message_id = 500
        progress.edit_reply_markup = AsyncMock()
        msg = MagicMock()
        msg.text = text
        msg.reply_document = AsyncMock()
        return SimpleNamespace(message=msg), progress

    @pytest.mark.asyncio
    async def test_runs_and_reports(self, topic, monkeypatch, tmp_path):
        update, progress = self._update("echo hello")
        monkeypatch.setattr(sh, "safe_reply", AsyncMock(return_value=progress))
        edit = AsyncMock()
        monkeypatch.setattr(sh, "safe_edit", edit)
        await topic.handle_text(update, SimpleNamespace(), "echo hello")
        final = edit.await_args.args[1]
        assert "hello" in final
        assert "exit 0" in final
        update.message.reply_document.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_long_output_attached(self, topic, monkeypatch):
        update, progress = self._update("seq 1 5000")
        monkeypatch.setattr(sh, "safe_reply", AsyncMock(return_value=progress))
        edit = AsyncMock()
        monkeypatch.setattr(sh, "safe_edit", edit)
        await topic.handle_text(update, SimpleNamespace(), "seq 1 5000")
        assert "attached file" in edit.await_args.args[1]
        update.message.reply_document.assert_awaited_once()
        assert (
            update.message.reply_document.await_args.kwargs["filename"] == "output.txt"
        )

    @pytest.mark.asyncio
    async def test_cwd_tracked_across_commands(self, topic, monkeypatch, tmp_path):
        (tmp_path / "d").mkdir()
        for cmd in ("cd d", "pwd"):
            update, progress = self._update(cmd)
            monkeypatch.setattr(sh, "safe_reply", AsyncMock(return_value=progress))
            edit = AsyncMock()
            monkeypatch.setattr(sh, "safe_edit", edit)
            await topic.handle_text(update, SimpleNamespace(), cmd)
        assert topic.cwd == str((tmp_path / "d").resolve())
        assert str((tmp_path / "d").resolve()) in edit.await_args.args[1]
