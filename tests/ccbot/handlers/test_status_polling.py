"""Tests for status_polling — Settings UI detection via the poller path.

Simulates the user workflow: /model is sent to Claude Code, the Settings
model picker renders in the terminal, and the status poller detects it
on its next 1s tick.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ccbot.handlers.status_polling import update_status_message


@pytest.fixture
def mock_bot():
    bot = AsyncMock()
    sent_msg = MagicMock()
    sent_msg.message_id = 999
    bot.send_message.return_value = sent_msg
    return bot


@pytest.fixture
def _clear_interactive_state():
    """Ensure interactive state is clean before and after each test."""
    from ccbot.handlers.interactive_ui import _interactive_mode, _interactive_msgs

    _interactive_mode.clear()
    _interactive_msgs.clear()
    yield
    _interactive_mode.clear()
    _interactive_msgs.clear()


@pytest.mark.usefixtures("_clear_interactive_state")
class TestStatusPollerSettingsDetection:
    """Simulate the status poller detecting a Settings UI in the terminal.

    This is the actual code path for /model: no JSONL tool_use entry exists,
    so the status poller (update_status_message) is the only detector.
    """

    @pytest.mark.asyncio
    async def test_settings_ui_detected_reads_transcript_then_queues_ui(
        self, mock_bot: AsyncMock, sample_pane_settings: str
    ):
        """Poller sees a UI → reads the transcript first, then queues the UI."""
        from ccbot.handlers import status_polling

        window_id = "@5"
        mock_window = MagicMock()
        mock_window.window_id = window_id
        calls: list[str] = []

        async def fake_poll() -> None:
            calls.append("poll")

        async def fake_enqueue(bot, user_id, wid, thread_id):
            calls.append(f"enqueue:{user_id}:{wid}:{thread_id}")

        status_polling.set_transcript_poller(fake_poll)
        try:
            with (
                patch("ccbot.handlers.status_polling.tmux_manager") as mock_tmux,
                patch(
                    "ccbot.handlers.status_polling.enqueue_interactive",
                    side_effect=fake_enqueue,
                ),
            ):
                mock_tmux.find_window_by_id = AsyncMock(return_value=mock_window)
                mock_tmux.capture_pane = AsyncMock(return_value=sample_pane_settings)

                await update_status_message(
                    mock_bot, user_id=1, window_id=window_id, thread_id=42
                )
        finally:
            status_polling.set_transcript_poller(None)

        assert calls == ["poll", "enqueue:1:@5:42"]

    @pytest.mark.asyncio
    async def test_normal_pane_no_interactive_ui(self, mock_bot: AsyncMock):
        """Normal pane text → no UI queued, just status check."""
        window_id = "@5"
        mock_window = MagicMock()
        mock_window.window_id = window_id
        normal_pane = (
            "some output\n"
            "✻ Reading file\n"
            "──────────────────────────────────────\n"
            "❯ \n"
            "──────────────────────────────────────\n"
            "  [Opus 4.6] Context: 50%\n"
        )

        with (
            patch("ccbot.handlers.status_polling.tmux_manager") as mock_tmux,
            patch(
                "ccbot.handlers.status_polling.enqueue_interactive",
                new_callable=AsyncMock,
            ) as mock_enqueue_ui,
            patch(
                "ccbot.handlers.status_polling.enqueue_status_update",
                new_callable=AsyncMock,
            ),
        ):
            mock_tmux.find_window_by_id = AsyncMock(return_value=mock_window)
            mock_tmux.capture_pane = AsyncMock(return_value=normal_pane)

            await update_status_message(
                mock_bot, user_id=1, window_id=window_id, thread_id=42
            )

            mock_enqueue_ui.assert_not_called()

    @pytest.mark.asyncio
    async def test_pending_interactive_task_keeps_mode(
        self, mock_bot: AsyncMock, sample_pane_settings: str
    ):
        """UI queued from the transcript but not drawn yet: poller leaves it be."""
        from ccbot.handlers import message_queue
        from ccbot.handlers.interactive_ui import (
            get_interactive_window,
            set_interactive_mode,
        )

        window_id = "@5"
        mock_window = MagicMock()
        mock_window.window_id = window_id
        set_interactive_mode(1, window_id, 42)
        message_queue._pending_interactive[(1, 42)] = 1
        try:
            with (
                patch("ccbot.handlers.status_polling.tmux_manager") as mock_tmux,
                patch(
                    "ccbot.handlers.status_polling.clear_interactive_msg",
                    new_callable=AsyncMock,
                ) as mock_clear,
            ):
                mock_tmux.find_window_by_id = AsyncMock(return_value=mock_window)
                mock_tmux.capture_pane = AsyncMock(return_value="idle\n")
                await update_status_message(
                    mock_bot, user_id=1, window_id=window_id, thread_id=42
                )
            mock_clear.assert_not_called()
            assert get_interactive_window(1, 42) == window_id
        finally:
            message_queue._pending_interactive.clear()

    @pytest.mark.asyncio
    async def test_settings_ui_end_to_end_sends_telegram_keyboard(
        self, mock_bot: AsyncMock, sample_pane_settings: str
    ):
        """Full path: poller → queue → worker → bot.send_message with keyboard."""
        from ccbot.handlers import message_queue

        window_id = "@5"
        mock_window = MagicMock()
        mock_window.window_id = window_id

        with (
            patch("ccbot.handlers.status_polling.tmux_manager") as mock_tmux_poll,
            patch("ccbot.handlers.interactive_ui.tmux_manager") as mock_tmux_ui,
            patch("ccbot.handlers.interactive_ui.session_manager") as mock_sm,
        ):
            mock_tmux_poll.find_window_by_id = AsyncMock(return_value=mock_window)
            mock_tmux_poll.capture_pane = AsyncMock(return_value=sample_pane_settings)
            mock_tmux_ui.find_window_by_id = AsyncMock(return_value=mock_window)
            mock_tmux_ui.capture_pane = AsyncMock(return_value=sample_pane_settings)
            mock_sm.resolve_chat_id.return_value = 100

            try:
                await update_status_message(
                    mock_bot, user_id=1, window_id=window_id, thread_id=42
                )
                queue = message_queue.get_message_queue(1)
                assert queue is not None
                await queue.join()
            finally:
                await message_queue.shutdown_workers()

            mock_bot.send_message.assert_called_once()
            call_kwargs = mock_bot.send_message.call_args.kwargs
            assert call_kwargs["chat_id"] == 100
            assert call_kwargs["message_thread_id"] == 42
            assert call_kwargs["reply_markup"] is not None
            assert "Select model" in call_kwargs["text"]


class TestVanishedWindow:
    """A topic whose window disappeared is unbound and offered ▶ Resume."""

    @pytest.mark.asyncio
    async def test_offers_resume_when_session_known(self):
        from ccbot.handlers import status_polling
        from ccbot.session import WindowState

        ws = WindowState(session_id="sid-9", cwd="/proj")
        sm = MagicMock()
        sm.window_states = {"@4": ws}
        sm.get_launch_info.return_value = {"mode": "plan"}
        sm.get_display_name.return_value = "proj"
        with (
            patch.object(status_polling, "session_manager", sm),
            patch.object(status_polling, "clear_topic_state", AsyncMock()) as clear,
            patch.object(status_polling, "offer_resume", AsyncMock()) as offer,
        ):
            await status_polling._handle_vanished_window(AsyncMock(), 1, 42, "@4")

        sm.unbind_thread.assert_called_once_with(1, 42)
        clear.assert_awaited_once()
        offer.assert_awaited_once()
        kwargs = offer.await_args.kwargs
        assert (kwargs["session_id"], kwargs["cwd"], kwargs["mode"]) == (
            "sid-9",
            "/proj",
            "plan",
        )

    @pytest.mark.asyncio
    async def test_no_offer_without_session(self):
        from ccbot.handlers import status_polling

        sm = MagicMock()
        sm.window_states = {}
        sm.get_launch_info.return_value = {}
        sm.get_display_name.return_value = "x"
        with (
            patch.object(status_polling, "session_manager", sm),
            patch.object(status_polling, "clear_topic_state", AsyncMock()),
            patch.object(status_polling, "offer_resume", AsyncMock()) as offer,
        ):
            await status_polling._handle_vanished_window(AsyncMock(), 1, 42, "@4")

        sm.unbind_thread.assert_called_once_with(1, 42)
        offer.assert_not_awaited()
