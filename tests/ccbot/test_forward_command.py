"""Tests for forward_command_handler — command forwarding to Claude Code."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _make_message(text: str, user_id: int = 1, thread_id: int = 42) -> MagicMock:
    """Build a minimal mock Message with text in a forum topic."""
    message = MagicMock()
    message.from_user = MagicMock()
    message.from_user.id = user_id
    message.text = text
    message.message_thread_id = thread_id
    message.is_topic_message = True
    message.chat = MagicMock()
    message.chat.type = "supergroup"
    message.chat.id = 100
    return message


class TestForwardCommand:
    @pytest.mark.asyncio
    async def test_model_sends_command_to_tmux(self):
        """/model → send_to_window called with "/model"."""
        message = _make_message("/model")
        bot = AsyncMock()

        with (
            patch("ccbot.bot.is_user_allowed", return_value=True),
            patch("ccbot.bot._get_thread_id", return_value=42),
            patch("ccbot.bot.session_manager") as mock_sm,
            patch("ccbot.bot.tmux_manager") as mock_tmux,
            patch("ccbot.bot.safe_reply", new_callable=AsyncMock),
        ):
            mock_sm.resolve_window_for_thread.return_value = "@5"
            mock_sm.get_display_name.return_value = "project"
            mock_tmux.find_window_by_id = AsyncMock(return_value=MagicMock())
            mock_sm.send_to_window = AsyncMock(return_value=(True, "ok"))

            from ccbot.bot import forward_command_handler

            await forward_command_handler(message, bot)

            mock_sm.send_to_window.assert_called_once_with("@5", "/model")
            # "typing…" goes to the message's topic
            bot.send_chat_action.assert_awaited_once()
            kwargs = bot.send_chat_action.await_args.kwargs
            assert kwargs["chat_id"] == 100
            assert kwargs["message_thread_id"] == 42

    @pytest.mark.asyncio
    async def test_cost_sends_command_to_tmux(self):
        """/cost → send_to_window called with "/cost"."""
        message = _make_message("/cost")
        bot = AsyncMock()

        with (
            patch("ccbot.bot.is_user_allowed", return_value=True),
            patch("ccbot.bot._get_thread_id", return_value=42),
            patch("ccbot.bot.session_manager") as mock_sm,
            patch("ccbot.bot.tmux_manager") as mock_tmux,
            patch("ccbot.bot.safe_reply", new_callable=AsyncMock),
        ):
            mock_sm.resolve_window_for_thread.return_value = "@5"
            mock_sm.get_display_name.return_value = "project"
            mock_tmux.find_window_by_id = AsyncMock(return_value=MagicMock())
            mock_sm.send_to_window = AsyncMock(return_value=(True, "ok"))

            from ccbot.bot import forward_command_handler

            await forward_command_handler(message, bot)

            mock_sm.send_to_window.assert_called_once_with("@5", "/cost")

    @pytest.mark.asyncio
    async def test_clear_clears_session(self):
        """/clear → send_to_window + clear_window_session."""
        message = _make_message("/clear")
        bot = AsyncMock()

        with (
            patch("ccbot.bot.is_user_allowed", return_value=True),
            patch("ccbot.bot._get_thread_id", return_value=42),
            patch("ccbot.bot.session_manager") as mock_sm,
            patch("ccbot.bot.tmux_manager") as mock_tmux,
            patch("ccbot.bot.safe_reply", new_callable=AsyncMock),
        ):
            mock_sm.resolve_window_for_thread.return_value = "@5"
            mock_sm.get_display_name.return_value = "project"
            mock_tmux.find_window_by_id = AsyncMock(return_value=MagicMock())
            mock_sm.send_to_window = AsyncMock(return_value=(True, "ok"))

            from ccbot.bot import forward_command_handler

            await forward_command_handler(message, bot)

            mock_sm.send_to_window.assert_called_once_with("@5", "/clear")
            mock_sm.clear_window_session.assert_called_once_with("@5")
