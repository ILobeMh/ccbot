# AGENTS.md

ccmux — Telegram bot that bridges Telegram Forum topics to Codex sessions via tmux windows. Each topic is bound to one tmux window running one Codex instance.

Tech stack: Python, aiogram 3, tmux, uv.

## Common Commands

```bash
uv run ruff check .                   # Lint — MUST pass before committing
uv run ruff format .                  # Format (incl. code blocks in *.md) — then verify with --check
uv run ty check                       # Type check (src/, see [tool.ty]) — MUST be 0 errors before committing
./scripts/restart.sh                  # Restart the ccbot service after code changes
ccbot hook --install                  # Auto-install Codex SessionStart hook
```

## Core Design Constraints

- **1 Topic = 1 Window = 1 Session** — all internal routing keyed by tmux window ID (`@0`, `@12`), not window name. Window names kept as display names. Same directory can have multiple windows.
- **Topic-only** — no backward-compat for non-topic mode. No `active_sessions`, no `/list`, no General topic routing.
- **No message truncation** at parse layer — splitting only at send layer (`split_message`, 4096 char limit).
- **MarkdownV2 only** — use `safe_reply`/`safe_edit`/`safe_send` helpers (auto fallback to plain text). Internal queue/UI code calls bot API directly with its own fallback.
- **Hook-based session tracking** — `SessionStart` hook writes `session_map.json`; monitor polls it to detect session changes.
- **Message queue per user** — FIFO ordering, message merging (3800 char limit), tool_use/tool_result pairing.
- **Rate limiting** — `telegram_client.TelegramRateLimiter`, a request middleware on the aiogram bot session (30/s global, 20 msg/min per group for message-creating calls, 429 → pause everything and retry, `max_retries=5`). The global bucket starts pre-filled to avoid a burst against Telegram's server-side counter on restart.

## Code Conventions

- Every `.py` file starts with a module-level docstring: purpose clear within 10 lines, one-sentence summary first line, then core responsibilities and key components.
- Telegram interaction: prefer inline keyboards over reply keyboards; use `edit_message_text` for in-place updates; keep callback data under 64 bytes; use `answer_callback_query` for instant feedback.

## Configuration

- Config directory: `~/.ccbot/` by default, override with `CCBOT_DIR` env var.
- `.env` loading priority: local `.env` > config dir `.env`.
- State files: `state.json` (thread bindings), `session_map.json` (hook-generated), `monitor_state.json` (byte offsets).

## Hook Configuration

Auto-install: `ccbot hook --install`

Or manually in `~/.Codex/settings.json`:
```json
{
  "hooks": {
    "SessionStart": [
      {
        "hooks": [{ "type": "command", "command": "ccbot hook", "timeout": 5 }]
      }
    ]
  }
}
```

## Architecture Details

See @.Codex/rules/architecture.md for full system diagram and module inventory.
See @.Codex/rules/topic-architecture.md for topic→window→session mapping details.
See @.Codex/rules/message-handling.md for message queue, merging, and rate limiting.
