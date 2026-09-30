# Message Handling

## Message Queue Architecture

Per-user message queues + worker pattern for all send tasks:
- Messages are sent in receive order (FIFO)
- Status messages always follow content messages
- Multi-user concurrent processing without interference

**Message merging**: The worker automatically merges consecutive mergeable content messages on dequeue:
- Content messages for the same window can be merged (including text, thinking)
- tool_use breaks the merge chain and is sent separately (message ID recorded for later editing)
- tool_result breaks the merge chain and is edited into the tool_use message (preventing order confusion)
- Merging stops when combined length exceeds 3800 characters (to avoid pagination)
- Rich style: a short one-line reply right after a thinking block (not the turn's final text) becomes that block's `<summary>` instead of a thinking preview — merged when both are queued together, otherwise edited into the thinking message if it is still the topic's last content (≤20 s)
- `<task-notification>` prompts (background command / agent / monitor done) are parsed into `task_notification` notices, never merged, and sent as a reply to the tool call that started the task when its message id is known

## Status Message Handling

**Conversion**: The status message is edited into the first content message, reducing message count:
- When a status message exists, the first content message updates it via edit
- Subsequent content messages are sent as new messages

**Polling**: Background task polls terminal status for all active windows at 1-second intervals. Send-layer rate limiting ensures flood control is not triggered.

**Deduplication**: The worker compares `last_text` when processing status updates; identical content skips the edit, reducing API calls.

## Rate Limiting

- `telegram_client.TelegramRateLimiter`: aiogram request middleware on the bot session (`build_bot`). 30 req/s global for every call with a `chat_id`; 20 msg/min per group only for message-creating calls (edits, deletes and chat actions don't count)
- On 429 (`TelegramRetryAfter`) it pauses all concurrent requests and retries after the ban (`max_retries=5`); a 429 that survives reaches the queue worker, which backs off
- The global bucket starts pre-filled (`_level=max_rate`) to avoid a burst against Telegram's persisted server-side counter on restart
- Updates are handled one at a time (`start_polling(handle_as_tasks=False)`), in arrival order
- Status polling interval: 1 second (skips enqueue when queue is non-empty)

## Performance Optimizations

**mtime cache**: The monitoring loop maintains an in-memory file mtime cache, skipping reads for unchanged files.

**Targeted transcript lookup**: the monitor resolves transcripts only for session ids in `session_map` (hook `transcript_path` → in-memory cache → one glob, negative-cached ~5 s); it never scans `~/.claude/projects`. `session_map.json` is re-parsed only when its (mtime, size, inode) changes. `poll_now()` runs an on-demand cycle (lock-serialised with the loop).

**Debounced state saves**: `update_user_window_offset` schedules one `state.json` write ~1 s later; direct `_save_state()` calls write immediately; `SessionManager.flush()` runs on shutdown.

**Single-fork pane capture**: `capture_pane` is one `tmux capture-pane -p` subprocess (5 s timeout).

**Byte offset incremental reads**: Each tracked session records `last_byte_offset`, reading only new content. File truncation (offset > file_size) is detected and offset is auto-reset.

## No Message Truncation

Historical messages (tool_use summaries, tool_result text, user/assistant messages) are always kept in full — no character-level truncation at the parsing layer. Long text is handled exclusively at the send layer: `split_message` splits by Telegram's 4096-character limit; real-time messages get `[1/N]` text suffixes, history pages get inline keyboard navigation.
