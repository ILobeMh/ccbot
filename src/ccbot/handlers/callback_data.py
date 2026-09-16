"""Callback data constants for Telegram inline keyboards.

Defines all CB_* prefixes used for routing callback queries in the bot.
Each prefix identifies a specific action or navigation target.

Constants:
  - CB_HISTORY_*: History pagination
  - CB_DIR_*: Directory browser navigation
  - CB_WIN_*: Window picker (bind existing unbound window)
  - CB_SCREENSHOT_*: Screenshot refresh
  - CB_ASK_*: Interactive UI navigation (arrows, enter, esc)
  - CB_KEYS_PREFIX: Screenshot control keys (kb:<key_id>:<window>)
"""

# History pagination
CB_HISTORY_PREV = "hp:"  # history page older
CB_HISTORY_NEXT = "hn:"  # history page newer

# Directory browser
CB_DIR_SELECT = "db:sel:"
CB_DIR_UP = "db:up"
CB_DIR_CONFIRM = "db:confirm"
CB_DIR_CANCEL = "db:cancel"
CB_DIR_PAGE = "db:page:"

# Window picker (bind existing unbound window)
CB_WIN_BIND = "wb:sel:"  # wb:sel:<index>
CB_WIN_NEW = "wb:new"  # proceed to directory browser
CB_WIN_CANCEL = "wb:cancel"

# Screenshot
CB_SCREENSHOT_REFRESH = "ss:ref:"

# Interactive UI (aq: prefix kept for backward compatibility)
CB_ASK_UP = "aq:up:"  # aq:up:<window>
CB_ASK_DOWN = "aq:down:"  # aq:down:<window>
CB_ASK_LEFT = "aq:left:"  # aq:left:<window>
CB_ASK_RIGHT = "aq:right:"  # aq:right:<window>
CB_ASK_ESC = "aq:esc:"  # aq:esc:<window>
CB_ASK_ENTER = "aq:enter:"  # aq:enter:<window>
CB_ASK_SPACE = "aq:spc:"  # aq:spc:<window>
CB_ASK_TAB = "aq:tab:"  # aq:tab:<window>
CB_ASK_REFRESH = "aq:ref:"  # aq:ref:<window>

# Session picker (resume existing session)
CB_SESSION_SELECT = "rs:sel:"  # rs:sel:<index>
CB_SESSION_NEW = "rs:new"  # start a new session
CB_SESSION_CANCEL = "rs:cancel"  # cancel

# Screenshot control keys
CB_KEYS_PREFIX = "kb:"  # kb:<key_id>:<window>

# Launch mode picker (new session) and mode switching (running session)
CB_MODE_SELECT = "md:sel:"  # md:sel:<mode>
CB_MODE_CANCEL = "md:cancel"
CB_MODE_SET = "md:set:"  # md:set:<mode>:<window>  (Shift+Tab cycling)

# Window lifecycle
CB_RESTART = "rw:go:"  # rw:go:<window>[:<mode>]
CB_KILL = "rw:kill:"  # rw:kill:<window>

# Special topics
CB_SHELL_KILL = "sh:kill:"  # sh:kill:<message_id>
CB_CCC_USE = "cc:use:"  # cc:use:<account id>
CB_CCC_NEXT = "cc:next:"  # cc:next:<provider>
CB_CCC_REFRESH = "cc:ref:"  # cc:ref:net
CB_CCC_RESTART = "cc:restart"  # restart all Claude sessions
CB_SETTING = "st:"  # st:<setting key>  (toggle / cycle)
