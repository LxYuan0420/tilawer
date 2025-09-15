<!-- Filename: 2025-09-15_til_ctrl-z-fg-vim.md -->

# Jump out and back to Vim with Ctrl+Z and fg

## What
Suspend Vim with Ctrl+Z, run a shell command, then use `fg` to return instantly —
no closing files or losing your place.

## Context
- Shells: bash/zsh with job control on (default in interactive shells).
- Works in a tmux pane; `fg` must be issued in the same shell session.
- Handy when working on a small screen and you need quick one‑off commands.

## Steps / Snippet
```bash
vim app.py
# Need a quick check without quitting Vim:
# Press CTRL+Z to suspend Vim (you'll see: [1]+  Stopped  vim app.py)
grep -n TODO -R .
jobs           # optional: list jobs
fg             # resume the last stopped job (Vim) in the foreground
# Or, explicitly by job:
fg %1
```

## How it works
- CTRL+Z sends the terminal’s suspend character (configured by `stty susp`), and the
  kernel delivers `SIGTSTP` to the foreground process group (Vim and any children).
- Vim stops; the shell records the job as "Stopped" and regains control of the TTY.
- `fg` is a shell builtin: it selects a job, sends `SIGCONT` to its process group, and
  uses `tcsetpgrp` to hand the terminal back so Vim resumes exactly where it paused.

## Pitfalls
- Use the same pane/tab: `fg` only sees jobs created by that shell.
- Some tools may trap suspension or alter TTY modes; Vim behaves well.
- Non‑interactive shells or disabled job control (`set +m` in bash) prevent this.

## Links
- GNU Bash Manual: Job Control Basics
  https://www.gnu.org/software/bash/manual/html_node/Job-Control-Basics.html
- POSIX `fg` specification
  https://pubs.opengroup.org/onlinepubs/9699919799/utilities/fg.html
- Signals overview (`SIGTSTP`/`SIGCONT`)
  https://man7.org/linux/man-pages/man7/signal.7.html

Now we really are not exiting Vim.

