<!-- Filename: 2025-09-22_til_tmux-prefix-l-previous-session.md -->

# Switch to the previous tmux session with Prefix+L

## What
Use tmux Prefix+L (uppercase L) to jump back to your previous session instantly—instead of opening the sessions list with Prefix+s and navigating with j/k.

## Context
- Tool: tmux 3.x (default prefix is Ctrl+b).
- Binding: `L` (capital L) runs `switch-client -l` to toggle to the last session.
- Note: lowercase `l` typically switches to the last window, not session.

## Steps / Snippet
```bash
# Inside tmux
# Jump to previous session (toggle):
# Press: PREFIX then L   # e.g., Ctrl+b then Shift+L

# Old way (for contrast):
# Press: PREFIX then s   # then j/k to select, Enter to attach

# Ensure the default binding exists (if your config overrides it):
tmux bind-key L switch-client -l

# Optional: also keep last-window on lowercase l
tmux bind-key l last-window
```

## Pitfalls
- It’s uppercase L: fonts can make l/1/I/L look similar.
- Works only after you have at least two sessions visited; nothing to toggle otherwise.
- The “last session” is tracked per client; a new terminal/SSH will have its own history.
- If you rebind keys in `.tmux.conf`, confirm `bind-key L switch-client -l` is present.

## Links
- tmux(1) manual — switch-client
  https://man7.org/linux/man-pages/man1/tmux.1.html#SWITCH-CLIENT
- tmux key bindings (reference)
  https://man7.org/linux/man-pages/man1/tmux.1.html#KEY_BINDINGS

