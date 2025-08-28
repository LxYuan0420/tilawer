# Amp CLI vs Codex CLI

IDEs have never worked for me. I like the terminal and Vim. I want a
terminal-first coding tool that’s fast and feels almost unlimited. Then, I
found Amp CLI and Codex CLI. I’m surprised not many people talk about these two
tools. My guess is that it is easy to overspend since they bill by usage and
every step makes a paid API call.


## Amp CLI. It just works until it eats up your credit
- Plug and play power: Installed in minutes and ran right away. No model picking.
- Quick results and quick burn: My ten dollar free credit was gone in under an hour.
- Unconstrained tokens and tool use: aimming for the best result.
- Collaboration. Threads can keep project context across sessions.

## Codex CLI. The local champion that just works too
- Local first agent: Runs in my terminal. Code stays on my machine unless I choose to
  share it.
- Approval controls speed: Modes for suggest, auto edit, and full auto.
- Recent release highlights: Web search. Queued messages. Copy paste or drag and drop
  image files. Transcript mode with scrolling.
- Sandboxed safely. In full auto it runs in a protected space with limited file and
  network access.

Amp CLI gave excellent results out of the box. In my first hour the free ten
dollar credit was gone. Amp is not constrained on token use and tool use. Amp
decides the model for you and sends large prompts to get strong results. That
makes the output great but it also burns through credit fast.

I moved my daily work to Codex CLI. It runs local first. It is fast and
reliable. It feels private. My code stays on my machine unless I decide to
share it. With our company API access (Thanks!) I do not worry about personal
billing. The tool keeps a clear plan, shows diffs for edits, and uses approvals
and sandbox modes so commands do not surprise me. It fits the way I work in the
terminal.

I tried Gemini CLI. It is free, but it was slow for me and too careful. It
asked for approval for simple commands like ls and pwd. That broke my flow.

Claude Code looks capable and promising, but I hit setup hurdles. I needed to
request an approval and use a case code before I could use it. That was too
much friction for a quick start.

## Usage examples
Amp CLI
```bash
# Start in your project root
cd YOUR_PROJECT

# Set auth if your install needs it
export AMP_API_KEY=YOUR_AMP_KEY

# Discover commands
amp --help

# Start an interactive session in the project
amp
```

Codex CLI
```bash
# Start in your project root
cd YOUR_PROJECT

# Use company OpenAI access
export OPENAI_API_KEY=YOUR_COMPANY_OPENAI_API_KEY

# Discover commands
codex --help

# Start an interactive session in the project
codex
```
