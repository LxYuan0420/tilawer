## llama.vim + llama-server: an offline copilot that just works

I found a small tool that made me smile. It's llama.vim, a Vim/Neovim
plugin that gives inline code suggestions from a local model. It just works, it's
fast, and it looks pretty in the editor. For me, it's a simple, solid
offline alternative to GitHub Copilot.

## What I like
- Local first. Runs a local server and keeps code on my machine.
- Simple setup. Install the plugin and start the server. Done.
- Low latency. Suggestions pop in quickly with little delay.
- Good defaults. It shows inline hints and lets me accept with Tab.

## setup I used
- plugin (vim-plug)
  ```vim
  Plug 'ggml-org/llama.vim'

  " Optional: point to your server if not default
  let g:llama_config = {
        \ 'endpoint': 'http://127.0.0.1:8012',
        \ 'auto_fim': 1,  " enable fill-in-the-middle inline suggestions
        \ }
  ```
- server (llama.cpp)

  ```bash
  # Run the local HTTP server with a coder model
  llama-server \
    -hf ggml-org/Qwen2.5-Coder-7B-Instruct-Q8_0-GGUF \
    --host 127.0.0.1 --port 8012
  ```
  - From my log: HTTP listens on `127.0.0.1:8012` and loads the GGUF model
    `qwen2.5-coder-7b-instruct-q8_0.gguf` (Qwen2.5 Coder 7B Instruct, Q8_0).
  - Keep this running in a terminal; the plugin will talk to it over HTTP.

## how I verified requests

Open Vim and type. The inline red text is the suggestion. Press Tab to accept
it.

Or you can probe the server directly (helpful for debugging or curiosity).

chat style (OpenAI-compatible)
```bash
curl -s http://127.0.0.1:8012/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 32
  }'
```

# llama.vim uses FIM for inline suggestions; you can call the infill endpoint directly
infill (fill‑in‑the‑middle)
```bash
curl -s http://127.0.0.1:8012/infill \
  -H 'Content-Type: application/json' \
  -d '{
    "input_prefix": "def add(a, b):",
    "input_suffix": "#",
    "prompt": "",
    "n_predict": 16
  }'
```

## Where this fits
- Online tools like GitHub Copilot are great when you have internet.
- There are many coding agents like Codex CLI and Claude Code that help with
  larger tasks.
- Llama.vim feels like a clean offline copilot. It keeps me in my editor and it
  just works.


## Links
- llama.vim: https://github.com/ggml-org/llama.vim
- llama.cpp (server): https://github.com/ggml-org/llama.cpp
