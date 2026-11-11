# DeepSeek Coder CLI

A lightweight terminal chat interface for **DeepSeek-Coder-6.7B-Instruct** using `llama.cpp`. Supports streaming responses, multi-turn conversations, file attachments, and CPU performance benchmarking.

## Requirements

- Python 3.10+
- [llama.cpp](https://github.com/ggml-org/llama.cpp) (built with `llama-server` and `llama-bench`)
- DeepSeek-Coder-6.7B-Instruct GGUF model
- `rich` (only for `deepseek_chat.py`; `deepseek_chat_no_style.py` needs no packages)

## Usage

Start an interactive chat:

```bash
python3 deepseek_chat.py
```

Send a one-shot prompt:

```bash
python3 deepseek_chat.py -p "Explain binary search trees"
```

Pipe a file into the model:

```bash
cat app.py | python3 deepseek_chat.py -p "Review this code"
```

Run a performance benchmark:

```bash
python3 deepseek_chat.py --bench
```

Keep the server running:

```bash
python3 deepseek_chat.py --serve-only
```

## Quick launch

`ds-chat.sh` sets up a virtual environment, installs `rich` if needed, and starts the chat (For this, please use deepseek_chat.py instead of deepseek_chat_no_style.py). Add an alias to your `.zshrc` or `.bashrc`:

```bash
alias ds-chat='~/llm/scripts/ds-chat.sh'
```

## Interactive Commands

| Command | Description |
|---|---|
| `/file PATH` | Attach a file to the next message |
| `/system [TXT]` | View or change the system prompt |
| `/temp X` | Change temperature |
| `/max N` | Set maximum response tokens |
| `/reset` | Clear conversation history |
| `/save PATH` | Save conversation as Markdown |
| `/help` | Display available commands |
| `/exit` | Exit the application |

## Configuration

Command-line options allow customization of context size, CPU threads, KV cache precision, sampling parameters, and server arguments. Run:

```bash
python3 deepseek_chat.py --help
```

## Files

- `deepseek_chat.py` - styled version (Markdown rendering via `rich`)
- `deepseek_chat_no_style.py` - plain text version (standard library only)
- `ds-chat.sh` - venv + launch wrapper

The code is commented with Doxygen blocks.
