# DeepSeek Coder CLI

A lightweight terminal-based chat interface for **DeepSeek-Coder-6.7B-Instruct** using `llama.cpp`. Supports streaming responses, multi-turn conversations, file attachments, and CPU performance benchmarking.

## Requirements

- Python 3.10+
- [llama.cpp](https://github.com/ggml-org/llama.cpp) (built with `llama-server` and `llama-bench`)
- DeepSeek-Coder-6.7B-Instruct GGUF model

No additional Python packages required.

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

What I did was to make it less wordy (.zshrc or bashrc):

```bash
alias ds-chat='python3 deepseek_chat.py'
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
