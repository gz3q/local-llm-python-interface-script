#!/usr/bin/env python3
"""
@file deepseek_chat.py
@brief Terminal interface for DeepSeek-Coder-6.7B-Instruct (GGUF) on llama.cpp.

@details
  * Launches llama-server (from your ~/llama.cpp build) with CPU-tuned flags, or attaches
    to one that is already running on the chosen port.
  * Chat REPL: streaming output, multi-turn memory, prompt-cache reuse, automatic history
    trimming when the context fills up, tokens/sec stats after every reply.
  * One-shot mode for pipes:   cat app.py | ./deepseek_chat.py -p "Review this code"
  * --bench    sweeps thread counts with llama-bench so you can pick the fastest setting.
  * --serve-only keeps the server running for the built-in web UI or editor plugins that
    speak the OpenAI API (http://127.0.0.1:8080/v1).

Standard library only - no pip installs needed.

@par Program flow
  -# parse_args()          read the command line
  -# server_ready()        is a llama-server already listening? attach : start_server()
  -# wait_ready()          poll /health until the model is loaded
  -# Chat / repl()         interactive loop, or Chat.ask() once in one-shot mode
  -# stop_server()         terminate the server if this script started it

@par Debugging tips
  - Server stdout/stderr go to @ref LOG_PATH (~/.cache/deepseek-chat/server.log).
  - The exact llama-server command line is printed at startup.
  - Run `llama-server` yourself with that command to see errors live; the script will
    then attach to it instead of spawning its own.

@defgroup cfg    Configuration and helpers
@defgroup http   HTTP client for llama-server
@defgroup server llama-server process management
@defgroup chat   Chat session and DeepSeek prompt format
@defgroup cli    Interactive REPL and command line
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

try:  # line editing + history for input()
    import readline  # noqa: F401
except ImportError:
    pass

# --------------------------------------------------------------------------- config

## @ingroup cfg
## @brief End-of-turn token DeepSeek Coder emits when it finishes a reply.
## @details Also written after each assistant turn when rebuilding the history prompt.
EOT = "<|EOT|>"

## @ingroup cfg
## @brief Strings that make the server stop generating.
## @details `<|EOT|>` is the normal end; `### Instruction:` guards against the model
## inventing the next user turn by itself.
STOP = [EOT, "### Instruction:"]

## @ingroup cfg
## @brief Default system prompt (override with --system or /system).
## @details The official DeepSeek default tells the model to refuse "security and
## privacy issues", which gets in the way of normal coding work (auth, crypto, etc.),
## so this script uses a coding-focused one.
DEFAULT_SYSTEM = (
    "You are an expert programming assistant. Give correct, idiomatic, production-quality "
    "code with brief explanations. Put code in fenced blocks with a language tag. "
    "If a request is ambiguous, state your assumption and proceed."
)

## @ingroup cfg
## @brief File that receives llama-server's stdout/stderr when this script spawns it.
LOG_PATH = Path.home() / ".cache" / "deepseek-chat" / "server.log"

## @ingroup cfg
## @brief True when stdout is a terminal; ANSI colours are only emitted in that case.
IS_TTY = sys.stdout.isatty()


def paint(code: str, text: str) -> str:
    """
    @ingroup cfg
    @brief Wrap text in an ANSI colour/style escape sequence.
    @param code ANSI SGR code(s), e.g. "1;31" for bold red, "2" for dim.
    @param text The text to colourise.
    @return The coloured string, or @p text unchanged when stdout is not a TTY
            (so piped output stays free of escape codes).
    """
    return f"\033[{code}m{text}\033[0m" if IS_TTY else text


def die(msg: str) -> None:
    """
    @ingroup cfg
    @brief Abort the program with a red "error:" message.
    @param msg Human-readable explanation, may span several lines.
    @throws SystemExit Always. The message goes to stderr and the exit code is 1;
            the `finally` block in main() still runs, so a spawned server is stopped.
    """
    raise SystemExit(paint("1;31", "error: ") + msg)


# --------------------------------------------------------------------------- HTTP


def http_json(url: str, payload: dict | None = None, timeout: float = 10) -> dict:
    """
    @ingroup http
    @brief Send a GET (no payload) or JSON POST (with payload) and decode the JSON reply.
    @param url     Full URL, e.g. "http://127.0.0.1:8080/health".
    @param payload Object to send as the JSON body. @c None makes this a GET request.
    @param timeout Socket timeout in seconds.
    @return The decoded JSON response.
    @throws urllib.error.URLError  Connection refused, DNS failure, timeout.
    @throws urllib.error.HTTPError Non-2xx status (llama-server answers 503 while loading).
    @throws json.JSONDecodeError   The body was not valid JSON.
    """
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def server_ready(base: str) -> bool:
    """
    @ingroup http
    @brief Check whether a llama-server is up and has finished loading the model.
    @param base Base URL of the server, e.g. "http://127.0.0.1:8080".
    @return True only if GET /health answers 200 with {"status": "ok"}.
            Any error (refused connection, 503 "loading model", bad JSON) gives False.
    @note Never raises, so it is safe to poll in a loop.
    """
    try:
        return http_json(base + "/health", timeout=2).get("status") == "ok"
    except Exception:  # connection refused, 503 while loading, etc.
        return False


def server_ctx(base: str, fallback: int) -> int:
    """
    @ingroup http
    @brief Ask the server for its real per-slot context size.
    @param base     Base URL of the server.
    @param fallback Value returned when /props is unavailable or has an unexpected shape.
    @return n_ctx from GET /props (default_generation_settings.n_ctx), else @p fallback.
    @note Matters when attaching to a server started elsewhere with a different -c value.
    """
    try:
        props = http_json(base + "/props", timeout=5)
        return int(props["default_generation_settings"]["n_ctx"])
    except Exception:
        return fallback


def stream_completion(base: str, payload: dict):
    """
    @ingroup http
    @brief Stream a completion from llama-server's /completion endpoint (Server-Sent Events).
    @param base    Base URL of the server.
    @param payload JSON body for /completion; must contain "stream": true.
    @return Generator yielding one decoded dict per SSE `data:` line. Intermediate chunks look
            like {"content": "tok", "stop": false}; the last has "stop": true plus
            "timings", "stop_type" and token counts.
    @throws urllib.error.URLError If the connection fails or drops.
    @note Blank lines, non-`data:` lines, "[DONE]" and malformed JSON are skipped silently.
    @note Closing the generator (e.g. on Ctrl+C) closes the socket, which makes the
          server cancel the generation.
    """
    req = urllib.request.Request(
        base + "/completion",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    # timeout is per socket read; the first token can take a while on a long prompt
    with urllib.request.urlopen(req, timeout=900) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            body = line[5:].strip()
            if not body or body == "[DONE]":
                continue
            try:
                yield json.loads(body)
            except json.JSONDecodeError:
                continue


# --------------------------------------------------------------------------- server


def tail(path: Path, n: int = 25) -> str:
    """
    @ingroup server
    @brief Return the last lines of a text file (used to show server errors).
    @param path File to read.
    @param n    Number of trailing lines to keep.
    @return The last @p n lines joined together, or "(no log)" if the file cannot be read.
    """
    try:
        return "".join(path.read_text(errors="replace").splitlines(keepends=True)[-n:])
    except OSError:
        return "(no log)"


def check_model(model: Path) -> None:
    """
    @ingroup server
    @brief Verify that @p model exists and really is a GGUF file.
    @param model Path to the .gguf model file.
    @throws SystemExit (via die()) If the file is missing, or does not start with the
            magic bytes "GGUF". The second case usually means `wget` was pointed at a
            Hugging Face `/blob/` URL and saved the HTML page; the message shows the
            correct `/resolve/` URL.
    """
    if not model.is_file():
        die(f"model not found: {model}")
    with model.open("rb") as f:
        magic = f.read(4)
    if magic != b"GGUF":
        die(
            f"{model} is not a GGUF file (size {model.stat().st_size:,} bytes).\n"
            "  If you used wget on a huggingface.co/.../blob/... URL you saved the web page.\n"
            "  Use the /resolve/ form instead:\n"
            "  wget -O ~/llm/models/gguf/deepseek-coder-6.7b-instruct.Q5_K_M.gguf \\\n"
            "    https://huggingface.co/TheBloke/deepseek-coder-6.7B-instruct-GGUF/resolve/main/"
            "deepseek-coder-6.7b-instruct.Q5_K_M.gguf"
        )


def start_server(a: argparse.Namespace) -> subprocess.Popen:
    """
    @ingroup server
    @brief Spawn llama-server in the background with the tuned flags.
    @param a Parsed command-line arguments. Fields used: llama_dir, model, ctx, threads,
             threads_batch, host, port, kv_cache, mlock, server_args.
    @return The running child process. The caller must pass it to wait_ready() and,
            eventually, stop_server().
    @throws SystemExit (via die()) If the llama-server binary or the model is missing/invalid.
    @par Flags passed to llama-server
      - `-c`   context size (--ctx)
      - `-t`   threads for token generation (--threads)
      - `-tb`  threads for prompt processing (--threads-batch)
      - `-np 1` a single slot, otherwise recent builds divide the context between slots
      - `--flash-attn on -ctk/-ctv` only when --kv-cache is not f16 (quantised V needs FA)
      - `--mlock` only when requested
      - anything in --server-args is appended last, so it can override the above
    @note stdout/stderr of the server are written to @ref LOG_PATH, and the full command
          line is echoed to the terminal.
    """
    server = Path(a.llama_dir).expanduser() / "build" / "bin" / "llama-server"
    if not server.is_file():
        die(
            f"llama-server not found at {server} (check --llama-dir and that the build finished)"
        )
    model = Path(a.model).expanduser()
    check_model(model)

    cmd = [
        str(server),
        "-m",
        str(model),
        "-c",
        str(a.ctx),
        "-t",
        str(a.threads),  # threads for token generation
        "-tb",
        str(a.threads_batch),  # threads for prompt processing
        "-np",
        "1",  # ONE slot: otherwise recent builds split the context
        "--host",
        a.host,
        "--port",
        str(a.port),
        "--alias",
        "deepseek-coder-6.7b-instruct",
    ]
    if a.kv_cache != "f16":  # quantized KV cache needs flash attention
        cmd += ["--flash-attn", "on", "-ctk", a.kv_cache, "-ctv", a.kv_cache]
    if a.mlock:
        cmd.append("--mlock")
    if a.server_args:
        cmd += shlex.split(a.server_args)

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    log = LOG_PATH.open("w")
    print(paint("2", "$ " + " ".join(shlex.quote(c) for c in cmd)))
    print(paint("2", f"(server log: {LOG_PATH})"))
    return subprocess.Popen(
        cmd, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT
    )


def wait_ready(base: str, proc: subprocess.Popen, timeout: float = 300) -> None:
    """
    @ingroup server
    @brief Block until the freshly started server answers /health, showing a progress line.
    @param base    Base URL of the server.
    @param proc    The process returned by start_server(); watched so a crash is noticed early.
    @param timeout Maximum seconds to wait.
    @throws SystemExit (via die()) If the process exits before becoming ready, or the
            timeout expires. In both cases the last lines of @ref LOG_PATH are shown.
    """
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            die(
                f"llama-server exited with code {proc.returncode}. Last log lines:\n{tail(LOG_PATH)}"
            )
        if server_ready(base):
            print(f"\rServer ready in {time.time() - t0:.1f}s." + " " * 20)
            return
        print(f"\rLoading model... {time.time() - t0:4.0f}s", end="", flush=True)
        time.sleep(0.5)
    die(f"server not ready after {timeout:.0f}s. Last log lines:\n{tail(LOG_PATH)}")


def stop_server(proc: subprocess.Popen | None) -> None:
    """
    @ingroup server
    @brief Terminate the server process if this script started it.
    @param proc Process from start_server(), or None when we attached to an existing server
                (in which case nothing is done, so someone else's server is never killed).
    @details Sends SIGTERM, waits up to 5 s, then SIGKILL.
    """
    if proc and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def run_bench(a: argparse.Namespace) -> None:
    """
    @ingroup server
    @brief Run llama-bench over several thread counts to find the fastest settings.
    @param a Parsed command-line arguments. Fields used: llama_dir, model.
    @throws SystemExit (via die()) If llama-bench or the model is missing/invalid.
    @details Tests 2, 4, 6, 8, 10 and the logical CPU count (those not above it) with a
             256-token prompt and 64 generated tokens, 2 repetitions each. Output is the
             native llama-bench table: use the best `tg` row for --threads and the best
             `pp` row for --threads-batch.
    """
    bench = Path(a.llama_dir).expanduser() / "build" / "bin" / "llama-bench"
    if not bench.is_file():
        die(f"llama-bench not found at {bench}")
    model = Path(a.model).expanduser()
    check_model(model)
    cpu = os.cpu_count() or 4
    counts = sorted({t for t in (2, 4, 6, 8, 10, cpu) if t <= cpu})
    cmd = [
        str(bench),
        "-m",
        str(model),
        "-t",
        ",".join(map(str, counts)),
        "-p",
        "256",
        "-n",
        "64",
        "-r",
        "2",
    ]
    print(paint("2", "$ " + " ".join(cmd)))
    print(
        "pp = prompt processing (t/s), tg = token generation (t/s). "
        "Use the best tg row for --threads and the best pp row for --threads-batch.\n"
        "Plug in AC power and close heavy apps first. This takes several minutes.\n"
    )
    subprocess.run(cmd, check=False)


# --------------------------------------------------------------------------- chat


class Chat:
    """
    @ingroup chat
    @brief A multi-turn conversation with the model.

    @details Keeps the history, renders it into DeepSeek's instruct format, trims old
    turns when the context window is full, and streams replies from the server.

    @par Prompt format (what the server actually receives)
    @code
    <system prompt>
    ### Instruction:
    <user turn 1>
    ### Response:
    <assistant turn 1>
    <|EOT|>
    ### Instruction:
    <current user message>
    ### Response:
    @endcode
    The BOS token is added by the server, so it is intentionally not in the text.

    @par Attributes
      - @c base       Base URL of llama-server.
      - @c n_ctx      Context window in tokens (real value from the server when available).
      - @c system     System prompt text.
      - @c temp       Sampling temperature.
      - @c top_p      Nucleus sampling threshold.
      - @c top_k      Top-k sampling cutoff.
      - @c max_tokens Upper bound on tokens generated per reply.
      - @c turns      History as a list of (user_message, assistant_reply) tuples.
    """

    def __init__(
        self,
        base: str,
        n_ctx: int,
        system: str,
        temp: float,
        top_p: float,
        top_k: int,
        max_tokens: int,
    ):
        """
        @brief Create an empty conversation.
        @param base       Base URL of llama-server, e.g. "http://127.0.0.1:8080".
        @param n_ctx      Context window in tokens.
        @param system     System prompt text.
        @param temp       Sampling temperature (lower = more deterministic).
        @param top_p      Nucleus sampling threshold.
        @param top_k      Top-k sampling cutoff.
        @param max_tokens Maximum tokens generated per reply.
        """
        self.base, self.n_ctx, self.system = base, n_ctx, system
        self.temp, self.top_p, self.top_k, self.max_tokens = (
            temp,
            top_p,
            top_k,
            max_tokens,
        )
        self.turns: list[tuple[str, str]] = []

    # DeepSeek-Coder instruct format. The server adds <BOS> itself, so it is not included.
    def build_prompt(self, user_msg: str) -> str:
        """
        @brief Render system prompt + history + the new message into the DeepSeek format.
        @param user_msg The message about to be sent.
        @return The full prompt string, ending with "### Response:\\n" so the model
                continues with its answer.
        @note Pure function of the current state; does not modify the history.
        """
        parts = [self.system.strip() + "\n"]
        for u, a in self.turns:
            parts.append(f"### Instruction:\n{u}\n")
            parts.append(f"### Response:\n{a}\n{EOT}\n")
        parts.append(f"### Instruction:\n{user_msg}\n### Response:\n")
        return "".join(parts)

    def count_tokens(self, text: str) -> int:
        """
        @brief Count tokens exactly, using the server's own tokenizer.
        @param text Text to measure (special tokens such as <|EOT|> are parsed, and BOS is counted).
        @return Number of tokens. If /tokenize fails, a conservative estimate of
                len(text)/3 + 1 is returned instead (overestimates, so trimming errs on the safe side).
        """
        try:
            r = http_json(
                self.base + "/tokenize",
                {"content": text, "add_special": True, "parse_special": True},
            )
            return len(r["tokens"])
        except Exception:
            return len(text) // 3 + 1  # conservative fallback

    def prepare(self, user_msg: str) -> tuple[str, int, int]:
        """
        @brief Build the prompt, trimming the oldest turns until it fits in the context.
        @param user_msg The message about to be sent.
        @return Tuple (prompt, n_prompt, n_predict): the final prompt text, its token count,
                and how many tokens the model may generate (min of max_tokens and the
                space left in the context).
        @throws ValueError If even with no history the message leaves fewer than 64 tokens
                for the answer. The original history is restored in that case.
        @note Side effect: pops entries from @c turns and prints a warning to stderr when
              turns are dropped. Dropping history invalidates the server's prompt cache,
              so the next request re-processes the whole prompt once.
        """
        dropped = 0
        backup = list(self.turns)
        while True:
            prompt = self.build_prompt(user_msg)
            n_prompt = self.count_tokens(prompt)
            if n_prompt + self.max_tokens + 8 <= self.n_ctx or not self.turns:
                break
            self.turns.pop(0)
            dropped += 1
        if dropped:
            print(
                paint("33", f"[context full: dropped {dropped} oldest turn(s)]"),
                file=sys.stderr,
            )
        n_predict = min(self.max_tokens, self.n_ctx - n_prompt - 8)
        if n_predict < 64:
            self.turns = backup  # don't lose history over a message that can't be sent
            raise ValueError(
                f"message is {n_prompt} tokens but the context is only {self.n_ctx}. "
                "Send less, or restart with a larger --ctx (and --kv-cache q8_0 to save RAM)."
            )
        return prompt, n_prompt, n_predict

    def ask(self, user_msg: str, out=sys.stdout) -> str | None:
        """
        @brief Send a message, stream the reply to @p out, and record the turn.
        @param user_msg The user's message (attached files are already merged into it).
        @param out      Writable text stream receiving the reply as it is generated.
        @return The full reply text, or None if the user pressed Ctrl+C (the turn is
                then discarded and not added to the history).
        @throws ValueError            Message too long for the context (see prepare()).
        @throws urllib.error.URLError The server is unreachable or dropped the connection.
        @par Request details
          - @c cache_prompt reuses the KV cache for the unchanged conversation prefix, so
            only new tokens are processed on follow-up turns.
          - @c repeat_penalty is 1.0 (disabled): penalties damage code, which repeats tokens.
          - The leading newline the model emits after "### Response:" is stripped.
        @par Output
          After the reply, one line is printed to stderr:
          `[ctx used/total | prompt N new tok @ x t/s | gen M tok @ y t/s]`. "new tok" is
          smaller than the real prompt size whenever the cache was hit. A second line warns
          when generation stopped because it reached max_tokens.
        """
        prompt, n_prompt, n_predict = self.prepare(user_msg)
        payload = {
            "prompt": prompt,
            "stream": True,
            "n_predict": n_predict,
            "temperature": self.temp,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "min_p": 0.05,
            "repeat_penalty": 1.0,  # penalties hurt code, which legitimately repeats tokens
            "stop": STOP,
            "cache_prompt": True,  # reuse the KV cache of the shared conversation prefix
        }
        pieces: list[str] = []
        started = False
        final: dict = {}
        try:
            for chunk in stream_completion(self.base, payload):
                text = chunk.get("content", "")
                if text and not started:
                    text = text.lstrip("\n")
                    started = bool(text)
                if text:
                    out.write(text)
                    out.flush()
                    pieces.append(text)
                if chunk.get("stop"):
                    final = chunk
                    break
        except KeyboardInterrupt:
            print(
                paint("33", "\n[interrupted - this turn was discarded]"),
                file=sys.stderr,
            )
            return None
        out.write("\n")
        reply = "".join(pieces).rstrip()
        self.turns.append((user_msg, reply))

        t = final.get("timings", {})
        used = n_prompt + int(
            t.get("predicted_n", final.get("tokens_predicted", 0)) or 0
        )
        stats = (
            f"[ctx {used}/{self.n_ctx} | prompt {t.get('prompt_n', '?')} new tok "
            f"@ {t.get('prompt_per_second', 0):.1f} t/s | gen {t.get('predicted_n', '?')} tok "
            f"@ {t.get('predicted_per_second', 0):.1f} t/s]"
        )
        print(paint("2", stats), file=sys.stderr)
        if final.get("stop_type") == "limit" or final.get("stopped_limit"):
            print(
                paint("33", "[hit --max-tokens; type 'continue' or use /max N]"),
                file=sys.stderr,
            )
        return reply


## @ingroup cli
## @brief Text printed by the /help command.
HELP = """\
Commands:
  /file PATH     attach a file to your next message (repeatable)
  /system [TXT]  show or replace the system prompt
  /temp X        set temperature (0.0-1.0; lower = more deterministic)
  /max N         set max tokens per reply
  /reset         clear conversation history
  /save PATH     write the conversation to a markdown file
  /help, /exit
Input:
  Type  \"\"\"  on its own line to start/stop a multi-line paste.
  Ctrl+C stops a generation in progress. Ctrl+D exits."""


def read_input() -> str:
    """
    @ingroup cli
    @brief Read one user message from the terminal, with multi-line paste support.
    @return The entered text. A line containing only `\"\"\"` starts a multi-line block that
            ends at the next such line; the lines in between are joined with newlines.
    @throws EOFError           On Ctrl+D (handled by repl() as "quit").
    @throws KeyboardInterrupt  On Ctrl+C (handled by repl() as "cancel input").
    @note The \\001 / \\002 characters around the colour codes tell readline they take no
          screen width; without them line editing and history redraw incorrectly.
    """
    prompt = "\001\033[1;36m\002>>> \001\033[0m\002" if IS_TTY else ">>> "
    line = input(prompt)
    if line.strip() == '"""':
        buf: list[str] = []
        while True:
            more = input("... ")
            if more.strip() == '"""':
                break
            buf.append(more)
        return "\n".join(buf)
    return line


def fence(name: str, text: str) -> str:
    """
    @ingroup cli
    @brief Format a file's contents as a labelled Markdown code block for the prompt.
    @param name File name (its extension becomes the code-fence language tag).
    @param text File contents.
    @return A string like "File `app.py`:" followed by a fenced block.
    """
    lang = Path(name).suffix.lstrip(".")
    return f"File `{name}`:\n```{lang}\n{text.rstrip()}\n```"


def handle_command(line: str, chat: Chat, files: list[tuple[str, str]]) -> bool:
    """
    @ingroup cli
    @brief Execute a slash command typed in the REPL.
    @param line  The full input line, starting with "/" (e.g. "/temp 0.3").
    @param chat  The active conversation; commands may change its settings or history.
    @param files Pending attachments list, appended to by /file and emptied by /reset.
    @return True if the REPL should quit (/exit, /quit, /q), otherwise False.
    @par Commands
      /help, /reset, /system [text], /temp X, /max N, /file PATH, /save PATH, /exit.
      Bad arguments print a usage line instead of raising.
    """
    cmd, _, arg = line.partition(" ")
    arg = arg.strip()
    if cmd in ("/exit", "/quit", "/q"):
        return True
    if cmd == "/help":
        print(HELP)
    elif cmd == "/reset":
        chat.turns.clear()
        files.clear()
        print("History cleared.")
    elif cmd == "/system":
        if arg:
            chat.system = arg
            print("System prompt updated.")
        else:
            print(chat.system)
    elif cmd == "/temp":
        try:
            chat.temp = float(arg)
            print(f"temperature = {chat.temp}")
        except ValueError:
            print("usage: /temp 0.2")
    elif cmd == "/max":
        try:
            chat.max_tokens = int(arg)
            print(f"max tokens = {chat.max_tokens}")
        except ValueError:
            print("usage: /max 1024")
    elif cmd == "/file":
        p = Path(arg).expanduser()
        try:
            files.append((p.name, p.read_text(errors="replace")))
            print(
                f"Attached {p.name} ({len(files[-1][1])} chars). It will go with your next message."
            )
        except OSError as e:
            print(f"cannot read file: {e}")
    elif cmd == "/save":
        if not arg:
            print("usage: /save session.md")
        else:
            out = [f"# DeepSeek Coder session\n\n_System:_ {chat.system}\n"]
            for u, a in chat.turns:
                out.append(f"\n## You\n\n{u}\n\n## Assistant\n\n{a}\n")
            Path(arg).expanduser().write_text("".join(out))
            print(f"Saved to {arg}")
    else:
        print("Unknown command. /help lists them.")
    return False


def repl(chat: Chat, a: argparse.Namespace) -> None:
    """
    @ingroup cli
    @brief Interactive read-eval-print loop.
    @param chat The conversation to drive.
    @param a    Parsed command-line arguments (currently unused, kept for future options).
    @details Loop: read input -> run slash command, or merge pending file attachments into
             the message and call Chat::ask(). Exits on Ctrl+D, /exit, or when the server
             connection is lost. A too-long message prints the error and keeps going.
    """
    print(
        paint("1", "DeepSeek Coder 6.7B Instruct")
        + f"  ctx={chat.n_ctx}  temp={chat.temp}  (/help for commands)"
    )
    files: list[tuple[str, str]] = []
    while True:
        try:
            msg = read_input().strip()
        except EOFError:
            print()
            return
        except KeyboardInterrupt:
            print("\n(Ctrl+D or /exit to quit)")
            continue
        if not msg:
            continue
        if msg.startswith("/"):
            if handle_command(msg, chat, files):
                return
            continue
        if files:
            msg = "\n\n".join(fence(n, t) for n, t in files) + "\n\n" + msg
            files.clear()
        try:
            chat.ask(msg)
        except ValueError as e:
            print(paint("31", str(e)))
        except (urllib.error.URLError, ConnectionError, OSError) as e:
            print(paint("31", f"server error: {e}"))
            return


# --------------------------------------------------------------------------- main


def parse_args() -> argparse.Namespace:
    """
    @ingroup cli
    @brief Define and parse the command-line options.
    @return Namespace with these fields: llama_dir, model, ctx, threads, threads_batch,
            kv_cache, mlock, host, port, server_args, system, temp, top_p, top_k,
            max_tokens, prompt, bench, serve_only.
    @note Thread defaults are derived from os.cpu_count(): cpu//2 for generation,
          cpu for prompt processing. They are starting points; run --bench to tune.
    """
    cpu = os.cpu_count() or 4
    p = argparse.ArgumentParser(
        description="Chat with DeepSeek-Coder-6.7B-Instruct via llama.cpp (CPU-tuned)."
    )
    p.add_argument(
        "--llama-dir",
        default="~/llama.cpp",
        help="llama.cpp checkout (default: %(default)s)",
    )
    p.add_argument(
        "--model", default="~/llm/models/gguf/deepseek-coder-6.7b-instruct.Q5_K_M.gguf"
    )
    p.add_argument(
        "--ctx",
        type=int,
        default=4096,
        help="context size. f16 KV costs ~0.5 MB/token for this model (4096 ~= 2 GB)",
    )
    p.add_argument(
        "--threads",
        type=int,
        default=max(1, cpu // 2),
        help="generation threads (default: %(default)s; tune with --bench)",
    )
    p.add_argument(
        "--threads-batch",
        type=int,
        default=cpu,
        help="prompt-processing threads (default: %(default)s)",
    )
    p.add_argument(
        "--kv-cache",
        choices=["f16", "q8_0"],
        default="f16",
        help="q8_0 halves KV-cache RAM (lets you raise --ctx) with negligible quality loss",
    )
    p.add_argument(
        "--mlock", action="store_true", help="pin the model in RAM (avoids paging)"
    )
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument(
        "--server-args",
        default="",
        help='extra llama-server flags, e.g. "-C 0-3 --cpu-strict 1"',
    )
    p.add_argument("--system", default=DEFAULT_SYSTEM)
    p.add_argument("--temp", type=float, default=0.2)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument(
        "-p",
        "--prompt",
        help="one-shot mode: answer this prompt and exit (stdin is appended)",
    )
    p.add_argument(
        "--bench",
        action="store_true",
        help="sweep thread counts with llama-bench and exit",
    )
    p.add_argument(
        "--serve-only", action="store_true", help="start the server and keep it running"
    )
    return p.parse_args()


def main() -> None:
    """
    @ingroup cli
    @brief Program entry point: wire together server, chat session and UI mode.
    @details Mode selection:
      - `--bench`       run_bench() and return (no server needed)
      - `--serve-only`  start/attach the server; block until Ctrl+C
      - `-p TEXT` or piped stdin   one-shot: ask once, print the reply, exit
      - otherwise       interactive repl()
    In one-shot mode with piped stdin, the piped text is appended to the -p text inside
    a code fence. The reply goes to stdout and the stats line to stderr, so
    `script -p ... > out.txt` captures only the answer.
    @note The `finally` block always calls stop_server(), so a server spawned by this
          run is shut down on normal exit, Ctrl+C, errors and SIGTERM. A server that was
          already running (attached) is never stopped.
    """
    a = parse_args()
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    if a.bench:
        return run_bench(a)

    base = f"http://{a.host}:{a.port}"
    proc: subprocess.Popen | None = None
    try:
        if server_ready(base):
            print(paint("2", f"Attached to llama-server already running at {base}"))
        else:
            proc = start_server(a)
            wait_ready(base, proc)

        if a.serve_only:
            if proc is None:
                return
            print(f"Serving. Web UI: {base}   OpenAI API: {base}/v1   (Ctrl+C to stop)")
            proc.wait()
            return

        chat = Chat(
            base,
            server_ctx(base, a.ctx),
            a.system,
            a.temp,
            a.top_p,
            a.top_k,
            a.max_tokens,
        )

        if a.prompt is not None or not sys.stdin.isatty():
            text = a.prompt or ""
            if not sys.stdin.isatty():
                piped = sys.stdin.read()
                if piped.strip():
                    text = (
                        (text + "\n\n" if text else "")
                        + "```\n"
                        + piped.rstrip()
                        + "\n```"
                    )
            if not text.strip():
                die("nothing to send (empty -p and empty stdin)")
            try:
                chat.ask(text)
            except ValueError as e:
                die(str(e))
        else:
            repl(chat, a)
    except KeyboardInterrupt:
        print()
    finally:
        stop_server(proc)


if __name__ == "__main__":
    main()
