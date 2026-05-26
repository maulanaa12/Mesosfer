"""
New and upgraded chat mode because a lot of the code has changed since the last one.

Intended to be run single GPU only atm:
python -m scripts.chat_cli
"""
import argparse
import os
import re
import shutil
import torch
from mesosfer.utils.common import compute_init, autodetect_device_type
from mesosfer.eval.engine import Engine
from mesosfer.utils.checkpoint_manager import load_model

parser = argparse.ArgumentParser(description='Chat with the model')
parser.add_argument('-i', '--source', type=str, default="sft", help="Source of the model: sft|rl")
parser.add_argument('-g', '--model-tag', type=str, default=None, help='Model tag to load')
parser.add_argument('-s', '--step', type=int, default=None, help='Step to load')
parser.add_argument('-p', '--prompt', type=str, default='', help='Prompt the model, get a single response back')
parser.add_argument('-t', '--temperature', type=float, default=0.6, help='Temperature for generation')
parser.add_argument('-k', '--top-k', type=int, default=50, help='Top-k sampling parameter')
parser.add_argument('-m', '--max-tokens', type=int, default=256, help='Maximum new tokens per response')
parser.add_argument('--plain', action='store_true', help='Disable styled terminal UI')
parser.add_argument('--device-type', type=str, default='', choices=['cuda', 'cpu', 'mps'], help='Device type for evaluation: cuda|cpu|mps. empty => autodetect')
args = parser.parse_args()


# Terminal UI helpers
USE_COLOR = (not args.plain) and os.isatty(1)
TERM_WIDTH = max(72, min(shutil.get_terminal_size((96, 24)).columns, 110))


class C:
    reset = "\033[0m" if USE_COLOR else ""
    bold = "\033[1m" if USE_COLOR else ""
    dim = "\033[2m" if USE_COLOR else ""
    orange = "\033[38;5;208m" if USE_COLOR else ""
    cyan = "\033[38;5;45m" if USE_COLOR else ""
    green = "\033[38;5;120m" if USE_COLOR else ""
    gray = "\033[38;5;245m" if USE_COLOR else ""
    white = "\033[38;5;255m" if USE_COLOR else ""


def strip_ansi(text):
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def visible_len(text):
    return len(strip_ansi(text))


def fit_text(text, width):
    if visible_len(text) <= width:
        return text + " " * (width - visible_len(text))
    plain = strip_ansi(text)
    return plain[:max(0, width - 1)] + "…"


def rule(label="", color=C.orange):
    if not label:
        print(f"{color}{'─' * TERM_WIDTH}{C.reset}")
        return
    text = f" {label} "
    side = max(0, TERM_WIDTH - visible_len(text))
    print(f"{color}{'─' * (side // 2)}{C.reset}{C.bold}{label}{C.reset}{color}{'─' * (side - side // 2)}{C.reset}")


def boxed(title, left_lines, right_lines):
    inner = TERM_WIDTH - 2
    left_w = min(34, max(26, inner // 3))
    sep = "│"
    right_w = inner - left_w - 1
    print(f"{C.orange}╭{'─' * (TERM_WIDTH - 2)}╮{C.reset}")
    print(f"{C.orange}│{C.reset}{C.orange}{fit_text(' ' + title, TERM_WIDTH - 2)}{C.reset}{C.orange}│{C.reset}")
    print(f"{C.orange}├{'─' * left_w}┬{'─' * right_w}┤{C.reset}")
    rows = max(len(left_lines), len(right_lines))
    for i in range(rows):
        left = left_lines[i] if i < len(left_lines) else ""
        right = right_lines[i] if i < len(right_lines) else ""
        print(f"{C.orange}│{C.reset}{fit_text(left, left_w)}{C.orange}{sep}{C.reset}{fit_text(right, right_w)}{C.orange}│{C.reset}")
    print(f"{C.orange}╰{'─' * left_w}┴{'─' * right_w}╯{C.reset}")


def print_welcome(meta):
    step = meta.get("step", "?") if isinstance(meta, dict) else "?"
    model_config = meta.get("model_config", {}) if isinstance(meta, dict) else {}
    depth = model_config.get("n_layer", "?")
    embd = model_config.get("n_embd", "?")
    left = [
        "",
        f"{C.bold}{C.white}Welcome back.{C.reset}",
        "",
        f"{C.orange}        ▄▄▄▄        {C.reset}",
        f"{C.orange}      ▄██████▄      {C.reset}",
        f"{C.orange}      ██ ██ ██      {C.reset}",
        f"{C.orange}    ▀████████▀      {C.reset}",
        f"{C.orange}      ▀█▀  ▀█▀      {C.reset}",
        "",
        f"{C.gray}{args.source.upper()} · step {step} · {device_type}{C.reset}",
    ]
    right = [
        f"{C.orange}{C.bold}Tips for getting started{C.reset}",
        f"{C.white}/help{C.reset} for commands and shortcuts",
        f"{C.white}/clear{C.reset} starts a fresh conversation",
        f"{C.white}/temperature 0.4{C.reset} changes sampling",
        "",
        f"{C.orange}{C.bold}Session{C.reset}",
        f"model tag: {args.model_tag or 'auto'}",
        f"layers: {depth} · embd: {embd}",
        f"top-k: {args.top_k} · temp: {args.temperature}",
        f"cwd: {os.getcwd()}",
    ]
    boxed(" Mesosfer Chat CLI ", left, right)
    print(f"{C.gray}? for shortcuts · Ctrl+C or /exit to quit{C.reset}")


def print_assistant_header():
    print()
    rule(" Mesosfer ", C.orange)


def print_footer():
    rule("", C.orange)
    print()


def print_console(message):
    print(f"{C.cyan}›{C.reset} {message}")

# Init the model and tokenizer

device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
model, tokenizer, meta = load_model(args.source, device, phase="eval", model_tag=args.model_tag, step=args.step)

# Special tokens for the chat state machine
bos = tokenizer.get_bos_token_id()
user_start, user_end = tokenizer.encode_special("<|user_start|>"), tokenizer.encode_special("<|user_end|>")
assistant_start, assistant_end = tokenizer.encode_special("<|assistant_start|>"), tokenizer.encode_special("<|assistant_end|>")

# Create Engine for efficient generation
engine = Engine(model, tokenizer)

if not args.prompt:
    print_welcome(meta)

conversation_tokens = [bos]

while True:

    if args.prompt:
        # Get the prompt from the launch command
        user_input = args.prompt
    else:
        # Get the prompt interactively from the console
        try:
            user_input = input(f"\n{C.orange}❯{C.reset} ").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{C.gray}Goodbye.{C.reset}")
            break

    # Handle special commands
    if user_input.lower() in ['quit', 'exit', '/quit', '/exit']:
        print(f"{C.gray}Goodbye.{C.reset}")
        break

    if user_input.lower() in ['clear', '/clear']:
        conversation_tokens = [bos]
        print_console("Conversation cleared.")
        continue

    if user_input.lower() in ['help', '/help', '?']:
        print_console("Commands: /clear, /temperature <0-2>, /topk <1-200>, /exit")
        print_console("Enter submits. Ctrl+C exits.")
        continue

    if user_input.lower().startswith('/temperature'):
        parts = user_input.split()
        if len(parts) == 1:
            print_console(f"Temperature: {args.temperature}")
        else:
            try:
                args.temperature = max(0.0, min(2.0, float(parts[1])))
                print_console(f"Temperature set to {args.temperature}")
            except ValueError:
                print_console("Invalid temperature.")
        continue

    if user_input.lower().startswith('/topk'):
        parts = user_input.split()
        if len(parts) == 1:
            print_console(f"Top-k: {args.top_k}")
        else:
            try:
                args.top_k = max(1, min(200, int(parts[1])))
                print_console(f"Top-k set to {args.top_k}")
            except ValueError:
                print_console("Invalid top-k.")
        continue

    if not user_input:
        continue

    # Add User message to the conversation
    conversation_tokens.append(user_start)
    conversation_tokens.extend(tokenizer.encode(user_input))
    conversation_tokens.append(user_end)

    # Kick off the assistant
    conversation_tokens.append(assistant_start)
    generate_kwargs = {
        "num_samples": 1,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_k": args.top_k,
    }
    response_tokens = []
    print_assistant_header()
    for token_column, token_masks in engine.generate(conversation_tokens, **generate_kwargs):
        token = token_column[0] # pop the batch dimension (num_samples=1)
        response_tokens.append(token)
        token_text = tokenizer.decode([token])
        print(token_text, end="", flush=True)
    print()
    print_footer()
    # we have to ensure that the assistant end token is the last token
    # so even if generation ends due to max tokens, we have to append it to the end
    if response_tokens[-1] != assistant_end:
        response_tokens.append(assistant_end)
    conversation_tokens.extend(response_tokens)

    # In the prompt mode, we only want a single response and exit
    if args.prompt:
        break
