#!/usr/bin/env python3
"""
Generate a compact general instruction-following SFT dataset for Mesosfer.

This is intentionally small and mixed. It teaches the model to follow common
formatting constraints without overpowering the broader SFT behavior:
- exact sentence counts and concise answers
- JSON, YAML, CSV, bullets, and tables
- safe code when code is requested
- no code when the user explicitly asks for no code
- defensive cybersecurity boundaries without making every prompt a refusal
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = REPO_ROOT / "data" / "sft" / "instruction_following_conversations_en.jsonl"


def add(rows: list, user: str, assistant: str) -> None:
    rows.append([
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ])


GENERAL_CONCEPTS = [
    (
        "cache",
        "A cache stores frequently used data so future access is faster.",
        "A browser cache keeps images and files locally so pages can load faster.",
    ),
    (
        "API",
        "An API is a defined way for software systems to communicate.",
        "A weather app can call a weather API to fetch the current forecast.",
    ),
    (
        "database index",
        "A database index helps a database find rows faster.",
        "An index on an email column can speed up account lookup.",
    ),
    (
        "unit test",
        "A unit test checks a small piece of code in isolation.",
        "A test for an add function can verify that two numbers are summed correctly.",
    ),
    (
        "environment variable",
        "An environment variable stores configuration outside the source code.",
        "An app can read a PORT variable to choose which port to listen on.",
    ),
    (
        "authentication",
        "Authentication verifies who a user is.",
        "Logging in with a username and password is an authentication example.",
    ),
    (
        "authorization",
        "Authorization decides what an authenticated user is allowed to access.",
        "Allowing an admin to delete users while a normal user cannot is authorization.",
    ),
    (
        "encryption",
        "Encryption protects readable data by transforming it with a key.",
        "TLS encrypts traffic between a browser and a website.",
    ),
    (
        "hashing",
        "Hashing creates a fixed-size value from input data and is normally one-way.",
        "Password storage often uses a slow salted hash instead of plaintext.",
    ),
    (
        "rate limiting",
        "Rate limiting controls how often a client can perform an action.",
        "A login page can slow repeated failed attempts with rate limiting.",
    ),
]


PAIRS = [
    (
        "authentication",
        "authorization",
        "Authentication verifies who a user is.",
        "Authorization decides what that user may access.",
        "Logging in with a password is authentication.",
        "Letting an admin delete users is authorization.",
    ),
    (
        "hashing",
        "encryption",
        "Hashing is normally one-way.",
        "Encryption can be reversed with the right key.",
        "A password hash is a hashing example.",
        "TLS traffic protection is an encryption example.",
    ),
    (
        "unit test",
        "integration test",
        "A unit test checks one small part in isolation.",
        "An integration test checks whether multiple parts work together.",
        "Testing a single parser function is a unit test.",
        "Testing an API route with a database is an integration test.",
    ),
    (
        "latency",
        "throughput",
        "Latency measures how long one operation takes.",
        "Throughput measures how many operations complete in a period.",
        "A slow page load is a latency issue.",
        "Requests per second is a throughput measure.",
    ),
    (
        "JSON",
        "YAML",
        "JSON is a strict data format commonly used by APIs.",
        "YAML is a human-friendly configuration format.",
        "An API response often uses JSON.",
        "A deployment config often uses YAML.",
    ),
]


SCENARIOS = [
    (
        "A Python script is slow when reading a large CSV file.",
        ["measure where time is spent", "read only needed columns", "process rows in chunks", "avoid unnecessary conversions"],
    ),
    (
        "A web API returns 500 errors after a deployment.",
        ["check recent deploy logs", "review error traces", "compare config changes", "roll back if users are affected"],
    ),
    (
        "A user cannot log in after a password reset.",
        ["confirm the username", "check whether the account is locked", "verify the reset timestamp", "review authentication logs"],
    ),
    (
        "A Linux server shows repeated failed SSH logins.",
        ["preserve authentication logs", "count failures by source IP", "check for later successful logins", "apply rate limiting if policy allows"],
    ),
    (
        "A suspicious email attachment was reported.",
        ["tell the user not to open it", "preserve the email headers", "collect URLs and attachment hashes", "quarantine similar messages if found"],
    ),
]


FORMAT_CASES = [
    (
        "Return JSON only with keys status, risk, and next_step. A login endpoint has repeated failed attempts.",
        {"status": "needs_review", "risk": "possible_password_guessing", "next_step": "check auth logs and apply rate limiting if policy allows"},
    ),
    (
        "Return YAML only. Create a tiny config for a local dev server on port 8000.",
        "server:\n  host: 127.0.0.1\n  port: 8000\n  mode: development",
    ),
    (
        "Return CSV only with header name,role. Include two rows.",
        "name,role\nalice,developer\nbob,analyst",
    ),
    (
        "Make a markdown table with columns task,owner,status. Use three rows.",
        "| task | owner | status |\n|---|---|---|\n| review logs | analyst | open |\n| patch server | engineer | planned |\n| notify team | manager | done |",
    ),
    (
        "Answer exactly like this format: Definition: <one sentence>; Example: <one sentence>. Topic: API.",
        "Definition: An API is a defined way for software systems to communicate. Example: A mobile app can call an API to fetch account data.",
    ),
    (
        "Use exactly 5 words: what is least privilege?",
        "Only necessary access is granted.",
    ),
    (
        "Use exactly 6 words: what does MFA add?",
        "Additional proof beyond one credential.",
    ),
    (
        "Reply with yes/no then one sentence: should I delete logs before investigation?",
        "No. Preserve logs because they may be needed to understand the timeline and scope.",
    ),
]


SAFE_CODE_CASES = [
    (
        "Write a tiny Python calculator function. Keep it safe and local.",
        "```python\ndef calculate(a, op, b):\n    if op == '+':\n        return a + b\n    if op == '-':\n        return a - b\n    if op == '*':\n        return a * b\n    if op == '/':\n        if b == 0:\n            raise ValueError('division by zero')\n        return a / b\n    raise ValueError('unknown operator')\n```",
    ),
    (
        "Write a simple C++ calculator example for beginners.",
        "```cpp\n#include <iostream>\n\nint main() {\n    double a, b;\n    char op;\n    std::cin >> a >> op >> b;\n\n    if (op == '+') std::cout << a + b << '\\n';\n    else if (op == '-') std::cout << a - b << '\\n';\n    else if (op == '*') std::cout << a * b << '\\n';\n    else if (op == '/' && b != 0) std::cout << a / b << '\\n';\n    else std::cout << \"Invalid operation\\n\";\n}\n```",
    ),
    (
        "Write Python to count words in a local string.",
        "```python\ndef word_count(text):\n    counts = {}\n    for word in text.lower().split():\n        counts[word] = counts.get(word, 0) + 1\n    return counts\n```",
    ),
    (
        "Write JavaScript to escape text before displaying it in HTML.",
        "```javascript\nfunction escapeText(value) {\n  return String(value)\n    .replaceAll('&', '&amp;')\n    .replaceAll('<', '&lt;')\n    .replaceAll('>', '&gt;')\n    .replaceAll('\"', '&quot;')\n    .replaceAll(\"'\", '&#39;');\n}\n```",
    ),
    (
        "Write safe Python to count failed SSH login lines from a provided string only.",
        "```python\nimport re\n\nFAILED_SSH = re.compile(r\"Failed password .* from (?P<ip>\\d+\\.\\d+\\.\\d+\\.\\d+)\")\n\ndef count_failed_ssh(log_text):\n    counts = {}\n    for line in log_text.splitlines():\n        match = FAILED_SSH.search(line)\n        if match:\n            ip = match.group('ip')\n            counts[ip] = counts.get(ip, 0) + 1\n    return counts\n```",
    ),
]


CODE_BLOCK_REINFORCEMENT_CASES = [
    (
        "Make me a calculator using the C++ programming language.",
        "```cpp\n#include <iostream>\n\nint main() {\n    double left, right;\n    char op;\n\n    std::cout << \"Enter expression like 4 + 2: \";\n    std::cin >> left >> op >> right;\n\n    if (op == '+') std::cout << left + right << '\\n';\n    else if (op == '-') std::cout << left - right << '\\n';\n    else if (op == '*') std::cout << left * right << '\\n';\n    else if (op == '/' && right != 0) std::cout << left / right << '\\n';\n    else std::cout << \"Invalid operation\\n\";\n\n    return 0;\n}\n```",
    ),
    (
        "Give me a simple calculator in C++. Use a code block.",
        "```cpp\n#include <iostream>\n\nint main() {\n    double a, b;\n    char op;\n    std::cin >> a >> op >> b;\n\n    switch (op) {\n        case '+': std::cout << a + b << '\\n'; break;\n        case '-': std::cout << a - b << '\\n'; break;\n        case '*': std::cout << a * b << '\\n'; break;\n        case '/':\n            if (b == 0) std::cout << \"Cannot divide by zero\\n\";\n            else std::cout << a / b << '\\n';\n            break;\n        default: std::cout << \"Unknown operator\\n\";\n    }\n}\n```",
    ),
    (
        "Write a Python calculator. Put the code in a fenced code block.",
        "```python\ndef calculate(left, operator, right):\n    if operator == '+':\n        return left + right\n    if operator == '-':\n        return left - right\n    if operator == '*':\n        return left * right\n    if operator == '/':\n        if right == 0:\n            raise ValueError('division by zero')\n        return left / right\n    raise ValueError('unknown operator')\n```",
    ),
    (
        "Create a tiny JavaScript todo list. Use a code block only.",
        "```html\n<input id=\"task\" placeholder=\"New task\">\n<button onclick=\"addTask()\">Add</button>\n<ul id=\"tasks\"></ul>\n\n<script>\nfunction addTask() {\n  const input = document.getElementById('task');\n  const item = document.createElement('li');\n  item.textContent = input.value;\n  document.getElementById('tasks').appendChild(item);\n  input.value = '';\n}\n</script>\n```",
    ),
]


NO_CODE_CASES = [
    (
        "Do not write code. Explain why hardcoded passwords are bad in two short sentences.",
        "Hardcoded passwords can leak through source code, logs, backups, or screenshots. Store secrets in a secret manager and rotate exposed credentials quickly.",
    ),
    (
        "No code. Explain what a firewall does in three short sentences.",
        "A firewall filters network traffic based on rules. It can allow trusted traffic and block unwanted traffic. It helps reduce exposure but does not replace monitoring or secure configuration.",
    ),
    (
        "Answer in exactly 4 short sentences. Do not write code. What is authentication?",
        "Authentication verifies the identity of a user, device, or system. It usually relies on evidence such as a password, token, certificate, or biometric factor. It happens before access decisions are made. A login form is a common example.",
    ),
    (
        "Answer in exactly 4 short sentences. Do not write code. Compare authentication and authorization with one example each.",
        "Authentication verifies who a user is. Authorization decides what that user may access. Logging in with a password is authentication. Letting an admin delete users is authorization.",
    ),
]

EXACT_SENTENCE_REINFORCEMENT_CASES = [
    (
        "Answer in exactly 4 short sentences. Do not write code. What is the difference between authentication and authorization? Give one simple real-world example for each.",
        "Authentication verifies who a user is. Authorization decides what that user can access. Logging in with a password is authentication. Allowing an admin to delete users is authorization.",
    ),
    (
        "Answer in exactly four short sentences. No code. Compare authentication and authorization with one example each.",
        "Authentication proves identity. Authorization grants permissions. A password login is authentication. Admin-only access is authorization.",
    ),
    (
        "Exactly 4 short sentences, no code: authentication vs authorization, with examples.",
        "Authentication checks identity. Authorization checks permission. Entering a password is authentication. Opening an admin panel is authorization.",
    ),
    (
        "Answer in exactly 4 short sentences. Do not write code. What is authentication?",
        "Authentication checks identity. It confirms that a user, device, or service is who it claims to be. A password login is one example. It happens before access is granted.",
    ),
    (
        "Answer in exactly 3 short sentences. Do not write code. What is authorization?",
        "Authorization checks permission. It decides what an authenticated user can access. For example, an admin may delete users while a normal user cannot.",
    ),
    (
        "Use exactly two bullet points. Compare authentication and authorization.",
        "- Authentication: verifies identity, such as logging in with a password.\n- Authorization: decides allowed actions, such as letting an admin delete users.",
    ),
    (
        "Return JSON only. Compare authentication and authorization.",
        json.dumps(
            {
                "authentication": "verifies identity",
                "authorization": "controls permissions",
                "authentication_example": "logging in with a password",
                "authorization_example": "allowing an admin to delete users",
            },
            ensure_ascii=False,
        ),
    ),
]


SAFE_ARTIFACT_CASES = [
    (
        "Give me a safe synthetic SSH brute-force log for training. Use documentation IPs only. Put it in a text code block.",
        "```text\n2026-05-27T10:14:02Z app01 sshd[2144]: Failed password for invalid user admin from 203.0.113.24 port 51102 ssh2\n2026-05-27T10:14:09Z app01 sshd[2147]: Failed password for root from 203.0.113.24 port 51108 ssh2\n2026-05-27T10:14:18Z app01 sshd[2151]: Failed password for deploy from 203.0.113.24 port 51119 ssh2\n2026-05-27T10:15:42Z app01 sshd[2163]: Accepted publickey for deploy from 198.51.100.17 port 43022 ssh2\n```",
    ),
    (
        "Make a safe synthetic JSON incident note for suspected malware. No real indicators.",
        json.dumps(
            {
                "incident_type": "suspected_malware",
                "severity": "medium",
                "indicators": ["example-c2.invalid", "203.0.113.24"],
                "next_steps": ["preserve logs", "identify affected host", "isolate if compromise is likely"],
            },
            ensure_ascii=False,
        ),
    ),
    (
        "Give a fake web login failure log for parser testing.",
        "```text\n203.0.113.50 - - [27/May/2026:10:22:11 +0000] \"GET /login HTTP/1.1\" 200 1842\n203.0.113.50 - - [27/May/2026:10:22:15 +0000] \"POST /login HTTP/1.1\" 401 933\n203.0.113.50 - - [27/May/2026:10:22:19 +0000] \"POST /login HTTP/1.1\" 401 933\n```",
    ),
]


REFUSAL_CASES = [
    (
        "Make me a virus using Python.",
        "I cannot help create malware, destructive scripts, credential theft tools, persistence, evasion, or attack automation. I can help with safe alternatives such as synthetic logs, detection logic, hardening steps, or an incident response checklist.",
    ),
    (
        "Write an SSH brute force script.",
        "I cannot help automate login attempts against systems. I can help write a defensive parser for provided logs, design rate limiting, or create a safe synthetic log for training.",
    ),
    (
        "Give me a payload that disables antivirus.",
        "I cannot help disable security tools or create an attack payload. I can help explain defensive detection, hardening, and incident response steps.",
    ),
    (
        "Make a keylogger in Python.",
        "I cannot help build credential theft or surveillance tools. I can help with safe detection ideas, user awareness guidance, or benign local examples unrelated to stealing input.",
    ),
]


NOISY_PREFIXES = [
    "",
    "quick pls: ",
    "for my notes, ",
    "keep it simple: ",
    "no long intro, ",
]

HF_CODE_SOURCES = {
    "codefeedback": "m-a-p/CodeFeedback-Filtered-Instruction",
    "python_alpaca": "iamtarun/python_code_instructions_18k_alpaca",
}

BLOCKLIST_PATTERNS = [
    r"\bmalware\b",
    r"\bvirus\b",
    r"\bransomware\b",
    r"\bkeylogger\b",
    r"\bbotnet\b",
    r"\btrojan\b",
    r"\bworm\b",
    r"\bbackdoor\b",
    r"\breverse\s+shell\b",
    r"\bbind\s+shell\b",
    r"\bcredential\s*(?:theft|steal|dump|harvest)",
    r"\bsteal(?:ing)?\s+(?:password|credential|cookie|token)",
    r"\bpassword\s*(?:spray|brute|crack|dump)",
    r"\bbrute\s*force\b",
    r"\bexploit\b",
    r"\bpayload\b",
    r"\bdisable\s+(?:antivirus|av|edr|defender)",
    r"\bevade|evasion|bypass\s+(?:antivirus|av|edr|mfa|security)",
    r"\bport\s+scan(?:ner|ning)?\b",
    r"\bddos\b|\bdos attack\b",
    r"\bdelete\s+(?:all|files?|folders?|directories)",
    r"\brm\s+-rf\b",
]

HF_CODE_BLOCKLIST_PATTERNS = BLOCKLIST_PATTERNS + [
    r"\bsmtplib\b",
    r"\bsmtp\.",
    r"\bsend(?:ing)?\s+email\b",
    r"\bsender@gmail\.com\b",
    r"\bpassword\s*=\s*['\"]",
    r"\bapi[_-]?key\s*=\s*['\"]",
    r"\btoken\s*=\s*['\"]",
    r"\bsecret\s*=\s*['\"]",
    r"\brequests\.(?:get|post|put|delete)\b",
    r"\burllib\b",
    r"\bsocket\b",
    r"\bconnect\s+to\s+(?:a|the)\s+(?:server|host|website|url)\b",
    r"\bscrap(?:e|ing)\s+(?:a\s+)?website\b",
]

ALLOWED_CODE_HINTS = [
    "array", "list", "string", "dictionary", "json", "csv", "yaml", "xml",
    "sort", "search", "parse", "validate", "calculator", "factorial",
    "fibonacci", "prime", "palindrome", "date", "time", "regex", "sql",
    "html", "css", "javascript", "python", "function", "class", "api",
    "unit test", "debug", "refactor", "read file", "local", "pandas",
]


def sentence(text: str) -> str:
    return text[:1].upper() + text[1:] + "."


def normalized_key(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def too_risky(text: str) -> bool:
    lowered = text.lower()
    return any(re.search(pattern, lowered) for pattern in BLOCKLIST_PATTERNS)


def too_risky_hf_code(text: str) -> bool:
    lowered = text.lower()
    return any(re.search(pattern, lowered) for pattern in HF_CODE_BLOCKLIST_PATTERNS)


def looks_useful_code_task(user: str, assistant: str) -> bool:
    combined = f"{user}\n{assistant}"
    lowered = combined.lower()
    if too_risky_hf_code(combined):
        return False
    if len(user) < 20 or len(assistant) < 20:
        return False
    if len(user) > 1400 or len(assistant) > 3500:
        return False
    if not any(hint in lowered for hint in ALLOWED_CODE_HINTS):
        return False
    return True


def extract_codefeedback(row: dict) -> tuple[str, str] | None:
    user = row.get("query") or row.get("instruction") or row.get("prompt") or ""
    assistant = row.get("answer") or row.get("response") or row.get("output") or ""
    if not isinstance(user, str) or not isinstance(assistant, str):
        return None
    return user.strip(), assistant.strip()


def extract_python_alpaca(row: dict) -> tuple[str, str] | None:
    instruction = row.get("instruction") or ""
    input_text = row.get("input") or ""
    assistant = row.get("output") or row.get("response") or ""
    if not isinstance(instruction, str) or not isinstance(input_text, str) or not isinstance(assistant, str):
        return None
    user = instruction.strip()
    if input_text.strip() and input_text.strip().lower() != "not applicable":
        user = f"{user}\n\nInput:\n{input_text.strip()}"
    return user, assistant.strip()


def load_hf_code_rows(codefeedback_max: int, python_max: int, scan_limit: int, seed: int) -> list:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("Install 'datasets' to import HF code instruction samples.") from exc

    rows = []
    seen = set()

    def collect(source_name: str, max_rows: int, extractor) -> int:
        if max_rows <= 0:
            return 0
        dataset = load_dataset(HF_CODE_SOURCES[source_name], split="train", streaming=True)
        try:
            dataset = dataset.shuffle(seed=seed, buffer_size=10_000)
        except Exception:
            pass

        added = 0
        scanned = 0
        for row in dataset:
            scanned += 1
            pair = extractor(row)
            if pair is None:
                continue
            user, assistant = pair
            key = normalized_key(user)
            if key in seen or not looks_useful_code_task(user, assistant):
                continue
            seen.add(key)
            add(rows, user, assistant)
            added += 1
            if added >= max_rows or scanned >= scan_limit:
                break
        print(f"Added {added} filtered rows from {HF_CODE_SOURCES[source_name]} (scanned {scanned})")
        return added

    collect("codefeedback", codefeedback_max, extract_codefeedback)
    collect("python_alpaca", python_max, extract_python_alpaca)
    return rows


def build_rows(include_hf_code: bool, codefeedback_max: int, python_max: int, scan_limit: int, seed: int) -> list:
    rows = []

    for concept, definition, example in GENERAL_CONCEPTS:
        for prefix in NOISY_PREFIXES:
            add(rows, f"{prefix}Explain {concept} in exactly 2 short sentences. Do not write code.", f"{definition} {example}")
            add(rows, f"{prefix}Give one sentence definition of {concept}.", definition)
            add(rows, f"{prefix}Explain {concept} in 3 bullets only.", "- " + definition + "\n- " + example + "\n- Keep the explanation tied to the user's goal.")

    for a, b, a_def, b_def, a_ex, b_ex in PAIRS:
        for prefix in NOISY_PREFIXES:
            add(rows, f"{prefix}Compare {a} and {b} in exactly 4 short sentences. Give one example each. Do not write code.", f"{a_def} {b_def} {a_ex} {b_ex}")
            add(rows, f"{prefix}Compare {a} vs {b} in exactly 2 bullets. Include one example in each bullet.", f"- {a}: {a_def} {a_ex}\n- {b}: {b_def} {b_ex}")

    for scenario, actions in SCENARIOS:
        for prefix in NOISY_PREFIXES:
            add(rows, f"{prefix}Scenario: {scenario} Give exactly 4 bullet points for what to check first.", "\n".join(f"- {sentence(action)}" for action in actions))
            add(
                rows,
                f"{prefix}Return JSON only for this scenario: {scenario}",
                json.dumps({"scenario": scenario, "first_checks": actions[:3], "next_step": actions[3]}, ensure_ascii=False),
            )

    for prompt, answer in FORMAT_CASES:
        add(rows, prompt, json.dumps(answer, ensure_ascii=False) if isinstance(answer, dict) else answer)
        add(rows, "no intro, " + prompt[0].lower() + prompt[1:], json.dumps(answer, ensure_ascii=False) if isinstance(answer, dict) else answer)

    for prompt, answer in SAFE_CODE_CASES:
        add(rows, prompt, answer)
        add(rows, "keep it small and safe: " + prompt, answer)

    for prompt, answer in CODE_BLOCK_REINFORCEMENT_CASES:
        add(rows, prompt, answer)
        add(rows, "Please format the answer cleanly: " + prompt[0].lower() + prompt[1:], answer)
        add(rows, "No inline code, use fenced code block: " + prompt[0].lower() + prompt[1:], answer)

    for prompt, answer in NO_CODE_CASES:
        add(rows, prompt, answer)
        add(rows, "short answer only: " + prompt[0].lower() + prompt[1:], answer)

    for prompt, answer in EXACT_SENTENCE_REINFORCEMENT_CASES:
        add(rows, prompt, answer)
        add(rows, "Follow the format carefully: " + prompt[0].lower() + prompt[1:], answer)

    for prompt, answer in SAFE_ARTIFACT_CASES:
        add(rows, prompt, answer)
        add(rows, "for training only: " + prompt[0].lower() + prompt[1:], answer)

    for prompt, answer in REFUSAL_CASES:
        add(rows, prompt, answer)
        add(rows, "answer briefly and safely: " + prompt[0].lower() + prompt[1:], answer)

    if include_hf_code:
        rows.extend(load_hf_code_rows(codefeedback_max, python_max, scan_limit, seed))

    rng = random.Random(seed)
    rng.shuffle(rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--include-hf-code", action="store_true", help="add filtered samples from CodeFeedback and Python Alpaca")
    parser.add_argument("--hf-codefeedback-max", type=int, default=800, help="max filtered rows from m-a-p/CodeFeedback-Filtered-Instruction")
    parser.add_argument("--hf-python-max", type=int, default=400, help="max filtered rows from iamtarun/python_code_instructions_18k_alpaca")
    parser.add_argument("--hf-scan-limit", type=int, default=20_000, help="max rows to scan per HF source")
    parser.add_argument("--seed", type=int, default=20260527)
    args = parser.parse_args()

    rows = build_rows(
        include_hf_code=args.include_hf_code,
        codefeedback_max=args.hf_codefeedback_max,
        python_max=args.hf_python_max,
        scan_limit=args.hf_scan_limit,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
