"""
Cybersecurity RL task for mesosfer.

Wraps cybersecurity SFT JSONL data into a Task that supports the .reward() and
.evaluate() interface used by chat_rl.py. Unlike GSM8K (which has a single
numerical answer extractable via regex), cybersecurity Q&A needs different
reward heuristics based on:

  1. Format adherence — does the response have structure (steps, sections)?
  2. Security keyword coverage — does it mention relevant security concepts?
  3. MITRE ATT&CK references — does it cite techniques where appropriate?
  4. Hallucination prevention — penalize obviously wrong tool/CVE references
  5. Length appropriateness — neither too short nor rambling

This is an approximation. Production RL on cybersecurity should use a learned
reward model trained on human preferences. This heuristic-based reward provides
a useful starting signal during the first RL phase.
"""

import os
import re
from pathlib import Path

from tasks.common import Task
from tasks.customjson import CustomJSON


_REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SFT_DIR = _REPO_ROOT / "data" / "sft"

# -----------------------------------------------------------------------------
# Security keyword vocabulary for reward shaping

# High-value defensive/analyst terms (positive signal)
DEFENSIVE_TERMS = (
    # Triage / IR
    "triage", "incident", "containment", "eradication", "recovery", "lessons learned",
    "indicators of compromise", "ioc", "iocs", "threat hunt", "hunting",
    "mean time to detect", "mttd", "mttr", "playbook", "runbook",
    # SOC operations
    "siem", "soar", "edr", "xdr", "ndr", "ueba", "log aggregation",
    "correlation rule", "alert tuning", "false positive", "true positive",
    # Forensics
    "memory dump", "disk image", "chain of custody", "artifact", "timeline",
    "volatile data", "non-volatile", "preserve evidence",
    # Identity / auth
    "mfa", "least privilege", "rbac", "abac", "service account", "privileged access",
    "just-in-time", "zero trust", "conditional access",
    # Network defense
    "segmentation", "microsegmentation", "egress filtering", "deny by default",
    "firewall", "ips", "ids", "waf", "proxy",
)

# Mechanism / vulnerability terms (technical depth signal)
MECHANISM_TERMS = (
    "buffer overflow", "heap overflow", "stack overflow", "use-after-free",
    "double free", "integer overflow", "format string", "rop chain", "rop gadget",
    "race condition", "deserialization", "path traversal", "directory traversal",
    "ssrf", "csrf", "xss", "sql injection", "sqli", "command injection",
    "privilege escalation", "lateral movement", "persistence", "exfiltration",
    "credential dumping", "pass the hash", "pass the ticket", "kerberoasting",
    "golden ticket", "silver ticket", "dcsync",
)

# MITRE ATT&CK technique pattern (e.g. T1055, T1078.004)
MITRE_TECHNIQUE_RE = re.compile(r"\bT\d{4}(?:\.\d{3})?\b")

# CVE pattern (e.g. CVE-2023-12345)
CVE_RE = re.compile(r"\bCVE-\d{4}-\d{4,7}\b", re.IGNORECASE)

# Marketing / fluff terms (negative signal — model should avoid sales-talk)
MARKETING_TERMS = (
    "book a demo", "free trial", "contact sales", "request a demo",
    "industry-leading", "best-in-class", "next-generation",
    "revolutionary", "cutting-edge", "world-class",
)

# Hedge / non-answer phrases (negative signal — model should commit to answers)
HEDGE_PHRASES = (
    "i cannot help with that",
    "i'm not able to",
    "as an ai language model",
    "i don't have access to",
    "consult a professional",
    "i recommend speaking with",
)


def _count_term_hits(text: str, terms) -> int:
    """Count how many distinct terms from `terms` appear in lowercased text."""
    lower = text.lower()
    return sum(1 for term in terms if term.lower() in lower)


def _has_structure(text: str) -> bool:
    """Heuristic: does the response have numbered steps, bullets, or headers?"""
    # Numbered list (1. 2. 3. or 1) 2) 3))
    if re.search(r"^\s*\d+[\.\)]\s+", text, re.MULTILINE):
        return True
    # Bullet list (- or *)
    if re.search(r"^\s*[-*]\s+", text, re.MULTILINE):
        return True
    # Markdown headers
    if re.search(r"^#+\s+", text, re.MULTILINE):
        return True
    return False


def _length_score(text: str, target_min: int = 50, target_max: int = 600) -> float:
    """
    Soft length reward: 1.0 in target range, decays outside.
    Too short → low reward (incomplete). Too long → low reward (rambling).
    """
    n_words = len(text.split())
    if n_words < target_min:
        return n_words / target_min
    if n_words <= target_max:
        return 1.0
    # Linear decay above target_max, hits 0 at 3× target
    decay = max(0.0, 1.0 - (n_words - target_max) / (2 * target_max))
    return decay


def cybersec_reward(
    response: str,
    reference: str | None = None,
    is_offensive_topic: bool = False,
) -> tuple[float, dict]:
    """
    Compute a heuristic reward for a cybersecurity response in [0, 1].

    Returns (reward, breakdown) where breakdown shows component scores for logging.

    Args:
        response: model's generated text
        reference: optional ground-truth assistant message (used for keyword overlap)
        is_offensive_topic: if True, penalize if model refuses (we want it to teach,
                            not refuse — the underlying SFT data is curated educational)
    """
    if not response or not response.strip():
        return 0.0, {"reason": "empty response"}

    breakdown = {}

    # 1. Defensive term coverage (cap at 5 hits = 1.0)
    def_hits = _count_term_hits(response, DEFENSIVE_TERMS)
    breakdown["defensive_terms"] = min(def_hits / 5.0, 1.0)

    # 2. Mechanism term coverage (cap at 3 hits = 1.0)
    mech_hits = _count_term_hits(response, MECHANISM_TERMS)
    breakdown["mechanism_terms"] = min(mech_hits / 3.0, 1.0)

    # 3. MITRE ATT&CK references (any reference = 1.0, no reference = 0.0)
    has_mitre = bool(MITRE_TECHNIQUE_RE.search(response))
    breakdown["mitre_reference"] = 1.0 if has_mitre else 0.0

    # 4. Structure (numbered steps / bullets / headers)
    breakdown["structure"] = 1.0 if _has_structure(response) else 0.0

    # 5. Length appropriateness
    breakdown["length"] = _length_score(response)

    # 6. Keyword overlap with reference (if provided)
    if reference:
        ref_terms = set(re.findall(r"\b\w{4,}\b", reference.lower()))
        resp_terms = set(re.findall(r"\b\w{4,}\b", response.lower()))
        if ref_terms:
            overlap = len(ref_terms & resp_terms) / len(ref_terms)
            breakdown["reference_overlap"] = min(overlap * 2.0, 1.0)  # scale: 50% overlap = full
        else:
            breakdown["reference_overlap"] = 0.0
    else:
        breakdown["reference_overlap"] = 0.5  # neutral when no reference

    # Penalties
    marketing_hits = _count_term_hits(response, MARKETING_TERMS)
    breakdown["marketing_penalty"] = -min(marketing_hits * 0.2, 1.0)

    hedge_hits = sum(1 for h in HEDGE_PHRASES if h in response.lower())
    breakdown["hedge_penalty"] = -min(hedge_hits * 0.3, 1.0)
    if is_offensive_topic and hedge_hits > 0:
        # Stronger penalty for refusing curated educational content
        breakdown["hedge_penalty"] -= 0.5

    # Weighted combination
    weights = {
        "defensive_terms":   0.20,
        "mechanism_terms":   0.15,
        "mitre_reference":   0.15,
        "structure":         0.15,
        "length":            0.15,
        "reference_overlap": 0.20,
        "marketing_penalty": 1.0,  # penalties have weight 1.0 (negative values)
        "hedge_penalty":     1.0,
    }

    reward = sum(breakdown[k] * weights[k] for k in breakdown)
    # Clip to [0, 1]
    reward = max(0.0, min(1.0, reward))
    breakdown["final_reward"] = reward

    return reward, breakdown


# -----------------------------------------------------------------------------
# Cybersecurity RL task

class CybersecRL(Task):
    """
    Generic cybersecurity RL task wrapping a JSONL of conversations.

    The reward is computed via heuristic `cybersec_reward()` rather than exact
    match (like GSM8K). This works for open-ended generative answers where
    multiple valid responses exist.

    Keyword indicators of offensive topics (for reward calibration) are auto-
    detected from the user prompt.
    """

    OFFENSIVE_INDICATORS = (
        "exploit", "payload", "shellcode", "bypass", "evasion",
        "metasploit", "msfvenom", "burp", "sqlmap", "hydra",
        "reverse shell", "buffer overflow", "rop chain",
    )

    def __init__(self, filepath, **kwargs):
        super().__init__(**kwargs)
        # Reuse CustomJSON's loading logic
        self._loader = CustomJSON(filepath=filepath)
        self.filepath = filepath

    @property
    def eval_type(self):
        return "generative"

    def num_examples(self):
        return self._loader.num_examples()

    def get_example(self, index):
        return self._loader.get_example(index)

    @staticmethod
    def _get_user_prompt(conversation) -> str:
        for msg in conversation["messages"]:
            if msg["role"] == "user":
                content = msg["content"]
                return content if isinstance(content, str) else str(content)
        return ""

    @staticmethod
    def _get_assistant_reference(conversation) -> str:
        for msg in conversation["messages"]:
            if msg["role"] == "assistant":
                content = msg["content"]
                return content if isinstance(content, str) else str(content)
        return ""

    def _is_offensive_topic(self, conversation) -> bool:
        prompt = self._get_user_prompt(conversation).lower()
        return any(ind in prompt for ind in self.OFFENSIVE_INDICATORS)

    def evaluate(self, conversation, assistant_response) -> int:
        """
        Binary correctness for pass@k: reward >= 0.5 is treated as "correct".
        """
        reward, _ = self.reward_with_breakdown(conversation, assistant_response)
        return int(reward >= 0.5)

    def reward(self, conversation, assistant_response) -> float:
        """
        Continuous reward in [0, 1] used by GRPO-style policy gradient.
        """
        reward, _ = self.reward_with_breakdown(conversation, assistant_response)
        return reward

    def reward_with_breakdown(self, conversation, assistant_response):
        """
        Returns (reward, breakdown_dict) for logging/debugging.
        """
        assert isinstance(assistant_response, str), "RL assumes string responses"
        reference = self._get_assistant_reference(conversation)
        is_offensive = self._is_offensive_topic(conversation)
        return cybersec_reward(assistant_response, reference=reference,
                                is_offensive_topic=is_offensive)


# -----------------------------------------------------------------------------
# Convenience: load specific cybersec RL datasets

def _sft_path(filename: str, sft_dir: str | None = None) -> str:
    base = Path(sft_dir) if sft_dir else DEFAULT_SFT_DIR
    return str(base / filename)


def CyberDefensiveRL(language: str = "id", sft_dir: str | None = None, **kwargs) -> CybersecRL:
    """5K-row defensive cybersec dataset for RL."""
    suffix = "" if language == "id" else "_en"
    return CybersecRL(filepath=_sft_path(f"cyber_defensive_conversations{suffix}.jsonl", sft_dir), **kwargs)


def MultiTurnSOCRL(sft_dir: str | None = None, **kwargs) -> CybersecRL:
    """Multi-turn SOC analyst dataset for RL."""
    return CybersecRL(filepath=_sft_path("multi_turn_soc_sft.jsonl", sft_dir), **kwargs)


def MythosCombinedRL(language: str = "id", sft_dir: str | None = None, **kwargs) -> CybersecRL:
    """Mythos combined dataset for RL."""
    suffix = "" if language == "id" else "_en"
    return CybersecRL(filepath=_sft_path(f"mythos_combined_sft{suffix}.jsonl", sft_dir), **kwargs)


if __name__ == "__main__":
    # Smoke test
    print("=" * 70)
    print("Testing cybersec_reward function")
    print("=" * 70)

    sample_responses = {
        "good": (
            "Initial triage for mass failed logins:\n"
            "1. Count failed login attempts per IP and per user within a 5-minute window.\n"
            "2. Check whether any successful login followed a burst of failures (T1110.001 brute force).\n"
            "3. Perform threat hunting in the SIEM for additional IOCs.\n"
            "4. Containment: block the IP at the firewall, reset passwords for targeted accounts.\n"
            "5. Enable MFA and review least privilege for admin accounts."
        ),
        "weak": "Just check the logs and block the IP.",
        "marketing": (
            "Our industry-leading next-generation SIEM platform offers world-class threat detection. "
            "Book a demo today to see how our cutting-edge solution can protect your enterprise."
        ),
        "hedge": "I cannot help with that. As an AI language model, I recommend speaking with a security professional.",
    }

    for name, resp in sample_responses.items():
        reward, breakdown = cybersec_reward(resp, reference=sample_responses["good"])
        print(f"\n[{name}] reward={reward:.3f}")
        for k, v in breakdown.items():
            if k != "final_reward":
                print(f"  {k}: {v:.3f}")

    print("\n" + "=" * 70)
    print("Testing CybersecRL task loading")
    print("=" * 70)

    test_files = [
        "cyber_defensive_conversations.jsonl",
        "multi_turn_soc_sft.jsonl",
        "mythos_combined_sft.jsonl",
    ]
    for fname in test_files:
        path = DEFAULT_SFT_DIR / fname
        if path.exists():
            task = CybersecRL(filepath=str(path))
            print(f"\n{fname}: {len(task)} conversations")
            ex = task[0]
            user = task._get_user_prompt(ex)[:80]
            ref = task._get_assistant_reference(ex)[:80]
            print(f"  user prompt: {user}...")
            print(f"  reference:   {ref}...")
        else:
            print(f"\n{fname}: not found")
