"""
Cybersecurity SFT tasks for mesosfer.

Wraps the JSONL files in data/sft/ as Task objects so they can be mixed into
the SFT training pipeline alongside SmolTalk, MMLU, GSM8K, etc.

Each file in data/sft/ follows the same format as identity_conversations.jsonl:
each line is a JSON array of {"role": ..., "content": ...} message objects.

Some files (mythos_combined, multi_turn_soc) contain `tool` role messages that
are not part of mesosfer's expected user/assistant alternation. These conversations
are filtered out with a single warning per file rather than crashing the pipeline.

Available datasets:
  - mesosfer_validation_conversations.jsonl  (300 rows, ID + EN variants)
  - cloud_security_sft.jsonl                 (6 rows)
  - cyber_defensive_conversations.jsonl      (5000 rows, ID + EN variants)
  - identity_conversations.jsonl             (1000 rows, ID + EN variants)
  - multi_turn_soc_sft.jsonl                 (4 rows, some skipped due to tool role)
  - mythos_combined_sft.jsonl                (110 rows, ~59 skipped due to tool role)
  - tool_oriented_cyber_sft.jsonl            (8 rows)
  - gemini_teacher_conversations.jsonl       (373 rows)
"""

import json
import os
from pathlib import Path

from tasks.common import Task


_REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SFT_DIR = _REPO_ROOT / "data" / "sft"

# Module-level cache: filepath → (conversations, skipped_count)
# Avoids re-reading and re-warning when the same file is loaded multiple times
# (which happens with epoch-based oversampling in build_cybersec_sft_tasks).
_FILE_CACHE: dict = {}


def _sft_path(filename: str, sft_dir: str | None = None) -> str:
    base = Path(sft_dir) if sft_dir else DEFAULT_SFT_DIR
    return str(base / filename)


def _is_valid_conversation(messages) -> bool:
    """
    Check if a conversation matches mesosfer's expected user/assistant alternation.

    Returns False if any message has a role other than user/assistant, or if the
    pattern doesn't alternate starting with user.
    """
    if not isinstance(messages, list) or len(messages) < 2:
        return False
    for i, message in enumerate(messages):
        if not isinstance(message, dict):
            return False
        if "role" not in message or "content" not in message:
            return False
        if not isinstance(message["content"], str):
            return False
        expected_role = "user" if i % 2 == 0 else "assistant"
        if message["role"] != expected_role:
            return False
    return True


class RobustCustomJSON(Task):
    """
    Like CustomJSON but tolerant of malformed conversations:
    - Skips conversations with non-standard roles (e.g. `tool`, `system`)
    - Skips conversations that don't alternate user/assistant
    - Logs a single summary line per file instead of crashing
    """

    def __init__(self, filepath, **kwargs):
        super().__init__(**kwargs)
        self.filepath = filepath

        # Reuse cached load if available (avoid re-reading + re-warning per epoch)
        if filepath in _FILE_CACHE:
            self.conversations, _ = _FILE_CACHE[filepath]
            self.length = len(self.conversations)
            return

        self.conversations = []
        skipped = 0

        if not os.path.exists(filepath):
            print(f"[cybersec_sft] WARNING: file not found: {filepath}")
            self.length = 0
            _FILE_CACHE[filepath] = (self.conversations, skipped)
            return

        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    messages = json.loads(line)
                except json.JSONDecodeError:
                    skipped += 1
                    continue
                if _is_valid_conversation(messages):
                    self.conversations.append(messages)
                else:
                    skipped += 1

        if skipped > 0:
            print(f"[cybersec_sft] {Path(filepath).name}: loaded {len(self.conversations)} valid, skipped {skipped} (non-standard schema)")

        self.length = len(self.conversations)
        _FILE_CACHE[filepath] = (self.conversations, skipped)

    def num_examples(self):
        return self.length

    def get_example(self, index):
        messages = self.conversations[index]
        return {"messages": messages}


# -----------------------------------------------------------------------------
# Cybersecurity SFT task wrappers

class CyberDefensiveConversations(RobustCustomJSON):
    """5K rows of defensive cybersecurity Q&A (SOC analyst, threat triage, IR)."""
    def __init__(self, language="id", sft_dir=None, **kwargs):
        assert language in ("id", "en"), f"language must be 'id' or 'en', got {language}"
        suffix = "" if language == "id" else "_en"
        super().__init__(filepath=_sft_path(f"cyber_defensive_conversations{suffix}.jsonl", sft_dir), **kwargs)


class CloudSecuritySFT(RobustCustomJSON):
    """Cloud security incident response (AWS, Azure, GCP). Small dataset (6 rows)."""
    def __init__(self, language="id", sft_dir=None, **kwargs):
        assert language in ("id", "en"), f"language must be 'id' or 'en', got {language}"
        suffix = "" if language == "id" else "_en"
        super().__init__(filepath=_sft_path(f"cloud_security_sft{suffix}.jsonl", sft_dir), **kwargs)


class MultiTurnSOC(RobustCustomJSON):
    """Multi-turn SOC analyst dialogues. Very small dataset (4 rows) — oversample heavily."""
    def __init__(self, sft_dir=None, **kwargs):
        super().__init__(filepath=_sft_path("multi_turn_soc_sft.jsonl", sft_dir), **kwargs)


class ToolOrientedCyber(RobustCustomJSON):
    """Tool-oriented cybersecurity conversations (nmap, burp, sqlmap, etc.). 8 rows."""
    def __init__(self, sft_dir=None, **kwargs):
        super().__init__(filepath=_sft_path("tool_oriented_cyber_sft.jsonl", sft_dir), **kwargs)


class MythosCombined(RobustCustomJSON):
    """Combined mythos (offensive + defensive narrative scenarios). 110 rows."""
    def __init__(self, language="id", sft_dir=None, **kwargs):
        assert language in ("id", "en"), f"language must be 'id' or 'en', got {language}"
        suffix = "" if language == "id" else "_en"
        super().__init__(filepath=_sft_path(f"mythos_combined_sft{suffix}.jsonl", sft_dir), **kwargs)


class mesosferValidation(RobustCustomJSON):
    """mesosfer validation conversations — domain alignment dataset. 300 rows."""
    def __init__(self, language="id", sft_dir=None, **kwargs):
        assert language in ("id", "en"), f"language must be 'id' or 'en', got {language}"
        suffix = "" if language == "id" else "_en"
        super().__init__(filepath=_sft_path(f"mesosfer_validation_conversations{suffix}.jsonl", sft_dir), **kwargs)


class GeminiTeacher(RobustCustomJSON):
    """Gemini-distilled teacher conversations. 373 rows."""
    def __init__(self, sft_dir=None, **kwargs):
        super().__init__(filepath=_sft_path("gemini_teacher_conversations.jsonl", sft_dir), **kwargs)


# -----------------------------------------------------------------------------
# Mixture builders

def build_cybersec_sft_tasks(
    cyber_defensive_epochs: int = 1,
    cloud_security_epochs: int = 20,
    multi_turn_soc_epochs: int = 30,
    tool_oriented_epochs: int = 20,
    mythos_epochs: int = 4,
    mesosfer_validation_epochs: int = 2,
    gemini_teacher_epochs: int = 2,
    include_english: bool = True,
    sft_dir: str | None = None,
) -> list:
    """
    Build a list of cybersecurity Task objects for use in TaskMixture.

    Default epoch counts oversample tiny datasets (cloud_security 6 rows × 20 = 120,
    multi_turn_soc 4 rows × 30 = 120) so they aren't drowned out by the larger
    cyber_defensive (5K rows). Tune these in chat_sft.py CLI flags.

    Args:
        cyber_defensive_epochs: epochs of cyber_defensive_conversations (5K rows each)
        cloud_security_epochs: epochs of cloud_security_sft (6 rows each, oversample)
        multi_turn_soc_epochs: epochs of multi_turn_soc_sft (4 rows, oversample)
        tool_oriented_epochs: epochs of tool_oriented_cyber_sft (8 rows, oversample)
        mythos_epochs: epochs of mythos_combined_sft (110 rows each)
        mesosfer_validation_epochs: epochs of mesosfer validation (300 rows each)
        gemini_teacher_epochs: epochs of gemini teacher (373 rows)
        include_english: whether to also include _en variants of bilingual datasets
        sft_dir: override path to data/sft/ (default: repo_root/data/sft)

    Returns:
        list of Task objects (each can be passed multiple times to oversample)
    """
    tasks = []

    languages = ["id", "en"] if include_english else ["id"]

    # Bilingual datasets: include each language separately × N epochs
    for lang in languages:
        for _ in range(cyber_defensive_epochs):
            tasks.append(CyberDefensiveConversations(language=lang, sft_dir=sft_dir))
        for _ in range(cloud_security_epochs):
            tasks.append(CloudSecuritySFT(language=lang, sft_dir=sft_dir))
        for _ in range(mythos_epochs):
            tasks.append(MythosCombined(language=lang, sft_dir=sft_dir))
        for _ in range(mesosfer_validation_epochs):
            tasks.append(mesosferValidation(language=lang, sft_dir=sft_dir))

    # Monolingual datasets
    for _ in range(multi_turn_soc_epochs):
        tasks.append(MultiTurnSOC(sft_dir=sft_dir))
    for _ in range(tool_oriented_epochs):
        tasks.append(ToolOrientedCyber(sft_dir=sft_dir))
    for _ in range(gemini_teacher_epochs):
        tasks.append(GeminiTeacher(sft_dir=sft_dir))

    # Filter out empty tasks (e.g. if all rows were skipped)
    tasks = [t for t in tasks if len(t) > 0]
    return tasks


def total_cybersec_rows(tasks: list) -> int:
    """Helper to sum the total row count across a list of cybersec SFT tasks."""
    return sum(len(t) for t in tasks)


if __name__ == "__main__":
    print(f"SFT dir: {DEFAULT_SFT_DIR}")
    print(f"Exists: {DEFAULT_SFT_DIR.exists()}")
    if DEFAULT_SFT_DIR.exists():
        tasks = build_cybersec_sft_tasks()
        print(f"\nBuilt {len(tasks)} task instances")
        print(f"Total rows (with epoch oversampling): {total_cybersec_rows(tasks):,}")
        from collections import Counter
        breakdown = Counter(type(t).__name__ for t in tasks)
        print("\nBreakdown by class:")
        for cls, n in sorted(breakdown.items()):
            print(f"  {cls}: {n} instances")
