# RLHF Feedback Data

Human preference feedback collected from the Mesosfer chat UI.

## Files

| File | Description |
|------|-------------|
| `feedback.jsonl` | Appended at runtime by `POST /feedback`. One JSON record per line. |

## Record Schema

```json
{
  "timestamp": "2026-05-20T10:00:00+00:00",
  "message_index": 1,
  "rating": "positive | negative",
  "reason": "inappropriate_response | continuous_repetition | factually_incorrect | too_verbose | formatting_issues | other | null",
  "comment": "optional free-text string | null",
  "conversation": [
    {"role": "user",      "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}
```

## Usage

This data is intended for:
1. **Reward model training** — use `rating` as the preference signal
2. **DPO / RLHF fine-tuning** — pair positive/negative responses for the same prompt
3. **Quality analysis** — aggregate `reason` counts to identify systematic failure modes

## Notes

- `feedback.jsonl` is excluded from git via `.gitignore` (add `data/rlhf/feedback.jsonl` if not already present).
- Comments are truncated to 2000 characters server-side.
- No PII is intentionally collected; review before using in training.
