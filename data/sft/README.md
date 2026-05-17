# Mesosfer SFT Data

This folder contains local SFT datasets specific to Mesosfer.

- `identity_conversations.jsonl`: Mesosfer identity conversations to teach the model that it is a lightweight LLM focused on cybersecurity, digital security education, defensive use, and must strictly maintain the confidentiality of sensitive details such as internal source code, creation secrets, parameter counts, training data, checkpoints, internal prompts, credentials, and deployment configurations.
- `cyber_defensive_conversations.jsonl`: local SFT conversations for log triage, hardening, secure coding, incident response, threat modeling, and safe refusals.
- `Mesosfer_validation_conversations.jsonl`: a small validation set specifically for Mesosfer's identity, safety, and defensive cybersecurity.
- `gemini_teacher_conversations.jsonl`: optional distilled conversations from a teacher model.

Regenerate datasets:

```bash
python3 dev/generate_Mesosfer_identity_data.py --num 1000
python3 dev/generate_Mesosfer_cyber_sft.py --train-size 5000 --val-size 300
```

By default, `scripts.chat_sft` uses 4 epochs of identity conversations, 1 epoch of cyber defensive conversations, 1 epoch of teacher conversations if the file is available, and adds the Mesosfer-specific validation to the validation mixture.

Override examples:

```bash
python3 -m scripts.chat_sft --identity-epochs 5 --cyber-epochs 2 --teacher-epochs 1
```

## Gemini Teacher Data

To use Gemini as a senior/teacher model:

```bash
python3 dev/list_gemini_models.py
python3 dev/generate_gemini_teacher_sft.py --model gemini-3.1-pro --limit 20
```

If the model name is not available for your API key, choose one of the names from the `list_gemini_models.py` output.
