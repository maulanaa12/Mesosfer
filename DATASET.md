# Dataset Configuration & Sampling Weights

## 1. Domain Sampling Weights

This section controls temperature-based interleaved shuffling, where higher weights make a domain appear more frequently in mixed shards. These numbers control the relative probability of document selection, not the total volume.

**Weight Guidelines:**
- **2.0 (High Priority):** For exploits, vuln patches, and critical SOC information.
- **1.5 (Elevated Priority):** For structured cybersecurity knowledge.
- **1.0 (Normal):** For general code, mathematics, and instructions.
- **0.7 (Reduced):** For wiki and general web content to prevent dataset domination.
- **0.5 (Low):** For telemetry and bulk text.

**Domain Weight Table:**

| Domain | Category | Weight | Specific Notes |
| :--- | :--- | :--- | :--- |
| `circl_vuln_patch` | Cybersecurity | 2.0 | Highest priority. |
| `trendyol_cyber` | Cybersecurity | 1.8 | - |
| `fenrir_v2` | Cybersecurity | 1.8 | - |
| `all_cve_records` | Cybersecurity | 1.5 | - |
| `nist_cybersec` | Cybersecurity | 1.5 | - |
| `nvd_cve` | Cybersecurity | 1.5 | - |
| `secure_code_python` | Code | 1.2 | - |
| `secure_code_c` | Code | 1.2 | - |
| `secure_code_shell`| Code | 1.2 | - |
| `secure_code_cpp` | Code | 1.0 | - |
| `secure_code_rust` | Code | 1.0 | - |
| `secure_code_go` | Code | 1.0 | - |
| `openhermes` | Instruction | 1.0 | - |
| `finemath` | Reasoning | 1.0 | - |
| `fineweb_edu` | General | 0.8 | - |
| `climbmix` | General | 0.7 | Very large, proportion needs suppression. |
| `wikipedia` | General | 0.6 | Large corpus, prevented from dominating. |

---

## 2. Dataset Sources

Dataset source definitions include data origin, token limits (as upper bounds), and data formats.

### A. Cybersecurity
* **`trendyol_cyber`**: Cybersecurity instruction tuning dataset from HuggingFace (`Trendyol/Trendyol-Cybersecurity-Instruction-Tuning-Dataset`) in instruction format, capped at 150M tokens.
* **`all_cve_records`**: Sourced from NVD JSON feeds (2002-present) with 400M token limit.
* **`circl_vuln_patch`**: Contains 39K vulnerabilities with real patches from GitHub/GitLab from HuggingFace (`CIRCL/vulnerability-cwe-patch`), in vuln_patch format, capped at 100M tokens.
* **`nist_cybersec`**: NIST cybersecurity training documents from HuggingFace (`ethanolivertroy/nist-cybersecurity-training`), streaming mode with messages format, capped at 300M tokens.
* **`fenrir_v2`**: Contains 99K high-quality cybersecurity Q&A (OWASP/MITRE/NIST) from HuggingFace (`AlicanKiraz0/Cybersecurity-Dataset-Fenrir-v2.1`), in chat format, capped at 150M tokens.
* **`nvd_cve`**: Sourced from CIRCL Vulnerability-Lookup CVE List v5 dump in ndjson format, capped at 300M tokens.

### B. General Knowledge
* **`climbmix`**: High-quality general pretraining data (`karpathy/climbmix-400b-shuffle`) in streaming mode, capped at 2.5B tokens.
* **`wikipedia`**: English Wikipedia articles (`wikimedia/wikipedia` subset `20231101.en`) in streaming mode, capped at 1B tokens.
* **`fineweb_edu`**: Educational web content (`HuggingFaceFW/fineweb-edu` subset `sample-10BT`) in streaming mode, capped at 1B tokens.

### C. Secure Code
Sourced from `bigcode/the-stack-dedup` in streaming mode:
* **`secure_code_python`**: Directory `data/python`, capped at 1B tokens.
* **`secure_code_c`**: Directory `data/c`, capped at 800M tokens.
* **`secure_code_cpp`**: Directory `data/cpp`, capped at 1B tokens.
* **`secure_code_rust`**: Directory `data/rust`, capped at 800M tokens.
* **`secure_code_go`**: Directory `data/go`, capped at 800M tokens.
* **`secure_code_shell`**: Directory `data/shell`, capped at 500M tokens.

### D. Instruction & Reasoning
* **`openhermes`**: Conversation/instruction data (`teknium/OpenHermes-2.5`) in messages format with streaming mode, capped at 400M tokens.
* **`finemath`**: Mathematical reasoning corpus (`HuggingFaceTB/finemath` subset `finemath-4plus`) in streaming mode, capped at 300M tokens.

---

## 3. Technical Notes

1. **Model Focus (Cybersecurity & Coding):** It is clear that this model is specifically optimized to become an expert in cybersecurity and secure coding. This is evidenced by assigning the highest priority scores (1.5-2.0) to vulnerability datasets such as CVE and security patches, compared to Wikipedia or general web content which are suppressed to (0.6-0.7).
2. **Shuffling Mechanism (Interleaving):** The system does not simply concatenate all text into one. The system uses sampling weights to determine how often data from a topic is "pulled" when mixing training data per batch. This keeps the model from "forgetting" how to write regular instructions when being fed millions of cybersecurity data.
3. **Scale Management via Max Tokens & Streaming:** The datasets used are massive in scale (such as Wikipedia and The Stack subsets). The `max_tokens` feature acts as a hard limit to prevent the model from leaning too heavily or overdosing on one language/topic. Additionally, `streaming: True` is enabled on giant corpora to avoid exhausting local RAM during training.