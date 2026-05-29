# Dataset Configuration & Sampling Weights

## 0. Token Budget Summary (Depth 32)

| Mode | Scaling Params | Ratio | Tokens Needed | Tokens Available | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Speedrun (`--target-param-data-ratio=10`) | ~6.5B | 10× | ~65B | ~100B+ | ✅ 1.5× surplus |
| Recommended (`--target-param-data-ratio=15`) | ~6.5B | 15× | ~98B (~100B) | ~100B+ | ✅ Fully Covered |
| Compute-optimal (`--target-param-data-ratio=18`) | ~6.5B | 18× | ~117B | ~100B+ | ⚠️ Scaled limits |

> **Depth 32 config:** `n_embd = 32 × 128 = 4096`, ~9.8B total params, ~6.5B scaling params (excl. embeddings).
> Recommended training command: `--depth=32 --target-param-data-ratio=15` (~100B tokens)

---

## 1. Domain Sampling Weights

This section controls temperature-based interleaved shuffling, where higher weights make a domain appear more frequently in mixed shards. These numbers control the relative probability of document selection, not the total volume.

**Weight Guidelines:**
- **2.0+ (Critical Priority):** Exploit/vuln patches, SOC-critical intel, curated seeds, local security data.
- **1.5–1.9 (High Priority):** Structured cybersecurity knowledge, CVE feeds, threat intel, coding feedback.
- **1.2–1.4 (Elevated):** Secure code, detection rules, exploit frameworks.
- **0.9–1.1 (Normal):** Competition math reasoning, general instruction.
- **0.5–0.7 (Reduced):** General web content to prevent domination.

> **Shuffle config (updated):** `cluster_size=32`, `sampling_temperature=1.2`
> Higher cluster_size (32 vs old 8) and temperature >1.0 reduce domain-burst loss spikes.

**Domain Weight Table:**

| Domain | Category | Weight | Max Tokens | Specific Notes |
| :--- | :--- | :--- | :--- | :--- |
| `circl_vuln_patch` | Cybersecurity | 2.3 | 2.4B | Highest priority. Real vuln+patch pairs. |
| `local_incident_response` | Cybersecurity | 2.2 | 1.44B | Synthetic IR reports. |
| `primus_seed` | Cybersecurity | 2.2 | 2.0B | Curated cybersecurity seed corpus (Trend Micro). |
| `local_soc_synthetic` | Cybersecurity | 2.1 | 1.44B | Synthetic SOC dialogues. |
| `primus_nemotron_cc` | Cybersecurity | 2.1 | 7.6B | Cybersecurity text filtered from Nemotron-CC. |
| `local_reverse_engineering` | Cybersecurity | 2.0 | 1.2B | RE/exploitation analysis. |
| `cybernative_vuln_dpo` | Cybersecurity | 2.0 | 360M | Synthetic vulnerable vs fixed code pairs. |
| `brightdata_cybersec` | Cybersecurity | 2.0 | 500M | Real-time threat intel scraped via BrightData proxy. |
| `primus_reasoning` | Cybersecurity | 2.0 | 1.5B | Distilled cybersecurity chain-of-thought. |
| `project_zero` | Cybersecurity | 2.0 | 40M | Google Project Zero RSS. |
| `mandiant` | Cybersecurity | 2.0 | 40M | Mandiant threat intel RSS. |
| `dfir_report` | Cybersecurity | 2.0 | 35M | DFIR Report intrusion analysis RSS. |
| `cisa_kev` | Cybersecurity | 2.0 | 10M | CISA Known Exploited Vulnerabilities. |
| `trendyol_cyber` | Cybersecurity | 1.9 | 2.4B | Cybersecurity instruction tuning. |
| `fenrir_v2` | Cybersecurity | 1.9 | 2.4B | 99K cybersec Q&A (OWASP/MITRE/NIST). |
| `primus_fineweb` | Cybersecurity | 1.9 | 5.0B | Cybersecurity-filtered FineWeb corpus. |
| `unit42` | Cybersecurity | 1.9 | 40M | Palo Alto Unit42 threat research RSS. |
| `talos` | Cybersecurity | 1.9 | 40M | Cisco Talos threat intel RSS. |
| `local_cloud_security` | Cybersecurity | 1.9 | 1.2B | Cloud audit logs (NL narratives). |
| `local_security_logs` | Cybersecurity | 1.9 | 1.2B | Auth/web/Sysmon logs (NL narratives). |
| `github_advisory` | Cybersecurity | 1.9 | 100M | GitHub Security Advisory database. |
| `all_cve_records` | Cybersecurity | 1.8 | 6.0B | NVD CVE JSON feeds 2002–present. |
| `nist_cybersec` | Cybersecurity | 1.8 | 4.8B | NIST cybersecurity training docs. |
| `nvd_cve` | Cybersecurity | 1.8 | 4.8B | CIRCL CVE List v5 ndjson dump. |
| `academic_security_papers` | Cybersecurity | 1.8 | 120M | Top-tier security conference papers. |
| `mitre_attack_stix` | Cybersecurity | 1.8 | 70M | MITRE ATT&CK STIX knowledge base. |
| `exploitdb` | Code | 1.8 | 140M | Exploit Database PoC corpus. |
| `metasploit` | Code | 1.7 | 180M | Metasploit Framework modules. |
| `cloudflare_security` | Cybersecurity | 1.6 | 30M | Cloudflare security research RSS. |
| `sigmahq_rules` | Cybersecurity | 1.6 | 70M | SigmaHQ detection rules. |
| `elastic_rules` | Cybersecurity | 1.6 | 70M | Elastic detection rules. |
| `code_feedback` | Code | 1.5 | 3.0B | Multi-language coding feedback instruction. |
| `threat_report_pdfs` | Cybersecurity | 1.5 | 90M | Vendor threat report PDFs. |
| `splunk_rules` | Cybersecurity | 1.5 | 60M | Splunk security detections. |
| `zeek_scripts` | Cybersecurity | 1.5 | 40M | Zeek scripts and packages. |
| `secure_code_python` | Code | 1.4 | 6.0B | Python from The Stack (dedup). |
| `secure_code_c` | Code | 1.4 | 4.2B | C from The Stack (dedup). |
| `secure_code_shell` | Code | 1.4 | 3.0B | Shell from The Stack (dedup). |
| `swallow_code_v2` | Code | 1.3 | 8.0B | Refined Python code via 4-stage LLM pipeline (TokyoTech). |
| `secure_code_cpp` | Code | 1.2 | 4.8B | C++ from The Stack (dedup). |
| `secure_code_rust` | Code | 1.2 | 3.6B | Rust from The Stack (dedup). |
| `secure_code_go` | Code | 1.2 | 3.6B | Go from The Stack (dedup). |
| `numinamath_cot` | Instruction | 1.1 | 3.0B | Competition math with chain-of-thought reasoning. |
| `openhermes` | Instruction | 0.9 | 4.2B | OpenHermes-2.5 conversations. |
| `finemath` | Instruction | 0.9 | 3.0B | finemath-4plus mathematical reasoning. |
| `fineweb_edu` | General | 0.7 | 4.0B | Educational web content. |
| `climbmix` | General | 0.6 | 5.0B | General pretraining, suppressed. |
| `wikipedia` | General | 0.5 | 3.0B | English Wikipedia, suppressed. |

**Total available tokens: ~100B+**

---

## 2. Dataset Sources

Dataset source definitions include data origin, token limits (as upper bounds), and data formats.

### A. Curated Cybersecurity & Reasoning (New in Depth 32)
* **`primus_seed`**: Trend Micro Curated Seed Corpus (`trendmicro-ailab/Primus-Seed`) containing high-quality security articles, capped at **2.0B tokens**.
* **`primus_reasoning`**: Distilled cybersecurity chain-of-thought reasoning (`trendmicro-ailab/Primus-Reasoning`) distilled from frontier models, capped at **1.5B tokens**.
* **`brightdata_cybersec`**: Scraped threat intelligence blogs via BrightData Proxy integration, capped at **500M tokens**.
* **`numinamath_cot`**: 860K+ competition math problems with chain-of-thought solutions (`AI-MO/NuminaMath-CoT`) in custom instruction format, capped at **3.0B tokens**.

### B. Scaled Cybersecurity
* **`trendyol_cyber`**: Cybersecurity instruction tuning dataset (`Trendyol/Trendyol-Cybersecurity-Instruction-Tuning-Dataset`) in instruction format, scaled to **2.4B tokens**.
* **`all_cve_records`**: Sourced from NVD JSON feeds (2002–present), scaled to **6.0B tokens**.
* **`circl_vuln_patch`**: 39K vulnerabilities with real patches (`CIRCL/vulnerability-cwe-patch`), vuln_patch format, scaled to **2.4B tokens**.
* **`nist_cybersec`**: NIST cybersecurity training documents (`ethanolivertroy/nist-cybersecurity-training`), streaming, scaled to **4.8B tokens**.
* **`fenrir_v2`**: 99K high-quality cybersecurity Q&A (OWASP/MITRE/NIST) (`AlicanKiraz0/Cybersecurity-Dataset-Fenrir-v2.1`), chat format, scaled to **2.4B tokens**.
* **`primus_nemotron_cc`**: 7.6B tokens of filtered cybersecurity text (`trend-cybertron/Primus-Nemotron-CC`), scaled to **7.6B tokens**.
* **`primus_fineweb`**: Cybersecurity-filtered FineWeb pretraining corpus (`trendmicro-ailab/Primus-FineWeb`), scaled to **5.0B tokens**.
* **`nvd_cve`**: CIRCL Vulnerability-Lookup CVE List v5 ndjson dump, scaled to **4.8B tokens**.

### C. Local Cybersecurity Data
> ⚠️ **Log files are pre-processed.** Raw `.log`, `.xml`, `.json` files are converted to natural language narratives via `scripts/data/convert_logs_to_nl.py` before training. Output stored in `data/log_nl/` and `data/cloud_nl/`.

* **`local_incident_response`**: Synthetic IR reports from `data/synthetic-ir/`, scaled to **1.44B tokens**.
* **`local_soc_synthetic`**: Synthetic SOC analyst conversations from `data/synthetic-soc/`, scaled to **1.44B tokens**.
* **`local_reverse_engineering`**: RE/exploitation analysis from `data/reverse-engineering/`, scaled to **1.2B tokens**.
* **`local_cloud_security`**: AWS CloudTrail, Azure Activity, GCP Audit logs as NL narratives from `data/cloud_nl/`, scaled to **1.2B tokens**.
* **`local_security_logs`**: Auth, Apache, Sysmon, Windows Event, CEF, Zeek, Suricata logs as NL narratives from `data/log_nl/`, scaled to **1.2B tokens**.

### D. General Reference & Math
* **`climbmix`**: High-quality general pretraining data (`karpathy/climbmix-400b-shuffle`), streaming, scaled to **5.0B tokens**.
* **`wikipedia`**: English Wikipedia (`wikimedia/wikipedia` subset `20231101.en`), streaming, scaled to **3.0B tokens**.
* **`fineweb_edu`**: Educational web content (`HuggingFaceFW/fineweb-edu` subset `sample-10BT`), streaming, scaled to **4.0B tokens**.

### E. Secure Code & Refined Snippets
Sourced from `bigcode/the-stack-dedup` in streaming mode:
* **`secure_code_python`**: `data/python`, scaled to **6.0B tokens**.
* **`secure_code_c`**: `data/c`, scaled to **4.2B tokens**.
* **`secure_code_cpp`**: `data/cpp`, scaled to **4.8B tokens**.
* **`secure_code_rust`**: `data/rust`, scaled to **3.6B tokens**.
* **`secure_code_go`**: `data/go`, scaled to **3.6B tokens**.
* **`secure_code_shell`**: `data/shell`, scaled to **3.0B tokens**.
* **`swallow_code_v2`**: Refined Python code from TokyoTech (`tokyotech-llm/swallow-code-v2` subset `stage4-llm-rewrite`), scaled to **8.0B tokens**.
* **`code_feedback`**: Multi-language coding feedback and instruction pairs (`m-a-p/Code-Feedback`), scaled to **3.0B tokens**.

### F. Exploit Frameworks
* **`metasploit`**: Metasploit Framework modules from GitHub (`rapid7/metasploit-framework`), scaled to **180M tokens**.
* **`exploitdb`**: Exploit Database PoC corpus from GitHub (`offensive-security/exploitdb`), scaled to **140M tokens**.

### G. General Instruction & Mathematics
* **`openhermes`**: Conversation/instruction data (`teknium/OpenHermes-2.5`), streaming, scaled to **4.2B tokens**.
* **`finemath`**: Mathematical reasoning corpus (`HuggingFaceTB/finemath` subset `finemath-4plus`), streaming, scaled to **3.0B tokens**.

---

## 3. Technical Notes

1. **Model Focus (Cybersecurity & Coding):** This model is specifically optimized for cybersecurity and secure coding. Vulnerability datasets (CVE, patches, SOC, Curated Seeds) carry the highest sampling weights (1.8–2.3), while general web content is suppressed (0.5–0.7).
2. **Loss Spike Fix:** Raw log files (`.log`, `.xml`, `.json`) previously caused loss spikes due to sudden domain shift. They are now pre-converted to natural language narratives via `scripts/data/convert_logs_to_nl.py` before being included in training.
3. **Shuffling Mechanism (Interleaving):** Temperature-weighted interleaved sampling with `cluster_size=32` and `sampling_temperature=1.2`. Higher cluster_size (vs old default of 8) reduces abrupt domain transitions. Temperature >1.0 flattens the weight distribution to prevent domain bursts.
4. **Scale Management via Max Tokens & Streaming:** `max_tokens` acts as a hard upper bound per source. Streaming is enabled on large corpora to avoid RAM exhaustion.
5. **Token Budget:** Total available ~100B+ tokens. Depth 32 requires ~98B tokens (ratio=15) to cover compute-optimal pretraining.
