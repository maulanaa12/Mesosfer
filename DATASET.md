# Dataset & Pretraining Data Architecture (mesosfer)

This document provides a comprehensive guide to the datasets, vocabulary markers, sampling weights, and natural language log narrative pipelines that power the pretraining phase of **mesosfer**.

---

## 1. Token Budget Summary (Depth 32)

| Mode | Scaling Params | Ratio | Tokens Needed | Tokens Available | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Speedrun (`--target-param-data-ratio=10`) | ~6.5B | 10× | ~65B | ~100B+ | ✅ 1.5× surplus |
| Recommended (`--target-param-data-ratio=15`) | ~6.5B | 15× | ~98B (~100B) | ~100B+ | ✅ Fully Covered |
| Compute-optimal (`--target-param-data-ratio=18`) | ~6.5B | 18× | ~117B | ~100B+ | ⚠️ Scaled limits |

> **Depth 32 Config:** `n_embd = 32 × 128 = 4096`, ~9.8B total params, ~6.5B scaling params (excl. embeddings).
> Recommended training command: `--depth=32 --target-param-data-ratio=15` (~100B tokens)

---

## 2. Domain Sampling Weights

Sampling weights control temperature-based interleaved shuffling, where higher weights make a domain appear more frequently in mixed shards. These weights determine the relative probability of document selection during data shuffling, not the total volume.

### A. Weight Guidelines:
- **2.0+ (Critical Priority):** Exploit/vuln patches, SOC-critical intel, curated seeds, local security data.
- **1.5–1.9 (High Priority):** Structured cybersecurity knowledge, CVE feeds, threat intel, coding feedback.
- **1.2–1.4 (Elevated):** Secure code, detection rules, exploit frameworks.
- **0.9–1.1 (Normal):** Competition math reasoning, general instruction.
- **0.5–0.7 (Reduced):** General web reference content to prevent domain domination.

> **Shuffle config (updated):** `cluster_size=32`, `sampling_temperature=1.2`
> Higher cluster_size (32 vs old 8) and temperature >1.0 reduce domain-burst loss spikes.

### B. Domain Weight Table:

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

---

## 3. Security Vocabulary Markers (Lexical Markers)

This section defines keyword tuples used for text filtering, quality scoring, and data routing in `prepare_data.py`:

* **`CAUSAL_MARKERS`**: Keywords indicating cause-effect or vulnerability impact (e.g., *"exploits"*, *"bypasses"*, *"allows attacker to"*).
* **`MECHANISM_TERMS`**: Specific memory corruption and reverse engineering concepts (e.g., *"buffer overflow"*, *"rop chain"*, *"use-after-free"*).
* **`MARKETING_TERMS`**: Promotional and sales phrases (e.g., *"book a demo"*, *"free trial"*). Used as a **negative filter** to eliminate non-technical vendor spam.
* **`SECURITY_STACK_TERMS`**: Infrastructure and defensive tooling (e.g., *"firewall"*, *"siem"*, *"yara"*).
* **`SECURITY_CODE_TERMS`**: Low-level technical vocabulary and exploitation frameworks (e.g., *"shellcode"*, *"ghidra"*, *"vtable"*, *"syscall"*).

---

## 4. Data Source Dataclass Schema

A frozen dataclass (`Source`) defines the configuration contract for each pretraining data source:
* `name`: Unique data source identifier.
* `source_type`: Integration pattern (e.g., `rss`, `github_repo`, `local_files`, `brightdata_scraper`, `url_json`).
* `domain` & `primary_subdomain`: High-level categorization.
* `expected_tier`: Quality assessment (`GOLD`, `SILVER`, or `BRONZE_CLEAN`).
* `estimated_tokens` & `max_tokens`: Size estimation and hard bounds.
* `description`: Contextual summary of the source.
* `config`: Source-specific parameters (URLs, directories, branches).

---

## 5. Pretraining Datasets & Categories

### A. Curated Seeds & Deep Reasoning
* **`primus_seed`**: Trend Micro Curated Seed Corpus (`trendmicro-ailab/Primus-Seed`) containing high-quality security articles, capped at **2.0B tokens**.
* **`primus_reasoning`**: Distilled cybersecurity chain-of-thought reasoning (`trendmicro-ailab/Primus-Reasoning`) distilled from frontier models, capped at **1.5B tokens**.
* **`brightdata_cybersec`**: Scraped threat intelligence blogs via BrightData Proxy integration, capped at **500M tokens**.
* **`numinamath_cot`**: 860K+ competition math problems with chain-of-thought solutions (`AI-MO/NuminaMath-CoT`), capped at **3.0B tokens**.

### B. Scaled Cybersecurity
* **`trendyol_cyber`**: Cybersecurity instruction tuning dataset (`Trendyol/Trendyol-Cybersecurity-Instruction-Tuning-Dataset`), scaled to **2.4B tokens**.
* **`all_cve_records`**: Sourced from NVD JSON feeds (2002–present), scaled to **6.0B tokens**.
* **`circl_vuln_patch`**: 39K vulnerabilities with real patches (`CIRCL/vulnerability-cwe-patch`), scaled to **2.4B tokens**.
* **`nist_cybersec`**: NIST cybersecurity training documents (`ethanolivertroy/nist-cybersecurity-training`), scaled to **4.8B tokens**.
* **`fenrir_v2`**: OWASP/MITRE/NIST Q&A (`AlicanKiraz0/Cybersecurity-Dataset-Fenrir-v2.1`), scaled to **2.4B tokens**.
* **`primus_nemotron_cc`**: Filtered cybersecurity text (`trend-cybertron/Primus-Nemotron-CC`), scaled to **7.6B tokens**.
* **`primus_fineweb`**: Cybersecurity-filtered FineWeb corpus (`trendmicro-ailab/Primus-FineWeb`), scaled to **5.0B tokens**.
* **`nvd_cve`**: CIRCL Vulnerability-Lookup CVE List v5 ndjson dump, scaled to **4.8B tokens**.

### C. Local Security & Natural Language Log Narratives
> ⚠️ **Loss Spike Prevention:** Raw log files (`.log`, `.xml`, `.json`) are converted to natural language narratives via `convert_logs_to_nl.py` before pretraining to prevent sudden gradient shocks.

```
data/log/*.{log,xml,jsonl}  ──► convert_logs_to_nl.py ──► data/log_nl/*.jsonl
data/cloud/*.json           ──► convert_logs_to_nl.py ──► data/cloud_nl/*.jsonl
```

**Supported Log Formats & Narrative Targets:**

| Input Format | Source Files | MITRE Mapping |
| :--- | :--- | :--- |
| Syslog (auth/privesc) | `auth.log`, `auth2.log`, `privesc.log`, `syslog_benign.log` | T1110.001, T1003.008, T1548.003 |
| Apache + ModSecurity | `apache.log`, `apache2.log` | T1190 |
| CEF | `cef.log` | Multi-vendor session |
| Zeek conn.log | `conn.log`, `conn2.log` | T1071.001 |
| Suricata/Zeek JSONL | `log.jsonl`, `c2_dns.jsonl` | T1071.001, T1071.004 |
| Sysmon XML | `sysmon.xml`, `webshell.xml` | T1059.001, T1547.001, T1505.003 |
| Windows Security Event XML | `winevent.xml`, `winevent2.xml` | T1550.002, T1053.005 |
| AWS CloudTrail | `cloudtrail.json`, `cloudtrail2.json` | T1562.008, T1136.003 |
| Azure Activity | `azure_activity.json`, `azure_activity2.json` | T1098.003 |
| GCP Audit | `gcp_audit.json`, `gcp_audit2.json` | T1078.004, T1530 |

* **`local_incident_response`**: Synthetic IR reports (`data/synthetic-ir/`), scaled to **1.44B tokens**.
* **`local_soc_synthetic`**: Synthetic SOC analyst conversations (`data/synthetic-soc/`), scaled to **1.44B tokens**.
* **`local_reverse_engineering`**: RE/exploitation analysis (`data/reverse-engineering/`), scaled to **1.2B tokens**.
* **`local_cloud_security`**: Audit logs as NL narratives (`data/cloud_nl/`), scaled to **1.2B tokens**.
* **`local_security_logs`**: System logs as NL narratives (`data/log_nl/`), scaled to **1.2B tokens**.

### D. Secure Code & Refined Snippets
Sourced from `bigcode/the-stack-dedup` in streaming mode:
* **`secure_code_python`**: `data/python`, scaled to **6.0B tokens**.
* **`secure_code_c`**: `data/c`, scaled to **4.2B tokens**.
* **`secure_code_cpp`**: `data/cpp`, scaled to **4.8B tokens**.
* **`secure_code_rust`**: `data/rust`, scaled to **3.6B tokens**.
* **`secure_code_go`**: `data/go`, scaled to **3.6B tokens**.
* **`secure_code_shell`**: `data/shell`, scaled to **3.0B tokens**.
* **`swallow_code_v2`**: Refined Python code from TokyoTech (`tokyotech-llm/swallow-code-v2 stage4-llm-rewrite`), scaled to **8.0B tokens**.
* **`code_feedback`**: Coding feedback and instruction pairs (`m-a-p/Code-Feedback`), scaled to **3.0B tokens**.

### E. Exploit Frameworks
* **`metasploit`**: Metasploit Framework modules (`rapid7/metasploit-framework`), scaled to **180M tokens**.
* **`exploitdb`**: Exploit Database PoC corpus (`offensive-security/exploitdb`), scaled to **140M tokens**.

### F. General Reference & Math
* **`climbmix`**: ClimbMix pretraining data (`karpathy/climbmix-400b-shuffle`), scaled to **5.0B tokens**.
* **`wikipedia`**: English Wikipedia (`wikimedia/wikipedia` subset `20231101.en`), scaled to **3.0B tokens**.
* **`fineweb_edu`**: Educational web content (`HuggingFaceFW/fineweb-edu` subset `sample-10BT`), scaled to **4.0B tokens**.
* **`openhermes`**: Conversation/instruction data (`teknium/OpenHermes-2.5`), scaled to **4.2B tokens**.
* **`finemath`**: Mathematical reasoning corpus (`HuggingFaceTB/finemath`), scaled to **3.0B tokens**.

---

## 6. Technical Notes & Implications

1. **Keyword-Based Filtering & Heuristics:** Constants such as `MARKETING_TERMS` ensure the model does not learn "sales" language from security vendor blogs, focusing instead on pure technical content containing `CAUSAL_MARKERS` and `MECHANISM_TERMS`.

2. **Dynamic Token Scaling:** Using `int(base_value * scale)` allows scaling from baseline target sizes to production (~100B+ tokens) without manually rewriting configs. The formula `scale = target_tokens / 500_000_000` keeps inter-domain ratios balanced.

3. **Data Tiering (Gold, Silver, Bronze):**
   - **Gold:** Structured, clean, high-weight cybersecurity knowledge (Academic Papers, Mandiant/Project Zero Intel, Synthetic SOC/IR data, Primus-Seed/Reasoning).
   - **Silver:** Valid but potentially noisy (GitHub repos, Sigma rules, ExploitDB, The Stack code).
   - **Bronze Clean → Gold:** Raw logs that have been converted to NL narratives via `convert_logs_to_nl.py`. Tier effectively upgraded from Bronze to Gold after conversion.

4. **Shuffling Mechanism (Interleaving):** Temperature-weighted interleaved sampling with `cluster_size=32` and `sampling_temperature=1.2` in `interleaved_shuffle_main`. Higher cluster_size reduces abrupt domain transitions; temperature >1.0 flattens the sampling distribution to prevent any single domain from dominating consecutive batches.
