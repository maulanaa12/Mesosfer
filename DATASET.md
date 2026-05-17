# Dataset Configuration & Sampling Weights

## 0. Token Budget Summary (Depth 24)

| Mode | Scaling Params | Ratio | Tokens Needed | Tokens Available | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Speedrun (`--target-param-data-ratio=8`) | ~350M | 8× | ~2.8B | ~8.5B | ✅ 3× surplus |
| Recommended (`--target-param-data-ratio=10`) | ~350M | 10× | ~3.5B | ~8.5B | ✅ 2.4× surplus |
| Compute-optimal (`--target-param-data-ratio=12`) | ~350M | 12× | ~4.2B | ~8.5B | ✅ 2× surplus |

> **Depth 24 config:** `n_embd = 24 × 64 = 1536`, ~500–550M total params, ~350M scaling params (excl. embeddings).
> Recommended training command: `--depth=24 --target-param-data-ratio=10`

---

## 1. Domain Sampling Weights

This section controls temperature-based interleaved shuffling, where higher weights make a domain appear more frequently in mixed shards. These numbers control the relative probability of document selection, not the total volume.

**Weight Guidelines:**
- **2.0+ (Critical Priority):** Exploit/vuln patches, SOC-critical intel, local security data.
- **1.5–1.9 (High Priority):** Structured cybersecurity knowledge, CVE feeds, threat intel.
- **1.2–1.4 (Elevated):** Secure code, detection rules, exploit frameworks.
- **0.9 (Normal):** General instruction and reasoning.
- **0.5–0.7 (Reduced):** General web content to prevent domination.

> **Shuffle config (updated):** `cluster_size=32`, `sampling_temperature=1.2`
> Higher cluster_size (32 vs old 8) and temperature >1.0 reduce domain-burst loss spikes.

**Domain Weight Table:**

| Domain | Category | Weight | Max Tokens | Specific Notes |
| :--- | :--- | :--- | :--- | :--- |
| `circl_vuln_patch` | Cybersecurity | 2.3 | 200M | Highest priority. Real vuln+patch pairs. |
| `local_incident_response` | Cybersecurity | 2.2 | 120M | Synthetic IR reports. |
| `local_soc_synthetic` | Cybersecurity | 2.1 | 120M | Synthetic SOC dialogues. |
| `local_reverse_engineering` | Cybersecurity | 2.0 | 100M | RE/exploitation analysis. |
| `project_zero` | Cybersecurity | 2.0 | 40M | Google Project Zero RSS. |
| `mandiant` | Cybersecurity | 2.0 | 40M | Mandiant threat intel RSS. |
| `dfir_report` | Cybersecurity | 2.0 | 35M | DFIR Report intrusion analysis RSS. |
| `cisa_kev` | Cybersecurity | 2.0 | 10M | CISA Known Exploited Vulnerabilities. |
| `trendyol_cyber` | Cybersecurity | 1.9 | 200M | Cybersecurity instruction tuning. |
| `fenrir_v2` | Cybersecurity | 1.9 | 200M | 99K cybersec Q&A (OWASP/MITRE/NIST). |
| `unit42` | Cybersecurity | 1.9 | 40M | Palo Alto Unit42 threat research RSS. |
| `talos` | Cybersecurity | 1.9 | 40M | Cisco Talos threat intel RSS. |
| `local_cloud_security` | Cybersecurity | 1.9 | 100M | Cloud audit logs (NL narratives). |
| `local_security_logs` | Cybersecurity | 1.9 | 100M | Auth/web/Sysmon logs (NL narratives). |
| `github_advisory` | Cybersecurity | 1.9 | 100M | GitHub Security Advisory database. |
| `all_cve_records` | Cybersecurity | 1.8 | 500M | NVD CVE JSON feeds 2002–present. |
| `nist_cybersec` | Cybersecurity | 1.8 | 400M | NIST cybersecurity training docs. |
| `nvd_cve` | Cybersecurity | 1.8 | 400M | CIRCL CVE List v5 ndjson dump. |
| `academic_security_papers` | Cybersecurity | 1.8 | 120M | Top-tier security conference papers. |
| `mitre_attack_stix` | Cybersecurity | 1.8 | 70M | MITRE ATT&CK STIX knowledge base. |
| `exploitdb` | Code | 1.8 | 140M | Exploit Database PoC corpus. |
| `metasploit` | Code | 1.7 | 180M | Metasploit Framework modules. |
| `cloudflare_security` | Cybersecurity | 1.6 | 30M | Cloudflare security research RSS. |
| `sigmahq_rules` | Cybersecurity | 1.6 | 70M | SigmaHQ detection rules. |
| `elastic_rules` | Cybersecurity | 1.6 | 70M | Elastic detection rules. |
| `threat_report_pdfs` | Cybersecurity | 1.5 | 90M | Vendor threat report PDFs. |
| `splunk_rules` | Cybersecurity | 1.5 | 60M | Splunk security detections. |
| `zeek_scripts` | Cybersecurity | 1.5 | 40M | Zeek scripts and packages. |
| `secure_code_python` | Code | 1.4 | 500M | Python from The Stack (dedup). |
| `secure_code_c` | Code | 1.4 | 350M | C from The Stack (dedup). |
| `secure_code_shell` | Code | 1.4 | 250M | Shell from The Stack (dedup). |
| `secure_code_cpp` | Code | 1.2 | 400M | C++ from The Stack (dedup). |
| `secure_code_rust` | Code | 1.2 | 300M | Rust from The Stack (dedup). |
| `secure_code_go` | Code | 1.2 | 300M | Go from The Stack (dedup). |
| `openhermes` | Instruction | 0.9 | 350M | OpenHermes-2.5 conversations. |
| `finemath` | Reasoning | 0.9 | 250M | finemath-4plus mathematical reasoning. |
| `fineweb_edu` | General | 0.7 | 600M | Educational web content. |
| `climbmix` | General | 0.6 | 900M | General pretraining, suppressed. |
| `wikipedia` | General | 0.5 | 500M | English Wikipedia, suppressed. |

**Total available tokens: ~8.5B**

---

## 2. Dataset Sources

Dataset source definitions include data origin, token limits (as upper bounds), and data formats.

### A. Cybersecurity
* **`trendyol_cyber`**: Cybersecurity instruction tuning dataset from HuggingFace (`Trendyol/Trendyol-Cybersecurity-Instruction-Tuning-Dataset`) in instruction format, capped at **200M tokens**.
* **`all_cve_records`**: Sourced from NVD JSON feeds (2002–present), capped at **500M tokens**.
* **`circl_vuln_patch`**: 39K vulnerabilities with real patches from HuggingFace (`CIRCL/vulnerability-cwe-patch`), vuln_patch format, capped at **200M tokens**.
* **`nist_cybersec`**: NIST cybersecurity training documents from HuggingFace (`ethanolivertroy/nist-cybersecurity-training`), streaming, capped at **400M tokens**.
* **`fenrir_v2`**: 99K high-quality cybersecurity Q&A (OWASP/MITRE/NIST) from HuggingFace (`AlicanKiraz0/Cybersecurity-Dataset-Fenrir-v2.1`), chat format, capped at **200M tokens**.
* **`nvd_cve`**: CIRCL Vulnerability-Lookup CVE List v5 ndjson dump, capped at **400M tokens**.

### B. Local Cybersecurity Data
> ⚠️ **Log files are pre-processed.** Raw `.log`, `.xml`, `.json` files are converted to natural language narratives via `scripts/data/convert_logs_to_nl.py` before training. Output stored in `data/log_nl/` and `data/cloud_nl/`.

* **`local_incident_response`**: Synthetic IR reports from `data/synthetic-ir/`, capped at **120M tokens**.
* **`local_soc_synthetic`**: Synthetic SOC analyst conversations from `data/synthetic-soc/`, capped at **120M tokens**.
* **`local_reverse_engineering`**: RE/exploitation analysis from `data/reverse-engineering/`, capped at **100M tokens**.
* **`local_cloud_security`**: AWS CloudTrail, Azure Activity, GCP Audit logs as NL narratives from `data/cloud_nl/`, capped at **100M tokens**.
* **`local_security_logs`**: Auth, Apache, Sysmon, Windows Event, CEF, Zeek, Suricata logs as NL narratives from `data/log_nl/`, capped at **100M tokens**.

### C. General Knowledge
* **`climbmix`**: High-quality general pretraining data (`karpathy/climbmix-400b-shuffle`), streaming, capped at **900M tokens**.
* **`wikipedia`**: English Wikipedia (`wikimedia/wikipedia` subset `20231101.en`), streaming, capped at **500M tokens**.
* **`fineweb_edu`**: Educational web content (`HuggingFaceFW/fineweb-edu` subset `sample-10BT`), streaming, capped at **600M tokens**.

### D. Secure Code
Sourced from `bigcode/the-stack-dedup` in streaming mode:
* **`secure_code_python`**: `data/python`, capped at **500M tokens**.
* **`secure_code_c`**: `data/c`, capped at **350M tokens**.
* **`secure_code_cpp`**: `data/cpp`, capped at **400M tokens**.
* **`secure_code_rust`**: `data/rust`, capped at **300M tokens**.
* **`secure_code_go`**: `data/go`, capped at **300M tokens**.
* **`secure_code_shell`**: `data/shell`, capped at **250M tokens**.

### E. Exploit Frameworks
* **`metasploit`**: Metasploit Framework modules from GitHub (`rapid7/metasploit-framework`), capped at **180M tokens**.
* **`exploitdb`**: Exploit Database PoC corpus from GitHub (`offensive-security/exploitdb`), capped at **140M tokens**.

### F. Instruction & Reasoning
* **`openhermes`**: Conversation/instruction data (`teknium/OpenHermes-2.5`), streaming, capped at **350M tokens**.
* **`finemath`**: Mathematical reasoning corpus (`HuggingFaceTB/finemath` subset `finemath-4plus`), streaming, capped at **250M tokens**.

---

## 3. Technical Notes

1. **Model Focus (Cybersecurity & Coding):** This model is specifically optimized for cybersecurity and secure coding. Vulnerability datasets (CVE, patches, SOC) carry the highest sampling weights (1.8–2.3), while general web content is suppressed (0.5–0.7).
2. **Loss Spike Fix:** Raw log files (`.log`, `.xml`, `.json`) previously caused loss spikes due to sudden domain shift. They are now pre-converted to natural language narratives via `scripts/data/convert_logs_to_nl.py` before being included in training.
3. **Shuffling Mechanism (Interleaving):** Temperature-weighted interleaved sampling with `cluster_size=32` and `sampling_temperature=1.2`. Higher cluster_size (vs old default of 8) reduces abrupt domain transitions. Temperature >1.0 flattens the weight distribution to prevent domain bursts.
4. **Scale Management via Max Tokens & Streaming:** `max_tokens` acts as a hard upper bound per source. Streaming is enabled on large corpora (climbmix, wikipedia, The Stack) to avoid RAM exhaustion.
5. **Token Budget:** Total available ~8.5B tokens. Depth 24 requires ~2.8B (ratio=8) to ~4.2B (ratio=12). Recommended: `--target-param-data-ratio=10` (~3.5B needed).
