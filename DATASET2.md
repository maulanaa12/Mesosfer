# Advanced Dataset Configuration: Vocabulary, Schema & Dynamic Sources

## 1. Security Vocabulary Markers (Lexical Markers)

This section defines keyword tuples used for text processing such as filtering, automatic tagging, or document quality scoring.

* **`CAUSAL_MARKERS`**: Keywords indicating cause-effect or vulnerability impact (examples: *"exploits"*, *"bypasses"*, *"allows attacker to"*). Very useful for detecting sentences explaining how exploits work.
* **`MECHANISM_TERMS`**: Specific terms related to vulnerability types, memory corruption, and reverse engineering techniques (examples: *"buffer overflow"*, *"rop chain"*, *"use-after-free"*).
* **`MARKETING_TERMS`**: Promotional or sales phrases (examples: *"book a demo"*, *"free trial"*). Usually used as a **negative filter** to discard text that contains no technical value (spam/promotional text).
* **`SECURITY_STACK_TERMS`**: General terms related to infrastructure, tools, and cybersecurity defense concepts (examples: *"firewall"*, *"siem"*, *"yara"*).
* **`SECURITY_CODE_TERMS`**: Low-level technical vocabulary, pentest tools, and system architecture for detecting code and hacking-related documents (examples: *"shellcode"*, *"ghidra"*, *"vtable"*, *"syscall"*).

---

## 2. Data Source Schema (Source Dataclass)

A data structure invariant (`@dataclass(frozen=True)`) named `Source` that defines the contract/required attributes for each dataset source:
* `name`: Data source name.
* `source_type`: Integration type (examples: *rss, pdf_manifest, github_repo, hf_text*).
* `domain` & `primary_subdomain`: High-level categorization (e.g.: *Cyber*, *Threat Intel*).
* `expected_tier`: Quality/confidence level (examples: *GOLD*, *SILVER*, *BRONZE_CLEAN*).
* `estimated_tokens` & `max_tokens`: Volume estimate and upper bound.
* `description`: Brief source description.
* `config`: Specific parameters (URL, environment variables, repository arguments).

---

## 3. Dynamic Scale Source Definitions (`_source_definitions`)

This function maps all baseline data sources using dynamic ratio calculations based on the target dataset size (`target_tokens`). Scale is calculated from a 500 million token baseline ratio (`scale = target_tokens / 500_000_000`).

### A. Curated Prose Intel & Threat Reports (Target: ~30% of total)

Focus on high-quality threat research, incident reports, and security advisories at **TIER_GOLD** and **TIER_SILVER** status.

| Source | Type | Tier | Max Tokens (at scale=1) |
| :--- | :--- | :--- | :--- |
| Project Zero | RSS | GOLD | 40M |
| Unit42 | RSS | GOLD | 40M |
| Mandiant | RSS | GOLD | 40M |
| Cloudflare Security | RSS | SILVER | 30M |
| Talos | RSS | GOLD | 40M |
| DFIR Report | RSS | GOLD | 35M |
| Academic Security Papers | PDF manifest | GOLD | 120M |
| Threat Report PDFs | PDF manifest | SILVER | 90M |
| GitHub Advisory (GHSA) | GitHub repo | GOLD | 100M |
| CISA KEV | URL JSON | GOLD | 10M |
| MITRE ATT&CK STIX | GitHub repo | GOLD | 70M |
| **Subtotal** | | | **~615M** |

### B. Curated Seeds & Deep Reasoning (New in Depth 32)

High-value curated corpora and multi-step reasoning outputs crucial for Depth 32 parameter capabilities.

| Source | Type | Tier | Max Tokens |
| :--- | :--- | :--- | :--- |
| `primus_seed` | HF streaming | GOLD | 2.0B |
| `primus_reasoning` | HF streaming | GOLD | 1.5B |
| `brightdata_cybersec` | Scraper Proxy | GOLD | 500M |
| `numinamath_cot` | HF streaming | GOLD | 3.0B |
| **Subtotal** | | | **7.0B** |

### C. Synthetic SOC & IR (Target: ~15% of total)

Focus on synthetic/simulated conversations at **TIER_GOLD** status.

| Source | Type | Tier | Max Tokens |
| :--- | :--- | :--- | :--- |
| `local_incident_response` | local JSONL | GOLD | 1.44B |
| `local_soc_synthetic` | local JSONL | GOLD | 1.44B |
| **Subtotal** | | | **2.88B** |

### D. Reverse Engineering (Target: ~10% of total)

| Source | Type | Tier | Max Tokens |
| :--- | :--- | :--- | :--- |
| `local_reverse_engineering` | local JSONL | GOLD | 1.2B |
| **Subtotal** | | | **1.2B** |

### E. Logs — Natural Language Narratives (Target: ~15% of total)

> ⚠️ **Log narratives eliminate loss spikes:** Raw log files (`.log`, `.xml`, `.json`) are no longer fed directly into training. They are pre-converted to natural language security narratives by `scripts/data/convert_logs_to_nl.py` to prevent distribution shifts.

**Conversion pipeline:**
```
data/log/*.{log,xml,jsonl}  ──► convert_logs_to_nl.py ──► data/log_nl/*.jsonl
data/cloud/*.json           ──► convert_logs_to_nl.py ──► data/cloud_nl/*.jsonl
```

**Supported input formats → narrative output:**

| Input Format | Source Files | MITRE Coverage |
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

| Source | Type | Tier | Max Tokens |
| :--- | :--- | :--- | :--- |
| `local_security_logs` | local JSONL (NL) | BRONZE_CLEAN → GOLD | 1.2B |
| `local_cloud_security` | local JSONL (NL) | BRONZE_CLEAN → GOLD | 1.2B |
| **Subtotal** | | | **2.4B** |

### F. Detections / Rules (Target: ~10% of total)

Focus on structured detection rule formats from GitHub repositories (**TIER_SILVER**).

| Source | Type | Tier | Max Tokens (at scale=1) |
| :--- | :--- | :--- | :--- |
| SigmaHQ Rules | GitHub repo | SILVER | 70M |
| Elastic Rules | GitHub repo | SILVER | 70M |
| Splunk Rules | GitHub repo | SILVER | 60M |
| Zeek Scripts | GitHub repo | SILVER | 40M |
| **Subtotal** | | | **240M** |

### G. Exploit / Secure Code (Target: ~30% of total)

Focus on secure coding patterns, multi-language coding feedback, and exploit payloads (**TIER_SILVER**).

| Source | Type | Tier | Max Tokens |
| :--- | :--- | :--- | :--- |
| `exploitdb` | GitHub repo | SILVER | 140M |
| `metasploit` | GitHub repo | SILVER | 180M |
| `swallow_code_v2` | HF streaming | SILVER | 8.0B |
| `code_feedback` | HF streaming | GOLD | 3.0B |
| `secure_code_python` | HF streaming | SILVER | 6.0B |
| `secure_code_c` | HF streaming | SILVER | 4.2B |
| `secure_code_cpp` | HF streaming | SILVER | 4.8B |
| `secure_code_rust` | HF streaming | SILVER | 3.6B |
| `secure_code_go` | HF streaming | SILVER | 3.6B |
| `secure_code_shell` | HF streaming | SILVER | 3.0B |
| **Subtotal** | | | **36.6B** |

### H. General Reference & Suppressed Web (Target: Balance remainder)

Focus on balancing general language distribution.

| Source | Type | Tier | Max Tokens |
| :--- | :--- | :--- | :--- |
| `climbmix` | HF streaming | SILVER | 5.0B |
| `wikipedia` | HF streaming | SILVER | 3.0B |
| `fineweb_edu` | HF streaming | SILVER | 4.0B |
| `openhermes` | HF streaming | GOLD | 4.2B |
| `finemath` | HF streaming | GOLD | 3.0B |
| **Subtotal** | | | **19.2B** |

---

## 4. Total Token Budget

| Category | Subtotal |
| :--- | :--- |
| Curated Seeds & Deep Reasoning | ~7.0B |
| Prose Intel / Reports | ~615M |
| Cybersecurity HF datasets (CVE, NIST, etc.) | ~24.8B |
| Synthetic SOC + RE | ~4.08B |
| Logs (NL narratives) | ~2.4B |
| Detections / Rules | ~240M |
| Exploit / Code | ~36.6B |
| General Reference | ~19.2B |
| **Grand Total** | **~95.0B (~100B with RSS/PDF base scales)** |

**Depth 32 requirements:**

| Ratio | Tokens Needed | Coverage |
| :--- | :--- | :--- |
| 10 (speedrun) | ~65B | ✅ Covered with massive headroom |
| 15 (recommended) | ~98B | ✅ Fully covered |
| 18 (compute-optimal) | ~117B | ⚠️ Reaches max token limit scaling |

---

## 5. Technical Notes & Implications

1. **Keyword-Based Filtering & Heuristics:** Constants such as `MARKETING_TERMS` ensure the model does not learn "sales" language from security vendor blogs, focusing instead on pure technical content containing `CAUSAL_MARKERS` and `MECHANISM_TERMS`.

2. **Dynamic Token Scaling:** Using `int(base_value * scale)` allows scaling from experiments (500M tokens) to production (100B+ tokens) without manually reworking `max_tokens` per source. The formula `scale = target_tokens / 500_000_000` keeps inter-domain ratios balanced.

3. **Data Tiering (Gold, Silver, Bronze):**
   - **Gold:** Structured, clean, high-weight cybersecurity knowledge (Academic Papers, Mandiant/Project Zero Intel, Synthetic SOC/IR data, Primus-Seed/Reasoning).
   - **Silver:** Valid but potentially noisy (GitHub repos, Sigma rules, ExploitDB, The Stack code).
   - **Bronze Clean → Gold:** Raw logs that have been converted to NL narratives via `convert_logs_to_nl.py`. Tier effectively upgraded from Bronze to Gold after conversion.

4. **Modality Diversity (Source Types):** The pipeline consumes data from RSS feeds, GitHub API pulls, PDF manifests, NVD JSON feeds, HuggingFace streaming datasets, and locally generated security logs — all normalized to the same `{"text": "..."}` JSONL format before training.

5. **Loss Spike Prevention:** The previous configuration fed raw `.log`, `.xml`, and `.json` files directly into training, causing loss spikes when the model encountered sudden token distribution shifts. The NL conversion pipeline resolves this by ensuring all data has consistent natural language token distributions.

6. **Shuffle Configuration:** `cluster_size=32` (up from 8) and `sampling_temperature=1.2` (up from 1.0) in `interleaved_shuffle_main`. Higher cluster_size reduces abrupt domain transitions between shards; temperature >1.0 flattens the sampling distribution to prevent any single domain from dominating consecutive batches.
