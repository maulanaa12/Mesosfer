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

This function maps all data sources using dynamic ratio calculations based on the target dataset size (`target_tokens`). Scale is calculated from a 500 million token baseline ratio (`scale = target_tokens / 500_000_000`).

### A. Prose Intel / Reports (Target: 25–35% of total, ~2.1–3.0B tokens)

Focus on threat research, incident reports, and security advisories at **TIER_GOLD** and **TIER_SILVER** status.

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

### B. Synthetic SOC (Target: 10–20%, ~850M–1.7B tokens)

Focus on synthetic/simulated data at **TIER_GOLD** status.

| Source | Type | Tier | Max Tokens |
| :--- | :--- | :--- | :--- |
| `local_incident_response` | local JSONL | GOLD | 120M |
| `local_soc_synthetic` | local JSONL | GOLD | 120M |
| **Subtotal** | | | **240M** |

### C. Reverse Engineering (Target: 10–15%, ~850M–1.3B tokens)

| Source | Type | Tier | Max Tokens |
| :--- | :--- | :--- | :--- |
| `local_reverse_engineering` | local JSONL | GOLD | 100M |
| **Subtotal** | | | **100M** |

### D. Logs — Natural Language Narratives (Target: 12–18%, ~1.0–1.5B tokens)

> ⚠️ **Breaking change from previous version:** Raw log files (`.log`, `.xml`, `.json`) are no longer fed directly into training. They are pre-converted to natural language security narratives by `scripts/data/convert_logs_to_nl.py`. This eliminates loss spikes caused by sudden token distribution shifts when the model encounters raw structured data.

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
| `local_security_logs` | local JSONL (NL) | BRONZE_CLEAN → GOLD | 100M |
| `local_cloud_security` | local JSONL (NL) | BRONZE_CLEAN → GOLD | 100M |
| **Subtotal** | | | **200M** |

### E. Detections / Rules (Target: 10–15%, ~850M–1.3B tokens)

Focus on structured detection rule formats from GitHub repositories (**TIER_SILVER**).

| Source | Type | Tier | Max Tokens (at scale=1) |
| :--- | :--- | :--- | :--- |
| SigmaHQ Rules | GitHub repo | SILVER | 70M |
| Elastic Rules | GitHub repo | SILVER | 70M |
| Splunk Rules | GitHub repo | SILVER | 60M |
| Zeek Scripts | GitHub repo | SILVER | 40M |
| **Subtotal** | | | **240M** |

### F. Exploit / Code (Target: 20–30%, ~1.7–2.6B tokens)

Focus on exploit code and system utilities (**TIER_SILVER**).

| Source | Type | Tier | Max Tokens |
| :--- | :--- | :--- | :--- |
| `exploitdb` | GitHub repo | SILVER | 140M |
| `metasploit` | GitHub repo | SILVER | 180M |
| `secure_code_python` | HF streaming | SILVER | 500M |
| `secure_code_c` | HF streaming | SILVER | 350M |
| `secure_code_cpp` | HF streaming | SILVER | 400M |
| `secure_code_rust` | HF streaming | SILVER | 300M |
| `secure_code_go` | HF streaming | SILVER | 300M |
| `secure_code_shell` | HF streaming | SILVER | 250M |
| **Subtotal** | | | **2,420M (~2.4B)** |

### G. General Reference (Target: balance remainder)

Focus on balancing general data distribution.

| Source | Type | Tier | Max Tokens |
| :--- | :--- | :--- | :--- |
| `climbmix` | HF streaming | SILVER | 900M |
| `wikipedia` | HF streaming | SILVER | 500M |
| `fineweb_edu` | HF streaming | SILVER | 600M |
| `openhermes` | HF streaming | GOLD | 350M |
| `finemath` | HF streaming | GOLD | 250M |
| **Subtotal** | | | **2,600M (~2.6B)** |

---

## 4. Total Token Budget

| Category | Subtotal |
| :--- | :--- |
| Prose Intel / Reports | ~615M |
| Cybersecurity HF datasets (CVE, NIST, etc.) | ~1,500M |
| Synthetic SOC + RE | ~340M |
| Logs (NL narratives) | ~200M |
| Detections / Rules | ~240M |
| Exploit / Code | ~2,420M |
| General Reference | ~2,600M |
| **Grand Total** | **~7,915M (~8.5B with dataset2 RSS/PDF sources)** |

**Depth 24 requirements:**

| Ratio | Tokens Needed | Coverage |
| :--- | :--- | :--- |
| 8 (speedrun) | ~2.8B | ✅ Covered by code + general alone |
| 10 (recommended) | ~3.5B | ✅ Covered with margin |
| 12 (compute-optimal) | ~4.2B | ✅ Covered with margin |

---

## 5. Technical Notes & Implications

1. **Keyword-Based Filtering & Heuristics:** Constants such as `MARKETING_TERMS` ensure the model does not learn "sales" language from security vendor blogs, focusing instead on pure technical content containing `CAUSAL_MARKERS` and `MECHANISM_TERMS`.

2. **Dynamic Token Scaling:** Using `int(base_value * scale)` allows scaling from experiments (500M tokens) to production (3B+ tokens) without manually reworking `max_tokens` per source. The formula `scale = target_tokens / 500_000_000` keeps inter-domain ratios balanced.

3. **Data Tiering (Gold, Silver, Bronze):**
   - **Gold:** Structured, clean, high-weight cybersecurity knowledge (Academic Papers, Mandiant/Project Zero Intel, Synthetic SOC/IR data).
   - **Silver:** Valid but potentially noisy (GitHub repos, Sigma rules, ExploitDB, The Stack code).
   - **Bronze Clean → Gold:** Raw logs that have been converted to NL narratives via `convert_logs_to_nl.py`. Tier effectively upgraded from Bronze to Gold after conversion.

4. **Modality Diversity (Source Types):** The pipeline consumes data from RSS feeds, GitHub API pulls, PDF manifests, NVD JSON feeds, HuggingFace streaming datasets, and locally generated security logs — all normalized to the same `{"text": "..."}` JSONL format before training.

5. **Loss Spike Prevention:** The previous configuration fed raw `.log`, `.xml`, and `.json` files directly into training, causing loss spikes when the model encountered sudden token distribution shifts (e.g., transitioning from prose to `<EventID>4624</EventID>` XML). The NL conversion pipeline resolves this by ensuring all data has consistent natural language token distributions.

6. **Shuffle Configuration:** `cluster_size=32` (up from 8) and `sampling_temperature=1.2` (up from 1.0) in `interleaved_shuffle_main`. Higher cluster_size reduces abrupt domain transitions between shards; temperature >1.0 flattens the sampling distribution to prevent any single domain from dominating consecutive batches.
