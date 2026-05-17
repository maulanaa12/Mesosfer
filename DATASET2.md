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

### A. Prose Intel / Reports (Target: 25-35%)
Focus on threat research, incident reports, and security advisories at **TIER_GOLD** and **TIER_SILVER** status.
* **Project Zero, Unit42, Mandiant, Cloudflare, Talos, DFIR Report**: Data pulled automatically from cybersecurity research blog RSS feeds.
* **Academic Security Papers**: Papers from top-tier security conferences (USENIX, NDSS, IEEE) via PDF search.
* **Threat Report PDFs**: Vendor reports in PDF format.
* **Advisories**: NVD CVE, GitHub Advisory (GHSA), CISA KEV, and MITRE ATT&CK STIX.

### B. Synthetic SOC (Target: 10-20%)
Focus on synthetic/simulated data at **TIER_GOLD** status.
* **`synthetic_soc_dialogue`**: SOC reasoning dialogue (analyst interaction simulation).
* **`synthetic_telemetry`**: Cloud incident response (IR) simulations and telemetry workflows.

### C. Reverse Engineering (Target: 10-15%)
* **`reverse_engineering_corpus`**: Contains disassembly, decompilation, exploit development, and CTF reports (TIER_GOLD).

### D. Logs (Target: 12-18%)
Focus on cleaned raw telemetry data (**TIER_BRONZE_CLEAN**).
* **`logs`**: Network/host logs (Loghub, CICIDS2017, LANL).
* **`cloud_telemetry`**: Audit logs from AWS CloudTrail, GCP, and Azure.

### E. Detections / Rules (Target: 10-15%)
Focus on structured detection rule formats from GitHub repositories (**TIER_SILVER**).
* **`sigmahq_rules`, `elastic_rules`, `splunk_rules`, `zeek_scripts`**: Sigma rules, Elastic detection logs, Splunk queries (SPL), and Zeek scripts.

### F. Exploit / Code (Target: 20-30%)
Focus on exploit code and system utilities (**TIER_SILVER**).
* **`exploitdb`** & **`metasploit`**: Vulnerability Proof of Concept (PoC) database and Metasploit framework modules.
* **`secure_code_*`**: Security-specific programming per language (Python, PowerShell, Bash, C, C++, Rust, Go) and Infrastructure as Code (Terraform, K8s).

### G. General Reference
Focus on balancing general data distribution.
* **`wikipedia`** & **`fineweb_edu`**: Scientific, technical, and educational samples from English Wikipedia and FineWeb-Edu.

---

## 4. Technical Notes & Implications

1. **Keyword-Based Filtering & Heuristics:** Constants such as `MARKETING_TERMS` are crucial in NLP (Natural Language Processing) to ensure the model does not learn "sales" language from security vendor blogs, but rather focuses on pure technical content containing `CAUSAL_MARKERS` and `MECHANISM_TERMS`.
2. **Dynamic Token Scaling:** Using the formula `int(base_value * scale)` allows architects to scale dataset size from experiments (e.g., 500M tokens) to production (e.g., 3 Billion tokens) without manually reworking `max_tokens` limits per source. Inter-domain distribution ratios will remain balanced.
3. **Data Tiering (Gold, Silver, Bronze):**
   - **Gold:** The most structured, clean, and high-weight cybersecurity knowledge (Academic Papers, Mandiant/Project Zero Intel, Synthetic Data).
   - **Silver:** Structured data that is valid but may have some noise (GitHub repos, Sigma rules, ExploitDB).
   - **Bronze Clean:** Raw logs cleaned from irrelevant information.
4. **Modality Diversity (Source Types):** This pipeline approach is very sophisticated because it consumes data directly from various original modes: RSS, direct GitHub API pulls, PDF manifests (text extraction), NVD JSON files, HF text repositories, to locally generated network logs.