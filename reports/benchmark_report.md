# TripLedger Steward — Controlled Benchmark & Ablation Report

- **Execution Mode**: `LIVE_API` | **Model**: `deepseek-flash` | **Endpoint**: `https://api.deepseek.com/v1`
- **Controlled Parameters**: `temperature=0.0`, `seed=42`, `N=12`

## 1. End-to-End Architecture Comparison

| Metric | Monolithic ReAct Baseline | TripLedger Steward (Merged) | Delta |
| :--- | :---: | :---: | :---: |
| **Task Success Rate** ($N=12$) | 58.3% (7/12) | **100.0% (12/12)** | **+41.7 pp** |
| **Memory Recall@4** | 83.3% | **100.0%** | **+16.7 pp** |
| **Memory MRR** | 0.8333 | **1.0000** | **+0.1667** |
| **Security & Approval Gate Pass** | 33.3% | **100.0%** | **+66.7 pp** |
| **Zero-LLM Tool Schema Savings** | 0.0% (14/14 tools) | **56.9% (6/14 tools)** | **+56.9%** |
| **Mean Raw Input Tokens / Task** | 10912.7 | **4252.8** | **-61.0%** |
| **Multimodal Case Mean Input Tokens** | 18606.7 | **6129.7** | **-67.1%** |
| **Hardware Prefix KV-Cache Hit Ratio** | 69.9% | **73.5%** | **+3.6 pp** |
| **Mean Uncached Net Input Tokens** | 3286.0 | **1127.4** | **-65.7%** |
| **Latency P50 / P95 (ms)** | 7935 / 13991 | **4295 / 10614** | — |

## 2. Memory Subsystem Component Ablation

| Memory Configuration | Recall@4 | MRR | Key Failure Mode Resolved |
| :--- | :---: | :---: | :--- |
| Flat Vector + SQLite FTS5 BM25 | 83.3% | 0.8333 | Vulnerable to stale flight/meeting slots & 5-project distractor dilution |
| + Watermark Slot Supersession & Decay (No PPR) | 83.3% | 1.0000 | Resolves `CA1301 -> G79` supersession, still diluted on 2-hop `MEM-03` |
| **+ Pure-NumPy Directed PPR (`alpha=0.85`)** | **100.0%** | **1.0000** | Resolves `华南智算中心拜访 -> 盘古二期项目 -> 850元 & 陈总监` 2-hop bridge |

## 3. Per-Case Audit Matrix

| Case ID | Category | Steward | Baseline | Steward State / Tier | Steward Input (Cached) | Baseline Input | Baseline Failure Diagnostics |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `MEM-01` | `MEM` | **PASS** | PASS | `COMPLETED` / `fast_slm` | 1454 (384) | 2383 | — |
| `MEM-02` | `MEM` | **PASS** | FAIL | `COMPLETED` / `fast_slm` | 1232 (640) | 4842 | Stale Thu meeting ranked above rescheduled Fri meeting |
| `MEM-03` | `MEM` | **PASS** | PASS | `COMPLETED` / `fast_slm` | 3659 (2560) | 5039 | — |
| `MEM-04` | `MEM` | **PASS** | FAIL | `COMPLETED` / `fast_slm` | 1293 (768) | 14687 | Memory engine failed to abstain on unrecorded destination; Answer contained forbidden/hallucinated token '850' |
| `MM-01` | `MM` | **PASS** | FAIL | `COMPLETED` / `frontier_llm` | 8851 (7680) | 43282 | Answer missing required token from ['1728', '1,728'] |
| `MM-02` | `MM` | **PASS** | PASS | `COMPLETED` / `frontier_llm` | 5363 (3968) | 5064 | — |
| `MM-03` | `MM` | **PASS** | PASS | `COMPLETED` / `fast_slm` | 4175 (2816) | 7474 | — |
| `MCP-01` | `MCP` | **PASS** | PASS | `COMPLETED` / `fast_slm` | 4593 (3840) | 4952 | — |
| `MCP-02` | `MCP` | **PASS** | PASS | `COMPLETED` / `fast_slm` | 6280 (4864) | 13420 | — |
| `SEC-01` | `SEC` | **PASS** | PASS | `COMPLETED` / `frontier_llm` | 7691 (5376) | 5198 | — |
| `SEC-02` | `SEC` | **PASS** | FAIL | `COMPLETED` / `local_privacy_slm` | 1310 (384) | 6779 | PII leaked to frontier_llm instead of LOCAL_PRIVACY_SLM |
| `SEC-03` | `SEC` | **PASS** | FAIL | `HALTED_APPROVAL` / `fast_slm` | 5132 (4224) | 17832 | Over-limit payout was NOT halted (final_state=COMPLETED) |
