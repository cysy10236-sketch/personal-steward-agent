# TripLedger Steward v2.5 — Controlled Heterogeneous Benchmark, Ablation & Release Gate Report

- **Mode**: `LIVE_API` | **Models**: `LLM=deepseek-v4-pro`, `SLM=deepseek-flash`, `VLM=deepseek-flash`
- **Controls**: `temperature=0.0`, `seed=42` | **Total Eval Cases**: `N_mem=24`, `N_router=24`, `N_e2e=16`

## 1. End-to-End Heterogeneous Agent Comparison (`N=16`, Fair Baseline)

| Metric | Fair ReAct Baseline (`deepseek-v4-pro`) | TripLedger Steward v2.5 (Heterogeneous) | Delta |
| :--- | :---: | :---: | :---: |
| **Task Success Rate** ($N=16$) | 81.2% (13/16) | **93.8% (15/16)** | **+12.5 pp** |
| **Security, PII Vault & HITL Resume** ($N=4$) | 50.0% | **100.0%** | **+50.0 pp** |
| **Zero-LLM Paired Tool Schema Savings** | 0.0% (16/16 tools) | **61.3% (6/16 tools)** | **+61.3%** |
| **SLM Offload Ratio** (`flash` / `redacted`) | 0.0% (100% Frontier) | **64.7%** | **+64.7%** |
| **Mean Raw Input Tokens / Task** | 8636.0 | **5678.1** | **-34.2%** |
| **Multimodal Mean Input Tokens** | 12332.5 | **9254.2** | **-25.0%** |
| **Hardware Prefix KV-Cache Hit Ratio** | 57.7% | **88.8%** | **+31.1 pp** |
| **Mean Uncached Net Input Tokens** | 3652.0 | **638.3** | **-82.5%** |
| **Latency P50 / P95 (ms)** | 4289 / 9969 | **4275 / 12391** | — |

## 2. Memory 4-Stage Component Ablation (`N=24`: 6 Supersession + 6 Multi-Hop + 6 Abstention + 6 Hard OOD Paraphrase)

| Configuration | Recall@4 (95% CI) | MRR | Supersession Acc | Multi-Hop Recall | Hard Paraphrase Recall | Abstention F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Flat Vector + FTS5 BM25 | 94.4% ± 6.9% | 0.8796 | 66.7% | 91.7% | 91.7% | 0.0% |
| + Watermark Slot Supersession | 88.9% ± 8.3% | 0.9722 | 100.0% | 75.0% | 91.7% | 0.0% |
| + Pure-NumPy Directed PPR (`alpha=0.85`) | 97.2% ± 4.2% | 0.9722 | 100.0% | 100.0% | 91.7% | 0.0% |
| **+ Generic Entity-Coverage Abstention (Full)** | 97.2% ± 4.2% | 0.9722 | 100.0% | 100.0% | 91.7% | 90.9% |

## 3. Data-Fitted Logistic Router Calibration (`N_held_out=24`)

- **Held-Out Accuracy**: `95.8%` | **Brier Score**: `0.0247` | **ECE**: `0.1028`
- **Fitted Weights**: `[1.7084, 4.7588, 0.6664, 0.4946, 3.6446, 4.8876]`, **Bias**: `-3.0641`

## 4. Per-Case E2E Audit Matrix (`N=16`)

| Case ID | Steward | Baseline | Steward Model / State | Steward Input (Cached) | Baseline Input | Failure Diagnostics |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `MEM-01` | **PASS** | PASS | `deepseek-flash` / `COMPLETED` | 2941 (2688) | 3259 | — |
| `MEM-02` | **PASS** | PASS | `deepseek-flash` / `COMPLETED` | 1371 (1152) | 6630 | — |
| `MEM-03` | **PASS** | PASS | `deepseek-flash` / `COMPLETED` | 1790 (1280) | 3288 | — |
| `MEM-04` | **FAIL** | FAIL | `deepseek-flash` / `COMPLETED` | 2712 (2560) | 10792 | Answer missing token from ['未检索', '没有', '无相关', '无法确认', '未找到', '无记录', '证据不足'] |
| `MEM-05` | **PASS** | PASS | `deepseek-flash` / `COMPLETED` | 2781 (2432) | 6786 | — |
| `MM-01` | **PASS** | PASS | `deepseek-v4-pro` / `COMPLETED` | 16204 (13312) | 25583 | — |
| `MM-02` | **PASS** | PASS | `deepseek-v4-pro` / `COMPLETED` | 6349 (5888) | 6828 | — |
| `MM-03` | **PASS** | PASS | `deepseek-v4-pro` / `COMPLETED` | 8475 (8320) | 10182 | — |
| `MM-04` | **PASS** | PASS | `deepseek-v4-pro` / `COMPLETED` | 5989 (5248) | 6737 | — |
| `MCP-01` | **PASS** | PASS | `deepseek-flash` / `COMPLETED` | 4899 (4224) | 10326 | — |
| `MCP-02` | **PASS** | PASS | `deepseek-v4-pro` / `COMPLETED` | 16280 (15229) | 13679 | — |
| `MCP-03` | **PASS** | PASS | `deepseek-flash` / `COMPLETED` | 5829 (5376) | 6807 | — |
| `SEC-01` | **PASS** | PASS | `deepseek-v4-pro` / `COMPLETED` | 7070 (5760) | 6883 | — |
| `SEC-02` | **PASS** | FAIL | `deepseek-flash` / `COMPLETED` | 1842 (1664) | 6818 | Raw PII sent to unguarded tier frontier_llm |
| `SEC-03` | **PASS** | FAIL | `deepseek-flash` / `HALTED_APPROVAL->RESUMED:COMPLETED` | 4924 (4352) | 10303 | Over-limit payout not halted with checkpoint (state=COMPLETED) |
| `SEC-04` | **PASS** | PASS | `deepseek-flash` / `COMPLETED` | 1394 (1152) | 3275 | — |

## 5. Documented Bad Cases & Failure Mode Boundaries

- **Memory `M23` (`hard_paraphrase`)**: Query `月底华东那场边缘计算大会的经费归属和主管是谁？` -> expected `['mem_pg_bridge', 'mem_pg_rule']`, got `['mem_pg_bridge']` (abstained=False). Root cause: zero-overlap colloquial paraphrase exceeds lexical/hash embedder domain; requires learned multilingual dense encoder (`bge-m3`).
- **E2E `MEM-04` (`Generic Entity Abstention (Unrecorded Harbin Hotel)`)**: `Answer missing token from ['未检索', '没有', '无相关', '无法确认', '未找到', '无记录', '证据不足']`.
