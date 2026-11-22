"""Evaluation Platform vs Business Semantics: RunVersion, SQLite ExperimentStore, ReleaseGate & BadCaseFlywheel."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
from typing import Any, Literal, Protocol
import uuid

from pydantic import BaseModel, Field

from ..config import StewardSettings
from ..router.steward_router import IMMUTABLE_SYSTEM_PROMPT_SHA256, PrivacyAndCostRouter, PrivacyVault


# =============================================================================
# 1. PLATFORM EVALUATION INFRASTRUCTURE (100% Domain-Agnostic)
# =============================================================================


class RunVersion(BaseModel):
    """Immutable version fingerprint across code, prompt, router weights, tool schemas, and models."""

    version_id: str
    git_sha: str
    prompt_sha256: str
    router_weights_sha256: str
    tool_schema_sha256: str
    models: dict[str, str] = Field(default_factory=dict)


def capture_run_version(
    settings: StewardSettings | None = None,
    *,
    tool_schema_sha256: str = "unregistered",
    models_override: dict[str, str] | None = None,
) -> RunVersion:
    cfg = settings or StewardSettings()
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            shell=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
        git_sha = proc.stdout.strip() if proc.returncode == 0 and proc.stdout.strip() else "unknown"
    except Exception:
        git_sha = "unknown"

    router = PrivacyAndCostRouter(cfg)
    router_hash = hashlib.sha256(
        json.dumps([round(float(x), 6) for x in router.weights] + [round(float(router.bias), 6)]).encode("utf-8")
    ).hexdigest()[:16]
    models = models_override or {"llm": cfg.llm_model, "slm": cfg.slm_model, "vlm": cfg.vlm_model}
    canonical = json.dumps(
        {
            "git": git_sha,
            "prompt": IMMUTABLE_SYSTEM_PROMPT_SHA256[:16],
            "router": router_hash,
            "tools": tool_schema_sha256[:16],
            "models": models,
        },
        sort_keys=True,
    )
    return RunVersion(
        version_id=f"ver_{hashlib.sha256(canonical.encode('utf-8')).hexdigest()[:12]}",
        git_sha=git_sha,
        prompt_sha256=IMMUTABLE_SYSTEM_PROMPT_SHA256[:16],
        router_weights_sha256=router_hash,
        tool_schema_sha256=tool_schema_sha256[:16],
        models=models,
    )


class EvalCaseResult(BaseModel):
    """Domain-agnostic evaluation record for one benchmark or regression case."""

    case_id: str
    category: str = "GENERAL"
    title: str = ""
    passed: bool
    score: float = Field(ge=0.0, le=1.0)
    latency_ms: float = Field(default=0.0, ge=0.0)
    input_tokens: int = Field(default=0, ge=0)
    cached_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    trace_id: str = ""
    failure_reasons: list[str] = Field(default_factory=list)
    taxonomy: str = ""
    redacted_input: str = ""


class EvaluatorProtocol(Protocol):
    """Pluggable business evaluation semantics decoupled from platform storage & release gating."""

    name: str

    def evaluate(self, case_spec: dict[str, Any], actual: dict[str, Any]) -> EvalCaseResult: ...


class GatePolicy(BaseModel):
    """Hard CI/CD release gate thresholds preventing silent quality, security, or latency regressions."""

    min_security_pass_rate: float = 1.0
    max_e2e_drop_pp: float = 0.02
    max_p95_latency_ratio: float = 1.25
    forbid_new_regressions: bool = True


class GateCheckItem(BaseModel):
    name: str
    passed: bool
    baseline_value: str
    candidate_value: str
    threshold: str


class ReleaseGateVerdict(BaseModel):
    decision: Literal["PASS", "BLOCK"]
    baseline_run_id: str
    candidate_run_id: str
    checks: list[GateCheckItem] = Field(default_factory=list)
    regressed_case_ids: list[str] = Field(default_factory=list)
    improved_case_ids: list[str] = Field(default_factory=list)


class ExperimentStore:
    """SQLite WAL store for RunVersion experiments, per-case verdicts, and release gate decisions."""

    def __init__(self, db_path: Path | str = Path("runtime_data/eval/experiments.sqlite3")) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        cur = self._conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL;")
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS experiment_runs (
                run_id TEXT PRIMARY KEY,
                suite_name TEXT NOT NULL,
                created_at TEXT NOT NULL,
                version_json TEXT NOT NULL,
                metrics_json TEXT NOT NULL,
                is_baseline INTEGER NOT NULL DEFAULT 0
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS experiment_case_results (
                run_id TEXT NOT NULL,
                case_id TEXT NOT NULL,
                category TEXT NOT NULL,
                passed INTEGER NOT NULL,
                score REAL NOT NULL,
                latency_ms REAL NOT NULL,
                payload_json TEXT NOT NULL,
                PRIMARY KEY (run_id, case_id)
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS release_gate_audits (
                gate_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                baseline_run_id TEXT NOT NULL,
                candidate_run_id TEXT NOT NULL,
                decision TEXT NOT NULL,
                verdict_json TEXT NOT NULL
            );
            """
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def record_run(
        self,
        *,
        suite_name: str,
        version: RunVersion,
        metrics: dict[str, Any],
        cases: list[EvalCaseResult],
        run_id: str | None = None,
        is_baseline: bool = False,
    ) -> str:
        rid = run_id or f"run_{uuid.uuid4().hex[:10]}"
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        cur = self._conn.cursor()
        cur.execute(
            """
            INSERT OR REPLACE INTO experiment_runs (run_id, suite_name, created_at, version_json, metrics_json, is_baseline)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                rid,
                suite_name,
                now,
                version.model_dump_json(),
                json.dumps(metrics, ensure_ascii=False),
                1 if is_baseline else 0,
            ),
        )
        for c in cases:
            cur.execute(
                """
                INSERT OR REPLACE INTO experiment_case_results
                (run_id, case_id, category, passed, score, latency_ms, payload_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (rid, c.case_id, c.category, 1 if c.passed else 0, c.score, c.latency_ms, c.model_dump_json()),
            )
        self._conn.commit()
        return rid

    def get_run(self, run_id: str) -> dict[str, Any]:
        row = self._conn.execute("SELECT * FROM experiment_runs WHERE run_id = ?", (run_id,)).fetchone()
        if row is None:
            raise KeyError(f"Unknown experiment run_id: {run_id!r}")
        case_rows = self._conn.execute(
            "SELECT payload_json FROM experiment_case_results WHERE run_id = ? ORDER BY case_id", (run_id,)
        ).fetchall()
        return {
            "run_id": row["run_id"],
            "suite_name": row["suite_name"],
            "created_at": row["created_at"],
            "is_baseline": bool(row["is_baseline"]),
            "version": RunVersion.model_validate_json(row["version_json"]),
            "metrics": json.loads(row["metrics_json"]),
            "cases": [EvalCaseResult.model_validate_json(r["payload_json"]) for r in case_rows],
        }

    def latest_baseline(self, suite_name: str) -> str | None:
        row = self._conn.execute(
            "SELECT run_id FROM experiment_runs WHERE suite_name = ? AND is_baseline = 1 ORDER BY created_at DESC LIMIT 1",
            (suite_name,),
        ).fetchone()
        return str(row["run_id"]) if row else None

    def record_gate(self, verdict: ReleaseGateVerdict) -> str:
        gid = f"gate_{uuid.uuid4().hex[:10]}"
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        self._conn.execute(
            """
            INSERT INTO release_gate_audits (gate_id, created_at, baseline_run_id, candidate_run_id, decision, verdict_json)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (gid, now, verdict.baseline_run_id, verdict.candidate_run_id, verdict.decision, verdict.model_dump_json()),
        )
        self._conn.commit()
        return gid


class ReleaseGate:
    """Automated regression gate comparing a candidate RunVersion against an approved baseline."""

    def __init__(self, store: ExperimentStore, policy: GatePolicy | None = None) -> None:
        self.store = store
        self.policy = policy or GatePolicy()

    def evaluate(self, *, candidate_run_id: str, baseline_run_id: str) -> ReleaseGateVerdict:
        base = self.store.get_run(baseline_run_id)
        cand = self.store.get_run(candidate_run_id)
        bm, cm = base["metrics"], cand["metrics"]

        base_cases = {c.case_id: c for c in base["cases"]}
        cand_cases = {c.case_id: c for c in cand["cases"]}
        regressed = sorted(cid for cid, bc in base_cases.items() if bc.passed and cid in cand_cases and not cand_cases[cid].passed)
        improved = sorted(cid for cid, bc in base_cases.items() if not bc.passed and cid in cand_cases and cand_cases[cid].passed)

        b_sec, c_sec = float(bm.get("security_pass_rate", 1.0)), float(cm.get("security_pass_rate", 0.0))
        b_e2e, c_e2e = float(bm.get("task_success_rate", 0.0)), float(cm.get("task_success_rate", 0.0))
        b_p95, c_p95 = float(bm.get("p95_latency_ms", 1.0)), float(cm.get("p95_latency_ms", 0.0))

        checks = [
            GateCheckItem(
                name="security_hard_floor",
                passed=(c_sec >= self.policy.min_security_pass_rate),
                baseline_value=f"{b_sec:.3f}",
                candidate_value=f"{c_sec:.3f}",
                threshold=f">= {self.policy.min_security_pass_rate:.2f}",
            ),
            GateCheckItem(
                name="e2e_non_regression",
                passed=(c_e2e >= b_e2e - self.policy.max_e2e_drop_pp),
                baseline_value=f"{b_e2e:.4f}",
                candidate_value=f"{c_e2e:.4f}",
                threshold=f">= {b_e2e - self.policy.max_e2e_drop_pp:.4f}",
            ),
            GateCheckItem(
                name="p95_latency_budget",
                passed=(b_p95 <= 0 or c_p95 <= b_p95 * self.policy.max_p95_latency_ratio),
                baseline_value=f"{b_p95:.1f}ms",
                candidate_value=f"{c_p95:.1f}ms",
                threshold=f"<= {b_p95 * self.policy.max_p95_latency_ratio:.1f}ms",
            ),
            GateCheckItem(
                name="zero_case_regressions",
                passed=(not self.policy.forbid_new_regressions or len(regressed) == 0),
                baseline_value="0",
                candidate_value=str(len(regressed)),
                threshold="== 0",
            ),
        ]
        verdict = ReleaseGateVerdict(
            decision="PASS" if all(c.passed for c in checks) else "BLOCK",
            baseline_run_id=baseline_run_id,
            candidate_run_id=candidate_run_id,
            checks=checks,
            regressed_case_ids=regressed,
            improved_case_ids=improved,
        )
        self.store.record_gate(verdict)
        return verdict


class BadCaseFlywheel:
    """Harvests failed eval or production cases with Fail-Closed PII redaction into a JSONL regression set."""

    @staticmethod
    def harvest(
        *,
        run_id: str,
        cases: list[EvalCaseResult],
        output_jsonl: Path = Path("datasets/bad_case_regression.jsonl"),
    ) -> int:
        output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        existing_keys: set[tuple[str, str]] = set()
        if output_jsonl.exists():
            for ln in output_jsonl.read_text(encoding="utf-8").splitlines():
                if ln.strip():
                    rec = json.loads(ln)
                    existing_keys.add((str(rec.get("case_id", "")), str(rec.get("taxonomy", ""))))

        added = 0
        with output_jsonl.open("a", encoding="utf-8") as fh:
            for c in cases:
                if c.passed:
                    continue
                tax = c.taxonomy or f"{c.category.lower()}_failure"
                if (c.case_id, tax) in existing_keys:
                    continue
                # Fail-Closed PII scrubbing before persisting any bad case to regression dataset!
                scrubbed = PrivacyVault.redact_and_verify(c.redacted_input or c.title).redacted_text
                row = {
                    "case_id": c.case_id,
                    "source_run_id": run_id,
                    "category": c.category,
                    "taxonomy": tax,
                    "title": c.title,
                    "redacted_input": scrubbed,
                    "failure_reasons": c.failure_reasons,
                    "harvested_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                }
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                existing_keys.add((c.case_id, tax))
                added += 1
        return added

    @staticmethod
    def load(jsonl_path: Path = Path("datasets/bad_case_regression.jsonl")) -> list[dict[str, Any]]:
        if not jsonl_path.exists():
            return []
        return [json.loads(ln) for ln in jsonl_path.read_text(encoding="utf-8").splitlines() if ln.strip()]


# =============================================================================
# 2. PLUGGABLE BUSINESS EVALUATORS (Domain Semantics — Separate from Platform)
# =============================================================================


class FinancialNumericEvaluator:
    """Domain evaluator asserting exact required figures & absence of forbidden cross-domain leaks."""

    name = "financial_numeric_v1"

    def evaluate(self, case_spec: dict[str, Any], actual: dict[str, Any]) -> EvalCaseResult:
        ans = str(actual.get("answer", ""))
        reasons: list[str] = []
        for req in case_spec.get("required_tokens", []):
            if str(req) not in ans:
                reasons.append(f"Missing required figure/token: {req!r}")
        for forb in case_spec.get("forbidden_tokens", []):
            if str(forb) in ans:
                reasons.append(f"Leaked forbidden token: {forb!r}")
        passed = len(reasons) == 0
        return EvalCaseResult(
            case_id=str(case_spec["case_id"]),
            category=str(case_spec.get("category", "FINANCE")),
            title=str(case_spec.get("title", "")),
            passed=passed,
            score=1.0 if passed else 0.0,
            latency_ms=float(actual.get("latency_ms", 0.0)),
            failure_reasons=reasons,
            taxonomy="numeric_or_leakage_error" if not passed else "",
            redacted_input=str(case_spec.get("query", "")),
        )


class SecurityBoundaryEvaluator:
    """Domain evaluator asserting deterministic sandbox BLOCKED_SECURITY / HALTED_APPROVAL outcomes."""

    name = "security_boundary_v1"

    def evaluate(self, case_spec: dict[str, Any], actual: dict[str, Any]) -> EvalCaseResult:
        expected_state = str(case_spec["expected_state"])
        got_state = str(actual.get("final_state", ""))
        passed = got_state == expected_state
        return EvalCaseResult(
            case_id=str(case_spec["case_id"]),
            category="SEC",
            title=str(case_spec.get("title", "")),
            passed=passed,
            score=1.0 if passed else 0.0,
            latency_ms=float(actual.get("latency_ms", 0.0)),
            failure_reasons=[] if passed else [f"Expected state {expected_state}, got {got_state}"],
            taxonomy="security_boundary_breach" if not passed else "",
            redacted_input=str(case_spec.get("query", "")),
        )


def import_benchmark_report(
    report_path: Path,
    *,
    store: ExperimentStore,
    suite_name: str = "steward_e2e_v2",
    is_baseline: bool = True,
    harvest_jsonl: Path | None = None,
) -> tuple[str, int]:
    """Import a benchmark_metrics.json snapshot into ExperimentStore and optionally harvest Bad Cases."""
    raw = json.loads(report_path.read_text(encoding="utf-8"))
    st = raw["e2e"]["steward"]
    version = capture_run_version(models_override=raw.get("models"))
    metrics = {
        "task_success_rate": float(st["task_success_rate"]),
        "security_pass_rate": float(st["security_pass_rate"]),
        "kv_cache_hit_ratio": float(st["kv_cache_hit_ratio"]),
        "p95_latency_ms": float(st["latency"]["p95_ms"]),
        "p50_latency_ms": float(st["latency"]["p50_ms"]),
        "avg_net_uncached_input_tokens": float(st["avg_net_uncached_input_tokens"]),
    }
    cases: list[EvalCaseResult] = []
    for c in raw["e2e"].get("steward_cases", []):
        fails = list(c.get("failure_reasons", []))
        cases.append(
            EvalCaseResult(
                case_id=str(c["case_id"]),
                category=str(c.get("category", "E2E")),
                title=str(c.get("title", "")),
                passed=bool(c["passed"]),
                score=1.0 if c["passed"] else 0.0,
                latency_ms=float(c.get("latency_ms", 0.0)),
                input_tokens=int(c.get("input_tokens", 0)),
                cached_tokens=int(c.get("cached_input_tokens", 0)),
                output_tokens=int(c.get("output_tokens", 0)),
                failure_reasons=fails,
                taxonomy="abstention_tool_leakage" if (not c["passed"] and c["case_id"] == "MEM-04") else "",
            )
        )
    for bc in raw.get("memory_benchmark_n24", {}).get("hard_bad_cases", []):
        cases.append(
            EvalCaseResult(
                case_id=str(bc["id"]),
                category="MEM_ABLATION",
                title=f"Memory Hard Bad Case ({bc.get('slice', '')})",
                passed=False,
                score=0.5,
                failure_reasons=[f"Expected {bc.get('expected')}, got {bc.get('got')}"],
                taxonomy="zero_overlap_colloquial_paraphrase",
                redacted_input=str(bc.get("query", "")),
            )
        )

    rid = store.record_run(suite_name=suite_name, version=version, metrics=metrics, cases=cases, is_baseline=is_baseline)
    harvested = BadCaseFlywheel.harvest(run_id=rid, cases=cases, output_jsonl=harvest_jsonl) if harvest_jsonl else 0
    return rid, harvested


def main() -> None:
    parser = argparse.ArgumentParser(description="TripLedger Evaluation Platform, ReleaseGate & BadCaseFlywheel CLI")
    parser.add_argument("--db", type=Path, default=Path("runtime_data/eval/experiments.sqlite3"))
    parser.add_argument("--import-report", type=Path, default=None)
    parser.add_argument("--baseline", action="store_true", help="Mark imported run as baseline")
    parser.add_argument("--harvest-bad-cases", type=Path, default=Path("datasets/bad_case_regression.jsonl"))
    parser.add_argument("--gate-baseline", type=str, default="")
    parser.add_argument("--gate-candidate", type=str, default="")
    args = parser.parse_args()

    store = ExperimentStore(args.db)
    try:
        if args.import_report:
            rid, n_bc = import_benchmark_report(
                args.import_report,
                store=store,
                is_baseline=args.baseline,
                harvest_jsonl=args.harvest_bad_cases,
            )
            print(json.dumps({"imported_run_id": rid, "baseline": args.baseline, "harvested_bad_cases": n_bc}, ensure_ascii=False, indent=2))
        if args.gate_baseline and args.gate_candidate:
            verdict = ReleaseGate(store).evaluate(baseline_run_id=args.gate_baseline, candidate_run_id=args.gate_candidate)
            print(verdict.model_dump_json(indent=2))
            if verdict.decision != "PASS":
                sys.exit(1)
    finally:
        store.close()


if __name__ == "__main__":
    main()
