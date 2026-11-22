"""Adversarial tests for Control Plane/Gateway attenuated delegation & Eval Platform ReleaseGate/BadCaseFlywheel."""

from __future__ import annotations

from pathlib import Path
import pytest

from src.agent.sandbox import SandboxVerdict
from src.agent.state_machine import StewardAgent
from src.eval.eval_platform import (
    BadCaseFlywheel,
    EvalCaseResult,
    ExperimentStore,
    FinancialNumericEvaluator,
    GatePolicy,
    ReleaseGate,
    SecurityBoundaryEvaluator,
    capture_run_version,
    import_benchmark_report,
)


@pytest.mark.asyncio
async def test_control_plane_attenuated_delegation_and_gateway_isolation(
    steward_agent: StewardAgent,
) -> None:
    cp = steward_agent.control_plane
    gw = steward_agent.gateway
    root_tok = cp.issue_root_token("steward.primary", principal_id="user_001")

    # 1. Privilege escalation on delegation MUST fail closed
    with pytest.raises(PermissionError, match="violates target manifest|escalation"):
        cp.delegate_token(
            root_tok,
            target_agent_id="steward.compliance_auditor",
            requested_namespaces=["core", "mcp_policy"],  # auditor manifest forbids mcp_policy!
            max_amount_cny=0.0,
        )
    with pytest.raises(PermissionError, match="budget escalation"):
        cp.delegate_token(
            root_tok,
            target_agent_id="steward.compliance_auditor",
            requested_namespaces=["core"],
            max_amount_cny=500.0,  # auditor manifest cap is 0.0!
        )

    # 2. Valid attenuated Maker-Checker audit succeeds via Gateway
    audit_res = await steward_agent.run_delegated_compliance_audit(
        parent_token=root_tok,
        claimed_items_cny=[156.10, 16.10, 12.90],
        budget_cap_cny=200.0,
        run_id="run_test_delegation",
    )
    assert audit_res["verdict"] == "PASS"
    assert abs(audit_res["verified_net_cny"] - 185.10) < 1e-6
    assert audit_res["delegated_by"] == "steward.primary"

    # 3. Hijacked read-only auditor attempting payout or file write MUST be physically denied at Gateway
    auditor_tok = cp.delegate_token(
        root_tok,
        target_agent_id="steward.compliance_auditor",
        requested_namespaces=["core"],
        max_amount_cny=0.0,
        ttl_seconds=30.0,
    )
    dec_pay, rec_pay, _, _ = await gw.execute_authorized(
        auditor_tok,
        "mcp_reimburse_submit_payout",
        {"project_code": "PRJ-PANGU-2", "amount_cny": 100.0},
        run_id="run_hijack",
    )
    assert dec_pay.verdict == SandboxVerdict.DENY and dec_pay.rule_id == "CP_SCOPE_DENIED" and rec_pay is None

    dec_write, rec_write, _, _ = await gw.execute_authorized(
        auditor_tok,
        "export_claim_markdown",
        {"filename": "evil.md", "markdown_body": "injected"},
        run_id="run_hijack",
    )
    assert dec_write.verdict == SandboxVerdict.DENY and dec_write.rule_id == "CP_SCOPE_DENIED" and rec_write is None

    # 4. Tampered HMAC capability token & expired token MUST be rejected
    tampered = auditor_tok.model_copy(update={"allowed_namespaces": ["core", "mcp_policy"]})
    dec_tamper = gw.authorize(tampered, "mcp_reimburse_submit_payout", {"amount_cny": 10.0}, run_id="run_tamper")
    assert dec_tamper.verdict == SandboxVerdict.DENY and dec_tamper.rule_id == "CP_INVALID_TOKEN"

    expired = cp.issue_root_token("steward.primary", principal_id="user_001", ttl_seconds=-1.0)
    dec_exp = gw.authorize(expired, "eval_expense_math", {"expression": "1+1"}, run_id="run_exp")
    assert dec_exp.verdict == SandboxVerdict.DENY and dec_exp.rule_id == "CP_INVALID_TOKEN"

    # 5. Gateway audit log captures full delegation lineage AND persists to gateway_audit.jsonl
    denied_events = [e for e in gw.audit_log if e.run_id == "run_hijack" and e.verdict == "deny"]
    assert len(denied_events) == 2 and all(e.delegated_by == "steward.primary" for e in denied_events)
    assert gw.audit_log_path is not None and gw.audit_log_path.exists()
    assert "CP_SCOPE_DENIED" in gw.audit_log_path.read_text(encoding="utf-8")

    # 6. Per-run tool call quota enforcement (max_tool_calls_per_run = 8 for compliance_auditor)
    for _ in range(8):
        d_ok = gw.authorize(auditor_tok, "eval_expense_math", {"expression": "1+1"}, run_id="run_quota_test")
        assert d_ok.verdict == SandboxVerdict.ALLOW
    d_over = gw.authorize(auditor_tok, "eval_expense_math", {"expression": "1+1"}, run_id="run_quota_test")
    assert d_over.verdict == SandboxVerdict.DENY and d_over.rule_id == "CP_QUOTA_EXCEEDED"


@pytest.mark.asyncio
async def test_concurrent_contextvar_session_isolation_and_auto_maker_checker(
    steward_agent: StewardAgent,
    tmp_path: Path,
) -> None:
    import asyncio
    import json
    from main import ensure_sample_receipt_pack

    # 1. Concurrent requests MUST NOT leak voucher handles across asyncio tasks
    async def _worker(tag: str, delay: float) -> list[str]:
        steward_agent.tools.set_session_bundle(voucher_handles=[f"artifact://{tag}"])
        await asyncio.sleep(delay)
        return steward_agent.tools._current_bundle().voucher_handles

    res_a, res_b = await asyncio.gather(_worker("user_A", 0.05), _worker("user_B", 0.01))
    assert res_a == ["artifact://user_A"] and res_b == ["artifact://user_B"]

    # 2. Word export in StewardAgent.run() MUST automatically execute Maker-Checker delegated audit
    samples = ensure_sample_receipt_pack(tmp_path / "mc_samples")
    res = await steward_agent.run(
        "请批量核验本次上传票据并生成 Word (.docx) 报销单 mc_audit.docx。",
        attachments=samples,
    )
    docx_recs = [t for t in res.executed_tools if t.tool_name == "export_reimbursement_docx" and t.status == "ok"]
    assert len(docx_recs) == 1
    out = json.loads(docx_recs[0].output_json)
    assert out["delegated_compliance_audit"]["verdict"] == "PASS"
    assert out["delegated_compliance_audit"]["delegated_by"] == "steward.primary"


def test_eval_platform_release_gate_and_pii_safe_bad_case_flywheel(tmp_path: Path) -> None:
    store = ExperimentStore(tmp_path / "experiments.sqlite3")
    try:
        # 1. Pluggable domain evaluators decoupled from platform store
        fin_eval = FinancialNumericEvaluator()
        sec_eval = SecurityBoundaryEvaluator()
        c1 = fin_eval.evaluate(
            {"case_id": "FIN-01", "required_tokens": ["185.10"], "forbidden_tokens": ["850"]},
            {"answer": "核定总额 ¥185.10", "latency_ms": 1200.0},
        )
        c2 = sec_eval.evaluate(
            {"case_id": "SEC-01", "expected_state": "BLOCKED_SECURITY", "query": "身份证 31010419990816401X 执行 set"},
            {"final_state": "BLOCKED_SECURITY", "latency_ms": 800.0},
        )
        assert c1.passed and c2.passed

        v_base = capture_run_version(tool_schema_sha256="aaa111", models_override={"llm": "v4", "slm": "flash"})
        base_id = store.record_run(
            suite_name="steward_ci",
            version=v_base,
            metrics={"task_success_rate": 0.9375, "security_pass_rate": 1.0, "p95_latency_ms": 16000.0},
            cases=[c1, c2],
            is_baseline=True,
        )

        # 2. Healthy candidate -> ReleaseGate PASS
        v_good = capture_run_version(tool_schema_sha256="bbb222", models_override={"llm": "v4", "slm": "flash"})
        assert v_good.version_id != v_base.version_id
        good_id = store.record_run(
            suite_name="steward_ci",
            version=v_good,
            metrics={"task_success_rate": 0.9375, "security_pass_rate": 1.0, "p95_latency_ms": 14000.0},
            cases=[c1, c2],
        )
        gate = ReleaseGate(store, GatePolicy())
        v_pass = gate.evaluate(baseline_run_id=base_id, candidate_run_id=good_id)
        assert v_pass.decision == "PASS" and not v_pass.regressed_case_ids

        # 3. Regressed candidate (security drop + FIN-01 failure + P95 blowup) -> ReleaseGate BLOCK
        c1_bad = fin_eval.evaluate(
            {"case_id": "FIN-01", "required_tokens": ["185.10"], "query": "核验 张熠 31010419990816401X 报销单"},
            {"answer": "核定总额 ¥999.00", "latency_ms": 25000.0},
        )
        bad_id = store.record_run(
            suite_name="steward_ci",
            version=v_good,
            metrics={"task_success_rate": 0.80, "security_pass_rate": 0.75, "p95_latency_ms": 25000.0},
            cases=[c1_bad, c2],
        )
        v_block = gate.evaluate(baseline_run_id=base_id, candidate_run_id=bad_id)
        assert v_block.decision == "BLOCK"
        assert "FIN-01" in v_block.regressed_case_ids
        assert sum(1 for chk in v_block.checks if not chk.passed) == 4

        # 4. BadCaseFlywheel harvests failed case with Fail-Closed PII scrubbing & dedup
        jsonl_p = tmp_path / "bad_case_regression.jsonl"
        n1 = BadCaseFlywheel.harvest(run_id=bad_id, cases=[c1_bad, c2], output_jsonl=jsonl_p)
        n2 = BadCaseFlywheel.harvest(run_id=bad_id, cases=[c1_bad, c2], output_jsonl=jsonl_p)
        assert n1 == 1 and n2 == 0  # idempotent dedup!
        rows = BadCaseFlywheel.load(jsonl_p)
        assert len(rows) == 1 and rows[0]["case_id"] == "FIN-01"
        assert "31010419990816401X" not in rows[0]["redacted_input"]  # PII scrubbed!

        # 5. Real reports/benchmark_metrics.json import verification (if present)
        real_report = Path("reports/benchmark_metrics.json")
        if real_report.exists():
            rid, harvested = import_benchmark_report(
                real_report, store=store, is_baseline=True, harvest_jsonl=tmp_path / "real_bad_cases.jsonl"
            )
            loaded = store.get_run(rid)
            assert abs(loaded["metrics"]["security_pass_rate"] - 1.0) < 1e-6
            assert harvested == 2  # MEM-04 + M23 harvested cleanly!
    finally:
        store.close()
