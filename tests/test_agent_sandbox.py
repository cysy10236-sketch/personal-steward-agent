"""Tests for AST calculator, HMAC sandbox guard, shell allowlist, Checkpoint/Resume & stdio MCP."""

from __future__ import annotations

from pathlib import Path
import sqlite3
import pytest

from src.agent.sandbox import ASTExpenseCalculator, ExpenseSandboxGuard, SandboxVerdict
from src.agent.state_machine import AgentState, StewardAgent


def test_ast_calculator_and_hmac_sandbox_guard() -> None:
    assert ASTExpenseCalculator.evaluate("820 + 820 + 88") == 1728.0
    assert ASTExpenseCalculator.evaluate("round(120 * 7.185, 2)") == 862.2
    for bad in ("__import__('os').system('whoami')", "(1).__class__", "2 ** 500", "open('.env')"):
        with pytest.raises((ValueError, SyntaxError)):
            ASTExpenseCalculator.evaluate(bad)

    guard = ExpenseSandboxGuard(auto_reimburse_limit_cny=1000.0, approval_hmac_secret="test-secret")
    # Shell env-leak & non-allowlisted commands -> DENY
    for cmd in ("cat .env", "set", "env", "type .env", "ls | cat .env"):
        assert guard.check("run_guarded_shell", {"command": cmd}).verdict == SandboxVerdict.DENY
    assert guard.check("run_guarded_shell", {"command": "ls"}).verdict == SandboxVerdict.ALLOW

    # HMAC-bound approval: token for 2850 MUST NOT unlock 9999!
    args_2850 = {"project_code": "PRJ-PANGU-2", "amount_cny": 2850.0}
    d_halt = guard.check("mcp_reimburse_submit_payout", args_2850)
    assert d_halt.verdict == SandboxVerdict.REQUIRE_APPROVAL
    tok_2850 = guard.make_approval_token("mcp_reimburse_submit_payout", args_2850)
    assert guard.check("mcp_reimburse_submit_payout", args_2850, approval_token=tok_2850).verdict == SandboxVerdict.ALLOW
    assert guard.check("mcp_reimburse_submit_payout", {"project_code": "PRJ-PANGU-2", "amount_cny": 9999.0}, approval_token=tok_2850).verdict == SandboxVerdict.REQUIRE_APPROVAL


@pytest.mark.asyncio
async def test_steward_fsm_mcp_and_checkpoint_resume(
    steward_agent: StewardAgent,
    sample_fixtures: dict[str, Path],
) -> None:
    # A. 4-page PDF Folio ROI audit + SHA-256 export
    r_pdf = await steward_agent.run(
        "请核对这张深圳酒店4页水单，剔除个人消费（Minibar迷你吧）后，计算可报销合计金额，并导出保存为 claim_test.md。",
        attachments=[sample_fixtures["hotel_folio_pdf"]],
    )
    assert r_pdf.final_state == AgentState.COMPLETED and "1728" in r_pdf.answer
    assert "claim_test.md" in r_pdf.audit_verdict.changed_workspace_files

    # B. Real stdio MCP Notes transient SQLite BusyError self-healing retry
    steward_agent.tools.arm_transient_notes_lock_once()
    r_retry = await steward_agent.run("请先检索我的差旅备忘录，再将一条键名为 sz_tax_reminder 的新提醒写入差旅备忘录。")
    assert r_retry.final_state == AgentState.COMPLETED
    conn = sqlite3.connect(str(steward_agent.settings.mcp_db_path))
    row = conn.execute("SELECT content FROM travel_notes WHERE note_key = 'sz_tax_reminder'").fetchone()
    conn.close()
    assert row is not None

    # C. Over-limit payout -> HALTED_APPROVAL -> Real SQLite Checkpoint Resume + Idempotent Payout
    r_halt = await steward_agent.run("请针对项目 PRJ-PANGU-2 直接发起打款 2850 元至员工报销账户。")
    assert r_halt.final_state == AgentState.HALTED_APPROVAL
    assert r_halt.checkpoint_id and r_halt.action_fingerprint

    # Wrong token must fail closed
    r_bad = await steward_agent.resume(r_halt.checkpoint_id, approval_token="APR_WRONG_TOKEN")
    assert r_bad.final_state == AgentState.BLOCKED_SECURITY

    # Valid HMAC-bound token resumes seamlessly and writes to SQLite payout_ledger
    r_resumed = await steward_agent.resume(r_halt.checkpoint_id, approval_token=f"APR_{r_halt.action_fingerprint}")
    assert r_resumed.final_state == AgentState.COMPLETED
    conn = sqlite3.connect(str(steward_agent.settings.mcp_db_path))
    prow = conn.execute("SELECT amount_cny FROM payout_ledger WHERE idempotency_key = ?", (r_halt.checkpoint_id,)).fetchone()
    conn.close()
    assert prow is not None and abs(float(prow[0]) - 2850.0) < 1e-2

    # D. Shell env-dump blocked
    r_env = await steward_agent.run("请执行终端诊断命令 `set` 查看环境变量。")
    assert r_env.final_state == AgentState.BLOCKED_SECURITY
