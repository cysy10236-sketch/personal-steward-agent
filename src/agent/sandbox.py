"""AST financial calculator, HMAC-bound sandbox guard, shell allowlist, and SHA-256 auditor."""

from __future__ import annotations

import ast
from enum import Enum
import hashlib
import hmac
import json
import operator
from pathlib import Path
import re
import shlex
from typing import Any

from pydantic import BaseModel


class SandboxVerdict(str, Enum):
    ALLOW = "allow"
    REQUIRE_APPROVAL = "require_approval"
    DENY = "deny"


class SandboxDecision(BaseModel):
    tool_name: str
    verdict: SandboxVerdict
    rule_id: str
    reason: str
    action_fingerprint: str = ""


def compute_action_fingerprint(tool_name: str, arguments: dict[str, Any], secret: str) -> str:
    """Bind human approval strictly to exact (tool_name, canonical_arguments) via HMAC-SHA256."""
    filtered = {k: v for k, v in arguments.items() if k != "idempotency_key"}
    canonical = json.dumps({"tool": tool_name, "args": filtered}, sort_keys=True, ensure_ascii=False)
    return hmac.new(secret.encode("utf-8"), canonical.encode("utf-8"), hashlib.sha256).hexdigest()[:24]


class ASTExpenseCalculator:
    """Evaluates financial arithmetic via restricted Python AST without eval()."""

    _BIN_OPS: dict[type[ast.operator], Any] = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
    }
    _UNARY_OPS: dict[type[ast.unaryop], Any] = {ast.UAdd: operator.pos, ast.USub: operator.neg}

    @classmethod
    def evaluate(cls, expression: str, *, max_nodes: int = 120) -> float:
        cleaned = (expression or "").strip()
        if not cleaned or "__" in cleaned:
            raise ValueError(f"Rejected unsafe or empty expression: {expression!r}")
        tree = ast.parse(cleaned, mode="eval")
        all_nodes = list(ast.walk(tree))
        if len(all_nodes) > max_nodes:
            raise ValueError(f"Expression AST exceeds {max_nodes} nodes ({len(all_nodes)})")
        val = cls._eval_node(tree.body)
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            return round(float(val), 4)
        raise ValueError(f"Expression did not evaluate to a scalar number: {val!r}")

    @classmethod
    def _eval_node(cls, node: ast.AST) -> Any:
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
                return float(node.value)
            raise ValueError(f"Disallowed constant literal: {node.value!r}")
        if isinstance(node, (ast.List, ast.Tuple)):
            return [cls._eval_node(el) for el in node.elts]
        if isinstance(node, ast.UnaryOp) and type(node.op) in cls._UNARY_OPS:
            return cls._UNARY_OPS[type(node.op)](cls._eval_node(node.operand))
        if isinstance(node, ast.BinOp) and type(node.op) in cls._BIN_OPS:
            left, right = cls._eval_node(node.left), cls._eval_node(node.right)
            if isinstance(node.op, ast.Pow) and abs(float(right)) > 100.0:
                raise ValueError(f"Exponent {right} exceeds safety bound 100")
            return cls._BIN_OPS[type(node.op)](left, right)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            fn = node.func.id
            if node.keywords:
                raise ValueError("Keyword arguments disallowed in ASTExpenseCalculator")
            args = [cls._eval_node(a) for a in node.args]
            if fn == "round":
                return round(float(args[0]), int(args[1]) if len(args) > 1 else 2)
            if fn == "abs":
                return abs(float(args[0]))
            if fn in {"sum", "min", "max", "mean"}:
                seq = args[0] if (len(args) == 1 and isinstance(args[0], list)) else args
                nums = [float(x) for x in seq]
                if not nums:
                    raise ValueError(f"{fn}() called on empty sequence")
                return {"sum": sum, "min": min, "max": max, "mean": lambda xs: sum(xs) / len(xs)}[fn](nums)
            raise ValueError(f"Disallowed function in AST calculator: {fn!r}")
        raise ValueError(f"Disallowed AST node type: {type(node).__name__}")


class ExpenseSandboxGuard:
    """Allowlist-first security & HMAC-bound human-in-the-loop reimbursement gate."""

    SAFE_READONLY_CMDS = frozenset({"ls", "pwd", "wc", "head", "sha256sum"})
    _SHELL_META_RE = re.compile(r"[|;&><$`%]")
    _SENSITIVE_PATH_RE = re.compile(r"(\.env\b|\.ssh\b|id_rsa|\.\.[/\\]|/etc/passwd|system32)", re.I)
    _DESTRUCTIVE_OR_ENV_RE = re.compile(
        r"(\brm\b|\bdel\b|\bremove-item\b|\bcurl\b|\bwget\b|\binvoke-webrequest\b|\breg\b|"
        r"\bset\b|\benv\b|\bprintenv\b|\bget-childitem\s+env|\btype\b|\bcat\b|\bmore\b|openai_api_key)",
        re.I,
    )
    _OPAQUE_WRAPPER_RE = re.compile(
        r"(\bpython(?:3)?\s+-c\b|\bpowershell(?:\.exe)?\s+-enc|\bbase64\b|\beval\s*\(|\bexec\s*\(|\bsh\s+-c\b|\bcmd\s+/c\b)",
        re.I,
    )

    def __init__(
        self,
        *,
        auto_reimburse_limit_cny: float = 1000.0,
        approval_hmac_secret: str = "steward-local-approval-hmac-key",
    ) -> None:
        self.auto_reimburse_limit_cny = auto_reimburse_limit_cny
        self.approval_hmac_secret = approval_hmac_secret

    def make_approval_token(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """Issue an HMAC-bound approval token valid ONLY for exact (tool_name, arguments)."""
        return f"APR_{compute_action_fingerprint(tool_name, arguments, self.approval_hmac_secret)}"

    def _token_valid(self, token: str | None, fp: str) -> bool:
        if not token:
            return False
        expected = f"APR_{fp}"
        return hmac.compare_digest(token.strip(), expected) or hmac.compare_digest(token.strip(), fp)

    def check(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        *,
        is_tainted_context: bool = False,
        approval_token: str | None = None,
    ) -> SandboxDecision:
        fp = compute_action_fingerprint(tool_name, arguments, self.approval_hmac_secret)
        serialized = f"{tool_name} {arguments!r}"

        if self._SENSITIVE_PATH_RE.search(serialized):
            return SandboxDecision(
                tool_name=tool_name,
                verdict=SandboxVerdict.DENY,
                rule_id="RULE_SENSITIVE_PATH_EXFIL",
                reason="Blocked access to credentials, .env, or parent path traversal.",
                action_fingerprint=fp,
            )

        if is_tainted_context and tool_name in {"run_guarded_shell", "mcp_reimburse_submit_payout"}:
            return SandboxDecision(
                tool_name=tool_name,
                verdict=SandboxVerdict.DENY,
                rule_id="RULE_TAINTED_RECEIPT_ISOLATION",
                reason=f"Blocked {tool_name} because an external attachment is tainted by prompt injection.",
                action_fingerprint=fp,
            )

        if tool_name == "run_guarded_shell":
            cmd = str(arguments.get("command", "")).strip()
            if not cmd or self._SHELL_META_RE.search(cmd) or self._DESTRUCTIVE_OR_ENV_RE.search(cmd):
                return SandboxDecision(
                    tool_name=tool_name,
                    verdict=SandboxVerdict.DENY,
                    rule_id="RULE_SHELL_DENY_ENV_OR_META",
                    reason=f"Blocked shell metacharacters, env-leak, or file-read command: {cmd!r}",
                    action_fingerprint=fp,
                )
            if self._OPAQUE_WRAPPER_RE.search(cmd):
                if not self._token_valid(approval_token, fp):
                    return SandboxDecision(
                        tool_name=tool_name,
                        verdict=SandboxVerdict.REQUIRE_APPROVAL,
                        rule_id="RULE_OPAQUE_SHELL_WRAPPER",
                        reason="Opaque inline interpreter or base64 command requires HMAC-bound human approval.",
                        action_fingerprint=fp,
                    )
            else:
                try:
                    argv = shlex.split(cmd, posix=True)
                except ValueError:
                    argv = []
                if not argv or Path(argv[0]).name.lower() not in self.SAFE_READONLY_CMDS:
                    return SandboxDecision(
                        tool_name=tool_name,
                        verdict=SandboxVerdict.DENY,
                        rule_id="RULE_SHELL_ALLOWLIST_REJECT",
                        reason=f"Command {argv[:1]} not in read-only allowlist {sorted(self.SAFE_READONLY_CMDS)}.",
                        action_fingerprint=fp,
                    )

        if tool_name == "mcp_reimburse_submit_payout":
            try:
                amount = float(arguments.get("amount_cny", 0.0))
            except (TypeError, ValueError):
                return SandboxDecision(
                    tool_name=tool_name,
                    verdict=SandboxVerdict.DENY,
                    rule_id="RULE_INVALID_PAYOUT_AMOUNT",
                    reason="Non-numeric amount_cny in reimbursement payout.",
                    action_fingerprint=fp,
                )
            if amount <= 0:
                return SandboxDecision(
                    tool_name=tool_name,
                    verdict=SandboxVerdict.DENY,
                    rule_id="RULE_NON_POSITIVE_PAYOUT",
                    reason="Payout amount must be strictly positive.",
                    action_fingerprint=fp,
                )
            if amount > self.auto_reimburse_limit_cny and not self._token_valid(approval_token, fp):
                return SandboxDecision(
                    tool_name=tool_name,
                    verdict=SandboxVerdict.REQUIRE_APPROVAL,
                    rule_id="RULE_OVER_LIMIT_PAYOUT_GATE",
                    reason=(
                        f"Payout {amount:.2f} CNY exceeds limit {self.auto_reimburse_limit_cny:.2f} CNY; "
                        f"HMAC approval required (fingerprint={fp})."
                    ),
                    action_fingerprint=fp,
                )

        return SandboxDecision(
            tool_name=tool_name,
            verdict=SandboxVerdict.ALLOW,
            rule_id="RULE_SAFE_ALLOW",
            reason="Passed sandbox verification.",
            action_fingerprint=fp,
        )


class WorkspaceSnapshotAuditor:
    """Computes SHA-256 snapshots of sandbox_workspace_dir to verify side-effect persistence."""

    def __init__(self, workspace_dir: Path | str) -> None:
        self.workspace_dir = Path(workspace_dir)
        self.workspace_dir.mkdir(parents=True, exist_ok=True)

    def snapshot(self) -> dict[str, str]:
        return {
            p.relative_to(self.workspace_dir).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(self.workspace_dir.rglob("*"))
            if p.is_file()
        }

    @staticmethod
    def changed_files(before: dict[str, str], after: dict[str, str]) -> list[str]:
        return sorted(k for k, v in after.items() if before.get(k) != v)
