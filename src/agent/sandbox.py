"""Zero-privilege AST financial calculator, taint/payout sandbox guard, and SHA-256 file auditor."""

from __future__ import annotations

import ast
from enum import Enum
import hashlib
import operator
from pathlib import Path
import re
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
    _UNARY_OPS: dict[type[ast.unaryop], Any] = {
        ast.UAdd: operator.pos,
        ast.USub: operator.neg,
    }

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
            left = cls._eval_node(node.left)
            right = cls._eval_node(node.right)
            if isinstance(node.op, ast.Pow) and abs(float(right)) > 100.0:
                raise ValueError(f"Exponent {right} exceeds safety bound 100")
            return cls._BIN_OPS[type(node.op)](left, right)

        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            fn = node.func.id
            args = [cls._eval_node(a) for a in node.args]
            if node.keywords:
                raise ValueError("Keyword arguments disallowed in ASTExpenseCalculator")
            if fn == "round":
                return round(float(args[0]), int(args[1]) if len(args) > 1 else 2)
            if fn == "abs":
                return abs(float(args[0]))
            if fn in {"sum", "min", "max", "mean"}:
                seq = args[0] if (len(args) == 1 and isinstance(args[0], list)) else args
                nums = [float(x) for x in seq]
                if not nums:
                    raise ValueError(f"{fn}() called on empty sequence")
                if fn == "sum":
                    return sum(nums)
                if fn == "min":
                    return min(nums)
                if fn == "max":
                    return max(nums)
                return sum(nums) / len(nums)
            raise ValueError(f"Disallowed function in AST calculator: {fn!r}")

        raise ValueError(f"Disallowed AST node type: {type(node).__name__}")


class ExpenseSandboxGuard:
    """Deterministic security & human-in-the-loop reimbursement gate."""

    _SENSITIVE_PATH_RE = re.compile(
        r"(\.env\b|\.ssh\b|id_rsa|\.\.[/\\]|/etc/passwd|c:\\windows\\system32)",
        re.I,
    )
    _DESTRUCTIVE_CMD_RE = re.compile(
        r"(\brm\s+-rf\b|\bdel\s+/s\b|\bremove-item\b.*-recurse|\bcurl\s+https?://|\bwget\s+https?://|\binvoke-webrequest\b|\breg\s+delete\b)",
        re.I,
    )
    _OPAQUE_WRAPPER_RE = re.compile(
        r"(\bpython(?:3)?\s+-c\b|\bpowershell(?:\.exe)?\s+-enc|\bbase64\b|\beval\s*\(|\bexec\s*\(|\bsh\s+-c\b|\bcmd\s+/c\b)",
        re.I,
    )

    def __init__(self, *, auto_reimburse_limit_cny: float = 1000.0) -> None:
        self.auto_reimburse_limit_cny = auto_reimburse_limit_cny

    def check(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        *,
        is_tainted_context: bool = False,
        approval_token: str | None = None,
    ) -> SandboxDecision:
        serialized = f"{tool_name} {arguments!r}"

        # 1. Sensitive path / credential exfiltration check across all tools
        if self._SENSITIVE_PATH_RE.search(serialized):
            return SandboxDecision(
                tool_name=tool_name,
                verdict=SandboxVerdict.DENY,
                rule_id="RULE_SENSITIVE_PATH_EXFIL",
                reason="Blocked access to credentials, .env, or parent path traversal.",
            )

        # 2. Tainted external attachment isolation: forbid dangerous side-effect tools
        if is_tainted_context and tool_name in {
            "run_guarded_shell",
            "mcp_reimburse_submit_payout",
        }:
            return SandboxDecision(
                tool_name=tool_name,
                verdict=SandboxVerdict.DENY,
                rule_id="RULE_TAINTED_RECEIPT_ISOLATION",
                reason=f"Blocked {tool_name} because an untrusted receipt attachment is tainted by prompt injection.",
            )

        # 3. Shell command structural inspection
        if tool_name == "run_guarded_shell":
            cmd = str(arguments.get("command", ""))
            if self._DESTRUCTIVE_CMD_RE.search(cmd):
                return SandboxDecision(
                    tool_name=tool_name,
                    verdict=SandboxVerdict.DENY,
                    rule_id="RULE_DESTRUCTIVE_OR_EXFIL_SHELL",
                    reason=f"Blocked destructive or network-exfiltration shell command: {cmd!r}",
                )
            if self._OPAQUE_WRAPPER_RE.search(cmd) and not approval_token:
                return SandboxDecision(
                    tool_name=tool_name,
                    verdict=SandboxVerdict.REQUIRE_APPROVAL,
                    rule_id="RULE_OPAQUE_SHELL_WRAPPER",
                    reason="Opaque inline interpreter or base64 command requires explicit human approval.",
                )

        # 4. Over-budget / high-value direct reimbursement payout gate
        if tool_name == "mcp_reimburse_submit_payout":
            try:
                amount = float(arguments.get("amount_cny", 0.0))
            except (TypeError, ValueError):
                return SandboxDecision(
                    tool_name=tool_name,
                    verdict=SandboxVerdict.DENY,
                    rule_id="RULE_INVALID_PAYOUT_AMOUNT",
                    reason="Invalid non-numeric amount_cny in reimbursement payout call.",
                )
            if amount <= 0:
                return SandboxDecision(
                    tool_name=tool_name,
                    verdict=SandboxVerdict.DENY,
                    rule_id="RULE_NON_POSITIVE_PAYOUT",
                    reason="Payout amount must be strictly positive.",
                )
            if amount > self.auto_reimburse_limit_cny and not approval_token:
                return SandboxDecision(
                    tool_name=tool_name,
                    verdict=SandboxVerdict.REQUIRE_APPROVAL,
                    rule_id="RULE_OVER_LIMIT_PAYOUT_GATE",
                    reason=(
                        f"Payout {amount:.2f} CNY exceeds auto-approval limit "
                        f"{self.auto_reimburse_limit_cny:.2f} CNY; human confirmation required."
                    ),
                )

        return SandboxDecision(
            tool_name=tool_name,
            verdict=SandboxVerdict.ALLOW,
            rule_id="RULE_SAFE_ALLOW",
            reason="Passed sandbox verification.",
        )


class WorkspaceSnapshotAuditor:
    """Computes SHA-256 snapshots of sandbox_workspace_dir to verify side-effect persistence."""

    def __init__(self, workspace_dir: Path | str) -> None:
        self.workspace_dir = Path(workspace_dir)
        self.workspace_dir.mkdir(parents=True, exist_ok=True)

    def snapshot(self) -> dict[str, str]:
        snap: dict[str, str] = {}
        for p in sorted(self.workspace_dir.rglob("*")):
            if p.is_file():
                rel = p.relative_to(self.workspace_dir).as_posix()
                snap[rel] = hashlib.sha256(p.read_bytes()).hexdigest()
        return snap

    @staticmethod
    def changed_files(before: dict[str, str], after: dict[str, str]) -> list[str]:
        changed: list[str] = []
        for path_k, sha_v in after.items():
            if before.get(path_k) != sha_v:
                changed.append(path_k)
        return sorted(changed)
