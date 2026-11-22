"""Control Plane / Data Plane separation: AgentManifest, HMAC Attenuated Delegation & ToolExecutionGateway."""

from __future__ import annotations

from collections import deque
import hashlib
import hmac
import json
from pathlib import Path
import time
from typing import Any
import uuid

from pydantic import BaseModel, Field

from ..agent.sandbox import ExpenseSandboxGuard, SandboxDecision, SandboxVerdict
from ..agent.tools import StewardToolRegistry, ToolExecutionRecord
from ..llm.provider import LLMProviderProtocol


class AgentManifest(BaseModel):
    """Declarative Agent Definition (Control Plane metadata — decoupled from execution Runtime)."""

    agent_id: str = Field(min_length=2)
    owner: str = "personal-steward-team"
    version: str = "2.4.0"
    policy_version: str = "policy-v2"
    allowed_namespaces: list[str] = Field(default_factory=list)
    max_payout_cny: float = Field(default=1000.0, ge=0.0)
    max_tool_calls_per_run: int = Field(default=24, ge=1)
    description: str = ""


class DelegatedCapabilityToken(BaseModel):
    """HMAC-SHA256 signed scoped capability credential supporting attenuated delegation."""

    token_id: str
    principal_id: str
    actor_agent_id: str
    delegated_by: str | None = None
    allowed_namespaces: list[str]
    max_amount_cny: float = Field(ge=0.0)
    issued_at: float
    expires_at: float
    signature: str


class GatewayAuditEvent(BaseModel):
    """Immutable Data Plane audit record logged for every tool invocation passing through the Gateway."""

    event_id: str
    timestamp: float
    run_id: str
    principal_id: str
    actor_agent_id: str
    delegated_by: str | None = None
    tool_name: str
    tool_namespace: str
    schema_sha256: str
    verdict: str
    rule_id: str
    reason: str


def default_steward_manifests() -> list[AgentManifest]:
    return [
        AgentManifest(
            agent_id="steward.primary",
            owner="personal-steward-team",
            version="2.4.0",
            policy_version="policy-v2",
            allowed_namespaces=[
                "core",
                "perception",
                "workspace",
                "system",
                "mcp_calendar",
                "mcp_finance",
                "mcp_notes",
                "mcp_policy",
                "mcp_fx",
            ],
            max_payout_cny=1000.0,
            max_tool_calls_per_run=24,
            description="Primary travel & reimbursement steward agent.",
        ),
        AgentManifest(
            agent_id="steward.compliance_auditor",
            owner="personal-steward-security",
            version="2.4.0",
            policy_version="policy-v2-readonly",
            allowed_namespaces=["core", "perception"],
            max_payout_cny=0.0,
            max_tool_calls_per_run=8,
            description="Read-only Maker-Checker compliance auditor sub-agent (zero write/payout privileges).",
        ),
    ]


class AgentControlPlane:
    """Control Plane: manages AgentManifests, governance policies, and HMAC attenuated delegation tokens."""

    def __init__(self, *, signing_secret: str = "steward-control-plane-hmac-v2") -> None:
        self._secret = signing_secret.encode("utf-8")
        self._manifests: dict[str, AgentManifest] = {}
        for m in default_steward_manifests():
            self.register_manifest(m)

    def register_manifest(self, manifest: AgentManifest) -> None:
        self._manifests[manifest.agent_id] = manifest

    def get_manifest(self, agent_id: str) -> AgentManifest:
        if agent_id not in self._manifests:
            raise KeyError(f"Unregistered agent_id in Control Plane: {agent_id!r}")
        return self._manifests[agent_id]

    def _sign_payload(self, fields: dict[str, Any]) -> str:
        raw = json.dumps(fields, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        return hmac.new(self._secret, raw, hashlib.sha256).hexdigest()

    def _build_signed_token(
        self,
        *,
        principal_id: str,
        actor_agent_id: str,
        delegated_by: str | None,
        allowed_namespaces: list[str],
        max_amount_cny: float,
        issued_at: float,
        expires_at: float,
    ) -> DelegatedCapabilityToken:
        tid = f"cap_{uuid.uuid4().hex[:12]}"
        norm_ns = sorted(set(allowed_namespaces))
        fields = {
            "token_id": tid,
            "principal_id": principal_id,
            "actor_agent_id": actor_agent_id,
            "delegated_by": delegated_by,
            "allowed_namespaces": norm_ns,
            "max_amount_cny": round(float(max_amount_cny), 2),
            "issued_at": round(float(issued_at), 3),
            "expires_at": round(float(expires_at), 3),
        }
        return DelegatedCapabilityToken(**fields, signature=self._sign_payload(fields))

    def issue_root_token(
        self,
        agent_id: str,
        *,
        principal_id: str = "user_session",
        ttl_seconds: float = 600.0,
    ) -> DelegatedCapabilityToken:
        manifest = self.get_manifest(agent_id)
        now = time.time()
        return self._build_signed_token(
            principal_id=principal_id,
            actor_agent_id=manifest.agent_id,
            delegated_by=None,
            allowed_namespaces=manifest.allowed_namespaces,
            max_amount_cny=manifest.max_payout_cny,
            issued_at=now,
            expires_at=now + ttl_seconds,
        )

    def delegate_token(
        self,
        parent_token: DelegatedCapabilityToken,
        *,
        target_agent_id: str,
        requested_namespaces: list[str],
        max_amount_cny: float = 0.0,
        ttl_seconds: float = 60.0,
    ) -> DelegatedCapabilityToken:
        """Issue an attenuated child token; strictly rejects any scope, budget, or TTL escalation."""
        ok, err = self.verify_token(parent_token)
        if not ok:
            raise PermissionError(f"Cannot delegate from invalid parent token: {err}")

        target_manifest = self.get_manifest(target_agent_id)
        req_set = set(requested_namespaces)
        parent_set = set(parent_token.allowed_namespaces)
        manifest_set = set(target_manifest.allowed_namespaces)

        if not req_set.issubset(parent_set):
            raise PermissionError(
                f"Delegation privilege escalation denied: requested {sorted(req_set - parent_set)} not in parent scopes."
            )
        if not req_set.issubset(manifest_set):
            raise PermissionError(
                f"Delegation violates target manifest '{target_agent_id}': disallowed {sorted(req_set - manifest_set)}."
            )
        if max_amount_cny > parent_token.max_amount_cny or max_amount_cny > target_manifest.max_payout_cny:
            raise PermissionError(
                f"Delegation budget escalation denied: requested {max_amount_cny} exceeds parent/manifest ceiling."
            )

        now = time.time()
        child_exp = min(now + ttl_seconds, parent_token.expires_at)
        return self._build_signed_token(
            principal_id=parent_token.principal_id,
            actor_agent_id=target_agent_id,
            delegated_by=parent_token.actor_agent_id,
            allowed_namespaces=sorted(req_set),
            max_amount_cny=max_amount_cny,
            issued_at=now,
            expires_at=child_exp,
        )

    def verify_token(self, token: DelegatedCapabilityToken, *, now: float | None = None) -> tuple[bool, str]:
        ts = time.time() if now is None else now
        if ts > token.expires_at:
            return False, "Capability token expired"
        fields = {
            "token_id": token.token_id,
            "principal_id": token.principal_id,
            "actor_agent_id": token.actor_agent_id,
            "delegated_by": token.delegated_by,
            "allowed_namespaces": sorted(set(token.allowed_namespaces)),
            "max_amount_cny": round(float(token.max_amount_cny), 2),
            "issued_at": round(float(token.issued_at), 3),
            "expires_at": round(float(token.expires_at), 3),
        }
        expected = self._sign_payload(fields)
        if not hmac.compare_digest(token.signature, expected):
            return False, "Invalid HMAC capability signature (tampering detected)"
        return True, "ok"


class ToolExecutionGateway:
    """Data Plane Gateway: single policy, capability, quota, sandbox & audit choke point over ToolRegistry."""

    def __init__(
        self,
        *,
        control_plane: AgentControlPlane,
        registry: StewardToolRegistry,
        sandbox: ExpenseSandboxGuard,
        audit_log_path: Path | str | None = None,
    ) -> None:
        self.control_plane = control_plane
        self.registry = registry
        self.sandbox = sandbox
        self.audit_log_path = Path(audit_log_path) if audit_log_path else None
        if self.audit_log_path is not None:
            self.audit_log_path.parent.mkdir(parents=True, exist_ok=True)
        self.audit_log: deque[GatewayAuditEvent] = deque(maxlen=256)
        self._run_call_counts: dict[str, int] = {}

    def _record_audit(
        self,
        *,
        run_id: str,
        token: DelegatedCapabilityToken,
        tool_name: str,
        tool_namespace: str,
        schema_sha256: str,
        decision: SandboxDecision,
    ) -> None:
        ev = GatewayAuditEvent(
            event_id=f"gw_{uuid.uuid4().hex[:10]}",
            timestamp=time.time(),
            run_id=run_id,
            principal_id=token.principal_id,
            actor_agent_id=token.actor_agent_id,
            delegated_by=token.delegated_by,
            tool_name=tool_name,
            tool_namespace=tool_namespace,
            schema_sha256=schema_sha256,
            verdict=decision.verdict.value,
            rule_id=decision.rule_id,
            reason=decision.reason,
        )
        self.audit_log.append(ev)
        if self.audit_log_path is not None:
            with self.audit_log_path.open("a", encoding="utf-8") as fh:
                fh.write(ev.model_dump_json() + "\n")

    def authorize(
        self,
        token: DelegatedCapabilityToken,
        tool_name: str,
        arguments: dict[str, Any],
        *,
        is_tainted_context: bool = False,
        approval_token: str | None = None,
        run_id: str = "default",
    ) -> SandboxDecision:
        """Enforce Control Plane capability token + namespace scope + quota + sandbox rules before execution."""
        spec_map = {s.name: s for s in self.registry.all_specs()}
        spec = spec_map.get(tool_name)
        ns = spec.namespace if spec else "unknown"
        schema_hash = (
            hashlib.sha256(json.dumps(spec.parameters, sort_keys=True).encode("utf-8")).hexdigest()[:12]
            if spec
            else "unknown"
        )

        # 1. Verify HMAC capability token & TTL
        ok, err = self.control_plane.verify_token(token)
        if not ok:
            dec = SandboxDecision(tool_name=tool_name, verdict=SandboxVerdict.DENY, rule_id="CP_INVALID_TOKEN", reason=err)
            self._record_audit(run_id=run_id, token=token, tool_name=tool_name, tool_namespace=ns, schema_sha256=schema_hash, decision=dec)
            return dec

        # 2. Enforce namespace scope (prevents delegated sub-agents from calling write/payout tools)
        if spec is None or ns not in token.allowed_namespaces:
            dec = SandboxDecision(
                tool_name=tool_name,
                verdict=SandboxVerdict.DENY,
                rule_id="CP_SCOPE_DENIED",
                reason=f"Agent '{token.actor_agent_id}' lacks namespace scope '{ns}' for tool '{tool_name}'.",
            )
            self._record_audit(run_id=run_id, token=token, tool_name=tool_name, tool_namespace=ns, schema_sha256=schema_hash, decision=dec)
            return dec

        # 3. Enforce delegated monetary cap strictly (sub-agents cannot bypass via HITL)
        if token.delegated_by is not None and "amount_cny" in arguments:
            try:
                amt = float(arguments["amount_cny"])
            except (TypeError, ValueError):
                amt = float("inf")
            if amt > token.max_amount_cny:
                dec = SandboxDecision(
                    tool_name=tool_name,
                    verdict=SandboxVerdict.DENY,
                    rule_id="CP_DELEGATED_BUDGET_DENIED",
                    reason=f"Delegated agent '{token.actor_agent_id}' amount {amt} exceeds cap {token.max_amount_cny}.",
                )
                self._record_audit(run_id=run_id, token=token, tool_name=tool_name, tool_namespace=ns, schema_sha256=schema_hash, decision=dec)
                return dec

        # 4. Enforce per-run tool call quota from Control Plane manifest
        manifest = self.control_plane.get_manifest(token.actor_agent_id)
        cnt = self._run_call_counts.get(run_id, 0)
        if cnt >= manifest.max_tool_calls_per_run:
            dec = SandboxDecision(
                tool_name=tool_name,
                verdict=SandboxVerdict.DENY,
                rule_id="CP_QUOTA_EXCEEDED",
                reason=f"Run '{run_id}' exceeded max_tool_calls_per_run ({manifest.max_tool_calls_per_run}).",
            )
            self._record_audit(run_id=run_id, token=token, tool_name=tool_name, tool_namespace=ns, schema_sha256=schema_hash, decision=dec)
            return dec

        # 5. Delegate to deterministic ExpenseSandboxGuard (shell allowlist, taint block, HMAC payout HITL)
        dec = self.sandbox.check(
            tool_name,
            arguments,
            is_tainted_context=is_tainted_context,
            approval_token=approval_token,
        )
        if dec.verdict == SandboxVerdict.ALLOW:
            self._run_call_counts[run_id] = cnt + 1
        self._record_audit(run_id=run_id, token=token, tool_name=tool_name, tool_namespace=ns, schema_sha256=schema_hash, decision=dec)
        return dec

    async def dispatch_approved(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        *,
        llm: LLMProviderProtocol | None = None,
    ) -> tuple[ToolExecutionRecord, int, int]:
        """Execute a tool call that has already passed Gateway.authorize()."""
        return await self.registry.execute_async(tool_name, arguments, llm=llm)

    async def execute_authorized(
        self,
        token: DelegatedCapabilityToken,
        tool_name: str,
        arguments: dict[str, Any],
        *,
        is_tainted_context: bool = False,
        approval_token: str | None = None,
        llm: LLMProviderProtocol | None = None,
        run_id: str = "default",
    ) -> tuple[SandboxDecision, ToolExecutionRecord | None, int, int]:
        """Atomic Gateway authorize -> execute pipeline."""
        dec = self.authorize(
            token,
            tool_name,
            arguments,
            is_tainted_context=is_tainted_context,
            approval_token=approval_token,
            run_id=run_id,
        )
        if dec.verdict != SandboxVerdict.ALLOW:
            return dec, None, 0, 0
        rec, v_in, v_out = await self.dispatch_approved(tool_name, arguments, llm=llm)
        return dec, rec, v_in, v_out
