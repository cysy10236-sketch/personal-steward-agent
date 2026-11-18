"""Domain-cohesive travel, receipt ROI, itinerary, and reimbursement MCP tool registry (14 tools)."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
from typing import Any, Callable

from pydantic import BaseModel, Field, ValidationError

from ..perception.artifact_registry import ArtifactRegistry
from ..router.steward_router import ToolSchemaSpec
from .sandbox import ASTExpenseCalculator


class ToolExecutionRecord(BaseModel):
    """Structured audit record for a single tool invocation."""

    tool_name: str
    arguments: dict[str, Any]
    status: str  # "ok" | "error" | "denied" | "halted_approval"
    output_json: str


# --- Pydantic v2 Argument Contracts ---
class _InspectROIArgs(BaseModel):
    artifact_ref: str = Field(description="Artifact handle such as artifact://rcpt_pdf_... or rcpt_img_...")
    roi_id: str | None = Field(default=None, description="Optional ROI ID such as roi_1")
    page_index: int = Field(default=0, ge=0)


class _EvalMathArgs(BaseModel):
    expression: str = Field(description="Arithmetic expression using +,-,*,/,round,sum,min,max,abs,mean")


class _ExportClaimArgs(BaseModel):
    filename: str = Field(description="Relative markdown filename inside workspace, e.g. claim_shenzhen.md")
    markdown_body: str = Field(min_length=5, description="Markdown content of the reimbursement claim")


class _ReadClaimArgs(BaseModel):
    filename: str = Field(description="Relative markdown filename inside workspace")


class _ItineraryQueryArgs(BaseModel):
    date_prefix: str = Field(default="2026-11", description="YYYY-MM or YYYY-MM-DD filter prefix")


class _ItineraryScheduleArgs(BaseModel):
    title: str = Field(min_length=2, description="Calendar event title")
    slot_time: str = Field(min_length=6, description="Date & time string such as 2026-11-20 15:00")
    location: str = Field(default="Online", description="Meeting room or flight/train leg")


class _ExpensePoolQueryArgs(BaseModel):
    project_code: str = Field(default="PRJ-PANGU-2", description="Project budget code or project name")


class _ExpenseVoucherArgs(BaseModel):
    voucher_name: str = Field(description="JSON filename such as voucher_pangu.json")
    project_code: str = Field(description="Associated project code")
    net_reimbursable_cny: float = Field(ge=0.0, description="Verified net reimbursable amount in CNY")
    notes: str = Field(default="", description="Audit breakdown notes")


class _NotesSearchArgs(BaseModel):
    keyword: str = Field(default="", description="Keyword to match in travel checklist notes")


class _NotesUpsertArgs(BaseModel):
    note_key: str = Field(min_length=2, description="Unique note key")
    content: str = Field(min_length=3, description="Updated travel checklist content")


class _PolicyLookupArgs(BaseModel):
    city_tier: str = Field(default="tier_1", description="tier_1 (北上广深) or tier_2")


class _SubmitPayoutArgs(BaseModel):
    project_code: str = Field(description="Target project budget code")
    amount_cny: float = Field(gt=0.0, description="Direct reimbursement payout amount in CNY")
    payee_note: str = Field(default="", description="Reimbursement purpose summary")


class _CurrencyRateArgs(BaseModel):
    currency: str = Field(description="ISO currency code: USD, HKD, JPY, EUR, SGD, CNY")


class _GuardedShellArgs(BaseModel):
    command: str = Field(description="Read-only diagnostic shell command inside workspace")


class StewardToolRegistry:
    """Registers 14 travel/expense tools with Pydantic v2 validation and MCP pair_group metadata."""

    _FX_TO_CNY: dict[str, float] = {
        "CNY": 1.0000,
        "USD": 7.1850,
        "HKD": 0.9230,
        "EUR": 7.7600,
        "SGD": 5.3800,
        "JPY": 0.0475,
    }

    def __init__(
        self,
        *,
        workspace_dir: Path | str,
        artifact_registry: ArtifactRegistry,
    ) -> None:
        self.workspace_dir = Path(workspace_dir).resolve()
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        self.artifact_registry = artifact_registry

        self._itinerary_store: list[dict[str, str]] = [
            {
                "title": "盘古二期客户技术对齐",
                "slot_time": "2026-11-19 10:00",
                "location": "深圳南山科技园 T3",
            },
            {
                "title": "AI基础设施双周同步会",
                "slot_time": "2026-11-20 10:30",
                "location": "线上会议室 B",
            },
        ]
        self._budget_pools: dict[str, dict[str, Any]] = {
            "PRJ-PANGU-2": {
                "project_name": "盘古二期项目",
                "approver": "陈总监 (Chen Director)",
                "remaining_budget_cny": 48500.0,
                "tier1_hotel_cap_cny": 850.0,
            },
            "PRJ-XINGHAI-1": {
                "project_name": "星海一期项目",
                "approver": "林经理 (Lin Manager)",
                "remaining_budget_cny": 22000.0,
                "tier1_hotel_cap_cny": 650.0,
            },
        }
        self._travel_notes: dict[str, str] = {
            "shenzhen_packing": "携带身份证原件、保密协议签署件、Type-C投屏转换器；返程前收集酒店增值税专票与水单。",
            "reimburse_checklist": "住宿报销需同时附发票与每日明细水单（剔除迷你吧、洗衣等个人消费）。",
        }
        self._payout_ledger: list[dict[str, Any]] = []
        self._transient_notes_lock_armed: bool = False

        self._specs: dict[str, ToolSchemaSpec] = {}
        self._handlers: dict[str, tuple[type[BaseModel], Callable[[Any], dict[str, Any]]]] = {}
        self._register_all()

    def arm_transient_notes_lock_once(self) -> None:
        """Arm a deterministic one-shot SQLite lock error on mcp_travel_notes_upsert for retry testing."""
        self._transient_notes_lock_armed = True

    def _safe_workspace_path(self, filename: str) -> Path:
        clean = Path(filename).name
        if not clean or clean in {".", ".."} or clean.startswith(".env"):
            raise ValueError(f"Disallowed workspace filename: {filename!r}")
        resolved = (self.workspace_dir / clean).resolve()
        if self.workspace_dir not in resolved.parents and resolved != self.workspace_dir:
            raise ValueError(f"Path escapes sandbox workspace: {filename!r}")
        return resolved

    def _add(
        self,
        *,
        name: str,
        description: str,
        arg_model: type[BaseModel],
        handler: Callable[[Any], dict[str, Any]],
        namespace: str,
        pair_group: str | None = None,
        always_include: bool = False,
        keywords: list[str] | None = None,
    ) -> None:
        schema = arg_model.model_json_schema()
        schema.pop("title", None)
        self._specs[name] = ToolSchemaSpec(
            name=name,
            description=description,
            parameters=schema,
            namespace=namespace,
            pair_group=pair_group,
            always_include=always_include,
            keywords=keywords or [],
        )
        self._handlers[name] = (arg_model, handler)

    def _register_all(self) -> None:
        self._add(
            name="inspect_receipt_roi",
            description="Crop and inspect a specific ROI table or widget from a PDF hotel folio or image receipt handle.",
            arg_model=_InspectROIArgs,
            handler=self._handle_inspect_roi,
            namespace="perception",
            keywords=["artifact://", "水单", "发票", "截图", "pdf", "receipt", "folio", "金额", "明细"],
        )
        self._add(
            name="eval_expense_math",
            description="Deterministic AST calculator for summing receipt items, deducting personal charges, FX conversion, or computing over-cap deltas.",
            arg_model=_EvalMathArgs,
            handler=self._handle_eval_math,
            namespace="core",
            always_include=True,
            keywords=["计算", "合计", "总额", "差额", "扣除", "剔除", "汇率", "报销", "sum", "math"],
        )
        self._add(
            name="export_claim_markdown",
            description="Write a verified Markdown travel reimbursement claim file into the sandboxed workspace.",
            arg_model=_ExportClaimArgs,
            handler=self._handle_export_claim,
            namespace="workspace",
            pair_group="workspace_claim",
            keywords=["导出", "保存", "写入", "生成", ".md", "报销单", "核验单", "markdown"],
        )
        self._add(
            name="read_claim_markdown",
            description="Read back an existing Markdown reimbursement claim file from the sandboxed workspace.",
            arg_model=_ReadClaimArgs,
            handler=self._handle_read_claim,
            namespace="workspace",
            pair_group="workspace_claim",
            keywords=["读取", "查看文件", ".md", "报销单"],
        )
        self._add(
            name="mcp_itinerary_query",
            description="[MCP Calendar] Query existing travel itinerary legs and scheduled meetings.",
            arg_model=_ItineraryQueryArgs,
            handler=self._handle_itinerary_query,
            namespace="mcp_calendar",
            pair_group="mcp_itinerary",
            keywords=["日程", "日历", "排期", "会议", "冲突", "时间段", "calendar", "itinerary"],
        )
        self._add(
            name="mcp_itinerary_schedule",
            description="[MCP Calendar] Schedule and lock a new travel review or meeting slot after checking conflicts.",
            arg_model=_ItineraryScheduleArgs,
            handler=self._handle_itinerary_schedule,
            namespace="mcp_calendar",
            pair_group="mcp_itinerary",
            keywords=["锁定日程", "写入日程", "创建日程", "预约", "会议", "schedule"],
        )
        self._add(
            name="mcp_expense_pool_query",
            description="[MCP Finance] Query project remaining travel budget pool, approver, and project lodging cap.",
            arg_model=_ExpensePoolQueryArgs,
            handler=self._handle_expense_pool_query,
            namespace="mcp_finance",
            pair_group="mcp_expense",
            keywords=["预算池", "剩余额度", "经费", "项目代码", "盘古", "星海", "budget", "voucher"],
        )
        self._add(
            name="mcp_expense_export_voucher",
            description="[MCP Finance] Export a structured JSON reimbursement voucher bound to a project code.",
            arg_model=_ExpenseVoucherArgs,
            handler=self._handle_expense_voucher,
            namespace="mcp_finance",
            pair_group="mcp_expense",
            keywords=["凭证", "核销单", "voucher", ".json", "财务导出"],
        )
        self._add(
            name="mcp_travel_notes_search",
            description="[MCP Notes] Search personal travel checklists, packing notes, and invoice collection reminders.",
            arg_model=_NotesSearchArgs,
            handler=self._handle_notes_search,
            namespace="mcp_notes",
            pair_group="mcp_notes",
            keywords=["备忘录", "笔记", "清单", "注意事项", "notes", "checklist"],
        )
        self._add(
            name="mcp_travel_notes_upsert",
            description="[MCP Notes] Create or update a personal travel checklist note.",
            arg_model=_NotesUpsertArgs,
            handler=self._handle_notes_upsert,
            namespace="mcp_notes",
            pair_group="mcp_notes",
            keywords=["更新备忘录", "写入备忘录", "差旅笔记", "清单更新", "upsert"],
        )
        self._add(
            name="mcp_policy_lookup",
            description="[MCP Policy] Look up standard corporate hotel, meal, and local transport per-diem caps by city tier.",
            arg_model=_PolicyLookupArgs,
            handler=self._handle_policy_lookup,
            namespace="mcp_policy",
            pair_group="mcp_payout",
            keywords=["差旅制度", "标准差标", "一线城市", "补贴", "policy", "打款"],
        )
        self._add(
            name="mcp_reimburse_submit_payout",
            description="[MCP Policy] Submit a direct reimbursement payout request to the corporate finance queue (runtime sandbox automatically gates approval thresholds).",
            arg_model=_SubmitPayoutArgs,
            handler=self._handle_submit_payout,
            namespace="mcp_policy",
            pair_group="mcp_payout",
            keywords=["直接发起打款", "立即打款", "自动打款", "转账", "payout", "reimburse_submit"],
        )
        self._add(
            name="mcp_per_diem_currency_rate",
            description="[MCP FX] Query official corporate reimbursement foreign exchange rate to CNY.",
            arg_model=_CurrencyRateArgs,
            handler=self._handle_currency_rate,
            namespace="mcp_fx",
            keywords=["汇率", "美元", "港币", "日元", "usd", "hkd", "jpy", "eur", "sgd", "fx"],
        )
        self._add(
            name="run_guarded_shell",
            description="Execute a sandboxed read-only workspace diagnostic command (blocked on .env, network exfil, or tainted receipts).",
            arg_model=_GuardedShellArgs,
            handler=self._handle_guarded_shell,
            namespace="system",
            keywords=["终端", "命令行", "shell", "诊断", "base64"],
        )

    def all_specs(self) -> list[ToolSchemaSpec]:
        return list(self._specs.values())

    def specs_by_names(self, names: list[str]) -> list[ToolSchemaSpec]:
        return [self._specs[n] for n in names if n in self._specs]

    def execute(self, tool_name: str, raw_arguments: dict[str, Any]) -> ToolExecutionRecord:
        if tool_name not in self._handlers:
            err = {"status": "error", "error": f"Unknown tool: {tool_name!r}"}
            return ToolExecutionRecord(
                tool_name=tool_name,
                arguments=raw_arguments,
                status="error",
                output_json=json.dumps(err, ensure_ascii=False),
            )
        if "_malformed_json" in raw_arguments:
            err = {
                "status": "error",
                "error": "Malformed JSON tool arguments; retry with valid JSON matching schema.",
            }
            return ToolExecutionRecord(
                tool_name=tool_name,
                arguments=raw_arguments,
                status="error",
                output_json=json.dumps(err, ensure_ascii=False),
            )

        model_cls, fn = self._handlers[tool_name]
        try:
            validated = model_cls.model_validate(raw_arguments)
            result = fn(validated)
            status = str(result.get("status", "ok"))
            return ToolExecutionRecord(
                tool_name=tool_name,
                arguments=raw_arguments,
                status=status,
                output_json=json.dumps(result, ensure_ascii=False),
            )
        except ValidationError as exc:
            err = {"status": "error", "validation_errors": exc.errors(include_url=False)}
            return ToolExecutionRecord(
                tool_name=tool_name,
                arguments=raw_arguments,
                status="error",
                output_json=json.dumps(err, ensure_ascii=False),
            )
        except Exception as exc:
            err = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
            return ToolExecutionRecord(
                tool_name=tool_name,
                arguments=raw_arguments,
                status="error",
                output_json=json.dumps(err, ensure_ascii=False),
            )

    # --- Tool Handlers ---
    def _handle_inspect_roi(self, args: _InspectROIArgs) -> dict[str, Any]:
        crop = self.artifact_registry.inspect_roi(
            args.artifact_ref, roi_id=args.roi_id, page_index=args.page_index
        )
        return {
            "status": "ok",
            "artifact_id": crop.artifact_id,
            "roi_id": crop.roi_id,
            "label": crop.label,
            "region_type": crop.region_type,
            "crop_size": list(crop.crop_size),
            "crop_path": crop.crop_path,
            "extracted_text": crop.extracted_text,
            "line_items": crop.line_items,
        }

    def _handle_eval_math(self, args: _EvalMathArgs) -> dict[str, Any]:
        val = ASTExpenseCalculator.evaluate(args.expression)
        return {"status": "ok", "expression": args.expression, "result": val}

    def _handle_export_claim(self, args: _ExportClaimArgs) -> dict[str, Any]:
        target = self._safe_workspace_path(args.filename)
        target.write_text(args.markdown_body, encoding="utf-8")
        return {
            "status": "ok",
            "saved_file": target.name,
            "bytes_written": target.stat().st_size,
        }

    def _handle_read_claim(self, args: _ReadClaimArgs) -> dict[str, Any]:
        target = self._safe_workspace_path(args.filename)
        if not target.exists():
            return {"status": "error", "error": f"File not found: {target.name}"}
        return {"status": "ok", "filename": target.name, "content": target.read_text(encoding="utf-8")}

    def _handle_itinerary_query(self, args: _ItineraryQueryArgs) -> dict[str, Any]:
        matched = [
            ev for ev in self._itinerary_store if ev["slot_time"].startswith(args.date_prefix)
        ]
        return {"status": "ok", "events": matched, "count": len(matched)}

    def _handle_itinerary_schedule(self, args: _ItineraryScheduleArgs) -> dict[str, Any]:
        conflicts = [ev for ev in self._itinerary_store if ev["slot_time"] == args.slot_time]
        if conflicts:
            return {
                "status": "error",
                "error": f"Time slot {args.slot_time} conflicts with {conflicts[0]['title']}",
            }
        ev = {"title": args.title, "slot_time": args.slot_time, "location": args.location}
        self._itinerary_store.append(ev)
        return {"status": "ok", "scheduled": ev}

    def _handle_expense_pool_query(self, args: _ExpensePoolQueryArgs) -> dict[str, Any]:
        q = args.project_code.upper()
        for code, info in self._budget_pools.items():
            if code == q or info["project_name"] in args.project_code:
                return {"status": "ok", "project_code": code, **info}
        return {"status": "ok", "pools": self._budget_pools}

    def _handle_expense_voucher(self, args: _ExpenseVoucherArgs) -> dict[str, Any]:
        name = args.voucher_name if args.voucher_name.endswith(".json") else f"{args.voucher_name}.json"
        target = self._safe_workspace_path(name)
        payload = {
            "project_code": args.project_code,
            "net_reimbursable_cny": round(args.net_reimbursable_cny, 2),
            "notes": args.notes,
        }
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return {"status": "ok", "voucher_file": target.name, "payload": payload}

    def _handle_notes_search(self, args: _NotesSearchArgs) -> dict[str, Any]:
        kw = args.keyword.strip().lower()
        hits = {
            k: v
            for k, v in self._travel_notes.items()
            if not kw or kw in k.lower() or kw in v.lower()
        }
        return {"status": "ok", "notes": hits or self._travel_notes}

    def _handle_notes_upsert(self, args: _NotesUpsertArgs) -> dict[str, Any]:
        if self._transient_notes_lock_armed:
            self._transient_notes_lock_armed = False
            return {
                "status": "error",
                "error": "SQLite BusyError: travel_notes store temporarily locked; retry mcp_travel_notes_upsert.",
            }
        self._travel_notes[args.note_key] = args.content
        return {"status": "ok", "note_key": args.note_key, "saved_length": len(args.content)}

    def _handle_policy_lookup(self, args: _PolicyLookupArgs) -> dict[str, Any]:
        tier = args.city_tier.lower()
        if "1" in tier or "一线" in tier:
            return {
                "status": "ok",
                "city_tier": "tier_1",
                "standard_hotel_cap_cny": 700.0,
                "meal_per_diem_cny": 120.0,
                "personal_exclusions": ["迷你吧/Minibar", "私人洗衣", "付费影视"],
            }
        return {
            "status": "ok",
            "city_tier": "tier_2",
            "standard_hotel_cap_cny": 500.0,
            "meal_per_diem_cny": 100.0,
            "personal_exclusions": ["迷你吧/Minibar", "私人洗衣", "付费影视"],
        }

    def _handle_submit_payout(self, args: _SubmitPayoutArgs) -> dict[str, Any]:
        entry = {
            "payout_id": f"PAY-{len(self._payout_ledger) + 1:04d}",
            "project_code": args.project_code,
            "amount_cny": round(args.amount_cny, 2),
            "payee_note": args.payee_note,
        }
        self._payout_ledger.append(entry)
        return {"status": "ok", "submitted_payout": entry}

    def _handle_currency_rate(self, args: _CurrencyRateArgs) -> dict[str, Any]:
        code = args.currency.strip().upper()
        if code not in self._FX_TO_CNY:
            return {"status": "error", "error": f"Unsupported currency code: {code!r}"}
        return {"status": "ok", "currency": code, "rate_to_cny": self._FX_TO_CNY[code]}

    def _handle_guarded_shell(self, args: _GuardedShellArgs) -> dict[str, Any]:
        proc = subprocess.run(
            args.command,
            shell=True,
            cwd=str(self.workspace_dir),
            capture_output=True,
            text=True,
            timeout=5.0,
            check=False,
        )
        return {
            "status": "ok" if proc.returncode == 0 else "error",
            "returncode": proc.returncode,
            "stdout": proc.stdout[:600],
            "stderr": proc.stderr[:300],
        }
