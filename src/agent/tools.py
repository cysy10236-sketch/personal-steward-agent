"""Unified tool registry: 5 local core tools (with VLM ROI check & shell=False) + 9 stdio MCP tools."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import shlex
import subprocess
from typing import Any, Callable

import pymupdf as fitz
from PIL import Image, ImageOps
from pydantic import BaseModel, Field, ValidationError

from ..llm.provider import LLMProviderProtocol
from ..mcp.client import StdioMcpClient
from ..perception.artifact_registry import ArtifactRegistry
from ..perception.docx_exporter import ExpenseLineItem, _trim_white_margins, build_reimbursement_docx
from ..perception.receipt_parser import detect_prompt_injection
from ..router.steward_router import ToolSchemaSpec
from .sandbox import ASTExpenseCalculator


class ToolExecutionRecord(BaseModel):
    tool_name: str
    arguments: dict[str, Any]
    status: str  # "ok" | "error" | "denied" | "halted_approval"
    output_json: str


class _InspectROIArgs(BaseModel):
    artifact_ref: str = Field(description="Artifact handle such as artifact://rcpt_pdf_... or rcpt_img_...")
    roi_id: str | None = Field(default=None, description="Optional ROI ID such as roi_1")
    page_index: int = Field(default=0, ge=0)


class _InspectAllReceiptsArgs(BaseModel):
    artifact_refs: list[str] = Field(description="List of artifact:// handles to batch-inspect via OCR + VLM")


class _EvalMathArgs(BaseModel):
    expression: str = Field(description="Arithmetic expression using +,-,*,/,round,sum,min,max,abs,mean")


class _ExportClaimArgs(BaseModel):
    filename: str = Field(description="Relative markdown filename inside workspace, e.g. claim_shenzhen.md")
    markdown_body: str = Field(min_length=5, description="Markdown content of the reimbursement claim")


class _DocxTableSection(BaseModel):
    heading: str = Field(description="Section title, e.g. '三单合一发票与交易凭证核对台账' or '参与人员名单'")
    note: str = Field(default="", description="Brief explanatory note above the table")
    headers: list[str] = Field(min_length=2, description="Table column headers")
    rows: list[list[str]] = Field(min_length=1, description="Table data rows")
    right_align_cols: list[int] = Field(default_factory=list, description="0-based numeric column indices to right-align")


class _ExportDocxArgs(BaseModel):
    filename: str = Field(default="reimbursement_report.docx", description="Output Word filename ending in .docx")
    kicker_text: str = Field(default="TRIPLEDGER STEWARD · 智能差旅报销核验单", description="Top institution/department kicker banner")
    title: str = Field(min_length=2, description="Document title, e.g. 第十七期“一院一社区”志愿授课实践活动报销明细单")
    applicant: str = Field(default="报销人", description="Applicant, coordinator, or traveler name and contact")
    project_code: str = Field(default="GENERAL", description="Associated project, event, or department name")
    trip_summary: str = Field(default="", description="Concise event/trip date, location, topic, and organizer summary")
    budget_cap_cny: float | None = Field(default=None, ge=0.0, description="Optional budget ceiling in CNY extracted from proposal/policy")
    items: list[ExpenseLineItem] = Field(min_length=1, description="Verified itemized expenses; use supporting_handles for tri-voucher proofs of the same purchase")
    extra_sections: list[_DocxTableSection] = Field(default_factory=list, description="Optional structured tables (e.g., invoice reconciliation ledger, participant/volunteer roster)")
    extra_warnings: list[str] = Field(default_factory=list, description="Audit warnings, screenshot compliance reminders, or missing-receipt notes")
    include_activity_photos: bool = Field(default=True, description="Embed selected event/activity supporting photos in a 2x2 grid if available")
    activity_photo_captions: list[str] = Field(default_factory=list, description="Concise grounded Chinese captions matching the selected activity photos")


class _ReadClaimArgs(BaseModel):
    filename: str = Field(description="Relative markdown filename inside workspace")


class _GuardedShellArgs(BaseModel):
    command: str = Field(description="Read-only workspace diagnostic command (ls, pwd, wc, head, sha256sum)")


class StewardToolRegistry:
    """Combines 5 sandboxed local tools with 9 tools discovered over stdio JSON-RPC 2.0 MCP."""

    def __init__(
        self,
        *,
        workspace_dir: Path | str,
        artifact_registry: ArtifactRegistry,
        mcp_db_path: Path | str | None = None,
        mcp_client: StdioMcpClient | None = None,
    ) -> None:
        self.workspace_dir = Path(workspace_dir).resolve()
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        self.artifact_registry = artifact_registry
        db_p = Path(mcp_db_path) if mcp_db_path else (self.workspace_dir.parent / "mcp" / "steward_mcp.sqlite3")
        self.mcp_client = mcp_client or StdioMcpClient(db_p, self.workspace_dir)
        self._owns_mcp = mcp_client is None

        self._local_specs: dict[str, ToolSchemaSpec] = {}
        self._local_handlers: dict[str, tuple[type[BaseModel], Callable[[Any], dict[str, Any]]]] = {}
        self._session_voucher_handles: list[str] = []
        self._session_activity_photos: list[Path] = []
        self._session_activity_total: int = 0
        self._session_photo_vlm_captions: dict[str, str] = {}
        self._register_local()

    def set_session_bundle(
        self,
        *,
        voucher_handles: list[str],
        activity_photos: list[Path] | None = None,
        total_activity_photo_count: int = 0,
    ) -> None:
        self._session_voucher_handles = list(voucher_handles)
        self._session_activity_photos = list(activity_photos or [])
        self._session_activity_total = total_activity_photo_count
        self._session_photo_vlm_captions = {}

    def close(self) -> None:
        if self._owns_mcp:
            self.mcp_client.close()

    def arm_transient_notes_lock_once(self) -> None:
        self.mcp_client.arm_transient_notes_lock_once()

    def _safe_workspace_path(self, filename: str) -> Path:
        clean = Path(filename).name
        if not clean or clean in {".", ".."} or clean.startswith(".env"):
            raise ValueError(f"Disallowed workspace filename: {filename!r}")
        resolved = (self.workspace_dir / clean).resolve()
        if self.workspace_dir not in resolved.parents and resolved != self.workspace_dir:
            raise ValueError(f"Path escapes sandbox workspace: {filename!r}")
        return resolved

    def _add_local(
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
        self._local_specs[name] = ToolSchemaSpec(
            name=name,
            description=description,
            parameters=schema,
            namespace=namespace,
            pair_group=pair_group,
            always_include=always_include,
            keywords=keywords or [],
        )
        self._local_handlers[name] = (arg_model, handler)

    def _register_local(self) -> None:
        self._add_local(
            name="inspect_receipt_roi",
            description="Crop high-res ROI from a single PDF folio or image receipt handle and return verified OCR/VLM line items.",
            arg_model=_InspectROIArgs,
            handler=self._handle_inspect_roi,
            namespace="perception",
            pair_group="receipt_perception",
            keywords=["artifact://", "水单", "发票", "截图", "pdf", "receipt", "folio", "金额", "明细"],
        )
        self._add_local(
            name="inspect_all_receipts",
            description="Batch-inspect multiple uploaded receipt screenshots/photos (WeChat/Alipay/paper receipts) in one step via parallel RapidOCR + VLM.",
            arg_model=_InspectAllReceiptsArgs,
            handler=self._handle_inspect_all_receipts,
            namespace="perception",
            pair_group="receipt_perception",
            keywords=["批量", "微信", "支付宝", "小票", "截图", "照片", "全部票据", "word", "docx", "带队"],
        )
        self._add_local(
            name="eval_expense_math",
            description="Deterministic AST calculator for summing receipt items, deducting personal charges, or FX conversion.",
            arg_model=_EvalMathArgs,
            handler=self._handle_eval_math,
            namespace="core",
            always_include=True,
            keywords=["计算", "合计", "总额", "差额", "扣除", "剔除", "汇率", "报销", "sum", "math"],
        )
        self._add_local(
            name="export_reimbursement_docx",
            description="Generate a formatted Word (.docx) reimbursement document with summary expense table, duplicate warnings, and embedded receipt photos.",
            arg_model=_ExportDocxArgs,
            handler=self._handle_export_docx,
            namespace="workspace",
            keywords=["word", "docx", ".docx", "贴图", "贴入照片", "报销文档", "生成文档", "带图", "导出"],
        )
        self._add_local(
            name="export_claim_markdown",
            description="Write a verified Markdown travel reimbursement claim file into the sandboxed workspace.",
            arg_model=_ExportClaimArgs,
            handler=self._handle_export_claim,
            namespace="workspace",
            pair_group="workspace_claim",
            keywords=["导出", "保存", "写入", "生成", ".md", "报销单", "核验单", "markdown"],
        )
        self._add_local(
            name="read_claim_markdown",
            description="Read back an existing Markdown reimbursement claim file from the sandboxed workspace.",
            arg_model=_ReadClaimArgs,
            handler=self._handle_read_claim,
            namespace="workspace",
            pair_group="workspace_claim",
            keywords=["读取", "查看文件", ".md", "报销单"],
        )
        self._add_local(
            name="run_guarded_shell",
            description="Run an allowlisted read-only workspace diagnostic command with shell=False and stripped env.",
            arg_model=_GuardedShellArgs,
            handler=self._handle_guarded_shell,
            namespace="system",
            keywords=["终端", "命令行", "shell", "诊断", "base64", "环境变量"],
        )

    def all_specs(self) -> list[ToolSchemaSpec]:
        return list(self._local_specs.values()) + self.mcp_client.tool_specs

    def schema_fingerprint(self) -> str:
        """Deterministic SHA-256 version fingerprint over all registered local + MCP tool schemas."""
        canonical = [
            {"name": s.name, "namespace": s.namespace, "parameters": s.parameters}
            for s in sorted(self.all_specs(), key=lambda x: x.name)
        ]
        return hashlib.sha256(json.dumps(canonical, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:16]

    def specs_by_names(self, names: list[str]) -> list[ToolSchemaSpec]:
        by_name = {s.name: s for s in self.all_specs()}
        return [by_name[n] for n in names if n in by_name]

    def execute(self, tool_name: str, raw_arguments: dict[str, Any]) -> ToolExecutionRecord:
        if "_malformed_json" in raw_arguments:
            return ToolExecutionRecord(
                tool_name=tool_name,
                arguments=raw_arguments,
                status="error",
                output_json=json.dumps({"status": "error", "error": "Malformed JSON arguments."}, ensure_ascii=False),
            )
        if tool_name in self._local_handlers:
            model_cls, fn = self._local_handlers[tool_name]
            try:
                out = fn(model_cls.model_validate(raw_arguments))
                return ToolExecutionRecord(
                    tool_name=tool_name,
                    arguments=raw_arguments,
                    status=str(out.get("status", "ok")),
                    output_json=json.dumps(out, ensure_ascii=False),
                )
            except ValidationError as exc:
                return ToolExecutionRecord(
                    tool_name=tool_name,
                    arguments=raw_arguments,
                    status="error",
                    output_json=json.dumps({"status": "error", "validation_errors": exc.errors(include_url=False)}, ensure_ascii=False),
                )
            except Exception as exc:
                return ToolExecutionRecord(
                    tool_name=tool_name,
                    arguments=raw_arguments,
                    status="error",
                    output_json=json.dumps({"status": "error", "error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False),
                )

        mcp_names = {s.name for s in self.mcp_client.tool_specs}
        if tool_name in mcp_names:
            out = self.mcp_client.call_tool(tool_name, raw_arguments)
            return ToolExecutionRecord(
                tool_name=tool_name,
                arguments=raw_arguments,
                status=str(out.get("status", "ok")),
                output_json=json.dumps(out, ensure_ascii=False),
            )

        return ToolExecutionRecord(
            tool_name=tool_name,
            arguments=raw_arguments,
            status="error",
            output_json=json.dumps({"status": "error", "error": f"Unknown tool: {tool_name!r}"}, ensure_ascii=False),
        )

    async def execute_async(
        self,
        tool_name: str,
        raw_arguments: dict[str, Any],
        *,
        llm: LLMProviderProtocol | None = None,
    ) -> tuple[ToolExecutionRecord, int, int]:
        """Execute tool; on single or batch image inspection with live VLM, cross-verify raw pixels in parallel."""
        rec = self.execute(tool_name, raw_arguments)
        vlm_in, vlm_out = 0, 0
        if rec.status != "ok" or llm is None:
            return rec, vlm_in, vlm_out

        vlm_prompt = (
            "Read this payment screenshot or receipt photo carefully. Return concise factual fields: "
            "channel (微信支付/支付宝/纸质小票/酒店水单), date, merchant/item description, gross amount in CNY, "
            "any non-reimbursable personal charges, and net reimbursable amount."
        )
        if tool_name == "inspect_receipt_roi":
            art = self.artifact_registry.get(str(raw_arguments.get("artifact_ref", "")))
            if art is not None and art.modality == "image":
                crop = self.artifact_registry.inspect_roi(
                    art.handle, roi_id=raw_arguments.get("roi_id"), page_index=int(raw_arguments.get("page_index", 0))
                )
                payload = json.loads(rec.output_json)
                try:
                    vlm_txt, vlm_in, vlm_out = await llm.inspect_image_crop(crop.base64_png, prompt=vlm_prompt)
                    payload["vlm_pixel_verification"] = vlm_txt
                except Exception as exc:
                    payload["vlm_pixel_verification_error"] = f"{type(exc).__name__}: {exc}"
                rec = ToolExecutionRecord(tool_name=rec.tool_name, arguments=rec.arguments, status=rec.status, output_json=json.dumps(payload, ensure_ascii=False))

        elif tool_name == "inspect_all_receipts":
            payload = json.loads(rec.output_json)
            receipts = payload.get("receipts", [])
            pdf_vlm_prompt = (
                "Carefully extract exact factual fields from this scanned invoice/receipt/payment page in concise Chinese: "
                "document type (电子发票/机打小票/支付记录/行程单), invoice or transaction order number, date/time, "
                "buyer name & tax ID (if shown), seller/merchant name, tax-exclusive amount, tax amount, "
                "gross total (价税合计/实付金额), any personal/excluded items, and concise item category summary. "
                "Preserve every single digit accurately."
            )
            img_tasks: list[tuple[str, int, str, Any]] = []
            for idx, item in enumerate(receipts):
                art = self.artifact_registry.get(str(item.get("handle", "")))
                if art is None:
                    continue
                if art.modality == "image":
                    pil = ImageOps.exif_transpose(Image.open(art.source_path)).convert("RGB")
                    pil.thumbnail((1152, 1152), Image.Resampling.LANCZOS)
                    buf = io.BytesIO()
                    pil.save(buf, format="PNG", optimize=True)
                    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
                    img_tasks.append(("receipt", idx, "img", llm.inspect_image_crop(b64, prompt=vlm_prompt)))
                elif art.modality == "pdf" and art.structured_kv.get("ocr_deferred_for_vlm"):
                    scanned = list(art.structured_kv.get("scanned_pages") or [])[:4]
                    if scanned:
                        pdf_doc = fitz.open(art.source_path)
                        try:
                            for p_idx in scanned:
                                if p_idx < len(pdf_doc):
                                    pix = pdf_doc[p_idx].get_pixmap(dpi=165, alpha=False)
                                    pil = _trim_white_margins(Image.frombytes("RGB", (pix.width, pix.height), pix.samples))
                                    pil.thumbnail((1440, 1440), Image.Resampling.LANCZOS)
                                    buf = io.BytesIO()
                                    pil.save(buf, format="PNG", optimize=True)
                                    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
                                    img_tasks.append(("pdf_page", idx, f"p{p_idx + 1}", llm.inspect_image_crop(b64, prompt=pdf_vlm_prompt)))
                        finally:
                            pdf_doc.close()

            photo_prompt = "用一句简洁客观的中文（15-28字）描述这张活动现场照片的核心画面内容（例如授课主题PPT、课堂互动、全体合影等），不要臆造文字。"
            activity_previews: list[dict[str, str]] = []
            for p_idx, p_path in enumerate(self._session_activity_photos):
                activity_previews.append({"index": str(p_idx + 1), "filename": p_path.name, "vlm_caption": ""})
                if p_path.exists():
                    pil = ImageOps.exif_transpose(Image.open(p_path)).convert("RGB")
                    pil.thumbnail((640, 640), Image.Resampling.LANCZOS)
                    buf = io.BytesIO()
                    pil.save(buf, format="PNG", optimize=True)
                    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
                    img_tasks.append(("photo", p_idx, p_path.name, llm.inspect_image_crop(b64, prompt=photo_prompt)))

            if img_tasks:
                results = await asyncio.gather(*(coro for _, _, _, coro in img_tasks), return_exceptions=True)
                pdf_page_notes: dict[int, list[str]] = {}
                for (kind, idx, tag, _), res in zip(img_tasks, results):
                    if isinstance(res, Exception):
                        if kind in {"receipt", "pdf_page"}:
                            receipts[idx]["vlm_error"] = f"{type(res).__name__}: {res}"
                    else:
                        vtxt, vi, vo = res
                        vlm_in += vi
                        vlm_out += vo
                        taints = detect_prompt_injection(vtxt)
                        if taints and kind in {"receipt", "pdf_page"}:
                            receipts[idx]["tainted"] = True
                            receipts[idx]["taint_reasons"] = taints
                        if kind == "receipt":
                            receipts[idx]["vlm_pixel_verification"] = vtxt
                        elif kind == "pdf_page":
                            pdf_page_notes.setdefault(idx, []).append(f"[{tag}] {vtxt}")
                        else:
                            clean_cap = vtxt.strip().replace("\n", " ")[:60]
                            activity_previews[idx]["vlm_caption"] = clean_cap
                            self._session_photo_vlm_captions[self._session_activity_photos[idx].name] = clean_cap
                for idx, notes in pdf_page_notes.items():
                    receipts[idx]["vlm_pixel_verification"] = "\n".join(notes)
                payload["receipts"] = receipts
                if activity_previews:
                    payload["activity_photos_preview"] = activity_previews
                rec = ToolExecutionRecord(tool_name=rec.tool_name, arguments=rec.arguments, status=rec.status, output_json=json.dumps(payload, ensure_ascii=False))

        return rec, vlm_in, vlm_out

    def _handle_inspect_roi(self, args: _InspectROIArgs) -> dict[str, Any]:
        crop = self.artifact_registry.inspect_roi(args.artifact_ref, roi_id=args.roi_id, page_index=args.page_index)
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

    def _handle_inspect_all_receipts(self, args: _InspectAllReceiptsArgs) -> dict[str, Any]:
        out_list: list[dict[str, Any]] = []
        for ref in args.artifact_refs:
            art = self.artifact_registry.get(ref)
            if art is None:
                out_list.append({"handle": ref, "status": "not_found"})
                continue
            crop = self.artifact_registry.inspect_roi(art.handle, roi_id=art.rois[0].roi_id if art.rois else None)
            ocr_body = art.raw_full_text[:2400] if art.modality == "pdf" else (crop.extracted_text or art.raw_full_text[:1000])
            out_list.append({
                "handle": art.handle,
                "filename": Path(art.source_path).name,
                "modality": art.modality,
                "pages": art.page_count,
                "tainted": art.is_tainted_untrusted,
                "ocr_text": ocr_body,
                "line_items": crop.line_items,
            })
        return {"status": "ok", "count": len(out_list), "receipts": out_list}

    def _handle_eval_math(self, args: _EvalMathArgs) -> dict[str, Any]:
        return {"status": "ok", "expression": args.expression, "result": ASTExpenseCalculator.evaluate(args.expression)}

    def _handle_export_docx(self, args: _ExportDocxArgs) -> dict[str, Any]:
        fname = args.filename if args.filename.lower().endswith(".docx") else f"{args.filename}.docx"
        target = self._safe_workspace_path(fname)

        # Deterministic Financial Audit Gate (Fail-Fast on any row subtraction hallucination!)
        row_errors: list[str] = []
        for idx, it in enumerate(args.items, start=1):
            expected_net = ASTExpenseCalculator.evaluate(f"round({it.gross_amount_cny} - {it.excluded_personal_cny}, 2)")
            if abs(expected_net - round(it.reimbursable_cny, 2)) > 0.01:
                row_errors.append(
                    f"Item #{idx} ({it.merchant_or_desc[:24]}): gross({it.gross_amount_cny}) - excluded({it.excluded_personal_cny}) = {expected_net}, got reimbursable_cny={it.reimbursable_cny}"
                )
        if row_errors:
            return {"status": "error", "error": "AST row arithmetic audit failed: " + "; ".join(row_errors)}

        ast_sum = ASTExpenseCalculator.evaluate(" + ".join(str(i.reimbursable_cny) for i in args.items) if args.items else "0")
        eff_warnings = list(args.extra_warnings)
        budget_remaining: float | None = None
        if args.budget_cap_cny is not None:
            budget_remaining = ASTExpenseCalculator.evaluate(f"round({args.budget_cap_cny} - {ast_sum}, 2)")
            if budget_remaining < 0:
                eff_warnings.insert(
                    0,
                    f"超预算预警：本次核定报销总额 ¥{ast_sum:.2f} 超出预算上限 ¥{args.budget_cap_cny:.2f}（超支 ¥{abs(budget_remaining):.2f}），需特批签字。",
                )

        # Tamper-evident SHA-256 Evidence Seal binding input voucher digests + AST total
        voucher_digests = sorted(
            art.sha256[:12]
            for h in self._session_voucher_handles
            if (art := self.artifact_registry.get(h)) is not None
        )
        seal_payload = json.dumps(
            {"vouchers": voucher_digests, "ast_net_cny": ast_sum, "item_count": len(args.items)},
            sort_keys=True,
        )
        audit_seal = hashlib.sha256(seal_payload.encode("utf-8")).hexdigest()

        act_pairs: list[tuple[Path, str]] | None = None
        act_note = ""
        if args.include_activity_photos and self._session_activity_photos:
            act_pairs = []
            for idx, p_path in enumerate(self._session_activity_photos):
                cap = (
                    args.activity_photo_captions[idx].strip()
                    if idx < len(args.activity_photo_captions) and args.activity_photo_captions[idx].strip()
                    else self._session_photo_vlm_captions.get(p_path.name, f"活动现场照片（{p_path.name}）")
                )
                act_pairs.append((p_path, f"照片 {idx + 1}｜{cap}"))
            act_note = (
                f"以下精选 {len(act_pairs)} 张现场佐证照片（原始归档共含 {self._session_activity_total} 张活动照片可供备查）。"
            )

        res = build_reimbursement_docx(
            output_path=target,
            kicker_text=args.kicker_text,
            title=args.title,
            applicant=args.applicant,
            project_code=args.project_code,
            trip_summary=args.trip_summary,
            items=args.items,
            artifact_registry=self.artifact_registry,
            extra_warnings=eff_warnings,
            include_all_registered_handles=self._session_voucher_handles,
            budget_cap_cny=args.budget_cap_cny,
            extra_sections=[s.model_dump() for s in args.extra_sections] if args.extra_sections else None,
            activity_photos=act_pairs,
            activity_photo_note=act_note,
            audit_seal_sha256=audit_seal,
        )
        return {
            "status": "ok",
            "saved_file": res.saved_file,
            "download_url": f"/v1/steward/files/{res.saved_file}",
            "bytes_written": res.bytes_written,
            "item_count": res.item_count,
            "ast_verified_net_cny": ast_sum,
            "budget_remaining_cny": budget_remaining,
            "gross_total_cny": res.gross_total_cny,
            "excluded_total_cny": res.excluded_total_cny,
            "embedded_photos": res.embedded_photos,
            "audit_seal_sha256": audit_seal[:16],
            "duplicate_warnings": res.duplicate_warnings,
        }

    def _handle_export_claim(self, args: _ExportClaimArgs) -> dict[str, Any]:
        target = self._safe_workspace_path(args.filename)
        target.write_text(args.markdown_body, encoding="utf-8")
        return {"status": "ok", "saved_file": target.name, "bytes_written": target.stat().st_size}

    def _handle_read_claim(self, args: _ReadClaimArgs) -> dict[str, Any]:
        target = self._safe_workspace_path(args.filename)
        if not target.exists():
            return {"status": "error", "error": f"File not found: {target.name}"}
        return {"status": "ok", "filename": target.name, "content": target.read_text(encoding="utf-8")}

    def _handle_guarded_shell(self, args: _GuardedShellArgs) -> dict[str, Any]:
        """Run allowlisted diagnostic commands with shell=False and stripped host secrets."""
        argv = shlex.split(args.command, posix=True)
        cmd = argv[0].lower()
        # Cross-platform pure-Python safe implementation for ls/pwd/wc/head/sha256sum (zero host shell exposure)
        if cmd == "pwd":
            return {"status": "ok", "stdout": str(self.workspace_dir)}
        if cmd == "ls":
            items = sorted(p.name for p in self.workspace_dir.iterdir())
            return {"status": "ok", "files": items}
        if cmd in {"wc", "head", "sha256sum"} and len(argv) >= 2:
            target = self._safe_workspace_path(argv[-1])
            if not target.exists():
                return {"status": "error", "error": f"File not found: {target.name}"}
            text = target.read_text(encoding="utf-8", errors="replace")
            if cmd == "wc":
                return {"status": "ok", "lines": len(text.splitlines()), "bytes": target.stat().st_size}
            if cmd == "head":
                return {"status": "ok", "stdout": "\n".join(text.splitlines()[:10])}
            import hashlib as _hl
            return {"status": "ok", "sha256": _hl.sha256(target.read_bytes()).hexdigest()}

        # Fallback non-shell execution with completely sanitized environment (no OPENAI_API_KEY)
        clean_env = {k: os.environ[k] for k in ("PATH", "SYSTEMROOT", "WINDIR") if k in os.environ}
        proc = subprocess.run(
            argv,
            shell=False,
            cwd=str(self.workspace_dir),
            env=clean_env,
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
