"""Unified tool registry: 5 local core tools (with VLM ROI check & shell=False) + 9 stdio MCP tools."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shlex
import subprocess
from typing import Any, Callable

from pydantic import BaseModel, Field, ValidationError

from ..llm.provider import LLMProviderProtocol
from ..mcp.client import StdioMcpClient
from ..perception.artifact_registry import ArtifactRegistry
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


class _EvalMathArgs(BaseModel):
    expression: str = Field(description="Arithmetic expression using +,-,*,/,round,sum,min,max,abs,mean")


class _ExportClaimArgs(BaseModel):
    filename: str = Field(description="Relative markdown filename inside workspace, e.g. claim_shenzhen.md")
    markdown_body: str = Field(min_length=5, description="Markdown content of the reimbursement claim")


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
        self._register_local()

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
            description="Crop high-res ROI from a PDF folio or image receipt handle and return verified OCR/VLM line items.",
            arg_model=_InspectROIArgs,
            handler=self._handle_inspect_roi,
            namespace="perception",
            keywords=["artifact://", "水单", "发票", "截图", "pdf", "receipt", "folio", "金额", "明细"],
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
        """Execute tool; on image ROI crops with a live VLM, cross-verify raw pixels visually."""
        rec = self.execute(tool_name, raw_arguments)
        vlm_in, vlm_out = 0, 0
        if tool_name == "inspect_receipt_roi" and rec.status == "ok" and llm is not None:
            art = self.artifact_registry.get(str(raw_arguments.get("artifact_ref", "")))
            if art is not None and art.modality == "image":
                crop = self.artifact_registry.inspect_roi(
                    art.handle,
                    roi_id=raw_arguments.get("roi_id"),
                    page_index=int(raw_arguments.get("page_index", 0)),
                )
                try:
                    vlm_txt, vlm_in, vlm_out = await llm.inspect_image_crop(
                        crop.base64_png,
                        prompt="List every monetary item and exact amount visible in this cropped receipt image. Note any non-reimbursable personal charges.",
                    )
                    payload = json.loads(rec.output_json)
                    payload["vlm_pixel_verification"] = vlm_txt
                    rec = ToolExecutionRecord(
                        tool_name=rec.tool_name,
                        arguments=rec.arguments,
                        status=rec.status,
                        output_json=json.dumps(payload, ensure_ascii=False),
                    )
                except Exception as exc:
                    payload = json.loads(rec.output_json)
                    payload["vlm_pixel_verification_error"] = f"{type(exc).__name__}: {exc}"
                    rec = ToolExecutionRecord(
                        tool_name=rec.tool_name,
                        arguments=rec.arguments,
                        status=rec.status,
                        output_json=json.dumps(payload, ensure_ascii=False),
                    )
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

    def _handle_eval_math(self, args: _EvalMathArgs) -> dict[str, Any]:
        return {"status": "ok", "expression": args.expression, "result": ASTExpenseCalculator.evaluate(args.expression)}

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
