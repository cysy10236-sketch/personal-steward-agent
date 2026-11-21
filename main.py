"""CLI, FastAPI Server & Minimalist Web Studio for TripLedger Steward."""

from __future__ import annotations

import argparse
import asyncio
from contextlib import asynccontextmanager
import json
from pathlib import Path
import shutil
from typing import Any
import uuid

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel, Field

from src.agent.state_machine import AgentState, StewardAgent
from src.agent.tools import StewardToolRegistry
from src.config import StewardSettings
from src.eval.run_benchmark import seed_steward_memories
from src.llm.provider import build_llm_provider
from src.memory.steward_memory import StewardMemoryEngine
from src.observability.tracer import StructuredTracer
from src.perception.artifact_registry import ArtifactRegistry

WEB_INDEX_PATH = Path(__file__).resolve().parent / "src" / "web" / "index.html"
DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


class StewardQueryRequest(BaseModel):
    query: str = Field(min_length=1)
    attachments: list[str] = Field(default_factory=list)
    session_id: str = "api_session"
    approval_token: str | None = None


class StewardResumeRequest(BaseModel):
    checkpoint_id: str = Field(min_length=4)
    approval_token: str = Field(min_length=4)


def _load_cjk_font(size: int) -> Any:
    for p in (
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
        Path("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"),
    ):
        if p.exists():
            return ImageFont.truetype(str(p), size)
    return ImageFont.load_default()


def _render_receipt_card(
    out_path: Path,
    *,
    header_bg: tuple[int, int, int],
    header_fg: tuple[int, int, int],
    canvas_bg: tuple[int, int, int],
    brand_title: str,
    amount_banner: str,
    rows: list[tuple[str, str]],
    footer_note: str = "",
) -> Path:
    """Render a realistic, pure-pixel PNG payment screenshot or paper receipt (zero PngInfo)."""
    w, h = 680, 740
    img = Image.new("RGB", (w, h), canvas_bg)
    draw = ImageDraw.Draw(img)
    f_hdr = _load_cjk_font(26)
    f_amt = _load_cjk_font(46)
    f_lbl = _load_cjk_font(21)
    f_val = _load_cjk_font(22)
    f_sub = _load_cjk_font(18)

    # Top bar
    draw.rectangle([0, 0, w, 92], fill=header_bg)
    draw.text((34, 28), brand_title, fill=header_fg, font=f_hdr)

    # Main receipt card
    draw.rounded_rectangle([28, 114, w - 28, h - 32], radius=18, fill=(255, 255, 255), outline=(226, 232, 240), width=2)
    draw.text((54, 146), amount_banner, fill=(15, 23, 42), font=f_amt)
    draw.line([(54, 224), (w - 54, 224)], fill=(226, 232, 240), width=2)

    y = 252
    for k, v in rows:
        draw.text((54, y), k, fill=(100, 116, 139), font=f_lbl)
        draw.text((224, y), v, fill=(15, 23, 42), font=f_val)
        y += 56

    if footer_note:
        draw.line([(54, h - 108), (w - 54, h - 108)], fill=(226, 232, 240), width=1)
        draw.text((54, h - 86), footer_note, fill=(71, 85, 105), font=f_sub)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, format="PNG")
    return out_path


def ensure_sample_receipt_pack(sample_dir: Path) -> list[Path]:
    """Generate 4 realistic pure-pixel roommate team-outing receipts (WeChat, Paper Dup, Alipay, Cafe)."""
    specs: list[tuple[str, dict[str, Any]]] = [
        (
            "01_wechat_pay_team_dinner.png",
            dict(
                header_bg=(7, 193, 96),
                header_fg=(255, 255, 255),
                canvas_bg=(245, 247, 250),
                brand_title="微信支付 · 账单详情",
                amount_banner="-268.00 元",
                rows=[
                    ("当前状态", "支付成功"),
                    ("收款商户", "南山湘味馆（团队晚餐）"),
                    ("商品说明", "带队聚餐工作晚餐 6人餐"),
                    ("支付时间", "2026-11-18 19:42:15"),
                    ("支付方式", "微信支付 / 招商银行借记卡"),
                    ("交易单号", "4200002418202611180012"),
                ],
                footer_note="微信支付电子凭证 · 可用于团队差旅餐费报销",
            ),
        ),
        (
            "02_paper_receipt_dinner_dup.png",
            dict(
                header_bg=(51, 65, 85),
                header_fg=(255, 255, 255),
                canvas_bg=(250, 248, 242),
                brand_title="南山湘味馆 · 结账小票照片",
                amount_banner="实付合计: 268.00 元",
                rows=[
                    ("消费日期", "2026-11-18 19:41"),
                    ("商户名称", "南山湘味馆(科技园店)"),
                    ("消费明细", "团队工作晚餐套餐 6人"),
                    ("应付金额", "268.00 元"),
                    ("结算方式", "微信扫码支付已结清"),
                    ("小票单号", "POS-20261118-092"),
                ],
                footer_note="提示: 本纸质小票与当日微信支付 268.00 元为同一笔晚餐消费",
            ),
        ),
        (
            "03_alipay_didi_trip.png",
            dict(
                header_bg=(22, 119, 255),
                header_fg=(255, 255, 255),
                canvas_bg=(245, 247, 250),
                brand_title="支付宝 · 账单详情",
                amount_banner="-91.50 元",
                rows=[
                    ("交易状态", "交易成功"),
                    ("商户名称", "高德打车（深圳北站至会场）"),
                    ("费用拆分", "基础车费 76.50 + 高速费 15.00"),
                    ("付款金额", "91.50 元"),
                    ("交易时间", "2026-11-18 09:15:30"),
                    ("用途备注", "带队团队接驳专车交通费"),
                ],
                footer_note="支付宝电子账单 · 全部属于因公团队交通费",
            ),
        ),
        (
            "04_coffee_personal_mix.png",
            dict(
                header_bg=(30, 41, 59),
                header_fg=(255, 255, 255),
                canvas_bg=(248, 250, 252),
                brand_title="瑞幸咖啡 · 电子消费小票",
                amount_banner="订单实付: 96.00 元",
                rows=[
                    ("消费日期", "2026-11-19 14:20"),
                    ("商户名称", "瑞幸咖啡(南山会场店)"),
                    ("团队会议饮品", "美式咖啡4杯 72.00 元"),
                    ("个人消费自费", "个人甜品蛋糕 24.00 元(不可报销)"),
                    ("可报销参考", "72.00 元（剔除个人甜品24元）"),
                    ("支付渠道", "微信支付"),
                ],
                footer_note="请财务核验时剔除个人甜品 24.00 元，仅报销会议饮品 72.00 元",
            ),
        ),
    ]
    paths: list[Path] = []
    for fname, kw in specs:
        p = sample_dir / fname
        if not p.exists():
            _render_receipt_card(p, **kw)
        paths.append(p)
    return paths


def build_steward_runtime(
    settings: StewardSettings | None = None,
    *,
    force_mock: bool = False,
) -> StewardAgent:
    cfg = settings or StewardSettings()
    memory = StewardMemoryEngine(
        cfg.memory_db_path,
        embedding_dim=cfg.embedding_dim,
        base_half_life_hours=cfg.memory_base_half_life_hours,
        ppr_alpha=cfg.memory_ppr_alpha,
        abstention_threshold=cfg.memory_abstention_threshold,
    )
    art_reg = ArtifactRegistry(cfg.artifact_cache_dir)
    tools = StewardToolRegistry(
        workspace_dir=cfg.sandbox_workspace_dir,
        artifact_registry=art_reg,
        mcp_db_path=cfg.mcp_db_path,
    )
    return StewardAgent(
        cfg,
        memory=memory,
        artifact_registry=art_reg,
        tools=tools,
        llm=build_llm_provider(cfg, force_mock=force_mock),
        tracer=StructuredTracer(cfg.trace_log_path),
    )


def _safe_child_file(base_dir: Path, filename: str) -> Path:
    clean = Path(filename).name
    if not clean or clean in {".", ".."} or clean != filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    target = (base_dir / clean).resolve()
    if target.parent != base_dir.resolve() or not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    return target


def create_app(*, force_mock: bool = False) -> FastAPI:
    agent = build_steward_runtime(force_mock=force_mock)
    if not agent.memory._records:
        seed_steward_memories(agent.memory)
    sample_dir = Path("runtime_data/sample_pack")
    upload_root = Path("runtime_data/uploads")
    upload_root.mkdir(parents=True, exist_ok=True)

    @asynccontextmanager
    async def _lifespan(_: FastAPI):
        try:
            yield
        finally:
            await agent.close()

    app = FastAPI(title="TripLedger Steward Studio", version="2.1.0", lifespan=_lifespan)

    @app.get("/", response_class=HTMLResponse)
    async def studio_ui() -> HTMLResponse:
        return HTMLResponse(WEB_INDEX_PATH.read_text(encoding="utf-8"))

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "mode": "mock" if (force_mock or agent.settings.use_mock_llm) else "live_api",
            "llm_model": agent.settings.llm_model,
            "slm_model": agent.settings.slm_model,
            "vlm_model": agent.settings.vlm_model,
            "registered_tools": len(agent.tools.all_specs()),
            "mcp_discovered_tools": len(agent.tools.mcp_client.tool_specs),
        }

    @app.get("/v1/steward/sample-pack")
    async def sample_pack() -> dict[str, Any]:
        files = ensure_sample_receipt_pack(sample_dir)
        return {"files": [{"name": p.name, "url": f"/v1/steward/samples/{p.name}"} for p in files]}

    @app.get("/v1/steward/samples/{filename}")
    async def get_sample_file(filename: str) -> FileResponse:
        return FileResponse(_safe_child_file(sample_dir, filename), media_type="image/png")

    @app.post("/v1/steward/reimburse-upload")
    async def reimburse_upload(
        query: str = Form(...),
        applicant: str = Form(default=""),
        project_code: str = Form(default=""),
        export_docx: str = Form(default="true"),
        files: list[UploadFile] = File(default_factory=list),
    ) -> dict[str, Any]:
        batch_id = uuid.uuid4().hex[:8]
        batch_dir = upload_root / batch_id
        saved_paths: list[Path] = []
        if files:
            batch_dir.mkdir(parents=True, exist_ok=True)
            for idx, uf in enumerate(files, start=1):
                safe_name = Path(uf.filename or f"receipt_{idx}.png").name
                dest = batch_dir / f"{idx:02d}_{safe_name}"
                dest.write_bytes(await uf.read())
                saved_paths.append(dest)

        want_docx = export_docx.lower() in {"true", "1", "yes", "on"}
        meta_ctx = f"报销人：{applicant or '团队带队人'}；归属项目：{project_code or 'PRJ-PANGU-2'}。"
        export_hint = (
            f"请将每张上传凭证均通过 inspect_all_receipts 核验、用 eval_expense_math 精确汇总后，"
            f"调用 export_reimbursement_docx 生成报销单（文件名 reimbursement_{batch_id}.docx），"
            f"并在每条费用项中准确填写对应的 artifact_handle 以便嵌入原始票据照片；若发现同日同额重复票据请明确标注并在 extra_warnings 说明。"
            if want_docx
            else ""
        )
        full_query = f"{query.strip()}\n\n[上下文约束] {meta_ctx} {export_hint}".strip()
        res = await agent.run(full_query, attachments=saved_paths, session_id=f"web_{batch_id}")
        return res.model_dump(mode="json")

    @app.get("/v1/steward/files/{filename}")
    async def download_workspace_file(filename: str) -> FileResponse:
        target = _safe_child_file(agent.settings.sandbox_workspace_dir, filename)
        media = DOCX_MIME if target.suffix.lower() == ".docx" else "application/octet-stream"
        return FileResponse(target, media_type=media, filename=target.name)

    @app.post("/v1/steward/run")
    async def run_steward(req: StewardQueryRequest) -> dict[str, Any]:
        res = await agent.run(
            req.query, attachments=req.attachments, session_id=req.session_id, approval_token=req.approval_token
        )
        return res.model_dump(mode="json")

    @app.post("/v1/steward/resume")
    async def resume_steward(req: StewardResumeRequest) -> dict[str, Any]:
        try:
            res = await agent.resume(req.checkpoint_id, approval_token=req.approval_token)
            return res.model_dump(mode="json")
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return app


async def run_demo(*, force_mock: bool = False) -> None:
    base_cfg = StewardSettings()
    demo_dir = Path("runtime_data/demo_session")
    if demo_dir.exists():
        shutil.rmtree(demo_dir)
    demo_dir.mkdir(parents=True, exist_ok=True)

    cfg = base_cfg.model_copy(
        update={
            "memory_db_path": demo_dir / "demo_memory.sqlite3",
            "mcp_db_path": demo_dir / "demo_mcp.sqlite3",
            "checkpoint_db_path": demo_dir / "demo_checkpoints.sqlite3",
            "sandbox_workspace_dir": demo_dir / "workspace",
            "artifact_cache_dir": demo_dir / "crops",
            "trace_log_path": demo_dir / "demo_trace.jsonl",
        }
    )
    sample_receipts = ensure_sample_receipt_pack(demo_dir / "sample_pack")
    agent = build_steward_runtime(cfg, force_mock=force_mock)
    try:
        seed_steward_memories(agent.memory)
        mode_str = "OFFLINE_MOCK" if (force_mock or cfg.use_mock_llm) else f"LIVE_API (SLM={cfg.slm_model}, LLM={cfg.llm_model}, VLM={cfg.vlm_model})"
        print(f"=== TripLedger Steward v2.1 Demo [{mode_str}] ===\n")

        scenarios: list[tuple[str, str, list[Path]]] = [
            ("1. Slot Supersession + 2-Hop PPR", "我11月18日去深圳参加「凌云视觉拜访」的最新改签车次票价是多少？对应项目住宿差标和审批人是谁？", []),
            ("2. Multi-Receipt Batch -> Word (.docx) with Photos", "请批量核验我带队出门上传的全部支付截图与小票照片，剔除个人自费项、标出重复票据并导出 Word (.docx) 报销文档 team_demo.docx。", sample_receipts),
            ("3. Fail-Closed PII PrivacyVault", "请帮我核对实名差旅备案要素：张熠，身份证号 31010419990816401X，公务卡 6222021001123456789。", []),
            ("4. Over-Limit Payout Halt -> Checkpoint Resume", "请针对项目 PRJ-PANGU-2 直接发起打款 2850 元至员工报销账户。", []),
        ]
        for title, q, files in scenarios:
            res = await agent.run(q, attachments=files)
            print(f"--- {title} ---")
            print(json.dumps({
                "final_state": res.final_state.value,
                "tier": res.routing_decision.tier.value,
                "model": res.routing_decision.model_id,
                "pruned_tools": f"{len(res.tool_selection.selected_names)}/{res.tool_selection.total_registered}",
                "changed_files": res.audit_verdict.changed_workspace_files,
                "tokens": {"in": res.total_input_tokens, "cached": res.total_cached_input_tokens, "out": res.total_output_tokens},
                "latency_ms": res.latency_ms,
            }, ensure_ascii=False, indent=2))
            print(f"Answer: {res.answer[:320]}\n")

            if res.final_state == AgentState.HALTED_APPROVAL and res.checkpoint_id and res.action_fingerprint:
                resumed = await agent.resume(res.checkpoint_id, approval_token=f"APR_{res.action_fingerprint}")
                print(f"  -> [RESUMED from {res.checkpoint_id}] final_state={resumed.final_state.value}, answer={resumed.answer[:200]}\n")
    finally:
        await agent.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="TripLedger Steward CLI & Web Studio Server")
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--query", type=str, default="")
    parser.add_argument("--file", action="append", default=[])
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--port", type=int, default=8088)
    args = parser.parse_args()

    if args.serve:
        import uvicorn
        uvicorn.run(create_app(force_mock=args.mock), host="127.0.0.1", port=args.port)
    elif args.demo:
        asyncio.run(run_demo(force_mock=args.mock))
    elif args.query:
        async def _once() -> None:
            ag = build_steward_runtime(force_mock=args.mock)
            try:
                r = await ag.run(args.query, attachments=args.file)
                print(json.dumps(r.model_dump(mode="json"), ensure_ascii=False, indent=2))
            finally:
                await ag.close()
        asyncio.run(_once())
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
