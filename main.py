"""CLI & FastAPI entrypoint for TripLedger Steward (personal-steward-agent)."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import shutil
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.agent.state_machine import StewardAgent
from src.agent.tools import StewardToolRegistry
from src.config import StewardSettings
from src.eval.run_benchmark import build_multimodal_fixtures, seed_steward_memories
from src.llm.provider import build_llm_provider
from src.memory.steward_memory import StewardMemoryEngine
from src.observability.tracer import StructuredTracer
from src.perception.artifact_registry import ArtifactRegistry


class StewardQueryRequest(BaseModel):
    query: str = Field(min_length=1)
    attachments: list[str] = Field(default_factory=list)
    session_id: str = "api_session"
    approval_token: str | None = None


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
    tools = StewardToolRegistry(workspace_dir=cfg.sandbox_workspace_dir, artifact_registry=art_reg)
    llm = build_llm_provider(cfg, force_mock=force_mock)
    tracer = StructuredTracer(cfg.trace_log_path)
    return StewardAgent(
        cfg,
        memory=memory,
        artifact_registry=art_reg,
        tools=tools,
        llm=llm,
        tracer=tracer,
    )


def create_app(*, force_mock: bool = False) -> FastAPI:
    app = FastAPI(
        title="TripLedger Steward API",
        version="1.0.0",
        description="Personal Travel, Receipt ROI Audit & Cross-Project Expense Steward",
    )
    agent = build_steward_runtime(force_mock=force_mock)

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "mode": "mock" if (force_mock or agent.settings.use_mock_llm) else "live_api",
            "model": agent.settings.llm_model,
            "registered_tools": len(agent.tools.all_specs()),
        }

    @app.post("/v1/steward/run")
    async def run_steward(req: StewardQueryRequest) -> dict[str, Any]:
        res = await agent.run(
            req.query,
            attachments=req.attachments,
            session_id=req.session_id,
            approval_token=req.approval_token,
        )
        return res.model_dump(mode="json")

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
            "sandbox_workspace_dir": demo_dir / "workspace",
            "artifact_cache_dir": demo_dir / "crops",
            "trace_log_path": demo_dir / "demo_trace.jsonl",
        }
    )
    fixtures = build_multimodal_fixtures(demo_dir / "fixtures")
    agent = build_steward_runtime(cfg, force_mock=force_mock)
    seed_steward_memories(agent.memory)

    mode_str = "OFFLINE_MOCK" if (force_mock or cfg.use_mock_llm) else f"LIVE_API ({cfg.llm_model})"
    print(f"=== TripLedger Steward End-to-End Demo [{mode_str}] ===\n")

    scenarios: list[tuple[str, str, list[Path]]] = [
        (
            "1. Slot Supersession + 2-Hop Directed PPR Bridge",
            "我11月18日去深圳参加「华南智算中心拜访」最新改签后的车次与票价是多少？对应项目的一线城市住宿差标和终审审批人是谁？",
            [],
        ),
        (
            "2. 4-Page PDF Hotel Folio ROI Zoom + AST Audit + Claim Export",
            "请核对这张深圳酒店4页水单，剔除个人消费（Minibar迷你吧）后，计算两晚房费与商务早餐的可报销合计金额，并将核验单导出保存为 claim_demo_sz.md。",
            [fixtures["hotel_folio_pdf"]],
        ),
        (
            "3. PII Local Privacy Isolation Gate",
            "请核对实名备案信息要素是否完整：张熠，身份证号 31010419990816401X，公务卡 6222021001123456789。",
            [],
        ),
        (
            "4. Over-Limit Reimbursement Payout Human-in-the-Loop Gate",
            "请针对项目 PRJ-PANGU-2 直接发起打款 2850 元至员工报销账户。",
            [],
        ),
    ]

    for title, q, files in scenarios:
        res = await agent.run(q, attachments=files)
        print(f"--- {title} ---")
        print(
            json.dumps(
                {
                    "final_state": res.final_state.value,
                    "tier": res.routing_decision.tier.value,
                    "privacy_triggered": res.routing_decision.privacy_triggered,
                    "pruned_tools": f"{len(res.tool_selection.selected_names)}/{res.tool_selection.total_registered} "
                    f"(-{res.tool_selection.token_savings_ratio*100:.1f}% schema tokens)",
                    "executed_tools": [t.tool_name for t in res.executed_tools],
                    "tokens": {
                        "input": res.total_input_tokens,
                        "cached_input": res.total_cached_input_tokens,
                        "output": res.total_output_tokens,
                    },
                    "latency_ms": res.latency_ms,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        print(f"Answer:\n{res.answer}\n")

    await agent.memory.flush_and_close()


async def run_single_query(query: str, files: list[str], *, force_mock: bool = False) -> None:
    agent = build_steward_runtime(force_mock=force_mock)
    try:
        res = await agent.run(query, attachments=files)
        print(json.dumps(res.model_dump(mode="json"), ensure_ascii=False, indent=2))
    finally:
        await agent.memory.flush_and_close()


def main() -> None:
    parser = argparse.ArgumentParser(description="TripLedger Steward CLI & API Server")
    parser.add_argument("--demo", action="store_true", help="Run 4-stage travel & expense showcase")
    parser.add_argument("--query", type=str, default="", help="Execute a single natural-language query")
    parser.add_argument("--file", action="append", default=[], help="Attach PDF or PNG receipt file(s)")
    parser.add_argument("--mock", action="store_true", help="Force deterministic offline MockLLMProvider")
    parser.add_argument("--serve", action="store_true", help="Start FastAPI server")
    parser.add_argument("--port", type=int, default=8088, help="Port for --serve")
    args = parser.parse_args()

    if args.serve:
        import uvicorn

        uvicorn.run(create_app(force_mock=args.mock), host="127.0.0.1", port=args.port)
    elif args.demo:
        asyncio.run(run_demo(force_mock=args.mock))
    elif args.query:
        asyncio.run(run_single_query(args.query, args.file, force_mock=args.mock))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
