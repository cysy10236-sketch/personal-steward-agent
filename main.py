"""CLI & FastAPI entrypoint for TripLedger Steward v2.0 (stdio MCP + Checkpoint Resume)."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import shutil
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.agent.state_machine import AgentState, StewardAgent
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


class StewardResumeRequest(BaseModel):
    checkpoint_id: str = Field(min_length=4)
    approval_token: str = Field(min_length=4)


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


def create_app(*, force_mock: bool = False) -> FastAPI:
    app = FastAPI(title="TripLedger Steward v2.0 API", version="2.0.0")
    agent = build_steward_runtime(force_mock=force_mock)

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
    fixtures = build_multimodal_fixtures(demo_dir / "fixtures")
    agent = build_steward_runtime(cfg, force_mock=force_mock)
    try:
        seed_steward_memories(agent.memory)
        mode_str = "OFFLINE_MOCK" if (force_mock or cfg.use_mock_llm) else f"LIVE_API (SLM={cfg.slm_model}, LLM={cfg.llm_model}, VLM={cfg.vlm_model})"
        print(f"=== TripLedger Steward v2.0 Demo [{mode_str}] ===\n")

        scenarios: list[tuple[str, str, list[Path]]] = [
            ("1. Slot Supersession + 2-Hop PPR", "我11月18日去深圳参加「凌云视觉拜访」的最新改签车次票价是多少？对应项目住宿差标和审批人是谁？", []),
            ("2. Pure-Pixel RapidOCR + Native VLM Ride Receipt", "请查看这张网约车电子行程单截图，剔除不可报销的个人绕路费后，计算基础车费与高速过路费的可报销总额是多少元？", [fixtures["ride_png"]]),
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
                "redacted_for_cloud": res.routing_decision.redacted_for_cloud,
                "pruned_tools": f"{len(res.tool_selection.selected_names)}/{res.tool_selection.total_registered}",
                "checkpoint_id": res.checkpoint_id,
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
    parser = argparse.ArgumentParser(description="TripLedger Steward v2.0 CLI & API Server")
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
