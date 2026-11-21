"""Shared pytest fixtures with isolated tmp_path SQLite stores and MockLLMProvider."""

from __future__ import annotations

from pathlib import Path
import pytest

from src.agent.state_machine import StewardAgent
from src.agent.tools import StewardToolRegistry
from src.config import StewardSettings
from src.eval.run_benchmark import build_multimodal_fixtures
from src.llm.provider import MockLLMProvider
from src.memory.steward_memory import StewardMemoryEngine
from src.observability.tracer import StructuredTracer
from src.perception.artifact_registry import ArtifactRegistry


@pytest.fixture()
def isolated_settings(tmp_path: Path) -> StewardSettings:
    return StewardSettings(
        OPENAI_API_KEY="mock",
        OPENAI_BASE_URL="https://api.deepseek.com/v1",
        LOCAL_SLM_BASE_URL="",
        MEMORY_DB_PATH=tmp_path / "mem" / "test_steward.sqlite3",
        MCP_DB_PATH=tmp_path / "mcp" / "test_mcp.sqlite3",
        CHECKPOINT_DB_PATH=tmp_path / "chk" / "test_chk.sqlite3",
        SANDBOX_WORKSPACE_DIR=tmp_path / "workspace",
        ARTIFACT_CACHE_DIR=tmp_path / "crops",
        TRACE_LOG_PATH=tmp_path / "logs" / "trace.jsonl",
    )


@pytest.fixture()
def sample_fixtures(tmp_path: Path) -> dict[str, Path]:
    return build_multimodal_fixtures(tmp_path / "fixtures")


@pytest.fixture()
def steward_agent(isolated_settings: StewardSettings) -> StewardAgent:
    mem = StewardMemoryEngine(isolated_settings.memory_db_path)
    reg = ArtifactRegistry(isolated_settings.artifact_cache_dir)
    tools = StewardToolRegistry(
        workspace_dir=isolated_settings.sandbox_workspace_dir,
        artifact_registry=reg,
        mcp_db_path=isolated_settings.mcp_db_path,
    )
    agent = StewardAgent(
        isolated_settings,
        memory=mem,
        artifact_registry=reg,
        tools=tools,
        llm=MockLLMProvider(isolated_settings),
        tracer=StructuredTracer(isolated_settings.trace_log_path),
    )
    yield agent
    tools.close()
    agent.checkpoints.close()
    try:
        mem._conn.close()
    except Exception:
        pass
