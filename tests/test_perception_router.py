"""Tests for pure-pixel RapidOCR, clip=cbox PDF, stdio MCP discovery, PrivacyVault & fitted router."""

from __future__ import annotations

from pathlib import Path
from PIL import Image

from src.agent.tools import StewardToolRegistry
from src.config import StewardSettings
from src.memory.tokenizer_embedder import DeterministicBilingualEmbedder
from src.perception.artifact_registry import ArtifactRegistry
from src.perception.receipt_parser import ReceiptMultimodalParser
from src.router.steward_router import (
    IMMUTABLE_SYSTEM_PROMPT,
    ModelTier,
    PairedMcpToolSelector,
    PrivacyAndCostRouter,
    PrivacyVault,
)


def test_pdf_clip_and_pure_pixel_rapidocr_without_pnginfo(
    tmp_path: Path, sample_fixtures: dict[str, Path]
) -> None:
    reg = ArtifactRegistry(tmp_path / "crops")
    parser = ReceiptMultimodalParser(reg)

    # 1. 4-page PDF cell-clip extraction (date column ignored)
    folio = parser.parse_file(sample_fixtures["hotel_folio_pdf"])
    assert folio.page_count == 4 and not folio.is_tainted_untrusted
    crop = reg.inspect_roi(folio.handle, roi_id="roi_1")
    assert crop.line_items["Deluxe King Room N1"] == 820.0
    assert crop.line_items["Business Breakfast"] == 88.0

    # 2. Pure-pixel PNG receipt (verify ZERO PngInfo metadata cheat!)
    ride_path = sample_fixtures["ride_png"]
    assert "structured_rois" not in (Image.open(ride_path).info or {})
    ride_art = parser.parse_file(ride_path)
    assert ride_art.modality == "image" and len(ride_art.rois) == 1
    assert "76.50" in ride_art.raw_full_text and "15.00" in ride_art.raw_full_text and "18.00" in ride_art.raw_full_text

    # 3. Tainted PDF detection
    tainted = parser.parse_file(sample_fixtures["tainted_pdf"])
    assert tainted.is_tainted_untrusted and "instruction_override" in tainted.taint_reasons


def test_stdio_mcp_discovery_paired_pruning_and_fail_closed_privacy_vault(
    isolated_settings: StewardSettings,
) -> None:
    tools = StewardToolRegistry(
        workspace_dir=isolated_settings.sandbox_workspace_dir,
        artifact_registry=ArtifactRegistry(isolated_settings.artifact_cache_dir),
        mcp_db_path=isolated_settings.mcp_db_path,
    )
    try:
        # Verify real stdio JSON-RPC MCP discovery (9 MCP + 7 local = 16 tools)
        assert len(tools.mcp_client.tool_specs) == 9
        assert len(tools.all_specs()) == 16

        selector = PairedMcpToolSelector(DeterministicBilingualEmbedder(dim=256))
        sel = selector.select("请先查询现有日程避免冲突，然后在明天下午锁定日程会议", tools.all_specs(), budget=6)
        assert "mcp_itinerary_query" in sel.selected_names and "mcp_itinerary_schedule" in sel.selected_names
        assert sel.token_savings_ratio > 0.45
    finally:
        tools.close()

    # Fail-Closed PrivacyVault when LOCAL_SLM_BASE_URL is empty
    router = PrivacyAndCostRouter(isolated_settings)
    dec, red = router.route(
        "乘机人张熠，身份证号 31010419990816401X，公务卡 6222021001123456789",
        artifacts=[],
        memory_hits=[],
    )
    assert dec.tier == ModelTier.REDACTED_CLOUD_SLM and dec.redacted_for_cloud is True
    assert "31010419990816401X" not in red.redacted_text
    assert "6222021001123456789" not in red.redacted_text
    assert PrivacyVault.verify_cn_id_checksum("31010419990816401X") == (False, "9")
    assert PrivacyVault.verify_luhn("6222021001123456789") is True

    # Byte-constant Prefix KV-Cache check
    m1 = router.build_kv_cache_messages("A", memory_hits=[], abstained=True, artifacts=[])
    m2 = router.build_kv_cache_messages("B", memory_hits=[], abstained=False, artifacts=[])
    assert m1[0]["content"] == IMMUTABLE_SYSTEM_PROMPT == m2[0]["content"]
