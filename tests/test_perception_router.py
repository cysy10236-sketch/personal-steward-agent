"""Unit tests for PDF/Image ROI perception, MCP paired tool pruning, and PII/KV-Cache routing."""

from __future__ import annotations

from pathlib import Path
from PIL import Image, ImageDraw

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
)


def test_pdf_cell_clip_roi_and_taint_detection(
    tmp_path: Path, sample_fixtures: dict[str, Path]
) -> None:
    reg = ArtifactRegistry(tmp_path / "crops")
    parser = ReceiptMultimodalParser(reg)

    folio = parser.parse_file(sample_fixtures["hotel_folio_pdf"])
    assert folio.modality == "pdf"
    assert folio.page_count == 4
    assert folio.is_tainted_untrusted is False
    assert len(folio.rois) >= 1

    # Idempotent SHA-256 registration
    folio_dup = parser.parse_file(sample_fixtures["hotel_folio_pdf"])
    assert folio_dup.artifact_id == folio.artifact_id

    crop = reg.inspect_roi(folio.handle, roi_id="roi_1")
    assert Path(crop.crop_path).exists()
    assert crop.line_items["Deluxe King Room N1"] == 820.0
    assert crop.line_items["Business Breakfast"] == 88.0

    # Unannotated PNG fallback via 2D spatial gradient band
    plain_png = tmp_path / "plain_receipt.png"
    im = Image.new("RGB", (400, 300), color=(255, 255, 255))
    dr = ImageDraw.Draw(im)
    dr.rectangle([40, 80, 360, 210], outline=(10, 10, 10), width=3)
    im.save(plain_png)
    plain_art = parser.parse_file(plain_png)
    assert len(plain_art.rois) == 1

    # Tainted PDF detection
    tainted = parser.parse_file(sample_fixtures["tainted_pdf"])
    assert tainted.is_tainted_untrusted is True
    assert "instruction_override" in tainted.taint_reasons


def test_paired_mcp_tool_selector_and_privacy_router(
    isolated_settings: StewardSettings,
) -> None:
    embedder = DeterministicBilingualEmbedder(dim=256)
    selector = PairedMcpToolSelector(embedder)
    tools = StewardToolRegistry(
        workspace_dir=isolated_settings.sandbox_workspace_dir,
        artifact_registry=ArtifactRegistry(isolated_settings.artifact_cache_dir),
    )

    sel = selector.select(
        "请先查询现有日程避免冲突，然后在明天下午锁定日程会议",
        tools.all_specs(),
        budget=6,
    )
    assert "mcp_itinerary_query" in sel.selected_names
    assert "mcp_itinerary_schedule" in sel.selected_names
    assert len(sel.selected_names) <= 6
    assert sel.token_savings_ratio > 0.45

    router = PrivacyAndCostRouter(isolated_settings)
    pii_dec = router.route(
        "乘机人张熠，身份证号 31010419990816401X，公务卡 6222021001123456789",
        artifacts=[],
        memory_hits=[],
    )
    assert pii_dec.tier == ModelTier.LOCAL_PRIVACY_SLM
    assert pii_dec.privacy_triggered is True
    assert "cn_resident_id" in pii_dec.pii_categories
    assert "bank_card_number" in pii_dec.pii_categories

    # Prefix KV-Cache byte-constant system prompt verification across distinct turns
    m1 = router.build_kv_cache_messages("Query A", memory_hits=[], abstained=True, artifacts=[])
    m2 = router.build_kv_cache_messages("Query B", memory_hits=[], abstained=False, artifacts=[])
    assert m1[0]["content"] == IMMUTABLE_SYSTEM_PROMPT
    assert m1[0]["content"] == m2[0]["content"]
