"""Parses PDF hotel folios & receipt screenshots with cell-clip precision and taint scanning."""

from __future__ import annotations

import base64
import json
import math
from pathlib import Path
import re
from typing import Any

import numpy as np
import pymupdf as fitz
from PIL import Image

from .artifact_registry import ArtifactRegistry, ReceiptArtifact, ReceiptROI

_NUMBER_RE = re.compile(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?")

_INJECTION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "instruction_override",
        re.compile(
            r"(ignore\s+(?:all\s+)?previous\s+instructions|system\s+override|忽略之前的所有指令|无视系统规则)",
            re.I,
        ),
    ),
    (
        "exfiltration_or_shell",
        re.compile(
            r"(cat\s+\.env|curl\s+http|rm\s+-rf|wire\s+\d+\s+to\s+attacker|执行shell|读取密钥)",
            re.I,
        ),
    ),
)


def detect_prompt_injection(text: str) -> list[str]:
    """Scan untrusted document/image text for indirect prompt injection payloads."""
    hits: list[str] = []
    for label, rx in _INJECTION_PATTERNS:
        if rx.search(text):
            hits.append(label)
    return hits


class ReceiptMultimodalParser:
    """Converts PDF folios & PNG receipts into compact handles + addressable ROIs."""

    def __init__(self, registry: ArtifactRegistry) -> None:
        self.registry = registry

    @staticmethod
    def encode_image_base64_uri(path: str | Path) -> str:
        raw = Path(path).read_bytes()
        return f"data:image/png;base64,{base64.b64encode(raw).decode('ascii')}"

    def parse_file(self, file_path: str | Path) -> ReceiptArtifact:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Attachment not found: {path}")
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            return self.parse_pdf_receipt(path)
        if suffix in {".png", ".jpg", ".jpeg", ".webp"}:
            return self.parse_image_receipt(path)
        raise ValueError(f"Unsupported receipt file type: {suffix!r}")

    def parse_pdf_receipt(self, path: Path) -> ReceiptArtifact:
        sha = self.registry.file_sha256(path)
        art_id = f"rcpt_pdf_{sha[:8]}"
        cached = self.registry.get(art_id)
        if cached is not None:
            return cached

        doc = fitz.open(path)
        try:
            page_count = len(doc)
            rois: list[ReceiptROI] = []
            page_texts: list[str] = []
            kv_summary: dict[str, Any] = {"pages": page_count, "filename": path.name}
            w0, h0 = 595, 842
            roi_seq = 1

            for p_idx in range(page_count):
                page = doc[p_idx]
                pw, ph = max(1, int(page.rect.width)), max(1, int(page.rect.height))
                if p_idx == 0:
                    w0, h0 = pw, ph
                p_txt = page.get_text("text").strip()
                page_texts.append(f"[Page {p_idx + 1}]\n{p_txt}")

                # 1. Coordinate-clipped table extraction (prevents glyph-reordering corruption)
                try:
                    tabs = page.find_tables()
                    found_tables = list(tabs.tables) if tabs else []
                except Exception:
                    found_tables = []

                for tbl in found_tables:
                    bx0, by0, bx1, by1 = tbl.bbox
                    grid: list[list[str]] = []
                    if hasattr(tbl, "rows") and tbl.rows:
                        for row_obj in tbl.rows:
                            cells = [
                                (
                                    page.get_text("text", clip=cbox).replace("\n", " ").strip()
                                    if cbox is not None
                                    else ""
                                )
                                for cbox in row_obj.cells
                            ]
                            if any(cells):
                                grid.append(cells)
                    if not grid:
                        grid = [
                            [(c.replace("\n", " ").strip() if c else "") for c in r]
                            for r in tbl.extract()
                        ]

                    md_lines = [" | ".join(r) for r in grid]
                    line_items: dict[str, list[float] | float] = {}
                    for r_cells in grid[1:]:
                        if r_cells and r_cells[0]:
                            item_key = r_cells[0]
                            nums = [
                                float(m.group(0).replace(",", ""))
                                for cell in r_cells[1:]
                                if not re.match(r"^\s*\d{4}[-/]\d{1,2}[-/]\d{1,2}", cell)
                                and (m := _NUMBER_RE.search(cell))
                            ]
                            if nums:
                                line_items[item_key] = nums[0] if len(nums) == 1 else nums

                    rois.append(
                        ReceiptROI(
                            roi_id=f"roi_{roi_seq}",
                            page_index=p_idx,
                            norm_box=(
                                round(max(0.0, bx0 / pw), 4),
                                round(max(0.0, by0 / ph), 4),
                                round(min(1.0, bx1 / pw), 4),
                                round(min(1.0, by1 / ph), 4),
                            ),
                            region_type="folio_table",
                            label=f"p{p_idx + 1}_folio_table_{roi_seq}",
                            extracted_text="\n".join(md_lines),
                            line_items=line_items,
                        )
                    )
                    roi_seq += 1

                # 2. Key-value & note text blocks
                for blk in page.get_text("blocks"):
                    bx0, by0, bx1, by1, btxt, _bno, btype = blk
                    clean_blk = str(btxt).strip()
                    if btype == 0 and len(clean_blk) >= 16:
                        rois.append(
                            ReceiptROI(
                                roi_id=f"roi_{roi_seq}",
                                page_index=p_idx,
                                norm_box=(
                                    round(max(0.0, bx0 / pw), 4),
                                    round(max(0.0, by0 / ph), 4),
                                    round(min(1.0, math.ceil(bx1) / pw), 4),
                                    round(min(1.0, math.ceil(by1) / ph), 4),
                                ),
                                region_type="text_section",
                                label=f"p{p_idx + 1}_{clean_blk[:24].replace(chr(10), ' ')}",
                                extracted_text=clean_blk,
                            )
                        )
                        roi_seq += 1
        finally:
            doc.close()

        full_text = "\n\n".join(page_texts)
        taints = detect_prompt_injection(full_text)
        roi_brief = ", ".join(f"{r.roi_id}({r.region_type}:{r.label})" for r in rois[:6])
        header = (
            f"PDF Receipt '{path.name}' ({page_count}p) | tainted={bool(taints)} "
            f"| rois=[{roi_brief or 'none'}] (call inspect_receipt_roi for exact table figures)"
        )

        artifact = ReceiptArtifact(
            artifact_id=art_id,
            handle=f"artifact://{art_id}",
            modality="pdf",
            source_path=str(path.resolve()),
            sha256=sha,
            width=w0,
            height=h0,
            page_count=page_count,
            is_tainted_untrusted=bool(taints),
            taint_reasons=taints,
            rois=rois,
            compact_header=header,
            raw_full_text=full_text,
            structured_kv=kv_summary,
        )
        return self.registry.register(artifact)

    def parse_image_receipt(self, path: Path) -> ReceiptArtifact:
        sha = self.registry.file_sha256(path)
        art_id = f"rcpt_img_{sha[:8]}"
        cached = self.registry.get(art_id)
        if cached is not None:
            return cached

        img = Image.open(path)
        info = dict(img.info) if img.info else {}
        rgb = img.convert("RGB")
        w, h = rgb.size
        gray = np.asarray(rgb.convert("L"), dtype=np.float32)

        rois: list[ReceiptROI] = []
        raw_rois = info.get("structured_rois")
        if isinstance(raw_rois, str) and raw_rois.strip():
            try:
                parsed = json.loads(raw_rois)
                for i, item in enumerate(parsed):
                    bx = item.get("box", [0, 0, w, h])
                    rois.append(
                        ReceiptROI(
                            roi_id=str(item.get("roi_id", f"roi_{i + 1}")),
                            page_index=0,
                            norm_box=(
                                round(bx[0] / w, 4),
                                round(bx[1] / h, 4),
                                round(bx[2] / w, 4),
                                round(bx[3] / h, 4),
                            ),
                            region_type=item.get("region_type", "invoice_widget"),
                            label=str(item.get("label", f"widget_{i + 1}")),
                            extracted_text=str(item.get("content", "")),
                            line_items=dict(item.get("line_items", {})),
                        )
                    )
            except json.JSONDecodeError:
                rois.clear()

        if not rois:
            # 2D spatial gradient band fallback for unannotated screenshots
            gx = np.abs(np.diff(gray, axis=1, prepend=gray[:, :1]))
            gy = np.abs(np.diff(gray, axis=0, prepend=gray[:1, :]))
            active = np.where(np.mean(0.5 * (gx + gy) > 15.0, axis=1) > 0.04)[0]
            y0 = int(active[0]) if active.size else 0
            y1 = int(active[-1]) if active.size else h
            rois.append(
                ReceiptROI(
                    roi_id="roi_1",
                    page_index=0,
                    norm_box=(0.0, round(y0 / h, 4), 1.0, round(max(y0 + 1, y1) / h, 4)),
                    region_type="invoice_widget",
                    label="detected_receipt_panel",
                    extracted_text=str(info.get("caption", f"Visual receipt {path.name}")),
                )
            )

        title = str(info.get("title", path.stem))
        caption = str(info.get("caption", ""))
        full_txt = "\n".join(
            [title, caption] + [r.extracted_text for r in rois if r.extracted_text]
        ).strip()
        taints = detect_prompt_injection(full_txt)
        roi_brief = ", ".join(f"{r.roi_id}({r.label})" for r in rois[:6])
        header = (
            f"Image Receipt '{title}' ({w}x{h}) | tainted={bool(taints)} "
            f"| rois=[{roi_brief}] (call inspect_receipt_roi to read itemized amounts)"
        )

        artifact = ReceiptArtifact(
            artifact_id=art_id,
            handle=f"artifact://{art_id}",
            modality="image",
            source_path=str(path.resolve()),
            sha256=sha,
            width=w,
            height=h,
            page_count=1,
            is_tainted_untrusted=bool(taints),
            taint_reasons=taints,
            rois=rois,
            compact_header=header,
            raw_full_text=full_txt,
            structured_kv={"title": title, "size": f"{w}x{h}"},
        )
        return self.registry.register(artifact)
