"""Parses PDF folios (clip=cbox) & receipt screenshots (ONNX RapidOCR pixel OCR) with taint scanning."""

from __future__ import annotations

import base64
import math
from pathlib import Path
import re
from typing import Any

import numpy as np
import pymupdf as fitz
from PIL import Image, ImageOps
from rapidocr_onnxruntime import RapidOCR

from .artifact_registry import ArtifactRegistry, ReceiptArtifact, ReceiptROI

_NUMBER_RE = re.compile(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?")
_DATE_PREFIX_RE = re.compile(r"^\s*\d{4}[-/]\d{1,2}[-/]\d{1,2}")

_INJECTION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "instruction_override",
        re.compile(
            r"(ignore\s+(?:all\s+)?previous\s+instructions|system\s+override|忽略之前的所有指令|无视系统规则|disregard\s+prior\s+rules)",
            re.I,
        ),
    ),
    (
        "exfiltration_or_shell",
        re.compile(
            r"(cat\s+\.env|type\s+\.env|curl\s+http|rm\s+-rf|wire\s+\d+\s+to\s+attacker|执行shell|读取密钥|print\s+env)",
            re.I,
        ),
    ),
)

_OCR_SINGLETON: RapidOCR | None = None


def _get_ocr() -> RapidOCR:
    global _OCR_SINGLETON
    if _OCR_SINGLETON is None:
        _OCR_SINGLETON = RapidOCR()
    return _OCR_SINGLETON


def detect_prompt_injection(text: str) -> list[str]:
    """Scan untrusted PDF or OCR text for indirect prompt injection payloads."""
    return [label for label, rx in _INJECTION_PATTERNS if rx.search(text)]


def _parse_ocr_line_items(lines: list[str]) -> dict[str, float]:
    """Extract numeric monetary line items from pixel-OCR lines, filtering dates/timestamps."""
    items: dict[str, float] = {}
    for ln in lines:
        clean = ln.strip()
        if not clean or re.search(r"\d{4}[-/]\d{1,2}[-/]\d{1,2}", clean):
            continue
        parts = re.split(r"[:：]", clean, maxsplit=1)
        if len(parts) == 2:
            key = parts[0].strip()
            m = _NUMBER_RE.search(parts[1])
            if key and m:
                items[key] = float(m.group(0).replace(",", ""))
    return items


class ReceiptMultimodalParser:
    """Converts PDF folios & raw receipt images into compact handles + addressable ROIs."""

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
            w0, h0, roi_seq = 595, 842, 1

            for p_idx in range(page_count):
                page = doc[p_idx]
                pw, ph = max(1, int(page.rect.width)), max(1, int(page.rect.height))
                if p_idx == 0:
                    w0, h0 = pw, ph
                p_txt = page.get_text("text").strip()
                if len(p_txt) < 20:
                    # Scanned / image-only PDF page fallback: render at 135 DPI (capped 1280px) and run local ONNX RapidOCR
                    pix = page.get_pixmap(dpi=135, alpha=False)
                    pil_page = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                    if max(pil_page.size) > 1280:
                        pil_page.thumbnail((1280, 1280), Image.Resampling.LANCZOS)
                    ocr_res, _ = _get_ocr()(np.asarray(pil_page))
                    ocr_lines = [str(it[1]).strip() for it in (ocr_res or []) if str(it[1]).strip()]
                    if ocr_lines:
                        p_txt = "\n".join(ocr_lines)
                        rois.append(
                            ReceiptROI(
                                roi_id=f"roi_{roi_seq}",
                                page_index=p_idx,
                                norm_box=(0.0, 0.0, 1.0, 1.0),
                                region_type="invoice_widget",
                                label=f"p{p_idx + 1}_scanned_pdf_ocr",
                                extracted_text=p_txt,
                                line_items=_parse_ocr_line_items(ocr_lines),
                            )
                        )
                        roi_seq += 1
                page_texts.append(f"[Page {p_idx + 1}]\n{p_txt}")

                tabs = page.find_tables()
                for tbl in (list(tabs.tables) if tabs else []):
                    bx0, by0, bx1, by1 = tbl.bbox
                    grid: list[list[str]] = []
                    for row_obj in (getattr(tbl, "rows", None) or []):
                        cells = [
                            page.get_text("text", clip=cbox).replace("\n", " ").strip() if cbox is not None else ""
                            for cbox in row_obj.cells
                        ]
                        if any(cells):
                            grid.append(cells)
                    if not grid:
                        grid = [[(c.replace("\n", " ").strip() if c else "") for c in r] for r in tbl.extract()]

                    line_items: dict[str, list[float] | float] = {}
                    for r_cells in grid[1:]:
                        if r_cells and r_cells[0]:
                            nums = [
                                float(m.group(0).replace(",", ""))
                                for cell in r_cells[1:]
                                if not _DATE_PREFIX_RE.match(cell) and (m := _NUMBER_RE.search(cell))
                            ]
                            if nums:
                                line_items[r_cells[0]] = nums[0] if len(nums) == 1 else nums

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
                            extracted_text="\n".join(" | ".join(r) for r in grid),
                            line_items=line_items,
                        )
                    )
                    roi_seq += 1

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
        return self.registry.register(
            ReceiptArtifact(
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
                structured_kv={"pages": page_count, "filename": path.name},
            )
        )

    def parse_image_receipt(self, path: Path) -> ReceiptArtifact:
        """Pure pixel perception via ONNX RapidOCR (zero PNG metadata shortcuts)."""
        sha = self.registry.file_sha256(path)
        art_id = f"rcpt_img_{sha[:8]}"
        cached = self.registry.get(art_id)
        if cached is not None:
            return cached

        rgb = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
        w, h = rgb.size
        # Downscale oversized phone screenshots only for RapidOCR speed (keep original full-res file on disk)
        ocr_img = rgb
        scale_x, scale_y = 1.0, 1.0
        if max(w, h) > 1200:
            ocr_img = rgb.copy()
            ocr_img.thumbnail((1200, 1200), Image.Resampling.LANCZOS)
            scale_x, scale_y = w / max(1, ocr_img.width), h / max(1, ocr_img.height)
        np_img = np.asarray(ocr_img)

        ocr_res, _ = _get_ocr()(np_img)
        rois: list[ReceiptROI] = []
        ocr_lines: list[str] = []
        xs: list[float] = []
        ys: list[float] = []
        confs: list[float] = []

        for item in (ocr_res or []):
            quad, txt, conf = item[0], str(item[1]).strip(), float(item[2])
            if not txt:
                continue
            ocr_lines.append(txt)
            confs.append(conf)
            for pt in quad:
                xs.append(float(pt[0]) * scale_x)
                ys.append(float(pt[1]) * scale_y)

        if ocr_lines:
            x0, x1 = max(0.0, min(xs) - 12.0), min(float(w), max(xs) + 12.0)
            y0, y1 = max(0.0, min(ys) - 12.0), min(float(h), max(ys) + 12.0)
            line_items = _parse_ocr_line_items(ocr_lines)
            rois.append(
                ReceiptROI(
                    roi_id="roi_1",
                    page_index=0,
                    norm_box=(round(x0 / w, 4), round(y0 / h, 4), round(x1 / w, 4), round(y1 / h, 4)),
                    region_type="invoice_widget",
                    label="ocr_detected_receipt_panel",
                    extracted_text="\n".join(ocr_lines),
                    line_items=line_items,
                )
            )
        else:
            # 2D spatial gradient fallback when image has graphical boxes without legible OCR text
            gray = np.asarray(rgb.convert("L"), dtype=np.float32)
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
                    label="gradient_fallback_panel",
                    extracted_text="",
                )
            )

        full_txt = "\n".join(ocr_lines).strip()
        taints = detect_prompt_injection(full_txt)
        mean_conf = round(float(np.mean(confs)), 3) if confs else 0.0
        roi_brief = ", ".join(f"{r.roi_id}({r.label})" for r in rois[:4])
        header = (
            f"Image Receipt '{path.name}' ({w}x{h}, ocr_conf={mean_conf}) | tainted={bool(taints)} "
            f"| rois=[{roi_brief}] (call inspect_receipt_roi for pixel crop & itemized amounts)"
        )
        return self.registry.register(
            ReceiptArtifact(
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
                structured_kv={"size": f"{w}x{h}", "ocr_mean_conf": mean_conf, "ocr_lines": len(ocr_lines)},
            )
        )
