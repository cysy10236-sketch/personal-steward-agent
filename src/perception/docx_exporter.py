"""Generates publication-grade Word (.docx) reimbursement reports with embedded receipt photos."""

from __future__ import annotations

from datetime import datetime
import io
from pathlib import Path
from typing import Any

from docx import Document
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn
from docx.shared import Cm, Pt, RGBColor
import pymupdf as fitz
from PIL import Image, ImageOps
from pydantic import BaseModel, Field

from .artifact_registry import ArtifactRegistry


class ExpenseLineItem(BaseModel):
    date: str = Field(default="", description="Expense date YYYY-MM-DD")
    category: str = Field(default="差旅费", description="Category: 餐饮/交通/住宿/办公/其他")
    merchant_or_desc: str = Field(description="Merchant name and business purpose description")
    payment_channel: str = Field(default="微信/支付宝", description="微信支付/支付宝/纸质小票/酒店水单")
    gross_amount_cny: float = Field(ge=0.0, description="Total amount shown on receipt")
    excluded_personal_cny: float = Field(default=0.0, ge=0.0, description="Deducted non-reimbursable personal amount")
    reimbursable_cny: float = Field(ge=0.0, description="Net reimbursable amount in CNY")
    artifact_handle: str = Field(default="", description="Source artifact:// handle for embedding photo")
    notes: str = Field(default="", description="Optional audit remarks or exclusion explanation")


class ReimbursementDocxResult(BaseModel):
    saved_file: str
    full_path: str
    bytes_written: int
    item_count: int
    gross_total_cny: float
    excluded_total_cny: float
    net_reimbursable_cny: float
    embedded_photos: int
    duplicate_warnings: list[str] = Field(default_factory=list)


def detect_duplicate_claims(items: list[ExpenseLineItem]) -> list[str]:
    """Detect likely duplicate claims (e.g., WeChat screenshot + paper receipt of the same payment)."""
    warnings: list[str] = []
    seen: list[tuple[int, ExpenseLineItem]] = []
    for idx, it in enumerate(items, start=1):
        for prev_idx, prev in seen:
            same_amt = abs(it.gross_amount_cny - prev.gross_amount_cny) <= 0.01 and it.gross_amount_cny > 0
            same_date = (not it.date) or (not prev.date) or (it.date == prev.date)
            diff_file = (not it.artifact_handle) or (it.artifact_handle != prev.artifact_handle)
            if same_amt and same_date and diff_file:
                warnings.append(
                    f"疑似同笔重复票据：第 {prev_idx} 项（{prev.payment_channel} / {prev.merchant_or_desc}，¥{prev.gross_amount_cny:.2f}）"
                    f"与第 {idx} 项（{it.payment_channel} / {it.merchant_or_desc}，¥{it.gross_amount_cny:.2f}）日期与金额一致，请财务/报销人复核避免重复报销。"
                )
        seen.append((idx, it))
    return warnings


def _set_cell_bg(cell: Any, hex_color: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    tc_pr.append(parse_xml(f'<w:shd {nsdecls("w")} w:fill="{hex_color}"/>'))


def _set_cell_margins(cell: Any, top: int = 90, bottom: int = 90, left: int = 120, right: int = 120) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    tc_mar = OxmlElement("w:tcMar")
    for side, val in (("top", top), ("bottom", bottom), ("left", left), ("right", right)):
        node = OxmlElement(f"w:{side}")
        node.set(qn("w:w"), str(val))
        node.set(qn("w:type"), "dxa")
        tc_mar.append(node)
    tc_pr.append(tc_mar)


def _set_table_borders(table: Any, color: str = "E2E8F0") -> None:
    tbl_pr = table._tbl.tblPr
    tbl_pr.append(
        parse_xml(
            f'<w:tblBorders {nsdecls("w")}>'
            f'<w:top w:val="single" w:sz="4" w:space="0" w:color="{color}"/>'
            f'<w:bottom w:val="single" w:sz="6" w:space="0" w:color="94A3B8"/>'
            f'<w:insideH w:val="single" w:sz="4" w:space="0" w:color="{color}"/>'
            f'<w:left w:val="none"/><w:right w:val="none"/><w:insideV w:val="none"/>'
            f"</w:tblBorders>"
        )
    )


def _style_run(run: Any, *, size_pt: float = 9.5, bold: bool = False, color: tuple[int, int, int] = (30, 41, 59)) -> None:
    run.font.name = "Microsoft YaHei"
    run._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
    run.font.size = Pt(size_pt)
    run.bold = bold
    run.font.color.rgb = RGBColor(*color)


def _prepare_embedded_png(source_path: Path, modality: str) -> tuple[io.BytesIO, float]:
    """Load receipt photo (EXIF-corrected) or PDF page 1 and compute clean fit width in cm."""
    if modality == "pdf":
        doc = fitz.open(source_path)
        try:
            pix = doc[0].get_pixmap(dpi=150, alpha=False)
            pil_img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        finally:
            doc.close()
    else:
        pil_img = ImageOps.exif_transpose(Image.open(source_path)).convert("RGB")

    w, h = pil_img.size
    max_w_cm, max_h_cm = 13.8, 12.8
    aspect = w / max(1, h)
    width_cm = min(max_w_cm, max_h_cm * aspect)
    width_cm = max(5.5, round(width_cm, 2))

    # Downsample very large mobile photos (>1600px) so .docx stays lightweight yet crisp
    if max(w, h) > 1600:
        pil_img.thumbnail((1600, 1600), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG", optimize=True)
    buf.seek(0)
    return buf, width_cm


def build_reimbursement_docx(
    *,
    output_path: Path,
    title: str,
    applicant: str,
    project_code: str,
    trip_summary: str,
    items: list[ExpenseLineItem],
    artifact_registry: ArtifactRegistry,
    extra_warnings: list[str] | None = None,
    include_all_registered_handles: list[str] | None = None,
) -> ReimbursementDocxResult:
    """Render a minimalist Apple/Corporate-styled .docx reimbursement report with embedded photos."""
    doc = Document()
    for sec in doc.sections:
        sec.top_margin = Cm(2.0)
        sec.bottom_margin = Cm(2.0)
        sec.left_margin = Cm(2.0)
        sec.right_margin = Cm(2.0)

    normal = doc.styles["Normal"]
    normal.font.name = "Microsoft YaHei"
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
    normal.font.size = Pt(10)

    gross_total = round(sum(i.gross_amount_cny for i in items), 2)
    excluded_total = round(sum(i.excluded_personal_cny for i in items), 2)
    net_total = round(sum(i.reimbursable_cny for i in items), 2)
    dup_warnings = detect_duplicate_claims(items)
    all_warnings = list(dict.fromkeys(dup_warnings + (extra_warnings or [])))

    # 1. Header Banner
    p_kicker = doc.add_paragraph()
    p_kicker.paragraph_format.space_after = Pt(2)
    _style_run(p_kicker.add_run("TRIPLEDGER STEWARD · 智能差旅报销核验单"), size_pt=8.5, bold=True, color=(37, 99, 235))

    p_title = doc.add_paragraph()
    p_title.paragraph_format.space_after = Pt(6)
    _style_run(p_title.add_run(title), size_pt=18, bold=True, color=(15, 23, 42))

    if trip_summary:
        p_sub = doc.add_paragraph()
        p_sub.paragraph_format.space_after = Pt(10)
        _style_run(p_sub.add_run(trip_summary), size_pt=9.5, color=(71, 85, 105))

    # 2. Metadata & KPI Card Grid
    meta_tbl = doc.add_table(rows=2, cols=4)
    meta_tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    _set_table_borders(meta_tbl, "E2E8F0")
    meta_pairs = [
        ("报销人 / 带队人", applicant or "未填写", "归属项目代码", project_code or "常规差旅"),
        ("生成日期", datetime.now().strftime("%Y-%m-%d %H:%M"), "核定可报销总额", f"¥ {net_total:,.2f} CNY"),
    ]
    for r_idx, (k1, v1, k2, v2) in enumerate(meta_pairs):
        row_cells = meta_tbl.rows[r_idx].cells
        vals = [(k1, True, (100, 116, 139)), (v1, r_idx == 1 and False, (15, 23, 42)), (k2, True, (100, 116, 139)), (v2, r_idx == 1, (37, 99, 235) if r_idx == 1 else (15, 23, 42))]
        for c_idx, (txt, is_b, col) in enumerate(vals):
            cell = row_cells[c_idx]
            _set_cell_bg(cell, "F8FAFC" if c_idx in (0, 2) else "FFFFFF")
            _set_cell_margins(cell, top=80, bottom=80, left=110, right=110)
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            p = cell.paragraphs[0]
            p.paragraph_format.space_after = Pt(0)
            _style_run(p.add_run(txt), size_pt=9.5 if not (r_idx == 1 and c_idx == 3) else 11.5, bold=is_b, color=col)

    doc.add_paragraph().paragraph_format.space_after = Pt(4)

    # 3. Audit / Duplicate Warning Callout
    box_tbl = doc.add_table(rows=1, cols=1)
    box_tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    box_cell = box_tbl.rows[0].cells[0]
    _set_cell_margins(box_cell, top=110, bottom=110, left=150, right=150)
    if all_warnings:
        _set_cell_bg(box_cell, "FFFBEB")
        _set_table_borders(box_tbl, "F59E0B")
        p_w = box_cell.paragraphs[0]
        p_w.paragraph_format.space_after = Pt(3)
        _style_run(p_w.add_run("核验预警与复核提示（请报销人/财务确认）"), size_pt=9.5, bold=True, color=(180, 83, 9))
        for w_txt in all_warnings:
            pw_i = box_cell.add_paragraph()
            pw_i.paragraph_format.space_after = Pt(2)
            _style_run(pw_i.add_run(f"• {w_txt}"), size_pt=8.8, color=(120, 53, 15))
    else:
        _set_cell_bg(box_cell, "F0FDF4")
        _set_table_borders(box_tbl, "86EFAC")
        p_ok = box_cell.paragraphs[0]
        p_ok.paragraph_format.space_after = Pt(0)
        _style_run(
            p_ok.add_run(
                f"✓ 自动算术核验通过：共 {len(items)} 笔消费，票面总额 ¥{gross_total:,.2f}，"
                f"已剔除个人自费 ¥{excluded_total:,.2f}，净可报销 ¥{net_total:,.2f}，未发现同日同额重复票据。"
            ),
            size_pt=9.0,
            bold=True,
            color=(22, 101, 52),
        )

    # 4. Detailed Expense Table
    h1 = doc.add_paragraph()
    h1.paragraph_format.space_before = Pt(12)
    h1.paragraph_format.space_after = Pt(6)
    _style_run(h1.add_run("一、费用报销明细汇总表"), size_pt=12.5, bold=True, color=(15, 23, 42))

    headers = ["#", "日期", "类目", "商户与用途说明", "凭证渠道", "票面(¥)", "剔除(¥)", "报销额(¥)"]
    col_widths = [Cm(0.8), Cm(2.2), Cm(1.5), Cm(5.4), Cm(2.0), Cm(1.7), Cm(1.6), Cm(1.8)]
    exp_tbl = doc.add_table(rows=len(items) + 2, cols=len(headers))
    exp_tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    _set_table_borders(exp_tbl, "E2E8F0")

    for c_i, h_txt in enumerate(headers):
        cell = exp_tbl.rows[0].cells[c_i]
        cell.width = col_widths[c_i]
        _set_cell_bg(cell, "0F172A")
        _set_cell_margins(cell, top=90, bottom=90, left=70, right=70)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.RIGHT if c_i >= 5 else WD_ALIGN_PARAGRAPH.LEFT
        p.paragraph_format.space_after = Pt(0)
        _style_run(p.add_run(h_txt), size_pt=8.5, bold=True, color=(255, 255, 255))

    for r_i, it in enumerate(items, start=1):
        desc_full = it.merchant_or_desc + (f"\n（备注：{it.notes}）" if it.notes else "")
        row_vals = [
            str(r_i),
            it.date or "—",
            it.category,
            desc_full,
            it.payment_channel,
            f"{it.gross_amount_cny:,.2f}",
            f"{it.excluded_personal_cny:,.2f}" if it.excluded_personal_cny > 0 else "0.00",
            f"{it.reimbursable_cny:,.2f}",
        ]
        bg = "FFFFFF" if r_i % 2 == 1 else "F8FAFC"
        for c_i, val in enumerate(row_vals):
            cell = exp_tbl.rows[r_i].cells[c_i]
            cell.width = col_widths[c_i]
            _set_cell_bg(cell, bg)
            _set_cell_margins(cell, top=75, bottom=75, left=70, right=70)
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.RIGHT if c_i >= 5 else WD_ALIGN_PARAGRAPH.LEFT
            p.paragraph_format.space_after = Pt(0)
            color = (185, 28, 28) if (c_i == 6 and it.excluded_personal_cny > 0) else (15, 23, 42)
            _style_run(p.add_run(val), size_pt=8.5, bold=(c_i == 7), color=color)

    # Total row
    tot_row = exp_tbl.rows[len(items) + 1].cells
    tot_vals = ["合计", "", "", f"共 {len(items)} 笔报销项", "", f"{gross_total:,.2f}", f"{excluded_total:,.2f}", f"¥{net_total:,.2f}"]
    for c_i, val in enumerate(tot_vals):
        cell = tot_row[c_i]
        cell.width = col_widths[c_i]
        _set_cell_bg(cell, "EFF6FF")
        _set_cell_margins(cell, top=95, bottom=95, left=70, right=70)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.RIGHT if c_i >= 5 else WD_ALIGN_PARAGRAPH.LEFT
        p.paragraph_format.space_after = Pt(0)
        _style_run(p.add_run(val), size_pt=9.0, bold=True, color=(30, 64, 175) if c_i == 7 else (15, 23, 42))

    # 5. Embedded Receipt Photos Appendix
    ordered_handles: list[tuple[str, str]] = []
    seen_handles: set[str] = set()
    for idx, it in enumerate(items, start=1):
        h = it.artifact_handle.strip()
        if h and h not in seen_handles:
            seen_handles.add(h)
            cap = f"图 {len(ordered_handles) + 1}｜序号 #{idx} · [{it.payment_channel}] {it.date or ''} {it.merchant_or_desc}（实报 ¥{it.reimbursable_cny:,.2f}）"
            ordered_handles.append((h, cap))
    for h in (include_all_registered_handles or []):
        if h and h not in seen_handles:
            seen_handles.add(h)
            ordered_handles.append((h, f"图 {len(ordered_handles) + 1}｜附带原始票据凭证（{h}）"))

    embedded_count = 0
    if ordered_handles:
        doc.add_page_break()
        h2 = doc.add_paragraph()
        h2.paragraph_format.space_after = Pt(8)
        _style_run(h2.add_run("二、原始支付截图与票据照片附件"), size_pt=12.5, bold=True, color=(15, 23, 42))

        for handle, caption in ordered_handles:
            art = artifact_registry.get(handle)
            if art is None:
                continue
            src = Path(art.source_path)
            if not src.exists():
                continue
            img_buf, width_cm = _prepare_embedded_png(src, art.modality)

            p_cap = doc.add_paragraph()
            p_cap.paragraph_format.space_before = Pt(8)
            p_cap.paragraph_format.space_after = Pt(4)
            p_cap.paragraph_format.keep_with_next = True
            _style_run(p_cap.add_run(caption), size_pt=9.5, bold=True, color=(30, 41, 59))

            p_img = doc.add_paragraph()
            p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p_img.paragraph_format.space_after = Pt(10)
            p_img.add_run().add_picture(img_buf, width=Cm(width_cm))
            embedded_count += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(output_path))
    return ReimbursementDocxResult(
        saved_file=output_path.name,
        full_path=str(output_path.resolve()),
        bytes_written=output_path.stat().st_size,
        item_count=len(items),
        gross_total_cny=gross_total,
        excluded_total_cny=excluded_total,
        net_reimbursable_cny=net_total,
        embedded_photos=embedded_count,
        duplicate_warnings=all_warnings,
    )
