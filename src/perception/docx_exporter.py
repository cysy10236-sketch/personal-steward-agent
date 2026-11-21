"""Generates publication-grade Word (.docx) reimbursement reports with multi-page PDF & photo embedding."""

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
    category: str = Field(default="差旅费", description="Category: 餐饮/交通/住宿/活动物资/办公/其他")
    merchant_or_desc: str = Field(description="Merchant name and business purpose description")
    payment_channel: str = Field(default="微信/支付宝", description="微信支付/支付宝/电子发票+小票+支付记录")
    gross_amount_cny: float = Field(ge=0.0, description="Total amount shown on receipt")
    excluded_personal_cny: float = Field(default=0.0, ge=0.0, description="Deducted non-reimbursable personal amount")
    reimbursable_cny: float = Field(ge=0.0, description="Net reimbursable amount in CNY")
    artifact_handle: str = Field(default="", description="Primary artifact:// handle for embedding photo")
    supporting_handles: list[str] = Field(default_factory=list, description="Additional linked voucher handles (三单合一)")
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


def cny_to_chinese_upper(amount: float) -> str:
    """Convert CNY amount to standard Chinese financial uppercase wording."""
    digits = "零壹贰叁肆伍陆柒捌玖"
    units = ["", "拾", "佰", "仟", "万", "拾", "佰", "仟", "亿"]
    cents = int(round(abs(amount) * 100))
    yuan, jiao, fen = cents // 100, (cents // 10) % 10, cents % 10
    if yuan == 0:
        yuan_str = "零"
    else:
        s = str(yuan)
        parts: list[str] = []
        zero_flag = False
        for i, ch in enumerate(s):
            d = int(ch)
            pos = len(s) - 1 - i
            if d == 0:
                if pos == 4:
                    parts.append("万")
                    zero_flag = False
                else:
                    zero_flag = True
            else:
                if zero_flag:
                    parts.append("零")
                    zero_flag = False
                parts.append(digits[d] + units[pos])
        yuan_str = "".join(parts)
    if jiao == 0 and fen == 0:
        return f"{yuan_str}圆整"
    tail = (f"{digits[jiao]}角" if jiao else "零") + (f"{digits[fen]}分" if fen else "整")
    return f"{yuan_str}圆{tail}"


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


def _set_cell_margins(cell: Any, top: int = 85, bottom: int = 85, left: int = 110, right: int = 110) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    tc_mar = OxmlElement("w:tcMar")
    for side, val in (("top", top), ("bottom", bottom), ("left", left), ("right", right)):
        node = OxmlElement(f"w:{side}")
        node.set(qn("w:w"), str(val))
        node.set(qn("w:type"), "dxa")
        tc_mar.append(node)
    tc_pr.append(tc_mar)


def _set_table_borders(table: Any, color: str = "E2E8F0", *, borderless: bool = False) -> None:
    tbl_pr = table._tbl.tblPr
    if borderless:
        tbl_pr.append(
            parse_xml(
                f'<w:tblBorders {nsdecls("w")}>'
                '<w:top w:val="none"/><w:bottom w:val="none"/><w:left w:val="none"/>'
                '<w:right w:val="none"/><w:insideH w:val="none"/><w:insideV w:val="none"/>'
                "</w:tblBorders>"
            )
        )
        return
    tbl_pr.append(
        parse_xml(
            f'<w:tblBorders {nsdecls("w")}>'
            f'<w:top w:val="single" w:sz="4" w:space="0" w:color="{color}"/>'
            f'<w:bottom w:val="single" w:sz="6" w:space="0" w:color="94A3B8"/>'
            f'<w:insideH w:val="single" w:sz="4" w:space="0" w:color="{color}"/>'
            '<w:left w:val="none"/><w:right w:val="none"/><w:insideV w:val="none"/>'
            "</w:tblBorders>"
        )
    )


def _style_run(run: Any, *, size_pt: float = 9.5, bold: bool = False, color: tuple[int, int, int] = (30, 41, 59)) -> None:
    run.font.name = "Microsoft YaHei"
    run._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
    run.font.size = Pt(size_pt)
    run.bold = bold
    run.font.color.rgb = RGBColor(*color)


def _prevent_row_split(row: Any, *, is_header: bool = False) -> None:
    tr_pr = row._tr.get_or_add_trPr()
    tr_pr.append(OxmlElement("w:cantSplit"))
    if is_header:
        tr_pr.append(OxmlElement("w:tblHeader"))


def _trim_white_margins(pil_img: Image.Image, *, threshold: int = 246, pad: int = 24) -> Image.Image:
    """Trim excessive blank white borders from scanned invoices/PDFs so content prints larger."""
    gray = pil_img.convert("L")
    mask = gray.point(lambda p: 255 if p < threshold else 0)
    bbox = mask.getbbox()
    if not bbox:
        return pil_img
    x0, y0, x1, y1 = bbox
    w, h = pil_img.size
    # Only crop if margins actually exceed 6% of width or height
    if (x1 - x0) > 0.94 * w and (y1 - y0) > 0.94 * h:
        return pil_img
    return pil_img.crop((max(0, x0 - pad), max(0, y0 - pad), min(w, x1 + pad), min(h, y1 + pad)))


def _encode_pil_fit(
    pil_img: Image.Image,
    *,
    max_w_cm: float = 15.8,
    max_h_cm: float = 20.5,
    trim_margins: bool = False,
    as_jpeg: bool = False,
) -> tuple[io.BytesIO, float, float]:
    if trim_margins:
        pil_img = _trim_white_margins(pil_img)
    w, h = pil_img.size
    aspect = w / max(1, h)
    width_cm = max(6.2, round(min(max_w_cm, max_h_cm * aspect), 2))
    if max(w, h) > 1800:
        pil_img = pil_img.copy()
        pil_img.thumbnail((1800, 1800), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    if as_jpeg:
        pil_img.convert("RGB").save(buf, format="JPEG", quality=88, optimize=True)
    else:
        pil_img.save(buf, format="PNG", optimize=True)
    buf.seek(0)
    return buf, width_cm, aspect


def _prepare_embedded_pages(source_path: Path, modality: str, *, max_pages: int = 4) -> list[tuple[io.BytesIO, float, float, str]]:
    """Render all PDF pages (up to max_pages at 180 DPI) or EXIF-corrected image."""
    pages: list[tuple[io.BytesIO, float, float, str]] = []
    if modality == "pdf":
        doc = fitz.open(source_path)
        try:
            total = min(len(doc), max_pages)
            for idx in range(total):
                pix = doc[idx].get_pixmap(dpi=180, alpha=False)
                pil_img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                buf, w_cm, aspect = _encode_pil_fit(pil_img, trim_margins=True, as_jpeg=True)
                suffix = f"（第 {idx + 1}/{len(doc)} 页）" if len(doc) > 1 else ""
                pages.append((buf, w_cm, aspect, suffix))
        finally:
            doc.close()
    else:
        pil_img = ImageOps.exif_transpose(Image.open(source_path)).convert("RGB")
        use_jpg = source_path.suffix.lower() in {".jpg", ".jpeg"}
        buf, w_cm, aspect = _encode_pil_fit(pil_img, as_jpeg=use_jpg)
        pages.append((buf, w_cm, aspect, ""))
    return pages


def _add_styled_table(
    doc: Document,
    headers: list[str],
    rows: list[list[str]],
    *,
    col_widths: list[Cm] | None = None,
    right_align_cols: set[int] | None = None,
) -> Any:
    tbl = doc.add_table(rows=len(rows) + 1, cols=len(headers))
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    _set_table_borders(tbl, "E2E8F0")
    r_cols = right_align_cols or set()
    _prevent_row_split(tbl.rows[0], is_header=True)
    for c_i, h_txt in enumerate(headers):
        cell = tbl.rows[0].cells[c_i]
        if col_widths and c_i < len(col_widths):
            cell.width = col_widths[c_i]
        _set_cell_bg(cell, "0F172A")
        _set_cell_margins(cell, top=65, bottom=65, left=65, right=65)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.RIGHT if c_i in r_cols else WD_ALIGN_PARAGRAPH.LEFT
        p.paragraph_format.space_after = Pt(0)
        _style_run(p.add_run(h_txt), size_pt=8.2, bold=True, color=(255, 255, 255))
    for r_i, r_vals in enumerate(rows, start=1):
        _prevent_row_split(tbl.rows[r_i])
        bg = "FFFFFF" if r_i % 2 == 1 else "F8FAFC"
        for c_i, val in enumerate(r_vals):
            cell = tbl.rows[r_i].cells[c_i]
            if col_widths and c_i < len(col_widths):
                cell.width = col_widths[c_i]
            _set_cell_bg(cell, bg)
            _set_cell_margins(cell, top=55, bottom=55, left=65, right=65)
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.RIGHT if c_i in r_cols else WD_ALIGN_PARAGRAPH.LEFT
            p.paragraph_format.space_after = Pt(0)
            _style_run(p.add_run(str(val)), size_pt=8.2, color=(15, 23, 42))
    return tbl


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
    kicker_text: str = "TRIPLEDGER STEWARD · 智能差旅报销核验单",
    budget_cap_cny: float | None = None,
    extra_sections: list[dict[str, Any]] | None = None,
    voucher_overrides: list[tuple[Path, str]] | None = None,
    activity_photos: list[tuple[Path, str]] | None = None,
    activity_photo_note: str = "",
    include_signoff: bool = True,
    audit_seal_sha256: str = "",
) -> ReimbursementDocxResult:
    """Render a minimalist Apple/Corporate-styled A4 .docx reimbursement report with embedded photos."""
    doc = Document()
    for sec in doc.sections:
        sec.top_margin = Cm(1.5)
        sec.bottom_margin = Cm(1.5)
        sec.left_margin = Cm(1.7)
        sec.right_margin = Cm(1.7)
        if audit_seal_sha256:
            fp = sec.footer.paragraphs[0]
            fp.alignment = WD_ALIGN_PARAGRAPH.RIGHT
            _style_run(
                fp.add_run(f"TripLedger Steward · AST 确定性核验封签 SHA-256: {audit_seal_sha256[:16]}"),
                size_pt=7.5,
                color=(148, 163, 184),
            )

    normal = doc.styles["Normal"]
    normal.font.name = "Microsoft YaHei"
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
    normal.font.size = Pt(9.5)

    gross_total = round(sum(i.gross_amount_cny for i in items), 2)
    excluded_total = round(sum(i.excluded_personal_cny for i in items), 2)
    net_total = round(sum(i.reimbursable_cny for i in items), 2)
    dup_warnings = detect_duplicate_claims(items)
    all_warnings = list(dict.fromkeys(dup_warnings + (extra_warnings or [])))

    # 1. Header Banner
    p_kicker = doc.add_paragraph()
    p_kicker.paragraph_format.space_after = Pt(2)
    _style_run(p_kicker.add_run(kicker_text), size_pt=8.5, bold=True, color=(0, 113, 227))

    p_title = doc.add_paragraph()
    p_title.paragraph_format.space_after = Pt(5)
    _style_run(p_title.add_run(title), size_pt=17.5, bold=True, color=(15, 23, 42))

    if trip_summary:
        p_sub = doc.add_paragraph()
        p_sub.paragraph_format.space_after = Pt(9)
        _style_run(p_sub.add_run(trip_summary), size_pt=9.2, color=(71, 85, 105))

    # 2. Metadata & KPI Card Grid
    budget_info = (
        f"预算 ¥{budget_cap_cny:,.2f}（结余 ¥{budget_cap_cny - net_total:,.2f}）"
        if budget_cap_cny is not None
        else datetime.now().strftime("%Y-%m-%d %H:%M")
    )
    budget_lbl = "预算额度 / 结余" if budget_cap_cny is not None else "生成日期"
    meta_tbl = doc.add_table(rows=2, cols=4)
    meta_tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    _set_table_borders(meta_tbl, "E2E8F0")
    meta_pairs = [
        ("报销人 / 统筹人", applicant or "未填写", "归属活动 / 项目", project_code or "常规差旅"),
        (budget_lbl, budget_info, "核定可报销总额", f"¥ {net_total:,.2f}（{cny_to_chinese_upper(net_total)}）"),
    ]
    for r_idx, (k1, v1, k2, v2) in enumerate(meta_pairs):
        row_cells = meta_tbl.rows[r_idx].cells
        vals = [
            (k1, True, (100, 116, 139)),
            (v1, False, (15, 23, 42)),
            (k2, True, (100, 116, 139)),
            (v2, r_idx == 1, (0, 113, 227) if r_idx == 1 else (15, 23, 42)),
        ]
        for c_idx, (txt, is_b, col) in enumerate(vals):
            cell = row_cells[c_idx]
            _set_cell_bg(cell, "F8FAFC" if c_idx in (0, 2) else "FFFFFF")
            _set_cell_margins(cell, top=55, bottom=55, left=85, right=85)
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            p = cell.paragraphs[0]
            p.paragraph_format.space_after = Pt(0)
            _style_run(p.add_run(txt), size_pt=9.2 if not (r_idx == 1 and c_idx == 3) else 10.2, bold=is_b, color=col)

    p_gap = doc.add_paragraph()
    p_gap.paragraph_format.space_after = Pt(3)

    # 3. Audit / Warning Callout
    box_tbl = doc.add_table(rows=1, cols=1)
    box_tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    box_cell = box_tbl.rows[0].cells[0]
    _prevent_row_split(box_tbl.rows[0])
    _set_cell_margins(box_cell, top=75, bottom=75, left=120, right=120)
    if all_warnings:
        _set_cell_bg(box_cell, "FFFBEB")
        _set_table_borders(box_tbl, "F59E0B")
        p_w = box_cell.paragraphs[0]
        p_w.paragraph_format.space_after = Pt(3)
        _style_run(
            p_w.add_run(
                f"✓ 金额核验通过（净报销 ¥{net_total:,.2f}）｜ 以下复核提示请经办人与财务确认："
            ),
            size_pt=9.2,
            bold=True,
            color=(180, 83, 9),
        )
        for w_txt in all_warnings:
            pw_i = box_cell.add_paragraph()
            pw_i.paragraph_format.space_after = Pt(2)
            _style_run(pw_i.add_run(f"• {w_txt}"), size_pt=8.5, color=(120, 53, 15))
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
    h1.paragraph_format.space_before = Pt(9)
    h1.paragraph_format.space_after = Pt(4)
    h1.paragraph_format.keep_with_next = True
    _style_run(h1.add_run("一、费用报销明细汇总表"), size_pt=11.5, bold=True, color=(15, 23, 42))

    headers = ["#", "日期", "类目", "商户与用途说明", "凭证形态", "票面(¥)", "剔除(¥)", "报销额(¥)"]
    col_widths = [Cm(1.0), Cm(2.15), Cm(1.7), Cm(5.75), Cm(2.15), Cm(1.6), Cm(1.45), Cm(1.8)]
    exp_tbl = doc.add_table(rows=len(items) + 2, cols=len(headers))
    exp_tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    _set_table_borders(exp_tbl, "E2E8F0")

    _prevent_row_split(exp_tbl.rows[0], is_header=True)
    for c_i, h_txt in enumerate(headers):
        cell = exp_tbl.rows[0].cells[c_i]
        cell.width = col_widths[c_i]
        _set_cell_bg(cell, "0F172A")
        _set_cell_margins(cell, top=65, bottom=65, left=55, right=55)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.RIGHT if c_i >= 5 else WD_ALIGN_PARAGRAPH.LEFT
        p.paragraph_format.space_after = Pt(0)
        _style_run(p.add_run(h_txt), size_pt=8.2, bold=True, color=(255, 255, 255))

    for r_i, it in enumerate(items, start=1):
        _prevent_row_split(exp_tbl.rows[r_i])
        desc_full = it.merchant_or_desc + (f"\n（说明：{it.notes}）" if it.notes else "")
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
            _set_cell_margins(cell, top=55, bottom=55, left=55, right=55)
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.RIGHT if c_i >= 5 else WD_ALIGN_PARAGRAPH.LEFT
            p.paragraph_format.space_after = Pt(0)
            color = (185, 28, 28) if (c_i == 6 and it.excluded_personal_cny > 0) else (15, 23, 42)
            _style_run(p.add_run(val), size_pt=8.2, bold=(c_i == 7), color=color)

    _prevent_row_split(exp_tbl.rows[len(items) + 1])
    tot_row = exp_tbl.rows[len(items) + 1].cells
    tot_vals = ["合计", "", "", f"共 {len(items)} 笔（大写：{cny_to_chinese_upper(net_total)}）", "", f"{gross_total:,.2f}", f"{excluded_total:,.2f}", f"¥{net_total:,.2f}"]
    for c_i, val in enumerate(tot_vals):
        cell = tot_row[c_i]
        cell.width = col_widths[c_i]
        _set_cell_bg(cell, "EFF6FF")
        _set_cell_margins(cell, top=70, bottom=70, left=55, right=55)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.RIGHT if c_i >= 5 else WD_ALIGN_PARAGRAPH.LEFT
        p.paragraph_format.space_after = Pt(0)
        _style_run(p.add_run(val), size_pt=8.5, bold=True, color=(0, 113, 227) if c_i == 7 else (15, 23, 42))

    # Optional structured sections (e.g. Tri-Voucher Invoice Ledger, Volunteer Roster)
    sec_num = 2
    cn_nums = ["一", "二", "三", "四", "五", "六"]
    for sec_spec in (extra_sections or []):
        h_sec = doc.add_paragraph()
        h_sec.paragraph_format.space_before = Pt(11)
        h_sec.paragraph_format.space_after = Pt(5)
        h_sec.paragraph_format.keep_with_next = True
        prefix = cn_nums[sec_num - 1] if sec_num - 1 < len(cn_nums) else str(sec_num)
        _style_run(h_sec.add_run(f"{prefix}、{sec_spec['heading']}"), size_pt=11.5, bold=True, color=(15, 23, 42))
        if sec_spec.get("note"):
            pn = doc.add_paragraph()
            pn.paragraph_format.space_after = Pt(4)
            pn.paragraph_format.keep_with_next = True
            _style_run(pn.add_run(sec_spec["note"]), size_pt=8.6, color=(71, 85, 105))
        _add_styled_table(
            doc,
            sec_spec["headers"],
            sec_spec["rows"],
            right_align_cols=set(sec_spec.get("right_align_cols", [])),
        )
        sec_num += 1

    # Sign-off block on summary pages before visual appendix
    if include_signoff:
        p_sp = doc.add_paragraph()
        p_sp.paragraph_format.space_before = Pt(12)
        p_sp.paragraph_format.space_after = Pt(4)
        _style_run(p_sp.add_run("报销审批签字栏"), size_pt=9.2, bold=True, color=(71, 85, 105))
        sign_tbl = doc.add_table(rows=2, cols=4)
        sign_tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
        _set_table_borders(sign_tbl, "CBD5E1")
        _prevent_row_split(sign_tbl.rows[0])
        _prevent_row_split(sign_tbl.rows[1])
        for c_i, role in enumerate(["经办人 / 报销人", "活动统筹负责人", "指导老师 / 团委审核", "学院财务审批"]):
            c_top = sign_tbl.rows[0].cells[c_i]
            _set_cell_bg(c_top, "F8FAFC")
            _set_cell_margins(c_top, top=60, bottom=60, left=80, right=80)
            pt0 = c_top.paragraphs[0]
            pt0.paragraph_format.space_after = Pt(0)
            _style_run(pt0.add_run(role), size_pt=8.4, bold=True, color=(71, 85, 105))

            c_bot = sign_tbl.rows[1].cells[c_i]
            _set_cell_margins(c_bot, top=180, bottom=140, left=80, right=80)
            pb0 = c_bot.paragraphs[0]
            pb0.paragraph_format.space_after = Pt(0)
            _style_run(pb0.add_run("签字：            日期："), size_pt=8.2, color=(148, 163, 184))

    # 5. Embedded Receipt / Invoice Vouchers Appendix
    voucher_entries: list[tuple[Path, str, str]] = []
    if voucher_overrides:
        for src_p, cap in voucher_overrides:
            mod = "pdf" if src_p.suffix.lower() == ".pdf" else "image"
            voucher_entries.append((src_p, mod, cap))
    else:
        seen_handles: set[str] = set()
        fig_idx = 0
        for idx, it in enumerate(items, start=1):
            all_h = [it.artifact_handle.strip()] + [h.strip() for h in it.supporting_handles]
            for sub_i, h in enumerate(all_h):
                if h and h not in seen_handles:
                    art = artifact_registry.get(h)
                    if art is None or not Path(art.source_path).exists():
                        continue
                    seen_handles.add(h)
                    fig_idx += 1
                    src_name = Path(art.source_path).name
                    tag = f"序号 #{idx} · [{it.payment_channel}] {it.date or ''} {it.merchant_or_desc}（实报 ¥{it.reimbursable_cny:,.2f}）"
                    cap = f"图 {fig_idx}｜{tag}" if sub_i == 0 else f"图 {fig_idx}｜序号 #{idx} 关联凭证（{src_name}）"
                    voucher_entries.append((Path(art.source_path), art.modality, cap))
        for h in (include_all_registered_handles or []):
            if h and h not in seen_handles:
                art = artifact_registry.get(h)
                if art is None or not Path(art.source_path).exists():
                    continue
                seen_handles.add(h)
                fig_idx += 1
                voucher_entries.append((Path(art.source_path), art.modality, f"图 {fig_idx}｜附带原始票据凭证（{Path(art.source_path).name}）"))

    # Flatten into rendered pages with smart 2-column pairing for tall portrait mobile screenshots
    rendered_pages: list[tuple[io.BytesIO, float, float, str]] = []
    for src_p, mod, base_cap in voucher_entries:
        if not src_p.exists():
            continue
        for buf, w_cm, aspect, page_suffix in _prepare_embedded_pages(src_p, mod):
            rendered_pages.append((buf, w_cm, aspect, f"{base_cap}{page_suffix}"))

    embedded_count = 0
    if rendered_pages:
        doc.add_page_break()
        h2 = doc.add_paragraph()
        h2.paragraph_format.space_after = Pt(8)
        prefix = cn_nums[sec_num - 1] if sec_num - 1 < len(cn_nums) else str(sec_num)
        _style_run(h2.add_run(f"{prefix}、原始发票、账单与支付记录附件（三单合一）"), size_pt=12.0, bold=True, color=(15, 23, 42))
        sec_num += 1

        idx = 0
        while idx < len(rendered_pages):
            buf, w_cm, aspect, cap = rendered_pages[idx]
            if idx > 0:
                doc.add_page_break()
            # Pair two consecutive tall portrait phone screenshots side-by-side cleanly on one page
            if (
                aspect < 0.52
                and idx + 1 < len(rendered_pages)
                and rendered_pages[idx + 1][2] < 0.52
            ):
                buf2, _w2, a2, cap2 = rendered_pages[idx + 1]
                pair_tbl = doc.add_table(rows=2, cols=2)
                pair_tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
                _set_table_borders(pair_tbl, borderless=True)
                _prevent_row_split(pair_tbl.rows[0])
                _prevent_row_split(pair_tbl.rows[1])
                for col_i, (b_i, a_i, c_i) in enumerate(((buf, aspect, cap), (buf2, a2, cap2))):
                    c_cap = pair_tbl.rows[0].cells[col_i]
                    _set_cell_margins(c_cap, top=40, bottom=40, left=60, right=60)
                    pc = c_cap.paragraphs[0]
                    pc.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    pc.paragraph_format.space_after = Pt(0)
                    pc.paragraph_format.keep_with_next = True
                    _style_run(pc.add_run(c_i), size_pt=8.5, bold=True, color=(30, 41, 59))

                    c_img = pair_tbl.rows[1].cells[col_i]
                    _set_cell_margins(c_img, top=30, bottom=40, left=60, right=60)
                    pi = c_img.paragraphs[0]
                    pi.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    pi.paragraph_format.space_after = Pt(0)
                    pair_w = min(7.4, max(4.8, round(19.5 * a_i, 2)))
                    pi.add_run().add_picture(b_i, width=Cm(pair_w))
                embedded_count += 2
                idx += 2
                continue

            p_cap = doc.add_paragraph()
            p_cap.paragraph_format.space_before = Pt(4)
            p_cap.paragraph_format.space_after = Pt(4)
            p_cap.paragraph_format.keep_with_next = True
            _style_run(p_cap.add_run(cap), size_pt=9.2, bold=True, color=(30, 41, 59))

            p_img = doc.add_paragraph()
            p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p_img.paragraph_format.space_after = Pt(4)
            p_img.add_run().add_picture(buf, width=Cm(w_cm))
            embedded_count += 1
            idx += 1

    # 6. Optional Activity Scene Supporting Photo Grid (2-column sleek layout)
    if activity_photos:
        doc.add_page_break()
        h3 = doc.add_paragraph()
        h3.paragraph_format.space_after = Pt(4)
        h3.paragraph_format.keep_with_next = True
        prefix = cn_nums[sec_num - 1] if sec_num - 1 < len(cn_nums) else str(sec_num)
        _style_run(h3.add_run(f"{prefix}、志愿授课活动现场佐证照片精选"), size_pt=12.0, bold=True, color=(15, 23, 42))
        if activity_photo_note:
            p_an = doc.add_paragraph()
            p_an.paragraph_format.space_after = Pt(8)
            _style_run(p_an.add_run(activity_photo_note), size_pt=8.8, color=(71, 85, 105))

        rows_needed = (len(activity_photos) + 1) // 2
        grid_tbl = doc.add_table(rows=rows_needed * 2, cols=2)
        grid_tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
        _set_table_borders(grid_tbl, borderless=True)
        for p_idx, (photo_path, photo_cap) in enumerate(activity_photos):
            if not photo_path.exists():
                continue
            r_base = (p_idx // 2) * 2
            c_idx = p_idx % 2
            _prevent_row_split(grid_tbl.rows[r_base])
            _prevent_row_split(grid_tbl.rows[r_base + 1])
            pil_p = ImageOps.exif_transpose(Image.open(photo_path)).convert("RGB")
            buf_p, _, _ = _encode_pil_fit(pil_p, max_w_cm=8.1, max_h_cm=6.2, as_jpeg=True)

            cell_img = grid_tbl.rows[r_base].cells[c_idx]
            _set_cell_margins(cell_img, top=60, bottom=30, left=60, right=60)
            pi = cell_img.paragraphs[0]
            pi.alignment = WD_ALIGN_PARAGRAPH.CENTER
            pi.paragraph_format.space_after = Pt(0)
            pi.add_run().add_picture(buf_p, width=Cm(8.0))

            cell_cap = grid_tbl.rows[r_base + 1].cells[c_idx]
            _set_cell_margins(cell_cap, top=20, bottom=90, left=60, right=60)
            pc = cell_cap.paragraphs[0]
            pc.alignment = WD_ALIGN_PARAGRAPH.CENTER
            pc.paragraph_format.space_after = Pt(0)
            _style_run(pc.add_run(photo_cap), size_pt=8.4, color=(71, 85, 105))
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
