"""Generic archive (.rar/.zip) & context document (.docx/.xlsx) ingestor for real reimbursement bundles."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
import re
import shutil
import subprocess
from typing import Sequence
import xml.etree.ElementTree as ET
import zipfile

from docx import Document as DocxDocument
from PIL import Image
from pydantic import BaseModel, Field

_ARCHIVE_EXTS = {".zip", ".rar"}
_CONTEXT_EXTS = {".docx", ".xlsx", ".txt", ".md"}
_VOUCHER_IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
_VOUCHER_PDF_EXTS = {".pdf"}

_SCENE_KEYWORDS = ("活动图片", "活动照片", "现场", "合影", "剪影", "照片", "activity", "scene", "photo", "gallery")
_VOUCHER_KEYWORDS = ("发票", "账单", "小票", "支付", "付款", "收据", "行程", "水单", "打车", "报销单", "receipt", "invoice", "pay", "bill", "ticket")


class IngestedBundle(BaseModel):
    """Structured separation of an arbitrary user upload (files or .rar/.zip archive)."""

    financial_vouchers: list[Path] = Field(default_factory=list)
    context_docs: list[Path] = Field(default_factory=list)
    activity_photos: list[Path] = Field(default_factory=list)
    selected_activity_photos: list[Path] = Field(default_factory=list)
    context_brief: str = ""
    archive_names: list[str] = Field(default_factory=list)


def _natural_sort_key(path: Path) -> list[int | str]:
    return [int(tok) if tok.isdigit() else tok.lower() for tok in re.split(r"(\d+)", str(path))]


def _find_rar_extractor() -> tuple[str, str] | None:
    for candidate in (
        shutil.which("7z"),
        r"C:\Program Files\7-Zip\7z.exe",
        r"C:\Program Files (x86)\7-Zip\7z.exe",
    ):
        if candidate and Path(candidate).exists():
            return ("7z", str(candidate))
    for candidate in (
        shutil.which("unrar"),
        r"C:\Program Files\WinRAR\UnRAR.exe",
        r"C:\Program Files (x86)\WinRAR\UnRAR.exe",
    ):
        if candidate and Path(candidate).exists():
            return ("unrar", str(candidate))
    tar_bin = shutil.which("tar")
    if tar_bin:
        return ("tar", tar_bin)
    return None


def _extract_archive(archive_path: Path, dest_dir: Path) -> list[Path]:
    """Safely extract .zip or .rar into dest_dir (shell=False, path-traversal protected)."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    ext = archive_path.suffix.lower()
    if ext == ".zip":
        with zipfile.ZipFile(archive_path) as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                raw_name = info.filename
                if not (info.flag_bits & 0x800):
                    try:
                        raw_name = raw_name.encode("cp437").decode("gbk")
                    except (UnicodeEncodeError, UnicodeDecodeError):
                        pass
                target = (dest_dir / raw_name).resolve()
                if dest_dir.resolve() not in target.parents:
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(zf.read(info.filename))
    elif ext == ".rar":
        tool = _find_rar_extractor()
        if tool is None:
            raise RuntimeError("No RAR extractor found (install 7-Zip or WinRAR).")
        kind, exe = tool
        if kind == "7z":
            cmd = [exe, "x", str(archive_path), f"-o{dest_dir}", "-y"]
        elif kind == "unrar":
            cmd = [exe, "x", "-o+", str(archive_path), str(dest_dir)]
        else:
            cmd = [exe, "-xf", str(archive_path), "-C", str(dest_dir)]
        proc = subprocess.run(cmd, shell=False, capture_output=True, text=False, timeout=30)
        if proc.returncode != 0:
            err_msg = (proc.stderr or proc.stdout or b"").decode("utf-8", errors="replace")
            raise RuntimeError(f"Failed to extract {archive_path.name}: {err_msg}")
    return sorted((p for p in dest_dir.rglob("*") if p.is_file()), key=_natural_sort_key)


def _extract_docx_text(path: Path, max_chars: int = 2400) -> str:
    doc = DocxDocument(str(path))
    parts: list[str] = []
    for p in doc.paragraphs:
        t = p.text.strip()
        if t:
            parts.append(t)
    for t_i, tbl in enumerate(doc.tables):
        rows_txt = [" | ".join(c.text.replace("\n", " ").strip() for c in r.cells) for r in tbl.rows]
        rows_txt = [r for r in rows_txt if r.replace("|", "").strip()]
        if rows_txt:
            parts.append(f"[表{t_i + 1}] " + " ; ".join(rows_txt[:12]))
    return "\n".join(parts)[:max_chars]


def _excel_serial_to_date(val: str) -> str:
    try:
        n = int(float(val))
        if 40000 <= n <= 60000:
            return (date(1899, 12, 30) + timedelta(days=n)).isoformat()
    except ValueError:
        pass
    return val


def _extract_xlsx_text(path: Path, max_chars: int = 1200) -> str:
    ns_uri = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    lines: list[str] = []
    with zipfile.ZipFile(path) as zf:
        ss: list[str] = []
        if "xl/sharedStrings.xml" in zf.namelist():
            sroot = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            for si in sroot.findall(f"{ns_uri}si"):
                ss.append("".join(t.text or "" for t in si.iter(f"{ns_uri}t")))
        for sheet_name in sorted(n for n in zf.namelist() if n.startswith("xl/worksheets/sheet") and n.endswith(".xml")):
            sh = ET.fromstring(zf.read(sheet_name))
            date_cols: set[int] = set()
            for r_idx, row in enumerate(sh.iter(f"{ns_uri}row")):
                vals: list[str] = []
                for c_idx, cell in enumerate(row.findall(f"{ns_uri}c")):
                    t = cell.attrib.get("t", "")
                    v = cell.find(f"{ns_uri}v")
                    raw = "" if (v is None or v.text is None) else (ss[int(v.text)] if t == "s" else v.text)
                    if r_idx == 0 and any(k in raw.lower() for k in ("日期", "时间", "date")):
                        date_cols.add(c_idx)
                    elif c_idx in date_cols and raw:
                        raw = _excel_serial_to_date(raw)
                    if raw.strip():
                        vals.append(raw.strip())
                if vals:
                    lines.append(" | ".join(vals))
    return "\n".join(lines)[:max_chars]


def _select_representative_photos(photos: list[Path], limit: int = 4) -> list[Path]:
    ordered = sorted(photos, key=_natural_sort_key)
    n = len(ordered)
    if n <= limit:
        return ordered
    start = 1 if n >= 8 else 0
    span = (n - 1) - start
    idxs = [start + round(k * span / max(1, limit - 1)) for k in range(limit)]
    return [ordered[i] for i in dict.fromkeys(idxs)]


def expand_attachments(attachments: Sequence[str | Path], extract_root: Path) -> IngestedBundle:
    """Expand any .rar/.zip archives and classify files into vouchers, context docs, and scene photos."""
    all_files: list[tuple[Path, str]] = []  # (path, relative_display_hint)
    archive_names: list[str] = []

    for idx, raw in enumerate(attachments):
        p = Path(raw)
        if not p.exists():
            raise FileNotFoundError(f"Attachment not found: {p}")
        if p.suffix.lower() in _ARCHIVE_EXTS:
            archive_names.append(p.name)
            sub_dir = extract_root / f"arc_{idx}_{p.stem}"
            if sub_dir.exists():
                shutil.rmtree(sub_dir)
            for extracted in _extract_archive(p, sub_dir):
                rel = str(extracted.relative_to(sub_dir))
                all_files.append((extracted, rel))
        else:
            all_files.append((p, p.name))

    vouchers: list[Path] = []
    context_docs: list[Path] = []
    activity_photos: list[Path] = []
    ambiguous_imgs: list[tuple[Path, str]] = []

    for fpath, rel in all_files:
        ext = fpath.suffix.lower()
        rel_low = rel.lower()
        if ext in _VOUCHER_PDF_EXTS:
            vouchers.append(fpath)
        elif ext in _CONTEXT_EXTS:
            context_docs.append(fpath)
        elif ext in _VOUCHER_IMG_EXTS:
            has_voucher_cue = any(k in rel_low for k in _VOUCHER_KEYWORDS)
            has_scene_cue = any(k in rel_low for k in _SCENE_KEYWORDS)
            if has_scene_cue and not has_voucher_cue:
                activity_photos.append(fpath)
            elif has_voucher_cue:
                vouchers.append(fpath)
            else:
                ambiguous_imgs.append((fpath, rel))

    # Secondary domain-agnostic heuristic: if >=4 unlabeled landscape camera photos share one folder, treat as scene photos
    by_parent: dict[Path, list[Path]] = {}
    for fp, _ in ambiguous_imgs:
        by_parent.setdefault(fp.parent, []).append(fp)
    for parent, imgs in by_parent.items():
        if len(imgs) >= 4:
            landscape_cnt = 0
            for im_p in imgs[:6]:
                try:
                    w, h = Image.open(im_p).size
                    if w > h and w >= 1000:
                        landscape_cnt += 1
                except Exception:
                    pass
            if landscape_cnt >= min(4, len(imgs)):
                activity_photos.extend(imgs)
                continue
        vouchers.extend(imgs)

    # Extract concise structured brief from context docs (.docx / .xlsx / .txt / .md)
    briefs: list[str] = []
    for doc_p in sorted(context_docs, key=_natural_sort_key):
        ext = doc_p.suffix.lower()
        try:
            if ext == ".docx":
                txt = _extract_docx_text(doc_p)
            elif ext == ".xlsx":
                txt = _extract_xlsx_text(doc_p)
            else:
                txt = doc_p.read_text(encoding="utf-8", errors="ignore")[:1500]
            if txt.strip():
                briefs.append(f"=== 伴随文档：{doc_p.name} ===\n{txt.strip()}")
        except Exception as exc:
            briefs.append(f"=== 伴随文档：{doc_p.name}（读取失败: {exc}）===")

    selected_photos = _select_representative_photos(activity_photos, limit=4)
    if activity_photos:
        briefs.append(
            f"=== 活动现场佐证照片归档 ===\n共检测到 {len(activity_photos)} 张活动现场照片"
            f"（已按时间/编号均匀预选 {len(selected_photos)} 张代表照用于 Word 附图："
            f"{', '.join(p.name for p in selected_photos)}）。"
        )

    return IngestedBundle(
        financial_vouchers=sorted(vouchers, key=_natural_sort_key),
        context_docs=sorted(context_docs, key=_natural_sort_key),
        activity_photos=sorted(activity_photos, key=_natural_sort_key),
        selected_activity_photos=selected_photos,
        context_brief="\n\n".join(briefs)[:4200],
        archive_names=archive_names,
    )
