"""Content-addressed multimodal receipt & document registry supporting ROI zoom-in."""

from __future__ import annotations

import base64
import hashlib
import io
from pathlib import Path
from typing import Any, Literal

import pymupdf as fitz
from PIL import Image
from pydantic import BaseModel, Field


class ReceiptROI(BaseModel):
    """Addressable spatial region (table, line-item box, or total panel) inside a receipt."""

    roi_id: str
    page_index: int = Field(default=0, ge=0)
    norm_box: tuple[float, float, float, float]
    region_type: Literal["folio_table", "invoice_widget", "text_section"]
    label: str
    extracted_text: str = ""
    line_items: dict[str, list[float] | float] = Field(default_factory=dict)


class ReceiptArtifact(BaseModel):
    """Registered PDF folio or image receipt represented in context by a compact handle."""

    artifact_id: str
    handle: str
    modality: Literal["pdf", "image"]
    source_path: str
    sha256: str
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    page_count: int = Field(default=1, ge=1)
    is_tainted_untrusted: bool = False
    taint_reasons: list[str] = Field(default_factory=list)
    rois: list[ReceiptROI] = Field(default_factory=list)
    compact_header: str
    raw_full_text: str = ""
    structured_kv: dict[str, Any] = Field(default_factory=dict)


class ROICropDetail(BaseModel):
    """Zoomed ROI payload returned when the agent calls inspect_receipt_roi."""

    artifact_id: str
    roi_id: str
    label: str
    region_type: str
    crop_path: str
    crop_size: tuple[int, int]
    extracted_text: str
    line_items: dict[str, list[float] | float] = Field(default_factory=dict)
    base64_png: str


class ArtifactRegistry:
    """Idempotent SHA-256 registry keeping multi-page images/tables out of global prompt history."""

    def __init__(self, cache_dir: Path | str = Path("artifacts_cache")) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._items: dict[str, ReceiptArtifact] = {}
        self._sha_map: dict[str, str] = {}

    @staticmethod
    def file_sha256(path: Path) -> str:
        h = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()

    def register(self, artifact: ReceiptArtifact) -> ReceiptArtifact:
        existing_id = self._sha_map.get(artifact.sha256)
        if existing_id and existing_id in self._items:
            return self._items[existing_id]
        self._items[artifact.artifact_id] = artifact
        self._sha_map[artifact.sha256] = artifact.artifact_id
        return artifact

    def get(self, ref: str) -> ReceiptArtifact | None:
        clean = ref.replace("artifact://", "").strip()
        return self._items.get(clean)

    def inspect_roi(
        self,
        artifact_ref: str,
        roi_id: str | None = None,
        page_index: int = 0,
    ) -> ROICropDetail:
        """Crop high-resolution visual region and return exact cell/line-item values."""
        art = self.get(artifact_ref)
        if art is None:
            raise KeyError(f"Unknown receipt artifact handle: {artifact_ref!r}")

        src_path = Path(art.source_path)
        target_roi: ReceiptROI | None = None
        if roi_id:
            for r in art.rois:
                if r.roi_id == roi_id:
                    target_roi = r
                    page_index = r.page_index
                    break
            if target_roi is None:
                valid = [r.roi_id for r in art.rois]
                raise ValueError(f"ROI {roi_id!r} not in {art.handle}; valid={valid}")
        elif art.rois:
            target_roi = art.rois[0]
            page_index = target_roi.page_index

        if art.modality == "image":
            pil_img = Image.open(src_path).convert("RGB")
        else:
            doc = fitz.open(src_path)
            try:
                p_idx = max(0, min(len(doc) - 1, page_index))
                pix = doc[p_idx].get_pixmap(dpi=150, alpha=False)
                pil_img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            finally:
                doc.close()

        w, h = pil_img.size
        if target_roi is not None:
            nx0, ny0, nx1, ny1 = target_roi.norm_box
            x0, y0 = max(0, int(nx0 * w)), max(0, int(ny0 * h))
            x1, y1 = max(x0 + 1, min(w, int(nx1 * w))), max(y0 + 1, min(h, int(ny1 * h)))
            eff_id = target_roi.roi_id
            label = target_roi.label
            rtype = target_roi.region_type
            txt = target_roi.extracted_text
            items = dict(target_roi.line_items)
        else:
            x0, y0, x1, y1 = 0, 0, w, h
            eff_id = "roi_full"
            label = "full_page"
            rtype = "text_section"
            txt = art.raw_full_text[:600]
            items = {}

        cropped = pil_img.crop((x0, y0, x1, y1))
        buf = io.BytesIO()
        cropped.save(buf, format="PNG")
        png_bytes = buf.getvalue()
        out_path = self.cache_dir / f"{art.artifact_id}_{eff_id}.png"
        out_path.write_bytes(png_bytes)

        return ROICropDetail(
            artifact_id=art.artifact_id,
            roi_id=eff_id,
            label=label,
            region_type=rtype,
            crop_path=str(out_path),
            crop_size=(cropped.width, cropped.height),
            extracted_text=txt,
            line_items=items,
            base64_png=base64.b64encode(png_bytes).decode("ascii"),
        )
