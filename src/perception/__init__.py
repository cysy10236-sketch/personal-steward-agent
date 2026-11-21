"""Multimodal receipt parsing, archive ingestion, artifact registry, and Word (.docx) exporter."""

from .artifact_registry import ArtifactRegistry, ReceiptArtifact, ReceiptROI, ROICropDetail
from .bundle_ingestor import IngestedBundle, expand_attachments
from .docx_exporter import (
    ExpenseLineItem,
    ReimbursementDocxResult,
    build_reimbursement_docx,
    detect_duplicate_claims,
)
from .receipt_parser import ReceiptMultimodalParser, detect_prompt_injection

__all__ = [
    "ArtifactRegistry",
    "ExpenseLineItem",
    "IngestedBundle",
    "ReceiptArtifact",
    "ReceiptMultimodalParser",
    "ReceiptROI",
    "ReimbursementDocxResult",
    "ROICropDetail",
    "build_reimbursement_docx",
    "detect_duplicate_claims",
    "detect_prompt_injection",
    "expand_attachments",
]
