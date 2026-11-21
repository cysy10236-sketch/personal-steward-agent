"""Multimodal receipt parsing, artifact registry, and Word (.docx) reimbursement exporter."""

from .artifact_registry import ArtifactRegistry, ReceiptArtifact, ReceiptROI, ROICropDetail
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
    "ReceiptArtifact",
    "ReceiptMultimodalParser",
    "ReceiptROI",
    "ReimbursementDocxResult",
    "ROICropDetail",
    "build_reimbursement_docx",
    "detect_duplicate_claims",
    "detect_prompt_injection",
]
