"""Taint-aware multimodal receipt & itinerary perception with addressable ROI handles."""

from .artifact_registry import ArtifactRegistry, ReceiptArtifact, ReceiptROI, ROICropDetail
from .receipt_parser import ReceiptMultimodalParser, detect_prompt_injection

__all__ = [
    "ArtifactRegistry",
    "ROICropDetail",
    "ReceiptArtifact",
    "ReceiptMultimodalParser",
    "ReceiptROI",
    "detect_prompt_injection",
]
