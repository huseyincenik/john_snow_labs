"""Pydantic models for data structures."""
from .schemas import (
    DocumentMetadata,
    ExtractedField,
    DocumentExtraction,
    ExtractionResult,
    ConsolidatedField,
    PatientConsolidation,
    ConsolidationResult,
    ProcessingRequest,
    ProcessingResponse,
)

__all__ = [
    "DocumentMetadata",
    "ExtractedField",
    "DocumentExtraction",
    "ExtractionResult",
    "ConsolidatedField",
    "PatientConsolidation",
    "ConsolidationResult",
    "ProcessingRequest",
    "ProcessingResponse",
]

