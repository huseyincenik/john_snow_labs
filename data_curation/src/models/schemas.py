"""Pydantic v2 schemas for extraction and consolidation."""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Literal
from datetime import datetime


class DocumentMetadata(BaseModel):
    """Document metadata."""

    patient_id: str
    doc_id: str
    doc_type: str
    doc_date: Optional[str] = None
    filename: Optional[str] = None
    content: str


class SplitInfo(BaseModel):
    """Chunking metadata for tagged documents."""

    is_split: bool = False
    split_number: int = 1
    total_splits: int = 1
    estimated_tokens: int = 0


class TaggedDocument(BaseModel):
    """Tagged document metadata exposed by the tagger stage."""

    doc_id: str
    filename: Optional[str] = None
    doc_type: str
    doc_date: Optional[str] = None
    type_confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    date_confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    content: str
    split_info: SplitInfo = Field(default_factory=SplitInfo)


class TaggerResult(BaseModel):
    """Chronologically sorted/tagged documents."""

    session_id: str
    generated_timestamp: datetime
    stage: Literal["stage_tagger"] = "stage_tagger"
    total_documents: int = 0
    sorted_documents: List[TaggedDocument] = Field(default_factory=list)
    processing_time_seconds: float = 0.0
    timestamp: datetime = Field(default_factory=datetime.now)


class FieldEvidence(BaseModel):
    """Evidence for extracted field."""

    explanation: str
    inferred: bool = False
    related_entities: List[str] = Field(default_factory=list)


class SourceReference(BaseModel):
    """Reference to source document."""

    doc_id: str
    split_number: Optional[int] = None
    reasoning_excerpt: str


class ExtractedField(BaseModel):
    """Single extracted field with evidence."""

    field_name: str
    category: str
    raw_value: Any
    normalized_value: Any
    units: Optional[str] = None
    vocabulary_code: Optional[str] = None
    field_evidence: FieldEvidence
    confidence_level: Literal["low", "medium", "high"] = "medium"
    confidence_score: float = Field(ge=0.0, le=1.0)
    sources: List[SourceReference] = Field(default_factory=list)
    extraction_timestamp: datetime
    data_type: str


class DocumentExtraction(BaseModel):
    """Extraction results for a single document."""

    doc_id: str
    doc_date: Optional[str] = None
    doc_type: Optional[str] = None
    extracted_fields: List[ExtractedField] = Field(default_factory=list)
    total_fields_extracted: int = 0
    processing_time_seconds: float = 0.0


class ExtractionResult(BaseModel):
    """Complete extraction result for all documents."""

    session_id: str
    generated_timestamp: datetime
    stage: Literal["stage_extractor"] = "stage_extractor"
    total_documents_processed: int = 0
    document_results: List[DocumentExtraction] = Field(default_factory=list)
    total_fields_extracted: int = 0
    processing_time_seconds: float = 0.0
    timestamp: datetime = Field(default_factory=datetime.now)


class SupportingEvidence(BaseModel):
    """Supporting evidence for consolidated field."""

    snippet: str
    explanation: str
    date: Optional[str] = None
    source_file: str
    confidence: float = Field(ge=0.0, le=1.0)
    span: Optional[Dict[str, Any]] = None
    page: Optional[int] = None


class ConsolidatedField(BaseModel):
    """Consolidated field with evidence."""

    field_name: str
    category: str
    consolidated_value: Dict[str, Any]
    all_values: List[Any] = Field(default_factory=list)
    units: Optional[str] = None
    vocabulary_code: Optional[str] = None
    confidence_score: float = Field(ge=0.0, le=1.0)
    source_documents: List[str] = Field(default_factory=list)
    consolidation_reasoning: str
    data_type: str


class PatientConsolidation(BaseModel):
    """Patient-level consolidation result."""

    patient_id: str
    consolidated_fields: List[ConsolidatedField] = Field(default_factory=list)
    total_fields_consolidated: int = 0
    processing_time_seconds: float = 0.0


class ConsolidationResult(BaseModel):
    """Complete consolidation result."""

    session_id: str
    generated_timestamp: datetime
    stage: Literal["stage_consolidator"] = "stage_consolidator"
    total_fields_consolidated: int = 0
    consolidated_fields: List[ConsolidatedField] = Field(default_factory=list)
    processing_time_seconds: float = 0.0
    timestamp: datetime = Field(default_factory=datetime.now)


class ProcessingRequest(BaseModel):
    """Request for document processing."""

    patient_ids: Optional[List[str]] = None
    doc_ids: Optional[List[str]] = None
    filenames: Optional[List[str]] = None  # Exact filenames to process
    process_all: bool = False
    llm_provider: Optional[Literal["openai", "qwen"]] = None
    llm_model: Optional[str] = None
    max_documents: Optional[int] = None
    parallel_patients: Optional[int] = None
    pipeline_threads: Optional[int] = None


class ProcessingResponse(BaseModel):
    """Response from processing request."""

    session_id: str
    status: Literal["pending", "processing", "completed", "failed"] = "pending"
    message: str
    tagger_result: Optional[TaggerResult] = None
    extraction_result: Optional[ExtractionResult] = None
    consolidation_result: Optional[ConsolidationResult] = None
    created_at: datetime = Field(default_factory=datetime.now)
