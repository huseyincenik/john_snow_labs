"""FastAPI routes for document processing."""

import asyncio
import uuid
from pathlib import Path
from typing import List, Optional

import aiofiles  # type: ignore
from fastapi import APIRouter, HTTPException, UploadFile, File, BackgroundTasks  # type: ignore

from config.settings import settings
from src.models.schemas import DocumentMetadata, ProcessingRequest, ProcessingResponse
from src.pipeline.consolidator import Consolidator
from src.pipeline.extractor import Extractor
from src.pipeline.tagger import Tagger
from src.utils.llm import get_llm_provider
from src.utils.logger import setup_logger
from src.utils.ontology import OntologyLoader
from src.utils.storage import StorageManager

router = APIRouter()
storage = StorageManager()
ontology = OntologyLoader()

# In-memory processing status (in production, use Redis or database)
processing_status: dict = {}


@router.post("/process", response_model=ProcessingResponse)
async def process_documents(
    request: ProcessingRequest,
    background_tasks: BackgroundTasks,
) -> ProcessingResponse:
    """Process documents for extraction and consolidation."""
    session_id = str(uuid.uuid4())

    # Load documents
    documents = await load_documents(request)

    if not documents:
        raise HTTPException(status_code=400, detail="No documents found")

    provider_name = request.llm_provider or settings.default_llm_provider
    model_override = request.llm_model

    # Start processing in background
    background_tasks.add_task(
        process_documents_task,
        session_id,
        documents,
        provider_name,
        model_override,
        request.parallel_patients,
        request.pipeline_threads,
    )

    processing_status[session_id] = {
        "status": "processing",
        "message": "Processing started",
    }

    return ProcessingResponse(
        session_id=session_id,
        status="processing",
        message="Processing started",
    )


@router.get("/status/{session_id}", response_model=ProcessingResponse)
async def get_status(session_id: str) -> ProcessingResponse:
    """Get processing status for a session."""
    if session_id not in processing_status:
        raise HTTPException(status_code=404, detail="Session not found")

    status = processing_status[session_id]

    # Try to load results
    tagger_result = storage.load_tagger(session_id)
    extraction_result = storage.load_extraction(session_id)
    consolidation_result = storage.load_consolidation(session_id)

    return ProcessingResponse(
        session_id=session_id,
        status=status["status"],
        message=status.get("message", ""),
        tagger_result=tagger_result,
        extraction_result=extraction_result,
        consolidation_result=consolidation_result,
    )


@router.post("/upload")
async def upload_documents(
    files: List[UploadFile] = File(...),
) -> dict:
    """Upload documents for processing."""
    uploaded_files = []
    input_dir = Path(settings.output_dir).parent / "input"
    input_dir.mkdir(parents=True, exist_ok=True)

    for file in files:
        file_path = input_dir / file.filename
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        uploaded_files.append(str(file_path))

    return {
        "message": f"Uploaded {len(uploaded_files)} files",
        "files": uploaded_files,
    }


async def load_documents(request: ProcessingRequest) -> List[DocumentMetadata]:
    """Load documents based on request."""
    input_dir = Path("data/input")
    file_paths: List[Path] = []

    if request.process_all:
        file_paths = sorted(input_dir.glob("*.txt"))
    elif request.patient_ids:
        for patient_id in request.patient_ids:
            file_paths.extend(sorted(input_dir.glob(f"*{patient_id}*.txt")))
    elif request.doc_ids:
        for doc_id in request.doc_ids:
            file_paths.extend(sorted(input_dir.glob(f"*{doc_id}*.txt")))

    if request.max_documents and request.max_documents > 0:
        file_paths = file_paths[: request.max_documents]

    if not file_paths:
        return []

    parsed = await asyncio.gather(*(parse_document_file(path) for path in file_paths))
    return [doc for doc in parsed if doc]


async def parse_document_file(file_path: Path) -> Optional[DocumentMetadata]:
    """Parse a document file and extract metadata."""
    try:
        async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
            content = await f.read()

        # Parse header metadata
        lines = content.split("\n")
        metadata = {}

        for line in lines[:10]:  # Check first 10 lines for metadata
            if ":" in line:
                key, value = line.split(":", 1)
                key = key.strip().lower().replace(" ", "_")
                value = value.strip()
                metadata[key] = value

        # Extract patient_id, doc_id, doc_type, doc_date
        patient_id = metadata.get("patient_id", "").replace("Patient Id: ", "").strip() or "unknown"
        raw_doc_id = metadata.get("doc_id", "").replace("Doc Id: ", "").strip() or file_path.stem
        doc_type = metadata.get("doc_type", "").replace("Doc Type: ", "").strip() or "unknown"
        doc_date = metadata.get("date", "").replace("Date: ", "").strip()
        
        # Format doc_id as: doc_{raw_doc_id}_{file_stem}
        # Example: doc_001_jsl_p01_001_summary_doc
        doc_id = f"doc_{raw_doc_id}_{file_path.stem}"

        return DocumentMetadata(
            patient_id=patient_id,
            doc_id=doc_id,
            doc_type=doc_type,
            doc_date=doc_date if doc_date else None,
            filename=str(file_path),
            content=content,
        )
    except Exception as e:
        return None


async def process_documents_task(
    session_id: str,
    documents: List[DocumentMetadata],
    provider_name: str,
    model_override: Optional[str],
    parallel_patients: Optional[int] = None,
    pipeline_threads: Optional[int] = None,
):
    """Background task for processing documents."""
    try:
        processing_status[session_id] = {
            "status": "processing",
            "message": "Tagging documents...",
        }

        llm = get_llm_provider(provider_name, model_override)

        # Tag documents with LLM-based confidence scoring (runs in parallel with extraction)
        tagger = Tagger(ontology, llm_provider=llm, provider_label=provider_name)
        
        # Start tagger and extractor in parallel
        # Tagger provides sorted documents, extractor can start processing
        tagger_task = asyncio.create_task(tagger.tag_documents(documents, session_id))
        
        processing_status[session_id] = {
            "status": "processing",
            "message": "Tagging documents and extracting fields in parallel...",
        }

        # Extract fields (can start with documents, tagger will provide sorted order)
        extractor = Extractor(
            ontology,
            llm_provider=llm,
            provider_label=provider_name,
            max_parallel_patients=parallel_patients,
            docetl_thread_override=pipeline_threads,
        )
        
        # Wait for tagger to complete to get sorted documents
        tagger_result, tagged_docs = await tagger_task
        storage.save_tagger(tagger_result, session_id)
        
        processing_status[session_id] = {
            "status": "processing",
            "message": "Extracting fields...",
        }
        
        extraction_result, pipeline_artifacts = await extractor.extract(
            tagged_docs,
            session_id,
        )

        # Save extraction result
        storage.save_extraction(extraction_result, session_id)

        processing_status[session_id] = {
            "status": "processing",
            "message": "Consolidating patient data...",
        }

        # Consolidate
        consolidator = Consolidator(
            ontology,
            llm_provider=llm,
            provider_label=provider_name,
        )
        consolidation_result = await consolidator.consolidate(
            extraction_result,
            pipeline_artifacts,
            session_id,
        )

        # Save consolidation result
        storage.save_consolidation(consolidation_result, session_id)

        processing_status[session_id] = {
            "status": "completed",
            "message": "Processing completed successfully",
        }
    except Exception as e:
        import traceback

        error_trace = traceback.format_exc()
        logger = setup_logger(session_id, "stage_error")
        logger.error(f"Processing failed: {str(e)}\n{error_trace}")
        processing_status[session_id] = {
            "status": "failed",
            "message": f"Processing failed: {str(e)}",
        }
