"""FastAPI routes for document processing."""
import uuid
from fastapi import APIRouter, HTTPException, UploadFile, File, BackgroundTasks
from typing import List, Optional
from pathlib import Path
from src.models.schemas import ProcessingRequest, ProcessingResponse
from src.pipeline.extractor import Extractor
from src.pipeline.consolidator import Consolidator
from src.pipeline.tagger import Tagger
from src.utils.storage import StorageManager
from src.utils.ontology import OntologyLoader
from src.models.schemas import DocumentMetadata
from config.settings import settings
from src.utils.llm import get_llm_provider

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
    documents = []
    
    if request.process_all:
        # Load all documents
        for txt_file in input_dir.glob("*.txt"):
            doc_metadata = await parse_document_file(txt_file)
            if doc_metadata:
                documents.append(doc_metadata)
    elif request.patient_ids:
        # Load documents for specific patients
        for patient_id in request.patient_ids:
            for txt_file in input_dir.glob(f"*{patient_id}*.txt"):
                doc_metadata = await parse_document_file(txt_file)
                if doc_metadata:
                    documents.append(doc_metadata)
    elif request.doc_ids:
        # Load specific documents
        for doc_id in request.doc_ids:
            for txt_file in input_dir.glob(f"*{doc_id}*.txt"):
                doc_metadata = await parse_document_file(txt_file)
                if doc_metadata:
                    documents.append(doc_metadata)
    
    if request.max_documents and request.max_documents > 0:
        documents = documents[: request.max_documents]
    
    return documents


async def parse_document_file(file_path: Path) -> Optional[DocumentMetadata]:
    """Parse a document file and extract metadata."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        
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
        doc_id = metadata.get("doc_id", "").replace("Doc Id: ", "").strip() or file_path.stem
        doc_type = metadata.get("doc_type", "").replace("Doc Type: ", "").strip() or "unknown"
        doc_date = metadata.get("date", "").replace("Date: ", "").strip()
        
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
):
    """Background task for processing documents."""
    try:
        processing_status[session_id] = {
            "status": "processing",
            "message": "Tagging documents...",
        }
        
        llm = get_llm_provider(provider_name, model_override)

        # Tag documents (optional)
        tagger = Tagger(ontology)
        tagger_result, tagged_docs = await tagger.tag_documents(documents, session_id)
        storage.save_tagger(tagger_result, session_id)
        
        processing_status[session_id] = {
            "status": "processing",
            "message": "Extracting fields...",
        }
        
        # Extract fields
        extractor = Extractor(
            ontology,
            llm_provider=llm,
            provider_label=provider_name,
        )
        extraction_result = await extractor.extract(tagged_docs, session_id)
        
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
        consolidation_result = await consolidator.consolidate(extraction_result, session_id)
        
        # Save consolidation result
        storage.save_consolidation(consolidation_result, session_id)
        
        processing_status[session_id] = {
            "status": "completed",
            "message": "Processing completed successfully",
        }
    except Exception as e:
        processing_status[session_id] = {
            "status": "failed",
            "message": f"Processing failed: {str(e)}",
        }

