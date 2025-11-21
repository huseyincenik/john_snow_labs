"""Document tagger for classification and ordering."""
from __future__ import annotations

from pathlib import Path
import time
from typing import List, Optional, Tuple
from datetime import datetime

from src.models.schemas import (
    DocumentMetadata,
    TaggerResult,
    TaggedDocument,
    SplitInfo,
)
from src.utils.ontology import OntologyLoader


class Tagger:
    """Tags and orders documents chronologically."""
    
    def __init__(self, ontology: Optional[OntologyLoader] = None):
        self.ontology = ontology or OntologyLoader()
    
    async def tag_documents(
        self,
        documents: List[DocumentMetadata],
        session_id: str,
    ) -> Tuple[TaggerResult, List[DocumentMetadata]]:
        """Tag and order documents by date."""
        start_time = time.time()
        sorted_docs = sorted(
            documents,
            key=lambda doc: self._parse_date(doc.doc_date) or datetime.min,
        )
        
        tagged_documents = [self._build_tagged_document(doc) for doc in sorted_docs]
        
        tagger_result = TaggerResult(
            session_id=session_id,
            generated_timestamp=datetime.utcnow(),
            total_documents=len(documents),
            sorted_documents=tagged_documents,
            processing_time_seconds=time.time() - start_time,
        )
        
        return tagger_result, sorted_docs
    
    def _parse_date(self, date_str: Optional[str]) -> Optional[datetime]:
        """Parse date string to datetime."""
        if not date_str:
            return None
        
        for fmt in ["%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y", "%d/%m/%Y"]:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        return None
    
    def _build_tagged_document(self, document: DocumentMetadata) -> TaggedDocument:
        """Convert DocumentMetadata into TaggedDocument schema."""
        filename = Path(document.filename).name if document.filename else document.doc_id
        type_confidence = 0.95 if document.doc_type and document.doc_type != "unknown" else 0.5
        date_confidence = 0.85 if document.doc_date else 0.0
        
        split_info = SplitInfo(
            is_split=False,
            split_number=1,
            total_splits=1,
            estimated_tokens=self._estimate_tokens(document.content),
        )
        
        return TaggedDocument(
            doc_id=document.doc_id,
            filename=filename,
            doc_type=document.doc_type,
            doc_date=document.doc_date,
            type_confidence=type_confidence,
            date_confidence=date_confidence,
            content=document.content,
            split_info=split_info,
        )
    
    def _estimate_tokens(self, content: str) -> int:
        """Rudimentary token estimation for logging/preview purposes."""
        if not content:
            return 0
        # Simple heuristic: assume ~4 characters per token
        return max(1, len(content) // 4)

