"""Patient-level consolidator using DocETL resolve/reduce operations."""
import time
from typing import List, Dict, Any, Optional
from datetime import datetime
from src.models.schemas import (
    ExtractionResult,
    ConsolidatedField,
    ConsolidationResult,
    SupportingEvidence,
)
from src.utils.llm import get_llm_provider
from src.utils.ontology import OntologyLoader
from src.utils.logger import setup_logger
from config.settings import settings


class Consolidator:
    """Consolidates document-level extractions to patient-level."""
    
    def __init__(
        self,
        ontology: Optional[OntologyLoader] = None,
        llm_provider=None,
        provider_label: str = "default",
    ):
        self.ontology = ontology or OntologyLoader()
        self.llm_provider = llm_provider or get_llm_provider()
        self.provider_label = provider_label
    
    async def consolidate(
        self,
        extraction_result: ExtractionResult,
        session_id: str,
    ) -> ConsolidationResult:
        """Consolidate extraction results at patient level."""
        logger = setup_logger(session_id, "stage_consolidator")
        start_time = time.time()
        
        logger.info("Starting patient-level consolidation")
        
        # Group documents by patient
        patient_docs = {}
        for doc_result in extraction_result.document_results:
            # Extract patient_id from doc_id or use first document's patient
            patient_id = self._extract_patient_id(doc_result.doc_id)
            if patient_id not in patient_docs:
                patient_docs[patient_id] = []
            patient_docs[patient_id].append(doc_result)
        
        # Consolidate for each patient
        consolidated_fields = []
        for patient_id, doc_results in patient_docs.items():
            patient_fields = await self._consolidate_patient_fields(
                patient_id,
                doc_results,
                logger,
            )
            consolidated_fields.extend(patient_fields)
        
        processing_time = time.time() - start_time
        
        result = ConsolidationResult(
            session_id=session_id,
            generated_timestamp=datetime.now(),
            stage="stage_consolidator",
            total_fields_consolidated=len(consolidated_fields),
            consolidated_fields=consolidated_fields,
            processing_time_seconds=processing_time,
        )
        
        logger.info(f"Consolidation completed in {processing_time:.2f} seconds")
        return result
    
    async def _consolidate_patient_fields(
        self,
        patient_id: str,
        doc_results: List,
        logger,
    ) -> List[ConsolidatedField]:
        """Consolidate fields for a single patient."""
        # Build consolidation prompt
        prompt = self._build_consolidation_prompt(patient_id, doc_results)
        
        try:
            response = await self.llm_provider.generate(
                prompt=prompt,
                system_prompt="You are a medical data consolidation expert. Consolidate field extractions from multiple documents into a single patient-level record, resolving conflicts and normalizing values.",
                temperature=0.0,
            )
            
            # Parse consolidation response
            consolidated_fields = self._parse_consolidation_response(
                response,
                patient_id,
                doc_results,
            )
            
            return consolidated_fields
        except Exception as e:
            logger.exception(
                "LLM consolidation error (patient=%s, provider=%s, model=%s, exc=%s)",
                patient_id,
                self.provider_label,
                getattr(self.llm_provider, "model", "unknown"),
                repr(e),
            )
            return []
    
    def _build_consolidation_prompt(
        self,
        patient_id: str,
        doc_results: List,
    ) -> str:
        """Build prompt for consolidation."""
        prompt_parts = [
            f"# Patient-Level Consolidation for Patient: {patient_id}",
            "\n## Document-Level Extractions:",
        ]
        
        for doc_result in doc_results:
            prompt_parts.append(f"\n### Document: {doc_result.doc_id}")
            for field in doc_result.extracted_fields:
                prompt_parts.append(
                    f"- {field.field_name}: {field.raw_value} -> {field.normalized_value} "
                    f"(confidence: {field.confidence_score})"
                )
        
        prompt_parts.extend([
            "\n## Consolidation Task:",
            "Consolidate the above extractions into a single patient-level record:",
            "1. Resolve conflicts between documents (prefer higher confidence, more recent dates)",
            "2. Normalize values to standard formats",
            "3. Aggregate list values (e.g., multiple diagnoses)",
            "4. Track supporting evidence from each document",
            "\nReturn consolidated fields as JSON.",
        ])
        
        return "\n".join(prompt_parts)
    
    def _parse_consolidation_response(
        self,
        response: str,
        patient_id: str,
        doc_results: List,
    ) -> List[ConsolidatedField]:
        """Parse consolidation response into ConsolidatedField objects."""
        import json
        import re
        
        consolidated_fields = []
        
        try:
            # Try to extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                consolidation_data = json.loads(json_match.group())
            else:
                consolidation_data = json.loads(response)
            
            # Convert to ConsolidatedField objects
            # This is a simplified version - full implementation would handle all field types
            for field_name, field_data in consolidation_data.items():
                consolidated_field = ConsolidatedField(
                    field_name=field_name,
                    category="consolidated",
                    consolidated_value=field_data,
                    all_values=field_data.get("all_values", []),
                    confidence_score=field_data.get("confidence_score", 0.5),
                    source_documents=[doc.doc_id for doc in doc_results],
                    consolidation_reasoning=field_data.get("reasoning", ""),
                    data_type="object",
                )
                consolidated_fields.append(consolidated_field)
        except Exception as e:
            pass
        
        return consolidated_fields
    
    def _extract_patient_id(self, doc_id: str) -> str:
        """Extract patient ID from document ID."""
        # Simple extraction - assumes format like "doc_001_jsl_p01_001_summary_doc"
        # Extract "p01" from the doc_id
        import re
        match = re.search(r'p\d+', doc_id)
        if match:
            return match.group()
        return "unknown"

