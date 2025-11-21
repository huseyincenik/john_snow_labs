"""Document extractor using DocETL map operations."""
import asyncio
import time
from typing import List, Dict, Any, Optional
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from src.models.schemas import (
    DocumentMetadata,
    ExtractedField,
    DocumentExtraction,
    ExtractionResult,
    FieldEvidence,
    SourceReference,
)
from src.utils.llm import get_llm_provider
from src.utils.ontology import OntologyLoader
from src.utils.logger import setup_logger
from config.settings import settings


class Extractor:
    """Extracts structured fields from documents using DocETL map operations."""
    
    def __init__(
        self,
        ontology: Optional[OntologyLoader] = None,
        llm_provider=None,
        provider_label: str = "default",
    ):
        self.ontology = ontology or OntologyLoader()
        self.fields_to_extract = self.ontology.get_all_fields()
        self.llm_provider = llm_provider or get_llm_provider()
        self.executor = ThreadPoolExecutor(max_workers=settings.max_workers)
        self.provider_label = provider_label
        self.response_format = self._build_response_schema(self.fields_to_extract)
    
    async def extract(
        self,
        documents: List[DocumentMetadata],
        session_id: str,
    ) -> ExtractionResult:
        """Extract fields from all documents."""
        logger = setup_logger(session_id, "stage_extractor")
        start_time = time.time()
        
        logger.info(f"Starting extraction for {len(documents)} documents")
        
        # Get all fields to extract
        fields_to_extract = self.fields_to_extract
        extraction_instructions = self.ontology.get_extraction_instructions()
        
        # Process documents concurrently with semaphore
        semaphore = asyncio.Semaphore(settings.max_concurrent_requests)
        tasks = [
            self._extract_document(doc, fields_to_extract, extraction_instructions, semaphore, logger)
            for doc in documents
        ]
        
        document_results = await asyncio.gather(*tasks)
        
        total_fields = sum(doc.total_fields_extracted for doc in document_results)
        processing_time = time.time() - start_time
        
        result = ExtractionResult(
            session_id=session_id,
            generated_timestamp=datetime.now(),
            stage="stage_extractor",
            total_documents_processed=len(documents),
            document_results=document_results,
            total_fields_extracted=total_fields,
            processing_time_seconds=processing_time,
        )
        
        logger.info(f"Extraction completed in {processing_time:.2f} seconds")
        return result
    
    async def _extract_document(
        self,
        document: DocumentMetadata,
        fields_to_extract: List[Dict[str, Any]],
        extraction_instructions: str,
        semaphore: asyncio.Semaphore,
        logger,
    ) -> DocumentExtraction:
        """Extract fields from a single document."""
        async with semaphore:
            start_time = time.time()
            logger.info(f"Extracting from document: {document.doc_id}")
            
            # Build extraction prompt
            prompt = self._build_extraction_prompt(
                document,
                fields_to_extract,
                extraction_instructions,
            )
            
            # Generate extraction using LLM
            try:
                provider_name = getattr(self.llm_provider, "provider_name", "").lower()
                response_format = (
                    self.response_format
                    if provider_name in {"openai", "qwen", "local"}
                    else None
                )
                response = await self.llm_provider.generate(
                    prompt=prompt,
                    system_prompt="You are a medical data extraction expert. Extract structured data from clinical documents following the provided schema.",
                    temperature=0.0,
                    response_format=response_format,
                )
                
                # Log response preview for debugging
                if not response or len(response.strip()) == 0:
                    logger.warning(
                        "Empty response from LLM for doc %s",
                        document.doc_id,
                    )
                else:
                    logger.debug(
                        "LLM response for doc %s (first 500 chars): %s",
                        document.doc_id,
                        response[:500],
                    )
                
                # Parse response
                extracted_fields = self._parse_extraction_response(
                    response,
                    document,
                    fields_to_extract,
                    logger,
                )
                
                processing_time = time.time() - start_time
                
                logger.info(f"Extracted {len(extracted_fields)} fields from {document.doc_id}")
                
                return DocumentExtraction(
                    doc_id=document.doc_id,
                    extracted_fields=extracted_fields,
                    total_fields_extracted=len(extracted_fields),
                    processing_time_seconds=processing_time,
                )
            except Exception as e:
                logger.exception(
                    "LLM extraction error (doc=%s, provider=%s, model=%s, exc=%s)",
                    document.doc_id,
                    self.provider_label,
                    getattr(self.llm_provider, "model", "unknown"),
                    repr(e),
                )
                return DocumentExtraction(
                    doc_id=document.doc_id,
                    extracted_fields=[],
                    total_fields_extracted=0,
                    processing_time_seconds=time.time() - start_time,
                )
    
    def _build_extraction_prompt(
        self,
        document: DocumentMetadata,
        fields_to_extract: List[Dict[str, Any]],
        extraction_instructions: str,
    ) -> str:
        """Build prompt for extraction."""
        field_names = [f.get("name") for f in fields_to_extract if f.get("name")]
        prompt_parts = [
            extraction_instructions,
            "\n## Document to Extract From:",
            f"Document ID: {document.doc_id}",
            f"Document Type: {document.doc_type}",
            f"Document Date: {document.doc_date or 'Not specified'}",
            f"Patient ID: {document.patient_id}",
            "\n## Document Content:",
            document.content,
            "\n## Extraction Task:",
            "You MUST output a strict JSON object with a top-level key `extractions` that contains an array of field extractions. "
            "Produce one entry for every field listed below, even if the value is `Not Reported`. "
            "Use the exact field_name strings provided and keep explanations concise.",
            "\n### Required Field Names:",
            ", ".join(field_names),
            "\nFor each extraction provide:",
            "1. Provide the raw_value found in the document",
            "2. Provide the normalized_value (standardized format)",
            "3. Include reasoning_excerpt showing where in the document the value was found",
            "4. Provide explanation for why this value was chosen",
            "5. Set confidence_score (0.0 to 1.0) based on certainty",
            "\nReturn ONLY valid JSON matching the schema; do not include narrative text outside the JSON payload.",
        ]
        return "\n".join(prompt_parts)
    
    def _parse_extraction_response(
        self,
        response: str,
        document: DocumentMetadata,
        fields_to_extract: List[Dict[str, Any]],
        logger,
    ) -> List[ExtractedField]:
        """Parse LLM response into ExtractedField objects."""
        import json
        import re
        extracted_fields = []
        field_lookup = {f.get("name"): f for f in fields_to_extract if f.get("name")}
        
        # Clean markdown code blocks
        cleaned_response = response.strip()
        # Remove ```json and ``` markers (opening or closing, even if same line)
        cleaned_response = re.sub(r"^```json\s*", "", cleaned_response, flags=re.IGNORECASE)
        cleaned_response = re.sub(r"^```\s*", "", cleaned_response)
        cleaned_response = re.sub(r"\s*```$", "", cleaned_response)
        if cleaned_response.startswith("```"):
            cleaned_response = cleaned_response[3:]
            if cleaned_response.lower().startswith("json"):
                cleaned_response = cleaned_response[4:]
        if cleaned_response.endswith("```"):
            cleaned_response = cleaned_response[:-3]
        cleaned_response = cleaned_response.replace("```json", "").replace("```", "")
        cleaned_response = cleaned_response.strip()
        
        try:
            parsed = json.loads(cleaned_response)
        except json.JSONDecodeError:
            try:
                # Attempt to extract JSON with balanced braces
                start = cleaned_response.find("{")
                if start == -1:
                    parsed = []
                else:
                    # Find matching closing brace by counting braces
                    brace_count = 0
                    end = start
                    in_string = False
                    escape_next = False
                    
                    for i, char in enumerate(cleaned_response[start:], start):
                        if escape_next:
                            escape_next = False
                            continue
                        if char == '\\':
                            escape_next = True
                            continue
                        if char == '"' and not escape_next:
                            in_string = not in_string
                        if not in_string:
                            if char == '{':
                                brace_count += 1
                            elif char == '}':
                                brace_count -= 1
                                if brace_count == 0:
                                    end = i
                                    break
                    
                    if brace_count == 0 and end > start:
                        json_str = cleaned_response[start : end + 1]
                        parsed = json.loads(json_str)
                    else:
                        # Fallback: try regex match
                        json_match = re.search(r'\{.*\}', cleaned_response, re.DOTALL)
                        if json_match:
                            parsed = json.loads(json_match.group())
                        else:
                            parsed = []
            except Exception as exc:
                logger.warning(
                    "Failed to parse JSON for doc %s: %s\nPayload (first 500 chars): %s",
                    document.doc_id,
                    exc,
                    cleaned_response[:500],
                )
                logger.debug(
                    "Full response for doc %s (first 1000 chars): %s",
                    document.doc_id,
                    cleaned_response[:1000],
                )
                parsed = []
        
        if isinstance(parsed, dict) and "extractions" in parsed:
            fields_data = parsed["extractions"]
            logger.debug(
                "Found 'extractions' key in parsed dict for doc %s with %d items",
                document.doc_id,
                len(fields_data),
            )
        elif isinstance(parsed, list):
            fields_data = parsed
            logger.debug(
                "Parsed response is a list for doc %s with %d items",
                document.doc_id,
                len(fields_data),
            )
        elif isinstance(parsed, dict):
            # Check if it's an empty dict or has different structure
            logger.warning(
                "Parsed dict for doc %s does not contain 'extractions' key. Keys: %s. Response preview: %s",
                document.doc_id,
                list(parsed.keys()) if parsed else "empty",
                cleaned_response[:200] if cleaned_response else "Empty response",
            )
            fields_data = []
        else:
            logger.warning(
                "Unexpected extraction payload type (%s) for doc %s. Response preview: %s",
                type(parsed),
                document.doc_id,
                cleaned_response[:200] if cleaned_response else "Empty response",
            )
            fields_data = []
        
        logger.debug(
            "Parsed %d fields from doc %s. Expected fields: %d",
            len(fields_data),
            document.doc_id,
            len(field_lookup),
        )
        
        skipped_fields: list[str] = []
        missing_field_name_count = 0
        for field_data in fields_data:
            field_name = field_data.get("field_name")
            if not field_name:
                missing_field_name_count += 1
                logger.debug(
                    "Skipping entry without field_name in doc %s: %s",
                    document.doc_id,
                    field_data,
                )
                continue

            field_def = field_lookup.get(field_name)
            if not field_def:
                skipped_fields.append(str(field_name))
                logger.debug(
                    "Skipping field '%s' for doc %s (not in ontology)",
                    field_name,
                    document.doc_id,
                )
                continue
            
            evidence = FieldEvidence(
                explanation=field_data.get("explanation", ""),
                inferred=field_data.get("inferred", False),
                related_entities=field_data.get("related_entities", []),
            )
            
            source = SourceReference(
                doc_id=document.doc_id,
                split_number=1,
                reasoning_excerpt=field_data.get("reasoning_excerpt", ""),
            )
            
            extracted_field = ExtractedField(
                field_name=field_name,
                category=field_def.get("category", "unknown"),
                raw_value=field_data.get("raw_value"),
                normalized_value=field_data.get("normalized_value"),
                units=field_data.get("units"),
                vocabulary_code=field_data.get("vocabulary_code"),
                field_evidence=evidence,
                confidence_level=field_data.get("confidence_level", "medium"),
                confidence_score=field_data.get("confidence_score", 0.5),
                sources=[source],
                extraction_timestamp=datetime.now(),
                data_type=field_def.get("data_type", "string"),
            )
            extracted_fields.append(extracted_field)
        
        if missing_field_name_count:
            logger.warning(
                "Doc %s contained %d extraction entries without field_name",
                document.doc_id,
                missing_field_name_count,
            )

        if skipped_fields:
            logger.warning(
                "Skipped %d fields for doc %s (not in ontology): %s",
                len(skipped_fields),
                document.doc_id,
                ", ".join(skipped_fields[:10]),  # İlk 10'unu göster
            )
        
        if len(fields_data) > 0 and len(extracted_fields) == 0:
            logger.warning(
                "No fields extracted from doc %s despite %d fields in response. "
                "Field names in response: %s",
                document.doc_id,
                len(fields_data),
                ", ".join([f.get("field_name", "unknown") for f in fields_data[:10]]),
            )
        
        return extracted_fields

    def _build_response_schema(self, fields: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Build JSON schema for OpenAI structured output."""
        field_names = [f.get("name") for f in fields if f.get("name")]
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "extraction_response",
                "strict": True,
                "schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "extractions": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "field_name": {
                                        "type": "string",
                                        "enum": field_names,
                                    },
                                    "raw_value": {"type": ["string", "number", "boolean", "null"]},
                                    "normalized_value": {"type": ["string", "number", "boolean", "null"]},
                                    "units": {"type": ["string", "null"]},
                                    "vocabulary_code": {"type": ["string", "null"]},
                                    "reasoning_excerpt": {"type": ["string", "null"]},
                                    "explanation": {"type": ["string", "null"]},
                                    "confidence_score": {"type": "number"},
                                },
                                "required": [
                                    "field_name",
                                    "raw_value",
                                    "normalized_value",
                                    "units",
                                    "vocabulary_code",
                                    "reasoning_excerpt",
                                    "explanation",
                                    "confidence_score",
                                ],
                            },
                        }
                    },
                    "required": ["extractions"],
                },
            },
        }

