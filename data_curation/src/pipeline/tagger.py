"""Document tagger for classification and ordering."""

from __future__ import annotations

import asyncio
import json
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
from src.utils.llm import get_llm_provider, LLMProvider
from src.utils.logger import setup_logger
from src.utils.ontology import OntologyLoader


class Tagger:
    """Tags and orders documents chronologically."""

    def __init__(
        self,
        ontology: Optional[OntologyLoader] = None,
        llm_provider: Optional[LLMProvider] = None,
        provider_label: str = "default",
    ):
        self.ontology = ontology or OntologyLoader()
        self.llm_provider = llm_provider or get_llm_provider()
        self.provider_label = provider_label
        # Check if using Qwen model (for Qwen-specific optimizations)
        model_name = getattr(self.llm_provider, "model", "") or ""
        self.is_qwen_model = any(
            keyword in model_name.lower() for keyword in ("qwen", "yi-", "glm", "deepseek")
        )

    async def tag_documents(
        self,
        documents: List[DocumentMetadata],
        session_id: str,
    ) -> Tuple[TaggerResult, List[DocumentMetadata]]:
        """Tag and order documents by date with LLM-based confidence scoring."""
        logger = setup_logger(session_id, "stage_tagger")
        logger.info(
            "Starting tagger for %d documents via provider=%s",
            len(documents),
            self.provider_label,
        )

        start_time = time.time()
        sorted_docs = sorted(
            documents,
            key=lambda doc: self._parse_date(doc.doc_date) or datetime.min,
        )

        # Calculate confidence scores in parallel using LLM
        try:
            tagged_documents = await self._build_tagged_documents_with_confidence(
                sorted_docs, session_id, logger
            )
        except Exception as e:
            logger.error("Error calculating confidence scores with LLM: %s", e, exc_info=True)
            logger.warning("Falling back to default confidence scores")
            # Fallback to default confidence scores if LLM fails
            tagged_documents = [
                self._build_tagged_document(
                    doc,
                    type_confidence=0.95 if doc.doc_type and doc.doc_type != "unknown" else 0.5,
                    date_confidence=0.85 if doc.doc_date else 0.0,
                )
                for doc in sorted_docs
            ]

        tagger_result = TaggerResult(
            session_id=session_id,
            generated_timestamp=datetime.utcnow(),
            total_documents=len(documents),
            sorted_documents=tagged_documents,
            processing_time_seconds=time.time() - start_time,
        )

        logger.info(
            "Tagger completed: %d documents processed in %.2f seconds",
            len(documents),
            time.time() - start_time,
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

    async def _build_tagged_documents_with_confidence(
        self,
        documents: List[DocumentMetadata],
        session_id: str,
        logger,
    ) -> List[TaggedDocument]:
        """Build tagged documents with LLM-calculated confidence scores in parallel."""
        # Optimized: Use higher concurrency for LLM calls (I/O-bound operations)
        # Increased from 10 to 20 for better throughput
        semaphore = asyncio.Semaphore(20)  # Increased concurrent LLM calls

        async def process_document(doc: DocumentMetadata) -> TaggedDocument:
            async with semaphore:
                try:
                    type_confidence, date_confidence = await self._calculate_confidence_scores(
                        doc, session_id, logger
                    )
                except Exception as e:
                    logger.warning(
                        "Error calculating confidence for doc %s: %s. Using defaults.",
                        doc.doc_id,
                        e,
                    )
                    # Fallback to default confidence scores
                    type_confidence = 0.95 if doc.doc_type and doc.doc_type != "unknown" else 0.5
                    date_confidence = 0.85 if doc.doc_date else 0.0

                return self._build_tagged_document(doc, type_confidence, date_confidence)

        tasks = [process_document(doc) for doc in documents]
        return await asyncio.gather(*tasks)

    async def _calculate_confidence_scores(
        self,
        document: DocumentMetadata,
        session_id: str,
        logger,
    ) -> Tuple[float, float]:
        """Calculate confidence scores for document type and date using LLM."""
        prompt = self._build_confidence_prompt(document)
        system_prompt = "You are a medical document classification expert. Analyze document metadata and content to assess confidence scores for document type and date classification."

        # Log prompt to file
        self._log_prompt(session_id, document.doc_id, prompt, system_prompt, logger)

        try:
            # Prepare extra_body for Qwen models (same as DocETL)
            extra_body = None
            if self.is_qwen_model:
                extra_body = {
                    "top_p": 0.95,
                    "temperature": 0.1,
                    "frequency_penalty": 0.0,
                    "presence_penalty": 0.0,
                    "reasoning_effort": "none",  # Disable reasoning completely
                }

            response = await self.llm_provider.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.1,
                max_tokens=150,  # Reduced from 200 for faster responses
                response_format={"type": "json_object"},  # Force JSON response format
                extra_body=extra_body,  # Qwen-specific optimizations
            )

            if not response:
                logger.error(
                    "Empty response from LLM for doc %s",
                    document.doc_id,
                )
                raise ValueError("Empty response from LLM")

            logger.debug(
                "LLM confidence response for doc %s: %s",
                document.doc_id,
                response[:200] if response else "None",
            )

            # Log response to file
            self._log_response(session_id, document.doc_id, response, logger)

            # Parse JSON response
            try:
                # Try to extract JSON from response (might be wrapped in markdown)
                response_clean = response.strip()
                if response_clean.startswith("```"):
                    # Remove markdown code blocks
                    lines = response_clean.split("\n")
                    response_clean = "\n".join(lines[1:-1]) if len(lines) > 2 else response_clean
                elif response_clean.startswith("```json"):
                    lines = response_clean.split("\n")
                    response_clean = "\n".join(lines[1:-1]) if len(lines) > 2 else response_clean

                result = json.loads(response_clean)
                type_confidence = float(result.get("type_confidence", 0.5))
                date_confidence = float(result.get("date_confidence", 0.0))

                # Validate scores are in valid range
                type_confidence = max(0.0, min(1.0, type_confidence))
                date_confidence = max(0.0, min(1.0, date_confidence))

                logger.debug(
                    "Parsed confidence scores for doc %s: type=%.2f, date=%.2f",
                    document.doc_id,
                    type_confidence,
                    date_confidence,
                )

                return type_confidence, date_confidence

            except (json.JSONDecodeError, ValueError, KeyError) as e:
                logger.warning(
                    "Failed to parse LLM response for doc %s: %s. Response: %s",
                    document.doc_id,
                    e,
                    response[:200] if response else "None",
                )
                raise

        except Exception as e:
            logger.error(
                "Error calling LLM for confidence scores (doc %s): %s",
                document.doc_id,
                e,
                exc_info=True,
            )
            raise

    def _build_confidence_prompt(self, document: DocumentMetadata) -> str:
        """Build prompt for confidence score calculation."""
        content_preview = document.content[:500] if document.content else ""

        return f"""Analyze the following medical document metadata and content preview to assess confidence scores for document type and date classification.

Document ID: {document.doc_id}
Filename: {document.filename or "N/A"}
Document Type (from metadata): {document.doc_type or "unknown"}
Document Date (from metadata): {document.doc_date or "N/A"}

Content Preview (first 500 chars):
{content_preview}

Task: Assess the confidence that the document type and date are correctly classified.

For type_confidence:
- 0.95-1.0: Document type is explicitly stated and clearly identifiable (e.g., "pathology report", "operative note", "summary")
- 0.85-0.94: Document type can be inferred with high confidence from structure/content
- 0.70-0.84: Document type is somewhat ambiguous but likely correct
- 0.50-0.69: Document type is uncertain or "unknown"
- 0.0-0.49: Document type appears incorrect or cannot be determined

For date_confidence:
- 0.90-1.0: Date is explicitly stated in standard format (YYYY-MM-DD) and clearly identifiable
- 0.75-0.89: Date can be inferred with high confidence from content or metadata
- 0.60-0.74: Date is somewhat ambiguous but likely correct
- 0.30-0.59: Date is uncertain or partially missing
- 0.0-0.29: Date cannot be determined or appears incorrect

CRITICAL: You MUST respond with ONLY valid JSON. Do not include any markdown, explanations, or additional text.

Return a JSON object with exactly these fields:
{{
    "type_confidence": <float between 0.0 and 1.0>,
    "date_confidence": <float between 0.0 and 1.0>,
    "reasoning": "<brief explanation of scores>"
}}

Start your response directly with {{ and end with }}. No code blocks, no markdown."""

    def _build_tagged_document(
        self,
        document: DocumentMetadata,
        type_confidence: float,
        date_confidence: float,
    ) -> TaggedDocument:
        """Convert DocumentMetadata into TaggedDocument schema with provided confidence scores."""
        filename = Path(document.filename).name if document.filename else document.doc_id

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

    def _log_prompt(
        self,
        session_id: str,
        doc_id: str,
        prompt: str,
        system_prompt: str,
        logger,
    ) -> None:
        """Log prompt to file for auditability."""
        try:
            from config.settings import settings

            log_path = Path(settings.log_dir) / session_id / "stage_tagger_prompts.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)

            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"\n{'='*80}\n")
                f.write(f"Document ID: {doc_id}\n")
                f.write(f"Timestamp: {datetime.utcnow().isoformat()}\n")
                f.write(f"{'='*80}\n\n")
                f.write(f"SYSTEM PROMPT:\n{system_prompt}\n\n")
                f.write(f"USER PROMPT:\n{prompt}\n\n")
                f.write(f"{'-'*80}\n")
        except Exception as e:
            logger.warning("Failed to log prompt: %s", e)

    def _log_response(
        self,
        session_id: str,
        doc_id: str,
        response: str,
        logger,
    ) -> None:
        """Log LLM response to file for auditability."""
        try:
            from config.settings import settings

            log_path = Path(settings.log_dir) / session_id / "stage_tagger_prompts.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)

            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"RESPONSE:\n{response}\n\n")
                f.write(f"{'='*80}\n\n")
        except Exception as e:
            logger.warning("Failed to log response: %s", e)
