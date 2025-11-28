"""Document extractor that orchestrates the DocETL pipeline."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
import traceback
from typing import Any, Dict, List, Optional, Tuple

from config.settings import settings
from src.models.schemas import (
    DocumentExtraction,
    DocumentMetadata,
    ExtractionResult,
    ExtractedField,
    FieldEvidence,
    SourceReference,
)
from src.pipeline.docetl_runner import (
    DocETLPipelineArtifacts,
    DocETLPipelineRunner,
)
from src.utils.llm import get_llm_provider
from src.utils.logger import setup_logger
from src.utils.ontology import OntologyLoader


class Extractor:
    """Runs DocETL map + resolve + reduce and exposes document-level outputs."""

    def __init__(
        self,
        ontology: Optional[OntologyLoader] = None,
        llm_provider=None,
        provider_label: str = "default",
        max_parallel_patients: Optional[int] = None,
        docetl_thread_override: Optional[int] = None,
    ):
        self.ontology = ontology or OntologyLoader()
        self.llm_provider = llm_provider or get_llm_provider()
        self.provider_label = provider_label
        self.model_name = getattr(self.llm_provider, "model", None)
        self._max_parallel_patients = (
            max_parallel_patients if (max_parallel_patients or 0) > 0 else None
        )
        self._docetl_thread_override = (
            docetl_thread_override if (docetl_thread_override or 0) > 0 else None
        )
        self._model_failover_order = self._build_model_failover_order()

    async def extract(
        self,
        documents: List[DocumentMetadata],
        session_id: str,
    ) -> Tuple[ExtractionResult, DocETLPipelineArtifacts]:
        """Execute DocETL and convert outputs into `ExtractionResult`."""
        logger = setup_logger(session_id, "stage_extractor")
        logger.info(
            "Launching DocETL pipeline for %d documents via provider=%s",
            len(documents),
            self.provider_label,
        )

        start_ts = time.time()
        grouped_docs = self._group_documents_by_patient(documents)
        patient_runs = await self._run_parallel_pipelines(
            grouped_docs,
            session_id,
            logger,
        )
        artifacts = self._combine_artifact_outputs(
            session_id,
            patient_runs,
            documents,
        )
        logger.info(
            "DocETL pipeline finished (map=%s, patient=%s)",
            artifacts.map_output_path,
            artifacts.patient_output_path,
        )

        map_records = artifacts.map_records or []
        document_results = self._build_document_extractions(map_records)
        total_fields = sum(doc.total_fields_extracted for doc in document_results)
        processing_time = time.time() - start_ts

        extraction_result = ExtractionResult(
            session_id=session_id,
            generated_timestamp=datetime.now(),
            stage="stage_extractor",
            total_documents_processed=len(documents),
            document_results=document_results,
            total_fields_extracted=total_fields,
            processing_time_seconds=processing_time,
        )

        logger.info(
            "Extraction complete: %d docs, %d fields, %.2fs",
            len(document_results),
            total_fields,
            processing_time,
        )
        return extraction_result, artifacts

    def _group_documents_by_patient(
        self,
        documents: List[DocumentMetadata],
    ) -> OrderedDict[str, List[DocumentMetadata]]:
        grouped: OrderedDict[str, List[DocumentMetadata]] = OrderedDict()
        for doc in documents:
            grouped.setdefault(doc.patient_id, []).append(doc)
        return grouped

    async def _run_parallel_pipelines(
        self,
        grouped_docs: OrderedDict[str, List[DocumentMetadata]],
        session_id: str,
        logger,
    ) -> List[Tuple[str, DocETLPipelineArtifacts]]:
        if not grouped_docs:
            return []

        parallel_limit = self._determine_parallel_limit(len(grouped_docs))
        logger.info(
            "Running %d patient batches with up to %d concurrent DocETL pipelines",
            len(grouped_docs),
            parallel_limit,
        )
        semaphore = asyncio.Semaphore(parallel_limit)

        async def run_patient(
            patient_id: str,
            patient_docs: List[DocumentMetadata],
        ) -> Tuple[str, DocETLPipelineArtifacts]:
            patient_session_id = self._build_patient_session_id(session_id, patient_id)
            last_error: Optional[Exception] = None

            for model_candidate in self._model_failover_order:
                runner = self._build_pipeline_runner(model_candidate)
                logger.info(
                    "Running DocETL for patient %s (%d documents) via model=%s",
                    patient_id,
                    len(patient_docs),
                    model_candidate,
                )
                try:
                    # Use semaphore to limit concurrent patients, but allow more parallelism
                    # within each patient's DocETL pipeline
                    async with semaphore:
                        # Use asyncio.to_thread for better async integration
                        # This allows other patients to proceed while this one waits for I/O
                        artifacts = await asyncio.to_thread(
                            runner.run_pipeline,
                            patient_docs,
                            patient_session_id,
                        )
                    artifacts.map_records = self._sort_map_records_by_source_docs(
                        artifacts.map_records,
                        patient_docs,
                    )
                    self._relocate_patient_outputs(
                        session_id,
                        patient_id,
                        patient_session_id,
                    )
                    return patient_id, artifacts
                except Exception as exc:
                    last_error = exc
                    if not self._should_failover_due_to_llm_error(exc):
                        raise
                    logger.warning(
                        "Model %s failed for patient %s due to %s. Trying fallback.",
                        model_candidate,
                        patient_id,
                        exc,
                    )
                    self._clear_patient_outputs(patient_session_id)

            if last_error:
                raise last_error
            raise RuntimeError(f"DocETL failed for patient {patient_id} without specific error.")

        # Create all tasks upfront for better async scheduling
        # This allows asyncio to optimize task execution order
        tasks = [asyncio.create_task(run_patient(pid, docs)) for pid, docs in grouped_docs.items()]

        # Use gather with return_exceptions=False to fail fast on errors
        # This provides better error handling and faster failure detection
        try:
            results = await asyncio.gather(*tasks, return_exceptions=False)
            return results
        except Exception as e:
            # Log any unhandled exceptions
            logger.error(f"Error in parallel pipeline execution: {e}")
            # Cancel remaining tasks
            for task in tasks:
                if not task.done():
                    task.cancel()
            raise

    def _combine_artifact_outputs(
        self,
        session_id: str,
        patient_runs: List[Tuple[str, DocETLPipelineArtifacts]],
        documents: List[DocumentMetadata],
    ) -> DocETLPipelineArtifacts:
        session_dir = Path(settings.output_dir) / session_id
        step_dir = session_dir / "docetl_intermediate" / DocETLPipelineRunner.STEP_NAME
        session_dir.mkdir(parents=True, exist_ok=True)
        step_dir.mkdir(parents=True, exist_ok=True)

        map_records: List[Dict[str, Any]] = []
        patient_records: List[Dict[str, Any]] = []
        resolve_records: List[Dict[str, Any]] = []

        for _, artifacts in patient_runs:
            if artifacts.map_records:
                map_records.extend(artifacts.map_records)
            if artifacts.patient_records:
                patient_records.extend(artifacts.patient_records)
            if artifacts.resolve_records:
                resolve_records.extend(artifacts.resolve_records)

        if documents:
            ordering = {doc.doc_id: idx for idx, doc in enumerate(documents)}
            map_records.sort(
                key=lambda record: ordering.get(
                    self._extract_doc_id(record),
                    len(ordering),
                )
            )

        map_output_path = step_dir / f"{DocETLPipelineRunner.MAP_OP}.json"
        resolve_output_path = step_dir / f"{DocETLPipelineRunner.RESOLVE_OP}.json"
        patient_output_path = session_dir / "docetl_patient_results.json"

        self._write_json(map_output_path, map_records)
        self._write_json(patient_output_path, patient_records)
        if resolve_records:
            self._write_json(resolve_output_path, resolve_records)
            resolve_path_value: Optional[Path] = resolve_output_path
        else:
            resolve_path_value = None

        return DocETLPipelineArtifacts(
            session_id=session_id,
            map_output_path=map_output_path,
            patient_output_path=patient_output_path,
            resolve_output_path=resolve_path_value,
            map_records=map_records,
            patient_records=patient_records,
            resolve_records=resolve_records or None,
        )

    def _build_pipeline_runner(
        self, model_name_override: Optional[str] = None
    ) -> DocETLPipelineRunner:
        return DocETLPipelineRunner(
            self.ontology,
            model_name_override or self.model_name,
            max_threads_override=self._docetl_thread_override,
        )

    def _determine_parallel_limit(self, patient_count: int) -> int:
        """Determine optimal parallel limit based on CPU, configured limits, and patient count."""
        if patient_count <= 1:
            return 1

        if self._max_parallel_patients:
            limit = self._max_parallel_patients
        else:
            configured = settings.max_parallel_patients
            if configured and configured > 0:
                limit = configured
            else:
                # Optimized CPU-based calculation: use more cores for parallel processing
                # Formula: (CPU cores * 2) for better I/O-bound LLM operations
                cpu_count = os.cpu_count() or 4
                # Use more aggressive parallelism: CPU cores * 2, minimum 8, maximum 32
                cpu_hint = max(8, min(int(cpu_count * 2), 32))
                limit = cpu_hint

        # Ensure we don't exceed patient count or configured max_concurrent_requests
        # Also consider max_concurrent_requests to avoid API rate limits
        # Increased multiplier from 2 to 3 for better parallel patient processing
        max_allowed = min(
            patient_count,
            limit,
            settings.max_concurrent_requests * 3,  # Allow more patients than concurrent requests
        )
        return max(1, max_allowed)

    def _build_model_failover_order(self) -> List[str]:
        """Build prioritized list of LLM model identifiers for automatic failover."""
        candidates = [
            self.model_name,
            settings.openrouter_model_openai,
            settings.openrouter_model_qwen,
        ]
        order: List[str] = []
        for candidate in candidates:
            if candidate and candidate not in order:
                order.append(candidate)
        return order or [settings.openrouter_model_openai]

    def _should_failover_due_to_llm_error(self, error: Exception) -> bool:
        """Return True if the error looks like a transient LLM/provider outage."""
        message = str(error).lower()
        transient_tokens = [
            "openrouterexception",
            "provider returned error",
            "bad gateway",
            "http 500",
            "http 502",
            "http 503",
            "http 504",
            "server error",
            "litellm.apierro",
            "apierror",
        ]
        return any(token in message for token in transient_tokens)

    def _clear_patient_outputs(self, patient_session_id: str) -> None:
        """Remove partially written outputs before retrying with a fallback model."""
        patient_dir = Path(settings.output_dir) / patient_session_id
        if patient_dir.exists():
            shutil.rmtree(patient_dir, ignore_errors=True)

    @staticmethod
    def _build_patient_session_id(session_id: str, patient_id: str) -> str:
        sanitized = str(patient_id).replace(" ", "_")
        return f"{session_id}__{sanitized}"

    @staticmethod
    def _write_json(path: Path, payload: List[Dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)

    def _relocate_patient_outputs(
        self,
        root_session_id: str,
        patient_id: str,
        patient_session_id: str,
    ) -> None:
        src = Path(settings.output_dir) / patient_session_id
        if not src.exists():
            return
        dest = Path(settings.output_dir) / root_session_id / "patients" / str(patient_id)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            shutil.rmtree(dest)
        shutil.move(str(src), str(dest))

    def _sort_map_records_by_source_docs(
        self,
        map_records: Optional[List[Dict[str, Any]]],
        source_docs: List[DocumentMetadata],
    ) -> List[Dict[str, Any]]:
        if not map_records:
            return []
        ordering = {doc.doc_id: idx for idx, doc in enumerate(source_docs)}
        return sorted(
            map_records,
            key=lambda record: ordering.get(
                self._extract_doc_id(record),
                len(ordering),
            ),
        )

    @staticmethod
    def _extract_doc_id(record: Optional[Dict[str, Any]]) -> Optional[str]:
        if not isinstance(record, dict):
            return None
        return record.get("doc_id") or record.get("document_id")

    def _build_document_extractions(
        self,
        map_records: List[Dict[str, Any]],
    ) -> List[DocumentExtraction]:
        """Convert DocETL map outputs into DocumentExtraction models."""
        from src.utils.logger import setup_logger

        logger = setup_logger("extractor_debug", "stage_extractor")

        if not map_records:
            logger.warning("map_records is empty or None")
            return []

        total_records = len(map_records)
        logger.info(f"Processing {total_records} map records in parallel")

        # Optimized worker count calculation for better parallel processing
        import os

        cpu_count = os.cpu_count() or 4
        # Use more workers: CPU cores * 3 for I/O-bound operations, but respect limits
        optimal_workers = min(
            settings.max_workers,
            max(cpu_count * 3, total_records, 1),
        )
        max_workers = min(optimal_workers, max(1, total_records))

        logger.debug(f"Using {max_workers} workers for map record processing")
        doc_results: List[Optional[DocumentExtraction]] = [None] * total_records

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(self._process_map_record, idx, record, logger)
                for idx, record in enumerate(map_records)
            ]
            for future in as_completed(futures):
                idx, result = future.result()
                if result:
                    doc_results[idx] = result

        return [doc for doc in doc_results if doc is not None]

    def _process_map_record(
        self,
        idx: int,
        record: Any,
        logger,
    ) -> Tuple[int, Optional[DocumentExtraction]]:
        """Process a single DocETL map record."""
        try:
            record_type = type(record).__name__
            logger.debug(
                f"Record {idx}: type={record_type}, value={str(record)[:200] if isinstance(record, str) else 'dict'}"
            )

            if not record:
                logger.warning(f"Record {idx} is None or empty, skipping")
                return idx, None

            if not isinstance(record, dict):
                logger.warning(
                    f"Record {idx} is not a dict, type={record_type}, attempting to parse"
                )
                if isinstance(record, str):
                    try:
                        parsed = json.loads(record)
                        if isinstance(parsed, dict):
                            record = parsed
                            logger.info(f"Record {idx} successfully parsed from string to dict")
                        elif isinstance(parsed, list) and parsed:
                            for item in parsed:
                                if isinstance(item, dict):
                                    record = item
                                    logger.info(
                                        f"Record {idx} extracted from list, using first dict item"
                                    )
                                    break
                            if not isinstance(record, dict):
                                logger.warning(
                                    f"Record {idx} could not be converted to dict from list, skipping"
                                )
                                return idx, None
                        else:
                            logger.warning(
                                f"Record {idx} parsed but not a dict or list with dicts, skipping"
                            )
                            return idx, None
                    except (json.JSONDecodeError, Exception) as parse_err:
                        logger.error(
                            f"Record {idx} JSON parse error: {parse_err}, traceback: {traceback.format_exc()}"
                        )
                        return idx, None
                else:
                    logger.warning(
                        f"Record {idx} is not a dict or string, type={record_type}, skipping"
                    )
                    return idx, None

            doc_id = record.get("doc_id") or record.get("document_id") or "unknown_doc"
            fields_payload = record.get("extractions")
            logger.debug(
                f"Record {idx} (doc_id={doc_id}): extractions type={type(fields_payload).__name__}"
            )

            if not isinstance(fields_payload, (list, tuple)):
                if fields_payload is None:
                    fields_payload = []
                elif isinstance(fields_payload, dict):
                    fields_payload = [fields_payload]
                elif isinstance(fields_payload, str):
                    try:
                        parsed = json.loads(fields_payload)
                        if isinstance(parsed, dict) and "extractions" in parsed:
                            extractions_value = parsed["extractions"]
                            if isinstance(extractions_value, list):
                                fields_payload = extractions_value
                            else:
                                fields_payload = [extractions_value]
                        elif isinstance(parsed, list):
                            fields_payload = parsed
                        elif isinstance(parsed, dict):
                            fields_payload = [parsed]
                        else:
                            fields_payload = []
                    except (json.JSONDecodeError, Exception):
                        fields_payload = []
                else:
                    fields_payload = []
            else:
                fields_payload = list(fields_payload)
                cleaned_payload: List[Dict[str, Any]] = []
                for payload in fields_payload:
                    if isinstance(payload, dict):
                        cleaned_payload.append(payload)
                    elif isinstance(payload, str):
                        try:
                            parsed = json.loads(payload)
                            if isinstance(parsed, dict) and "extractions" in parsed:
                                nested = parsed["extractions"]
                                if isinstance(nested, list):
                                    cleaned_payload.extend(
                                        [item for item in nested if isinstance(item, dict)]
                                    )
                                elif isinstance(nested, dict):
                                    cleaned_payload.append(nested)
                            elif isinstance(parsed, dict):
                                cleaned_payload.append(parsed)
                            elif isinstance(parsed, list):
                                cleaned_payload.extend(
                                    [item for item in parsed if isinstance(item, dict)]
                                )
                        except (json.JSONDecodeError, Exception):
                            continue
                fields_payload = cleaned_payload

            extracted_fields = []
            for payload_idx, payload in enumerate(fields_payload):
                try:
                    if not payload:
                        logger.warning(
                            f"Record {idx}, payload {payload_idx} is None or empty, skipping"
                        )
                        continue

                    if not isinstance(payload, dict):
                        logger.warning(
                            f"Record {idx}, payload {payload_idx} is not a dict, type={type(payload).__name__}, attempting to parse"
                        )
                        if isinstance(payload, str):
                            try:
                                parsed = json.loads(payload)
                                if isinstance(parsed, dict):
                                    payload = parsed
                                    logger.info(
                                        f"Record {idx}, payload {payload_idx} successfully parsed from string to dict"
                                    )
                                else:
                                    logger.warning(
                                        f"Record {idx}, payload {payload_idx} parsed but not a dict, skipping"
                                    )
                                    continue
                            except (json.JSONDecodeError, Exception) as parse_err:
                                logger.error(
                                    f"Record {idx}, payload {payload_idx} JSON parse error: {parse_err}"
                                )
                                continue
                        else:
                            logger.warning(
                                f"Record {idx}, payload {payload_idx} is not a dict or string, skipping"
                            )
                            continue

                    field = self._convert_field_payload(doc_id, payload, record, logger)
                    extracted_fields.append(field)
                except Exception as field_err:
                    logger.error(
                        f"Record {idx}, payload {payload_idx} conversion error: {field_err}, traceback: {traceback.format_exc()}"
                    )
                    continue

            document_extraction = DocumentExtraction(
                doc_id=doc_id,
                extracted_fields=extracted_fields,
                total_fields_extracted=len(extracted_fields),
                processing_time_seconds=0.0,
            )
            return idx, document_extraction
        except Exception as record_err:
            logger.error(
                f"Record {idx} processing error: {record_err}, traceback: {traceback.format_exc()}"
            )
            return idx, None

    def _convert_field_payload(
        self,
        doc_id: str,
        payload: Dict[str, Any],
        record: Optional[Dict[str, Any]] = None,
        logger=None,
    ) -> ExtractedField:
        """Create an ExtractedField from a DocETL payload."""
        if logger is None:
            from src.utils.logger import setup_logger

            logger = setup_logger("extractor_debug", "stage_extractor")

        # Ensure payload is a dict, not a string (Qwen tool call responses)
        if not isinstance(payload, dict):
            logger.error(
                f"_convert_field_payload: payload is not a dict, type={type(payload).__name__}, value={str(payload)[:200]}"
            )
            logger.error(f"_convert_field_payload: traceback: {traceback.format_exc()}")
            if isinstance(payload, str):
                try:
                    parsed = json.loads(payload)
                    if isinstance(parsed, dict):
                        payload = parsed
                        logger.info(
                            "_convert_field_payload: successfully parsed payload from string to dict"
                        )
                    else:
                        raise ValueError(f"Parsed payload is not a dict: {type(parsed).__name__}")
                except (json.JSONDecodeError, Exception) as parse_err:
                    logger.error(f"_convert_field_payload: JSON parse error: {parse_err}")
                    raise ValueError(
                        f"Payload must be a dict, got {type(payload).__name__}: {str(payload)[:200]}"
                    )
            else:
                raise ValueError(
                    f"Payload must be a dict, got {type(payload).__name__}: {str(payload)[:200]}"
                )

        # Ensure record is a dict if provided
        if record is not None and not isinstance(record, dict):
            logger.warning(
                f"_convert_field_payload: record is not a dict, type={type(record).__name__}, converting to empty dict"
            )
            if isinstance(record, str):
                try:
                    import json

                    record = json.loads(record)
                    if not isinstance(record, dict):
                        record = {}
                except (json.JSONDecodeError, Exception):
                    record = {}
            else:
                record = {}

        field_name = payload.get("field_name") or "unknown_field"
        field_def = self.ontology.get_field_definition(field_name) or {}

        # Get base confidence from LLM
        base_confidence = self._safe_float(payload.get("confidence_score"), 0.5)

        # Apply quality-based adjustments
        confidence_score = self._adjust_confidence_by_quality(
            base_confidence,
            payload,
            field_name,
        )

        confidence_level = self._normalize_confidence_level(
            payload.get("confidence_level"),
            confidence_score,
        )

        evidence = FieldEvidence(
            explanation=payload.get("explanation", ""),
            inferred=bool(payload.get("inferred", False)),
            related_entities=self._ensure_list(payload.get("related_entities")),
        )
        source = SourceReference(
            doc_id=doc_id,
            split_number=1,
            reasoning_excerpt=payload.get("reasoning_excerpt", ""),
        )

        # Convert empty strings to None for units and vocabulary_code
        units_value = payload.get("units")
        if units_value == "":
            units_value = None

        vocab_code_value = payload.get("vocabulary_code")
        if vocab_code_value == "":
            vocab_code_value = None

        return ExtractedField(
            field_name=field_name,
            category=payload.get("category") or field_def.get("category", "unknown"),
            raw_value=payload.get("raw_value"),
            normalized_value=payload.get("normalized_value"),
            units=units_value,
            vocabulary_code=vocab_code_value,
            field_evidence=evidence,
            confidence_level=confidence_level,
            confidence_score=confidence_score,
            sources=[source],
            extraction_timestamp=datetime.now(),
            data_type=payload.get("data_type") or field_def.get("data_type", "string"),
        )

    def _adjust_confidence_by_quality(
        self,
        base_confidence: float,
        payload: Dict[str, Any],
        field_name: str,
    ) -> float:
        """Adjust confidence score based on evidence quality indicators."""
        adjusted = base_confidence

        # Quality indicators
        reasoning_excerpt = payload.get("reasoning_excerpt", "")
        explanation = payload.get("explanation", "")
        raw_value = payload.get("raw_value", "")
        inferred = payload.get("inferred", False)
        related_entities = payload.get("related_entities", [])

        # Penalize if reasoning excerpt is missing or generic
        if not reasoning_excerpt or reasoning_excerpt.strip() == "":
            adjusted *= 0.7  # -30% for missing evidence
        elif len(reasoning_excerpt) < 10:
            adjusted *= 0.85  # -15% for very short evidence
        elif "not reported" in reasoning_excerpt.lower():
            adjusted *= 0.5  # -50% for "not reported" in evidence

        # Penalize if raw_value is "Not Reported" but confidence is high
        if isinstance(raw_value, str) and "not reported" in raw_value.lower():
            if adjusted > 0.5:
                adjusted = min(adjusted, 0.45)  # Cap at 0.45 for "Not Reported"

        # Boost if inferred=false (explicitly stated) and has good evidence
        if not inferred and reasoning_excerpt and len(reasoning_excerpt) > 20:
            adjusted = min(adjusted * 1.05, 1.0)  # +5% boost, max 1.0

        # Penalize if inferred=true but confidence is too high
        if inferred and adjusted > 0.85:
            adjusted = min(adjusted, 0.82)  # Cap inferred at 0.82

        # Boost if has multiple related entities (shows context understanding)
        if isinstance(related_entities, list) and len(related_entities) >= 3:
            adjusted = min(adjusted * 1.03, 1.0)  # +3% for rich context

        # Ensure score stays in valid range and round to 2 decimals
        # This prevents floating point precision issues like 0.9974999999999999
        adjusted = max(0.0, min(1.0, adjusted))
        return round(adjusted, 2)

    @staticmethod
    def _safe_float(value: Any, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _normalize_confidence_level(
        level: Optional[str],
        score: float,
    ) -> str:
        normalized = (level or "").lower()
        if normalized in {"low", "medium", "high"}:
            return normalized
        if score >= 0.8:
            return "high"
        if score >= 0.5:
            return "medium"
        return "low"

    @staticmethod
    def _ensure_list(value: Any) -> List[str]:
        if isinstance(value, list):
            return [str(item) for item in value]
        if value is None:
            return []
        return [str(value)]
