"""Utilities for building and executing DocETL pipelines."""

from __future__ import annotations

import ast
import json
import logging
import re
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

try:  # Optional dependency used only for error inspection
    import httpx  # type: ignore
except Exception:  # pragma: no cover - optional import
    httpx = None  # type: ignore

try:  # Optional dependency used only for error inspection
    import httpcore  # type: ignore
except Exception:  # pragma: no cover - optional import
    httpcore = None  # type: ignore

try:  # litellm exposes dedicated exception types
    from litellm.exceptions import APIError as LiteLLMAPIError  # type: ignore
except Exception:  # pragma: no cover - optional import
    LiteLLMAPIError = None  # type: ignore

try:
    from docetl.api import (  # type: ignore[import]
        Dataset,
        MapOp,
        Pipeline,
        PipelineOutput,
        PipelineStep,
        ReduceOp,
        ResolveOp,
        UnnestOp,
    )

    # Try to import CodeMapOp from api first, fallback to schemas
    try:
        from docetl.api import CodeMapOp  # type: ignore[import]
    except ImportError:
        try:
            from docetl.schemas import CodeMapOp  # type: ignore[import]
        except ImportError:
            CodeMapOp = None  # Will use dict-based operation instead
except ImportError as exc:  # pragma: no cover - surfaced during misconfiguration
    raise RuntimeError(
        "DocETL Python package is not installed. "
        "Run `uv sync` or ensure `docetl>=0.2.5` is in pyproject dependencies."
    ) from exc

from jinja2 import Environment, StrictUndefined, TemplateError

from config.settings import settings
from src.models.schemas import DocumentMetadata
from src.utils.ontology import OntologyLoader


class _DictOperation:
    """Lightweight shim when docetl CodeMapOp class is unavailable."""

    def __init__(self, payload: Dict[str, Any]):
        self._payload = payload

    def dict(self) -> Dict[str, Any]:
        return self._payload


_JSON_OBJECT_PATTERN = re.compile(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", re.DOTALL)
_JSON_ARRAY_PATTERN = re.compile(r"\[[^\[\]]*(?:\[[^\[\]]*\][^\[\]]*)*\]", re.DOTALL)


def _strip_code_fences(payload: str) -> str:
    """Remove markdown-style fences and surrounding quotes from a payload."""
    text = payload.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text, count=1).strip()
        text = re.sub(r"```$", "", text).strip()
    if text.lower().startswith("json:"):
        text = text[5:].strip()
    if text and text[0] in ("'", '"') and text[-1] == text[0]:
        text = text[1:-1].strip()
    return text


def _balance_pairs(text: str, open_char: str, close_char: str) -> str:
    """Balance open/close delimiters by trimming/adding trail characters."""
    balanced = text
    diff = balanced.count(open_char) - balanced.count(close_char)
    if diff > 0:
        balanced = balanced + (close_char * diff)
    elif diff < 0:
        while diff < 0:
            stripped = balanced.rstrip()
            if not stripped.endswith(close_char):
                break
            balanced = stripped[:-1]
            diff += 1
    return balanced


def _fix_truncated_json(payload: str) -> str:
    """Attempt to fix truncated JSON by completing missing closing brackets/quotes."""
    if not payload or not isinstance(payload, str):
        return payload

    text = payload.strip()

    # Check if text ends with incomplete string (most common truncation)
    # Look for patterns like: "adenocarcinoma (without closing quote)
    # Find the last quote that's not escaped
    last_quote_idx = -1
    for i in range(len(text) - 1, -1, -1):
        if text[i] == '"' and (i == 0 or text[i - 1] != "\\"):
            last_quote_idx = i
            break

    # If we found a quote, check if we're inside a string
    if last_quote_idx >= 0:
        # Count unescaped quotes before this position
        quotes_before = 0
        for i in range(last_quote_idx):
            if text[i] == '"' and (i == 0 or text[i - 1] != "\\"):
                quotes_before += 1

        # If odd number of quotes, we're inside a string - close it
        if quotes_before % 2 == 1:
            # Check if there's already content after the quote (shouldn't be if truncated)
            if last_quote_idx == len(text) - 1 or text[last_quote_idx + 1] in [
                ",",
                "]",
                "}",
                "\n",
                " ",
            ]:
                # String is complete, but check if we need to close it anyway
                pass
            else:
                # String appears incomplete - close it
                text = text + '"'

    # Count unclosed brackets (accounting for escaped characters)
    open_braces = 0
    open_brackets = 0
    in_string = False
    escape_next = False

    for char in text:
        if escape_next:
            escape_next = False
            continue
        if char == "\\":
            escape_next = True
            continue
        if char == '"' and not escape_next:
            in_string = not in_string
            continue
        if not in_string:
            if char == "{":
                open_braces += 1
            elif char == "}":
                open_braces -= 1
            elif char == "[":
                open_brackets += 1
            elif char == "]":
                open_brackets -= 1

    # Close arrays first (they're usually nested inside objects)
    if open_brackets > 0:
        text = text + "]" * open_brackets

    # Close objects
    if open_braces > 0:
        text = text + "}" * open_braces

    return text


def _sanitize_json_payload(payload: str) -> str:
    """Normalize loose JSON fragments so they can be decoded safely."""
    text = _strip_code_fences(payload)

    # First try to fix truncated JSON
    text = _fix_truncated_json(text)

    # Then balance pairs
    text = _balance_pairs(text, "{", "}")
    text = _balance_pairs(text, "[", "]")
    return text.strip()


def _coerce_structured_payload(payload: Any) -> Any:
    """Best-effort conversion of malformed JSON-ish payloads into Python data."""
    if isinstance(payload, (dict, list)):
        return payload
    if payload is None or not isinstance(payload, str):
        return payload if isinstance(payload, (dict, list)) else None

    candidate = _sanitize_json_payload(payload)
    decoders = (json.loads, ast.literal_eval)
    for decoder in decoders:
        try:
            return decoder(candidate)
        except Exception:
            continue

    for pattern in (_JSON_OBJECT_PATTERN, _JSON_ARRAY_PATTERN):
        match = pattern.search(candidate)
        if not match:
            continue
        snippet = _sanitize_json_payload(match.group(0))
        for decoder in decoders:
            try:
                return decoder(snippet)
            except Exception:
                continue

    return None


def _parse_argument_payload(arguments: Any) -> List[Dict[str, Any]]:
    """Convert tool call arguments (of varying formats) into dict outputs."""
    parsed_args = _coerce_structured_payload(arguments)
    if parsed_args is None:
        return []

    if isinstance(parsed_args, dict):
        return [parsed_args]

    if isinstance(parsed_args, list):
        parsed_results: List[Dict[str, Any]] = []
        for item in parsed_args:
            if isinstance(item, dict):
                parsed_results.append(item)
            else:
                nested = _coerce_structured_payload(item)
                if isinstance(nested, dict):
                    parsed_results.append(nested)
        return parsed_results

    return [{"extractions": parsed_args}]


@dataclass
class DocETLPipelineArtifacts:
    """Holds the raw outputs emitted by DocETL for downstream parsing."""

    session_id: str
    map_output_path: Path
    patient_output_path: Path
    resolve_output_path: Optional[Path]
    map_records: List[Dict[str, Any]]
    patient_records: List[Dict[str, Any]]
    resolve_records: Optional[List[Dict[str, Any]]]


class DocETLPipelineRunner:
    """Builds and executes a DocETL map → resolve → reduce pipeline."""

    STEP_NAME = "clinical_registry"
    MAP_OP = "extract_clinical_fields"
    NORMALIZE_OP = "normalize_extractions"
    UNNEST_OP = "explode_field_records"
    RESOLVE_OP = "resolve_patient_fields"
    REDUCE_OP = "reduce_patient_summary"

    def __init__(
        self,
        ontology: OntologyLoader,
        model_name: str,
        max_threads_override: Optional[int] = None,
    ):
        self.ontology = ontology
        self.model_name = model_name or settings.openrouter_model_openai
        self.map_prompt_template: Optional[str] = None
        self.reduce_prompt_template: Optional[str] = None
        self.max_threads_override = (
            max_threads_override if (max_threads_override or 0) > 0 else None
        )
        # Patch DocETL's parse_llm_response to handle markdown code blocks in structured output
        self._patch_docetl_parser()

    def run_pipeline(
        self,
        documents: List[DocumentMetadata],
        session_id: str,
    ) -> DocETLPipelineArtifacts:
        """Execute DocETL and return the captured artifacts."""
        logger = logging.getLogger(__name__)

        output_dir = Path(settings.output_dir) / session_id
        output_dir.mkdir(parents=True, exist_ok=True)
        intermediate_dir = output_dir / "docetl_intermediate"
        intermediate_dir.mkdir(parents=True, exist_ok=True)

        raw_records = [
            {
                "patient_id": doc.patient_id,
                "doc_id": doc.doc_id,
                "doc_type": doc.doc_type,
                "doc_date": doc.doc_date or "",
                "filename": doc.filename or "",
                "content": doc.content,
            }
            for doc in documents
        ]

        dataset = Dataset(
            type="memory",
            path=raw_records,
        )

        operations = [
            self._build_map_operation(),
        ]
        operation_names = [self.MAP_OP]

        # Always normalize map outputs so downstream operations receive lists.
        normalize_operation = self._build_normalize_operation()
        if normalize_operation:
            operations.append(normalize_operation)
            operation_names.append(self.NORMALIZE_OP)

        operations.extend(
            [
                self._build_unnest_operation(),
                self._build_resolve_operation(),
                self._build_reduce_operation(),
            ]
        )
        operation_names.extend(
            [
                self.UNNEST_OP,
                self.RESOLVE_OP,
                self.REDUCE_OP,
            ]
        )

        step = PipelineStep(
            name=self.STEP_NAME,
            input="clinical_docs",
            operations=operation_names,
        )

        pipeline = Pipeline(
            name=f"docetl_session_{session_id}",
            datasets={"clinical_docs": dataset},
            operations=operations,
            steps=[step],
            output=PipelineOutput(
                type="file",
                path=str(output_dir / "docetl_patient_results.json"),
                intermediate_dir=str(intermediate_dir),
            ),
            default_model=self.model_name,
        )

        self._log_map_prompts(session_id, raw_records)

        # Define output paths
        map_output_path = intermediate_dir / self.STEP_NAME / f"{self.MAP_OP}.json"
        resolve_output_path = intermediate_dir / self.STEP_NAME / f"{self.RESOLVE_OP}.json"
        patient_output_path = output_dir / "docetl_patient_results.json"

        # Check for existing checkpoints and resume from there
        map_records = None
        resolve_records = None
        patient_records = None

        # Try to load existing checkpoints
        if map_output_path.exists():
            try:
                map_records = self._load_json(map_output_path, "map")
                map_records = self._normalize_map_records(map_records)
            except Exception as e:
                logger.warning(f"Failed to load map checkpoint: {e}, will re-run map operation")
                map_records = None

        if resolve_output_path.exists():
            try:
                resolve_records = self._load_json(resolve_output_path, "resolve")
            except Exception as e:
                logger.warning(
                    f"Failed to load resolve checkpoint: {e}, will re-run resolve operation"
                )
                resolve_records = None

        if patient_output_path.exists():
            try:
                patient_records = self._load_json(patient_output_path, "reduce")
                patient_records = self._normalize_patient_records(patient_records)
            except Exception as e:
                logger.warning(
                    f"Failed to load reduce checkpoint: {e}, will re-run reduce operation"
                )
                patient_records = None

        # Only run pipeline if we don't have all checkpoints
        if map_records is None or resolve_records is None or patient_records is None:
            # Optimized thread calculation for better parallel processing
            # Use more aggressive parallelism: consider document count, CPU cores, and settings
            import os

            cpu_count = os.cpu_count() or 4

            # Calculate optimal thread count:
            # 1. Base on document count (at least 1 thread per document, but cap reasonably)
            # 2. Consider CPU cores (use 4-6x CPU cores for I/O-bound LLM operations)
            # 3. Respect configured limits
            base_threads = max(len(documents), 1)
            cpu_based_threads = (
                cpu_count * 5
            )  # Increased from 3x to 5x CPU for I/O-bound operations
            settings_based_threads = max(
                settings.max_workers,
                settings.max_concurrent_requests * 3,  # Increased from 2x to 3x
            )

            # Take the maximum of all considerations, but cap at docetl_max_threads
            desired_threads = min(
                settings.docetl_max_threads,
                max(
                    base_threads,
                    cpu_based_threads,
                    settings_based_threads,
                    1,
                ),
            )

            if self.max_threads_override:
                desired_threads = max(
                    1,
                    min(settings.docetl_max_threads, self.max_threads_override),
                )

            logger.info(
                "DocETL pipeline will use %d threads (documents=%d, cpu=%d, max_threads=%d)",
                desired_threads,
                len(documents),
                cpu_count,
                settings.docetl_max_threads,
            )

            self._run_pipeline_with_retries(
                pipeline=pipeline,
                max_threads=desired_threads,
                map_output_path=map_output_path,
                resolve_output_path=resolve_output_path,
                patient_output_path=patient_output_path,
            )

        # Load final results (in case pipeline completed successfully)
        if map_records is None:
            map_records = self._load_json(map_output_path, "map")
            map_records = self._normalize_map_records(map_records)

        if resolve_records is None:
            resolve_records = (
                self._load_json(resolve_output_path, "resolve")
                if resolve_output_path.exists()
                else None
            )

        if patient_records is None:
            if patient_output_path.exists():
                patient_records = self._load_json(patient_output_path, "reduce")
                patient_records = self._normalize_patient_records(patient_records)
            else:
                logger = logging.getLogger(__name__)
                logger.warning(
                    "DocETL reduce output missing at %s. Continuing with empty patient records; downstream steps will rebuild from resolve outputs if available.",
                    patient_output_path,
                )
                patient_records = []

        self._log_reduce_prompts(session_id, resolve_records)

        return DocETLPipelineArtifacts(
            session_id=session_id,
            map_output_path=map_output_path,
            patient_output_path=patient_output_path,
            resolve_output_path=resolve_output_path if resolve_output_path.exists() else None,
            map_records=map_records,
            patient_records=patient_records,
            resolve_records=resolve_records,
        )

    def _run_pipeline_with_retries(
        self,
        pipeline: Pipeline,
        max_threads: int,
        map_output_path: Path,
        resolve_output_path: Path,
        patient_output_path: Path,
    ) -> None:
        """Execute DocETL with automatic retries + resume on transient failures."""

        logger = logging.getLogger(__name__)
        attempts = max(settings.docetl_pipeline_retries, 1)
        last_error: Optional[Exception] = None

        for attempt in range(1, attempts + 1):
            try:
                pipeline.run(max_threads=max_threads)
                return
            except Exception as exc:  # pragma: no cover - exercised via integration tests
                last_error = exc
                retryable = self._is_retryable_pipeline_error(exc)
                logger.warning(
                    "DocETL attempt %s/%s failed (%s). Retryable=%s",
                    attempt,
                    attempts,
                    exc,
                    retryable,
                )

                if self._has_successful_patient_output(patient_output_path):
                    logger.info(
                        "Reduce output detected despite error. Skipping further retries and resuming from checkpoint."
                    )
                    return

                if attempt >= attempts or not retryable:
                    has_partial = any(
                        (
                            self._checkpoint_has_records(map_output_path, "map"),
                            self._checkpoint_has_records(resolve_output_path, "resolve"),
                            self._checkpoint_has_records(patient_output_path, "reduce"),
                        )
                    )
                    if has_partial:
                        logger.info(
                            "Partial DocETL outputs detected. Will resume from checkpoints without raising."
                        )
                        return
                    raise

                sleep_seconds = min(
                    settings.docetl_retry_backoff_seconds * attempt,
                    settings.docetl_retry_backoff_max_seconds,
                )
                time.sleep(sleep_seconds)

        if last_error:  # Safety net – should not be reached
            raise last_error

    def _checkpoint_has_records(self, path: Path, label: str) -> bool:
        if not path.exists():
            return False
        try:
            records = self._load_json(path, label)
        except Exception:
            return False
        return bool(records)

    def _has_successful_patient_output(self, patient_output_path: Path) -> bool:
        return self._checkpoint_has_records(patient_output_path, "reduce")

    def _is_retryable_pipeline_error(self, error: Exception) -> bool:
        """Identify transient OpenRouter/LiteLLM/Cloudflare failures."""

        if LiteLLMAPIError is not None and isinstance(error, LiteLLMAPIError):
            return True

        retryable_types = []
        if httpx is not None:
            retryable_types.extend(
                [
                    getattr(httpx, "ConnectError", Exception),
                    getattr(httpx, "ReadTimeout", Exception),
                ]
            )
        if httpcore is not None:
            retryable_types.append(getattr(httpcore, "ConnectError", Exception))

        if any(isinstance(error, err_type) for err_type in retryable_types):
            return True

        message = str(error).lower()
        transient_markers = [
            "openrouterexception",
            "<!doctype html",
            "<html",
            "cloudflare",
            "apierror",
            "all connection attempts failed",
            "connecterror",
            "bad gateway",
            "service unavailable",
            "timeout",
            "http 500",
            "http 502",
            "http 503",
            "http 504",
        ]
        return any(marker in message for marker in transient_markers)

    def _build_map_operation(self) -> MapOp:
        instructions = self.ontology.get_extraction_instructions()
        map_prompt = textwrap.dedent(
            f"""
            You are a certified oncology registrar. Extract every NAACCR field exactly as
            defined in the ontology below and emit strictly valid JSON.

            ⚠️⚠️⚠️ CRITICAL: CONFIDENCE SCORE RULES - READ THIS FIRST ⚠️⚠️⚠️
            
            **DO NOT DEFAULT TO 1.0!** Most fields should be 0.85-0.94 or lower.
            **1.0 is EXTREMELY RARE** - only use when EXACT medical term appears verbatim.
            
            **MANDATORY CONFIDENCE SCORE CALIBRATION PROTOCOL:**
            
            For EVERY field, you MUST follow this 5-step process before assigning a confidence_score:
            
            **Step A – Evidence Classification:**
            - **Explicit**: The exact value appears verbatim in the document (e.g., "ECOG 1", "Gleason 3+4", "2015-07-15")
            - **Interpreted**: Value is stated but requires minor calculation or mapping (e.g., "60-year-old" → birth year, "Stage IIA" → TNM code)
            - **Inferred**: Value must be derived from context or indirect evidence (e.g., "limited mobility" → ECOG 2, treatment type suggests stage)
            - **Absence-driven**: Field is "Not Reported" or missing with minimal context
            
            **Step B – Source Quality Assessment:**
            - **Pathology reports**: Highest quality (pathology > operative > imaging > clinical note > administrative)
            - **Count corroborating sources**: Multiple mentions increase confidence
            - **Single source**: Lower confidence unless it's a high-quality source with exact terminology
            
            **Step C – Consistency Check:**
            - **No conflicts**: If all sources agree, maintain or slightly increase score
            - **Conflicts exist**: If any document contradicts the value, subtract at least 0.08 from the score
            - **Multiple cancers**: If document mentions multiple cancers, extract the MOST RECENT or PRIMARY one
            
            **Step D – Numeric Score Mapping (MANDATORY):**
            - **Explicit & verbatim** (exact term appears):
              * Start at 0.92
              * Add +0.03 per corroborating quote (maximum cap: 0.98)
              * Example: "ECOG 1" appears verbatim → 0.95-0.98
              * Example: "Gleason Grade Group 3 (3+4)" verbatim → 0.98
              * **NEVER assign 1.0 unless absolutely certain the EXACT term appears with zero interpretation**
            
            - **Interpreted** (minor reasoning required):
              * Start at 0.85
              * Adjust ±0.05 based on source quality and clarity
              * Example: "60-year-old male" → calculate birth year → 0.85-0.90
              * Example: "Stage IIA" → map to TNM → 0.88-0.92
              * **Maximum for interpreted: 0.94**
            
            - **Inferred** (requires inference from context):
              * Start at 0.65
              * Adjust ±0.10 based on strength of contextual clues
              * Example: "restricted in strenuous activity" → infer ECOG 1 → 0.70-0.80
              * Example: "well-differentiated" → infer Grade 1 → 0.72-0.82
              * **Maximum for inferred: 0.82 (hard cap)**
            
            - **Absence-driven** ("Not Reported" or minimal evidence):
              * Start at 0.30
              * Adjust ±0.10 based on any related context
              * Example: "Not Reported" with no context → 0.30-0.40
              * Example: "Not Reported" but related field mentioned → 0.35-0.45
              * **Maximum for "Not Reported": 0.40 (hard cap)**
            
            **Step E – Sanity Clamp (FINAL CHECK):**
            - Fields containing "Not Reported" or blank reasoning_excerpt → MUST be ≤0.40
            - inferred=true entries → CAN NEVER exceed 0.82
            - Always round to two decimal places (e.g., 0.85, 0.92, 0.35)
            - If you assigned 0.95-1.0, verify the EXACT term appears verbatim - if not, lower to 0.94 or below
            
            **CONFIDENCE SCORE RUBRIC (QUICK REFERENCE):**
            - **0.95-0.98**: Explicitly stated with exact medical terminology (VERY RARE - only when exact term appears)
            - **0.85-0.94**: Explicitly stated but needs minor interpretation (MOST COMMON for good extractions)
            - **0.75-0.84**: Strong inference from clear clinical context
            - **0.60-0.74**: Moderate inference from indirect evidence
            - **0.45-0.59**: Weak inference or ambiguous evidence
            - **0.30-0.44**: Very uncertain, minimal evidence or "Not Reported"
            - **0.0-0.29**: No evidence or completely contradictory information
            
            **CRITICAL WARNINGS:**
            1. **DO NOT assign 1.0** - This is reserved for perfect, unambiguous, verbatim matches (almost never happens)
            2. **DO NOT default to 1.0** - Every field needs individual calibration
            3. **If calculating, inferring, or interpreting → score MUST be 0.94 or lower**
            4. **"Not Reported" fields → score MUST be 0.30-0.44 (hard cap: 0.40)**
            5. **inferred=true → score CAN NEVER exceed 0.82**
            6. **When in doubt, LOWER the score - reliability > optimism**
            
            **CONFIDENCE SCORE EXAMPLES:**
            - Document says "ECOG 1" verbatim → confidence_score: 0.95-0.98 (exact term)
            - Document says "Gleason Grade Group 3 (3+4)" verbatim → confidence_score: 0.98 (exact terminology)
            - Document says "60-year-old male" → calculate birth year → confidence_score: 0.88 (minor interpretation)
            - Document says "limited mobility" → infer ECOG 2 → confidence_score: 0.72 (moderate inference)
            - Document says "Not Reported" with no context → confidence_score: 0.35 (very uncertain) + inferred: true
            - Document says "well-differentiated adenocarcinoma" → infer Grade 1 → confidence_score: 0.78 (inferred, max 0.82)
            
            **REMEMBER**: Most confidence scores should be 0.85-0.94 for good extractions. 1.0 is EXTREMELY RARE and should almost never be used.
            
            ### Ontology Guidance
            {instructions}

            ### Document Metadata
            - Patient ID: {{{{ input.patient_id }}}}
            - Document ID: {{{{ input.doc_id }}}}
            - Document Type: {{{{ input.doc_type }}}}
            - Document Date: {{{{ input.doc_date }}}}
            - Filename: {{{{ input.filename }}}}

            ### Extraction Task - READ CAREFULLY
            You MUST read the ENTIRE document word-by-word and extract REAL values. "Not Reported" is ONLY allowed if you have read every sentence and confirmed the information is completely absent.

            STEP-BY-STEP PROCESS:
            1. READ THE FULL DOCUMENT: Start from the first line and read every sentence, paragraph, and section
            2. SEARCH FOR EACH FIELD: For each NAACCR field, actively search the document for relevant information
            3. EXTRACT REAL VALUES: If ANY information exists (even partial), extract it - NEVER use "Not Reported" if information exists
            4. MULTIPLE CANCERS: If document mentions multiple cancers, extract information for the MOST RECENT or PRIMARY cancer
            5. VERBATIM QUOTES: Copy the EXACT text from document in `reasoning_excerpt` - never use "Not Reported" as reasoning_excerpt

            ### Confidence Score Application (REVIEW ABOVE RULES)
            Before assigning confidence_score for each field, review the CRITICAL CONFIDENCE SCORE RULES at the top of this prompt.
            Apply the 5-step calibration protocol (Steps A-E) for EVERY field individually.
            DO NOT copy/paste default values - each field requires individual assessment.

            FIELD-SPECIFIC EXTRACTION RULES:

            **naaccr_diagnosis_dt (Diagnosis Date):**
            - Search for: "Diagnosed in [year]", "Diagnosis date", "On [date]", "Recently diagnosed in [month] [year]"
            - Examples from document: "Diagnosed in 1987", "Diagnosed in 2005", "Recently diagnosed in July 2015"
            - Extract the MOST RECENT diagnosis date if multiple cancers exist
            - Format: YYYY-MM-DD (e.g., "2015-07-15" for "July 2015")
            - If only month/year given, use 15th day of that month
            - reasoning_excerpt example: "Recently diagnosed in July 2015 via transperineal biopsy"

            **ca_site (Cancer Site):**
            - Search for: Cancer names with ICD-O codes like "Prostate Cancer (C61.9)", "Rectal Cancer (C20.9)", "Appendiceal Cancer (C18.1)"
            - Extract the MOST RECENT or PRIMARY cancer site
            - Format: "Site Name (Code)/Malignant" (e.g., "Prostate (C61.9)/Malignant")
            - If code not given, infer from cancer name using ICD-O-3 standards
            - reasoning_excerpt example: "Prostate Cancer (C61.9): Recently diagnosed in July 2015"

            **naaccr_histology_cd (Histology Code):**
            - Search for: Pathology terms like "adenocarcinoma", "carcinoma", "mucinous adenocarcinoma", "prostatic adenocarcinoma"
            - Look for ICD-O-3 codes or map pathology terms to codes
            - Format: "XXXX/3 - Description" (e.g., "8140/3 - Adenocarcinoma, NOS")
            - Extract from pathology reports, biopsy results, or diagnosis descriptions
            - reasoning_excerpt example: "Pathology showed prostatic adenocarcinoma, Gleason Grade Group 3"

            **Staging Fields (ca_clinical_t_stage, ca_clinical_n_stage, ca_clinical_m_stage, ca_path_t_stage, etc.):**
            - Search for: Explicit staging mentions like "pT3", "cT2", "pN0", "cM0", "TNM staging"
            - Clinical staging: Look for "cT", "cN", "cM" prefixes or staging from imaging/clinical exam
            - Pathological staging: Look for "pT", "pN", "pM" prefixes or staging from pathology reports
            - Extract exact values as written in document
            - reasoning_excerpt example: "Pathology revealed a moderately differentiated mucinous adenocarcinoma, pT3"

            **ecog (Performance Status):**
            - Search for: "ECOG", "performance status", "ECOG 0", "ECOG 1", etc.
            - Extract numeric value (0-5)
            - If not explicitly stated, infer from activity level descriptions

            Return a JSON object with a single key `extractions` that contains one entry for *every* ontology field.

            Each extraction object MUST include:
            - `field_name`, `category`, `data_type`
            - `raw_value`: The EXACT value found in document (e.g., "2015-07-15", "Prostate (C61.9)/Malignant", "8140/3 - Adenocarcinoma, NOS")
            - `normalized_value`: Normalized version following NAACCR/ICD-O-3 standards
            - `units`, `vocabulary_code`: Include ICD-O-3 codes when applicable
            - `reasoning_excerpt`: EXACT verbatim quote from document (e.g., "Recently diagnosed in July 2015 via transperineal biopsy" or "Prostate Cancer (C61.9): Recently diagnosed")
            - `explanation`: Brief explanation (e.g., "Extracted from most recent cancer diagnosis section")
            - `confidence_level`: "high" if explicitly stated, "medium" if inferred, "low" if uncertain
            - `confidence_score`: **CRITICAL - REFER TO CONFIDENCE SCORE RULES AT TOP OF PROMPT!**
              Apply the 5-step calibration protocol (Steps A-E) from the top of this prompt.
              **DO NOT DEFAULT TO 1.0** - Most fields should be 0.85-0.94 or lower.
              **1.0 is EXTREMELY RARE** - only use when EXACT medical term appears verbatim with zero interpretation.
              
              Quick reference (see detailed rules at top):
              * **0.95-0.98**: Explicit & verbatim (exact term appears) - VERY RARE
              * **0.85-0.94**: Explicit but interpreted (MOST COMMON for good extractions)
              * **0.75-0.84**: Strong inference from context
              * **0.60-0.74**: Moderate inference
              * **0.45-0.59**: Weak inference
              * **0.30-0.44**: "Not Reported" or minimal evidence (hard cap: 0.40)
              * **0.0-0.29**: No evidence
              
              **HARD CAPS:**
              - "Not Reported" fields: Maximum 0.40
              - inferred=true fields: Maximum 0.82
              - Calculated/interpreted fields: Maximum 0.94
              - Verbatim fields: Maximum 0.98 (almost never 1.0)
              
            - `inferred`: false if explicitly stated, true only if you had to infer from context
            - `related_entities`: List of related terms (e.g., ["Prostate", "C61.9", "adenocarcinoma", "Gleason 3+4"])

            ABSOLUTE REQUIREMENTS:
            - READ EVERY WORD: Do not skip any section of the document
            - EXTRACT REAL VALUES: If information exists anywhere in the document, extract it
            - NO "Not Reported" FOR reasoning_excerpt: Always provide actual text from document
            - MULTIPLE CANCERS: Extract the most recent/primary cancer information
            - VERBATIM QUOTES: Copy exact sentences/phrases from document for reasoning_excerpt
            - DATES: Convert relative dates to YYYY-MM-DD format
            - CODES: Extract or map to standard ICD-O-3 codes

            FINAL CHECKLIST BEFORE RESPONDING:
            - Did I read the ENTIRE document from start to finish?
            - For each field, did I search for relevant information?
            - Did I extract REAL values instead of using "Not Reported"?
            - Is every `reasoning_excerpt` an actual quote from the document (not "Not Reported")?
            - If multiple cancers exist, did I extract the most recent/primary one?
            - Are dates in YYYY-MM-DD format?
            - Are ICD-O-3 codes properly formatted?
            - **CONFIDENCE SCORES** (CRITICAL - CHECK EVERY FIELD):
              * Did I review the CONFIDENCE SCORE RULES at the top of this prompt?
              * Did I apply the 5-step calibration protocol (Steps A-E) for each field?
              * Am I giving 0.95-0.98 ONLY when the EXACT term appears verbatim in the document?
              * Did I lower the score to 0.85-0.94 for any interpretation or calculation?
              * Did I avoid defaulting to 1.0? (1.0 is EXTREMELY RARE - almost never use it)
              * For "Not Reported" fields, did I set score ≤0.40?
              * For inferred=true fields, did I set score ≤0.82?
              * Did I round all scores to two decimal places?

            **FINAL REMINDERS:**
            1. "Not Reported" is a LAST RESORT. If you see ANY mention of the information, extract it!
            2. DO NOT assign confidence_score of 1.0 - use 0.95-0.98 for verbatim matches instead
            3. Most confidence scores should be 0.85-0.94 for good extractions
            4. When in doubt, lower—not raise—the score. Reliability > optimism.

            Respond with JSON containing a top-level `extractions` array where every object repeats
            the requested fields. Do not include narrative text—only strict JSON.
            """
        ).strip()

        self.map_prompt_template = map_prompt
        validation_rules = [
            "isinstance(output, dict) or isinstance(output, list)",
        ]

        model_name_lower = (self.model_name or "").lower()
        is_qwen_model = any(
            keyword in model_name_lower for keyword in ("qwen", "yi-", "glm", "deepseek")
        )

        # Qwen models need faster timeouts and fewer retries for speed
        if is_qwen_model:
            # Daha agresif timeout'lar - Qwen modelleri genellikle daha hızlı yanıt verir
            qwen_timeout = min(
                settings.llm_request_timeout, 20.0
            )  # Reduced from 25s to 20s max for Qwen (faster responses)
            qwen_retries = min(
                settings.llm_retry_attempts, 1
            )  # Reduced from 2 to 1 retry for Qwen (faster failure)
        else:
            qwen_timeout = min(settings.llm_request_timeout, 40.0)  # Reduced from 45.0 to 40.0
            qwen_retries = max(settings.llm_retry_attempts, 3)  # Reduced from 5 to 3

        completion_kwargs = {
            "temperature": 0.0,
            "timeout": qwen_timeout,
            "stream": False,  # Disable streaming to avoid connection issues
            "max_retries": qwen_retries,
        }

        # OpenRouter için api_base belirt (LiteLLM'in model adını tanıması için)
        if is_qwen_model or "openrouter" in (self.model_name or "").lower():
            completion_kwargs["api_base"] = settings.openrouter_api_base

        # Qwen-specific optimizations for speed
        if is_qwen_model:
            completion_kwargs.update(
                {
                    "max_tokens": 10000,  # Increased to prevent JSON truncation (Qwen models can handle this)
                    "extra_body": {
                        "top_p": 0.95,  # Biraz daha yüksek top_p daha hızlı token üretimi sağlar
                        "temperature": 0.1,  # Düşük temperature daha hızlı ve tutarlı çıktı
                        "frequency_penalty": 0.0,
                        "presence_penalty": 0.0,
                        # Reasoning'i tamamen kapatmak için (Qwen 3 dual-mode architecture)
                        "reasoning_effort": "none",  # Reasoning'i tamamen kapat
                    },
                }
            )
        else:
            # For non-Qwen models, also increase max_tokens to prevent truncation
            completion_kwargs["max_tokens"] = 8000

        # Qwen models on OpenRouter don't support tool use/function calling
        # Use structured_output mode instead which works with JSON schema validation
        # DocETL's structured_output mode already handles response_format internally
        output_mode = "structured_output"
        if is_qwen_model:
            # Keep structured_output mode for Qwen - it uses LiteLLM's JSON schema validation
            # which doesn't require tool use/function calling
            # Note: Don't add response_format here - DocETL handles it in structured_output mode
            output_mode = "structured_output"

        # Qwen needs shorter timeouts and fewer validation retries for speed
        if is_qwen_model:
            pipeline_timeout = min(
                settings.docetl_timeout, 120
            )  # Reduced from 150s to 120s (2 min) for Qwen (faster processing)
            validation_retries = min(
                settings.docetl_validation_retries, 1
            )  # Reduced from 2 to 1 validation retry for faster failure
        else:
            pipeline_timeout = settings.docetl_timeout
            validation_retries = settings.docetl_validation_retries

        return MapOp(
            name=self.MAP_OP,
            type="map",
            prompt=map_prompt,
            model=self.model_name,
            num_retries_on_validate_failure=validation_retries,
            timeout=pipeline_timeout,
            output={
                "schema": {
                    "extractions": (
                        "list[{field_name: string, category: string, data_type: string, "
                        "raw_value: string, normalized_value: string, units: string, "
                        "vocabulary_code: string, reasoning_excerpt: string, explanation: string, "
                        "confidence_level: string, confidence_score: number, inferred: bool, "
                        "related_entities: list[string]}]"
                    )
                },
                "mode": output_mode,
            },
            validate=validation_rules,
            litellm_completion_kwargs=completion_kwargs,
        )

    def _build_normalize_operation(self):
        """Normalize map output to ensure 'extractions' is always a list."""
        normalize_code = """
import json
def transform(doc: dict) -> dict:
    # Ensure 'extractions' key exists and is always a list
    if 'extractions' not in doc:
        doc['extractions'] = []
    else:
        extractions_value = doc['extractions']
        if isinstance(extractions_value, str):
            try:
                parsed = json.loads(extractions_value)
                if isinstance(parsed, dict) and 'extractions' in parsed:
                    extractions_value = parsed['extractions']
                else:
                    extractions_value = parsed
            except Exception:
                extractions_value = []
        doc['extractions'] = extractions_value

    if not isinstance(doc['extractions'], (list, tuple)):
        # If extractions is None or not a list, convert to empty list
        if doc['extractions'] is None:
            doc['extractions'] = []
        elif isinstance(doc['extractions'], dict):
            # If it's a dict, wrap it in a list
            doc['extractions'] = [doc['extractions']]
        else:
            # For any other type, default to empty list
            doc['extractions'] = []
    else:
        # Handle case where extractions is a list containing JSON strings (from tool calls)
        if len(doc['extractions']) > 0 and isinstance(doc['extractions'][0], str):
            first_item = doc['extractions'][0]
            if first_item.strip().startswith('{'):
                # This is likely a tool call response with JSON string in list
                try:
                    parsed_list = []
                    for item in doc['extractions']:
                        if isinstance(item, str):
                            try:
                                parsed_item = json.loads(item)
                                if isinstance(parsed_item, dict) and 'extractions' in parsed_item:
                                    # Extract the nested extractions array
                                    nested = parsed_item['extractions']
                                    if isinstance(nested, list):
                                        parsed_list.extend(nested)
                                    elif isinstance(nested, dict):
                                        parsed_list.append(nested)
                                elif isinstance(parsed_item, list):
                                    parsed_list.extend(parsed_item)
                                elif isinstance(parsed_item, dict):
                                    parsed_list.append(parsed_item)
                            except json.JSONDecodeError:
                                # If JSON parsing fails, skip this item
                                continue
                        else:
                            parsed_list.append(item)
                    doc['extractions'] = parsed_list if parsed_list else []
                except Exception:
                    # If processing fails, keep original
                    pass
        # Ensure it's a list (not tuple)
        doc['extractions'] = list(doc['extractions'])
    return doc
"""
        # Use CodeMapOp if available, otherwise return dict for Pipeline._update_from_dict
        if CodeMapOp is not None:
            return CodeMapOp(
                name=self.NORMALIZE_OP,
                type="code_map",
                code=normalize_code,
            )
        # Fallback to dict-based operation definition with dict() shim.
        return _DictOperation(
            {
                "name": self.NORMALIZE_OP,
                "type": "code_map",
                "code": normalize_code,
            }
        )

    def _build_unnest_operation(self) -> UnnestOp:
        expand_fields = [
            "field_name",
            "category",
            "data_type",
            "raw_value",
            "normalized_value",
            "units",
            "vocabulary_code",
            "reasoning_excerpt",
            "explanation",
            "confidence_level",
            "confidence_score",
            "inferred",
            "related_entities",
        ]
        return UnnestOp(
            name=self.UNNEST_OP,
            type="unnest",
            unnest_key="extractions",
            expand_fields=expand_fields,
            keep_empty=False,
            recursive=True,  # ensure we drill into dict values and expose expand_fields
            depth=2,
        )

    def _build_resolve_operation(self) -> ResolveOp:
        comparison_prompt = textwrap.dedent(
            """
            You are comparing two candidate values for the same oncology registry field.

            Field 1 ({{ input1.field_name }}) from {{ input1.doc_id }}:
            - Patient: {{ input1.patient_id }}
            - Value: {{ input1.normalized_value or input1.raw_value }}
            - Evidence: {{ input1.reasoning_excerpt }}
            - Explanation: {{ input1.explanation }}

            Field 2 ({{ input2.field_name }}) from {{ input2.doc_id }}:
            - Patient: {{ input2.patient_id }}
            - Value: {{ input2.normalized_value or input2.raw_value }}
            - Evidence: {{ input2.reasoning_excerpt }}
            - Explanation: {{ input2.explanation }}

            Respond with JSON: {"is_match": true} when both entries represent the same registry fact
            after normalization, else {"is_match": false}.
            """
        ).strip()

        resolution_prompt = textwrap.dedent(
            """
            You are consolidating oncology registry evidence for patient {{ inputs[0].patient_id }}
            and field {{ inputs[0].field_name }}.

            Evidence set:
            {% for item in inputs %}
            - Document {{ item.doc_id }} ({{ item.doc_type }}) on {{ item.doc_date or "unknown date" }}
              raw="{{ item.raw_value }}" | normalized="{{ item.normalized_value }}"
              confidence={{ item.confidence_score if item.confidence_score is not none else "not provided" }} | reason="{{ item.reasoning_excerpt }}"
            {% endfor %}

            TASK: Resolve conflicts and determine the most reliable value across documents. Apply the following calibration checklist before emitting JSON:

            1. **Consistency Audit**: Are values identical, compatible, or conflicting? Note the number of corroborating docs.
            2. **Source Tiering**: Rank evidence quality (Pathology > Operative > Imaging > Clinical note > Administrative).
            3. **Specificity & Timeliness**: Prefer precise dates/codes and the most recent records.
            4. **Conflict Penalty**: If any document contradicts the resolved value, subtract at least 0.08 from the final score for each contradiction you override.

            **CONFIDENCE SCORE CALCULATION (MANDATORY & RE-COMPUTED):**
            - Start from these anchors:
              * 0.95-0.99 → 3+ agreeing sources OR 2 high-quality sources with identical language.
              * 0.85-0.94 → Clear agreement but limited to one high-quality source + secondary mention.
              * 0.75-0.84 → Majority consensus with minor interpretation.
              * 0.60-0.74 → Conflicts exist yet one source is clearly superior (explain why).
              * 0.45-0.59 → Significant ambiguity; you still pick one value but warn in notes.
              * 0.30-0.44 → Barely any evidence; mostly inferred.
              * Below 0.30 → No reliable evidence; consider leaving normalized_value empty.
            - After anchoring, adjust ±0.03 per corroborating snippet and −0.1 for each conflict you dismiss.
            - Cap inferred outcomes at 0.82.

            DO NOT average or reuse upstream confidence scores. Recompute from scratch using the checklist and explain the reasoning in `consolidation_notes`.
            
            Output JSON containing exactly these keys: `patient_id`, `field_name`, `category`, `data_type`,
            `normalized_value`, `resolved_value`, `units`, `vocabulary_code`, `confidence_score`,
            `supporting_docs`, and `consolidation_notes`.
            
            In `consolidation_notes`, briefly explain:
            - How you resolved any conflicts
            - Why you assigned the confidence_score you chose
            - Which sources were most reliable

            Preserve every supporting document entry so downstream reducers can trace provenance.
            """
        ).strip()

        model_name_lower = (self.model_name or "").lower()
        is_qwen_model = any(
            keyword in model_name_lower for keyword in ("qwen", "yi-", "glm", "deepseek")
        )

        # Qwen models need faster timeouts and fewer retries for speed
        if is_qwen_model:
            qwen_timeout = min(
                settings.llm_request_timeout, 20.0
            )  # Reduced from 25s to 20s max for Qwen (faster responses)
            qwen_retries = min(settings.llm_retry_attempts, 1)  # Reduced from 2 to 1 retry for Qwen
            pipeline_timeout = min(
                settings.docetl_timeout, 120
            )  # Reduced from 150s to 120s (2 min) for Qwen
        else:
            qwen_timeout = min(settings.llm_request_timeout, 40.0)  # Reduced from 45.0 to 40.0
            qwen_retries = max(settings.llm_retry_attempts, 3)  # Reduced from 5 to 3
            pipeline_timeout = settings.docetl_timeout

        resolve_completion_kwargs = {
            "temperature": 0.0,
            "timeout": qwen_timeout,
            "stream": False,  # Disable streaming to avoid connection issues
            "max_retries": qwen_retries,
            "max_tokens": 8000,  # Increased from default to prevent truncation
        }

        # OpenRouter için api_base belirt (LiteLLM'in model adını tanıması için)
        if is_qwen_model or "openrouter" in (self.model_name or "").lower():
            resolve_completion_kwargs["api_base"] = settings.openrouter_api_base

        # Qwen-specific optimizations for speed
        if is_qwen_model:
            resolve_completion_kwargs.update(
                {
                    "extra_body": {
                        "top_p": 0.95,  # Biraz daha yüksek top_p daha hızlı token üretimi
                        "temperature": 0.1,  # Düşük temperature daha hızlı ve tutarlı
                        "frequency_penalty": 0.0,
                        "presence_penalty": 0.0,
                        # Reasoning'i tamamen kapatmak için (Qwen 3 dual-mode architecture)
                        "reasoning_effort": "none",  # Reasoning'i tamamen kapat
                        # max_tokens sınırı kaldırıldı - model kendi limitini kullanacak
                    }
                    # Note: Don't add response_format here - DocETL handles it in structured_output mode
                }
            )

        # Qwen models on OpenRouter don't support tool use/function calling
        # Use structured_output mode which works with JSON schema validation
        resolve_output_mode = "structured_output"
        if is_qwen_model:
            # Keep structured_output mode - it doesn't require tool use
            resolve_output_mode = "structured_output"

        return ResolveOp(
            name=self.RESOLVE_OP,
            type="resolve",
            comparison_prompt=comparison_prompt,
            resolution_prompt=resolution_prompt,
            blocking_keys=["patient_id", "field_name"],
            blocking_conditions=[
                "input1.get('patient_id') == input2.get('patient_id') and input1.get('field_name') == input2.get('field_name')"
            ],  # ensure DocETL skips interactive confirmation in non-TTY environments
            comparison_model=self.model_name,
            resolution_model=self.model_name,
            timeout=pipeline_timeout,
            litellm_completion_kwargs=resolve_completion_kwargs,
            output={
                "schema": {
                    "patient_id": "string",
                    "field_name": "string",
                    "category": "string",
                    "data_type": "string",
                    "normalized_value": "string",
                    "resolved_value": "string",
                    "units": "string",
                    "vocabulary_code": "string",
                    "confidence_score": "number",
                    "supporting_docs": (
                        "list[{doc_id: string, patient_id: string, field_name: string, "
                        "raw_value: string, normalized_value: string, reasoning_excerpt: string, "
                        "explanation: string, doc_type: string, doc_date: string, confidence_score: number}]"
                    ),
                    "consolidation_notes": "string",
                },
                "mode": resolve_output_mode,
            },
        )

    def _build_reduce_operation(self) -> ReduceOp:
        prompt = textwrap.dedent(
            """
            You are a certified oncology registrar consolidating patient-level cancer registry data.

            TASK: Generate a patient-level cancer registry row for patient {{ reduce_key }} by consolidating the resolved field extractions below.

            IMPORTANT: The data below contains REAL patient information extracted from medical documents. You MUST use this actual data to generate your response. Do NOT create hypothetical examples or sample data.

            RESOLVED FIELD EXTRACTIONS FOR PATIENT {{ reduce_key }}:
            {% if inputs %}
            {% for item in inputs %}
            Field: {{ item.field_name }} (Category: {{ item.category }}, Data Type: {{ item.data_type }})
            - Normalized value: {{ item.normalized_value or item.resolved_value or "Not Reported" }}
            - Resolved value: {{ item.resolved_value or item.normalized_value or "Not Reported" }}
            - Units: {{ item.units or "Not Reported" }}
            - Vocabulary code: {{ item.vocabulary_code or "Not Reported" }}
            - Confidence score: {{ item.confidence_score if item.confidence_score is not none else "Not provided" }}
            - Consolidation notes: {{ item.consolidation_notes or "No notes" }}
            - Supporting documents:
            {% if item.supporting_docs %}
            {% for doc in item.supporting_docs %}
              {% if doc is mapping %}
              * Document ID: {{ doc.get('doc_id', 'unknown') }}, Type: {{ doc.get('doc_type', 'unknown') }}, Raw value: "{{ doc.get('raw_value', '') }}", Normalized: "{{ doc.get('normalized_value', '') }}", Confidence: {{ doc.get('confidence_score', 'Not provided') }}
              {% else %}
              * {{ doc }}
              {% endif %}
            {% endfor %}
            {% else %}
              * No supporting documents
            {% endif %}
            ---
            {% endfor %}
            {% else %}
            No resolved inputs available for this patient.
            {% endif %}

            CRITICAL INSTRUCTIONS:
            1. You MUST use the ACTUAL data provided above - do NOT create hypothetical examples
            2. Respond with ONLY valid JSON - no markdown code blocks, no explanations, no additional text
            3. Do NOT wrap your response in ```json``` or any code blocks
            4. Start your response directly with { and end with }
            5. Return pure JSON that can be parsed directly
            
            CONFIDENCE SCORE CALCULATION:
            For each consolidated field, you MUST calculate a confidence_score based on the recomputation checklist below (never copy upstream numbers):
            1. Count the number of unique documents supporting the chosen value.
            2. Rank the highest-quality source that agrees (Pathology > Operative > Imaging > Clinical > Admin).
            3. Identify any conflicting statements and deduct at least 0.08 per conflict overridden.
            4. Decide whether the value is explicit, lightly interpreted, or heavily inferred.

            Use this scoring rubric after applying the checklist:
            * 0.95-0.99: 2+ high-quality sources with identical wording and zero conflicts.
            * 0.85-0.94: One high-quality explicit source + corroboration from another note.
            * 0.75-0.84: Mixed-quality but generally consistent evidence.
            * 0.60-0.74: Conflict resolved in favor of a clearly better source; explain why.
            * 0.45-0.59: Ambiguous/weak support; you kept value solely to avoid data loss.
            * 0.30-0.44: Almost no evidence; mostly inferred or historical.
            * Below 0.30: No trustworthy evidence—consider leaving normalized_value empty.

            If you truly cannot determine a confidence score from the evidence, set it to null (not 0.5 or any default value) and justify in `consolidation_notes`.
            
            REQUIRED JSON STRUCTURE:
            {
              "patient_id": "{{ reduce_key }}",
              "consolidated_fields": [
                {
                  "field_name": "<field_name from inputs>",
                  "category": "<category from inputs>",
                  "data_type": "<data_type from inputs>",
                  "normalized_value": "<normalized_value from inputs>",
                  "resolved_value": "<resolved_value from inputs>",
                  "units": "<units from inputs>",
                  "vocabulary_code": "<vocabulary_code from inputs>",
                  "confidence_score": <calculated confidence score as number between 0 and 1, or null if cannot determine>,
                  "consolidation_notes": "<brief explanation of how confidence was determined and any conflicts resolved>"
                },
                ... (one object for each field in inputs)
              ],
              "patient_summary": "<short narrative paragraph summarizing staging, therapy, performance status, etc. based on the actual data above>"
            }

            IMPORTANT:
            - Copy ALL fields from the inputs above into consolidated_fields array
            - Do NOT drop or reformat the supporting_docs arrays—copy them verbatim from the inputs
            - ALWAYS calculate confidence_score based on evidence quality - do NOT copy input confidence blindly
            - Generate patient_summary based on the ACTUAL data provided, not hypothetical examples
            - If a field value is "Not Reported", keep it as "Not Reported" in your response
            - consolidation_notes should explain your confidence assessment and any conflicts you resolved
            """
        ).strip()

        validation_rules = [
            "isinstance(output, dict)",
        ]

        model_name_lower = (self.model_name or "").lower()
        is_qwen_model = any(
            keyword in model_name_lower for keyword in ("qwen", "yi-", "glm", "deepseek")
        )

        # Qwen models need faster timeouts and fewer retries for speed
        if is_qwen_model:
            qwen_timeout = min(
                settings.llm_request_timeout * 1.0, 40.0
            )  # Reduced from 50s to 40s max for Qwen reduce ops (faster processing)
            qwen_retries = min(settings.llm_retry_attempts, 1)  # Reduced from 2 to 1 retry for Qwen
        else:
            qwen_timeout = min(
                settings.llm_request_timeout * 1.5, 75.0
            )  # Reduced from 90.0 to 75.0
            qwen_retries = max(settings.llm_retry_attempts, 3)  # Reduced from 5 to 3

        completion_kwargs = {
            "temperature": 0.0,
            "timeout": qwen_timeout,
            "stream": False,  # Disable streaming to avoid connection issues
            "max_retries": qwen_retries,
        }

        # OpenRouter için api_base belirt (LiteLLM'in model adını tanıması için)
        if is_qwen_model or "openrouter" in (self.model_name or "").lower():
            completion_kwargs["api_base"] = settings.openrouter_api_base

        # Qwen-specific optimizations for speed
        if is_qwen_model:
            completion_kwargs.update(
                {
                    "max_tokens": 10000,  # Increased for reduce operations (consolidation can be large)
                    "extra_body": {
                        "top_p": 0.95,  # Biraz daha yüksek top_p daha hızlı token üretimi
                        "temperature": 0.1,  # Düşük temperature daha hızlı ve tutarlı
                        "frequency_penalty": 0.0,
                        "presence_penalty": 0.0,
                        # Reasoning'i tamamen kapatmak için (Qwen 3 dual-mode architecture)
                        "reasoning_effort": "none",  # Reasoning'i tamamen kapat
                    },
                    # Note: Don't add response_format here - DocETL handles it in structured_output mode
                }
            )
        else:
            # For non-Qwen models, also increase max_tokens to prevent truncation
            completion_kwargs["max_tokens"] = 10000

        model_name_lower = (self.model_name or "").lower()
        # Qwen models on OpenRouter don't support tool use/function calling
        # Use structured_output mode which works with JSON schema validation
        output_mode = "structured_output"
        if any(keyword in model_name_lower for keyword in ("qwen", "yi-", "glm", "deepseek")):
            # Keep structured_output mode - it doesn't require tool use
            output_mode = "structured_output"

        # Qwen needs shorter timeouts and fewer validation retries for speed
        if is_qwen_model:
            reduce_pipeline_timeout = min(
                settings.docetl_timeout, 120
            )  # Reduced from 150s to 120s (2 min) for Qwen (faster processing)
            reduce_validation_retries = min(
                settings.docetl_validation_retries, 1
            )  # Reduced from 2 to 1 validation retry for faster failure
        else:
            reduce_pipeline_timeout = settings.docetl_timeout
            reduce_validation_retries = settings.docetl_validation_retries

        return ReduceOp(
            name=self.REDUCE_OP,
            type="reduce",
            reduce_key="patient_id",
            prompt=prompt,
            model=self.model_name,
            num_retries_on_validate_failure=reduce_validation_retries,
            timeout=reduce_pipeline_timeout,
            output={
                # DocETL's schema parser cannot handle nested `list[{...}]` declarations
                # inside another object (it splits on commas without tracking depth),
                # so we omit the supporting_docs definition here even though the prompt
                # still asks the model to emit it. Downstream normalization treats the
                # field as optional and defaults to [] when missing.
                "schema": {
                    "patient_id": "string",
                    "consolidated_fields": (
                        "list[{field_name: string, category: string, data_type: string, "
                        "normalized_value: string, resolved_value: string, units: string, "
                        "vocabulary_code: string, confidence_score: number, "
                        "consolidation_notes: string}]"
                    ),
                    "patient_summary": "string",
                },
                "mode": output_mode,
            },
            validate=validation_rules,
            litellm_completion_kwargs=completion_kwargs,
        )

        self.reduce_prompt_template = prompt

    def _load_json(self, path: Path, label: str) -> List[Dict[str, Any]]:
        if not path.exists():
            raise RuntimeError(
                f"DocETL {label} output missing at {path}. Check pipeline logs for errors."
            )
        with open(path, "r", encoding="utf-8") as handle:
            content = handle.read().strip()
            if not content:
                return []
            try:
                data = json.loads(content)
            except json.JSONDecodeError as e:
                raise RuntimeError(
                    f"DocETL {label} output at {path} contains invalid JSON: {e}"
                ) from e
        if data is None:
            return []
        if isinstance(data, dict):
            # Some DocETL ops emit dicts; wrap for consistency
            return [data]
        if isinstance(data, list):
            return data
        return []

    def _normalize_map_records(self, map_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Normalize map records to ensure 'extractions' is always a list for unnest operation."""
        normalized = []
        for record in map_records:
            # Ensure record is a dict, not a string or None
            if record is None:
                continue
            if not isinstance(record, dict):
                # Try to parse if it's a string
                if isinstance(record, str):
                    try:
                        parsed = json.loads(record)
                        if parsed is not None and isinstance(parsed, dict):
                            record = parsed
                        else:
                            continue
                    except (json.JSONDecodeError, Exception):
                        continue
                else:
                    continue
            # Ensure 'extractions' key exists and is a list
            if "extractions" not in record:
                record["extractions"] = self._generate_synthetic_extractions(record)
            elif not isinstance(record["extractions"], (list, tuple)):
                value = record["extractions"]
                if value is None:
                    record["extractions"] = self._generate_synthetic_extractions(record)
                elif isinstance(value, dict):
                    # If it's a dict, wrap it in a list
                    record["extractions"] = [value]
                elif isinstance(value, str):
                    # If it's a JSON string, attempt to parse it (and extract nested payload)
                    try:
                        parsed = json.loads(value)
                        if (
                            parsed is not None
                            and isinstance(parsed, dict)
                            and "extractions" in parsed
                        ):
                            record["extractions"] = parsed["extractions"]
                        elif parsed is not None and isinstance(parsed, list):
                            record["extractions"] = parsed
                        else:
                            record["extractions"] = self._generate_synthetic_extractions(
                                record, "json_decode_shape"
                            )
                    except Exception:
                        record["extractions"] = self._generate_synthetic_extractions(
                            record, "json_decode_error"
                        )
                else:
                    # For any other type, default to empty list
                    record["extractions"] = self._generate_synthetic_extractions(
                        record, "non_iterable"
                    )
            elif isinstance(record["extractions"], list) and len(record["extractions"]) > 0:
                # Handle case where extractions is a list containing JSON strings (from tool calls)
                first_item = record["extractions"][0]
                if isinstance(first_item, str):
                    # This is likely a tool call response with JSON string in list
                    # Handle both cases: single JSON string or list of JSON strings
                    try:
                        parsed_list = []
                        for item in record["extractions"]:
                            if isinstance(item, str):
                                # Try to parse as JSON
                                try:
                                    parsed_item = json.loads(item)
                                    if parsed_item is not None:
                                        if (
                                            isinstance(parsed_item, dict)
                                            and "extractions" in parsed_item
                                        ):
                                            # Extract the nested extractions array
                                            nested = parsed_item["extractions"]
                                            if isinstance(nested, list):
                                                parsed_list.extend(nested)
                                            elif isinstance(nested, dict):
                                                parsed_list.append(nested)
                                        elif isinstance(parsed_item, dict):
                                            # If it's a dict but no "extractions" key, treat as single extraction
                                            parsed_list.append(parsed_item)
                                        elif isinstance(parsed_item, list):
                                            parsed_list.extend(parsed_item)
                                except json.JSONDecodeError:
                                    # If JSON parsing fails, skip this item
                                    continue
                            elif isinstance(item, dict):
                                # Already a dict, add it
                                parsed_list.append(item)
                            else:
                                # Other types, skip
                                continue
                        record["extractions"] = (
                            parsed_list
                            if parsed_list
                            else self._generate_synthetic_extractions(record)
                        )
                    except Exception:
                        # If processing fails, keep original or generate synthetic
                        record["extractions"] = self._generate_synthetic_extractions(
                            record, "parse_error"
                        )
            else:
                # Ensure it's a list (not tuple)
                record["extractions"] = list(record["extractions"])
            normalized.append(record)
        return normalized

    def _normalize_patient_records(
        self, patient_records: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Normalize patient records from reduce operation.

        Handles cases where reduce operation returns content with markdown code blocks
        instead of proper tool call responses (e.g., Qwen models).
        """
        if not patient_records:
            return []

        normalized = []
        for record in patient_records:
            if record is None or not isinstance(record, dict):
                continue

            # Check if record has invalid structure (e.g., empty dict from failed tool call)
            # This might happen when reduce operation fails due to missing tool calls
            if not record.get("patient_id") and not record.get("consolidated_fields"):
                # Try to extract from any string content that might contain JSON
                # This handles cases where DocETL returns empty dict but we have content elsewhere
                continue

            # Handle case where record might contain raw content that needs parsing
            # (e.g., from failed tool call that returned content instead)
            if "_raw_content" in record or "_content" in record:
                content = record.get("_raw_content") or record.get("_content", "")
                if isinstance(content, str) and content.strip():
                    # Try to extract JSON from markdown code blocks
                    # Look for JSON in markdown code blocks
                    json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
                    if json_match:
                        try:
                            parsed = json.loads(json_match.group(1))
                            if isinstance(parsed, dict):
                                # Merge parsed content into record
                                record.update(parsed)
                        except json.JSONDecodeError:
                            pass
                    else:
                        # Try to find JSON object directly
                        json_match = re.search(
                            r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", content, re.DOTALL
                        )
                        if json_match:
                            try:
                                parsed = json.loads(json_match.group(0))
                                if isinstance(parsed, dict):
                                    record.update(parsed)
                            except json.JSONDecodeError:
                                pass

            # Ensure consolidated_fields is a list
            if "consolidated_fields" in record:
                consolidated_fields = record["consolidated_fields"]
                if not isinstance(consolidated_fields, (list, tuple)):
                    if consolidated_fields is None:
                        record["consolidated_fields"] = []
                    elif isinstance(consolidated_fields, dict):
                        record["consolidated_fields"] = [consolidated_fields]
                    elif isinstance(consolidated_fields, str):
                        # Try to parse JSON string
                        try:
                            parsed = json.loads(consolidated_fields)
                            if isinstance(parsed, list):
                                record["consolidated_fields"] = parsed
                            elif isinstance(parsed, dict):
                                record["consolidated_fields"] = [parsed]
                            else:
                                record["consolidated_fields"] = []
                        except json.JSONDecodeError:
                            record["consolidated_fields"] = []
                    else:
                        record["consolidated_fields"] = []
                else:
                    record["consolidated_fields"] = list(consolidated_fields)
            else:
                record["consolidated_fields"] = []

            # Ensure patient_summary is a string
            if "patient_summary" not in record:
                record["patient_summary"] = ""
            elif not isinstance(record["patient_summary"], str):
                record["patient_summary"] = str(record["patient_summary"])

            normalized.append(record)

        return normalized

    def _patch_docetl_parser(self):
        """Patch DocETL's parse_llm_response to handle markdown code blocks in structured output."""
        try:
            from docetl.operations.utils.api import APIWrapper

            original_parse = APIWrapper.parse_llm_response

            def patched_parse_llm_response(
                self,
                response: Any,
                schema: dict[str, Any] = {},
                tools: list[dict[str, str]] | None = None,
                manually_fix_errors: bool = False,
                use_structured_output: bool = False,
            ) -> list[dict[str, Any]]:
                """Patched version that extracts JSON from markdown code blocks and tool calls."""
                # First, try to handle tool calls directly before calling original parser
                # This is critical for Qwen and similar models that use tool calls
                # We MUST handle tool calls before original parser to avoid list wrapping issues
                try:
                    if hasattr(response, "choices") and len(response.choices) > 0:
                        choice = response.choices[0]
                        message = choice.message

                        # Check for tool calls - handle these first (CRITICAL: before original parser)
                        if message.tool_calls:
                            parsed_results: List[Dict[str, Any]] = []
                            for tool_call in message.tool_calls:
                                if not hasattr(tool_call, "function"):
                                    continue
                                arguments = getattr(tool_call.function, "arguments", None)
                                parsed_results.extend(_parse_argument_payload(arguments))
                            if parsed_results:
                                return parsed_results
                except Exception as parse_err:
                    # If tool call parsing fails, continue to original parser
                    pass

                # Call original parser only if tool calls weren't found or failed
                try:
                    result = original_parse(
                        self, response, schema, tools, manually_fix_errors, use_structured_output
                    )
                    # Check if result is a list containing JSON strings (Qwen tool call issue)
                    # This happens when original parser wraps tool call arguments in a list
                    if isinstance(result, list) and len(result) > 0:
                        first_item = result[0]
                        # If first item is a string that looks like JSON, try to parse it
                        if isinstance(first_item, str) and first_item.strip().startswith("{"):
                            try:
                                parsed = json.loads(first_item)
                                if parsed is not None and isinstance(parsed, dict):
                                    # If it has extractions or consolidated_fields, return it as a dict in a list
                                    if (
                                        "extractions" in parsed
                                        or "consolidated_fields" in parsed
                                        or "patient_summary" in parsed
                                    ):
                                        return [parsed]
                            except json.JSONDecodeError:
                                pass
                        # If first item is a list containing JSON strings
                        elif isinstance(first_item, list) and len(first_item) > 0:
                            if isinstance(first_item[0], str) and first_item[0].strip().startswith(
                                "{"
                            ):
                                try:
                                    parsed = json.loads(first_item[0])
                                    if parsed is not None and isinstance(parsed, dict):
                                        if (
                                            "extractions" in parsed
                                            or "consolidated_fields" in parsed
                                            or "patient_summary" in parsed
                                        ):
                                            return [parsed]
                                except json.JSONDecodeError:
                                    pass
                        # If first item is already a dict with the right structure, use it directly
                        elif isinstance(first_item, dict):
                            if (
                                "extractions" in first_item
                                or "consolidated_fields" in first_item
                                or "patient_summary" in first_item
                            ):
                                return [first_item]
                    return result
                except Exception as e:
                    error_msg = str(e)
                    # Handle various error types including validation errors and markdown code blocks
                    # Also check if error message contains the problematic list format
                    if (
                        "Could not decode LLM JSON response" in error_msg
                        or "Could not decode structured output JSON response" in error_msg
                        or "No tool calls" in error_msg
                        or "validation" in error_msg.lower()
                        or "schema" in error_msg.lower()
                        or "Expected schema" in error_msg
                        or "Invalid output" in error_msg
                        or ("[" in error_msg and "extractions" in error_msg and "]" in error_msg)
                    ):
                        # Check if error message contains a list with JSON string (Qwen tool call issue)
                        # Pattern: Invalid output: ['{"extractions": [...]}']
                        patterns = [
                            r"Invalid output:\s*\[['\"](\{.*?\})['\"]\]",
                            r"\[['\"](\{.*?extractions.*?\})['\"]\]",
                            r"\[['\"](\{.*?\})['\"]\]",
                        ]
                        for pattern in patterns:
                            list_match = re.search(pattern, error_msg, re.DOTALL)
                            if not list_match:
                                continue
                            json_str = list_match.group(1)
                            json_str = json_str.replace("\\'", "'").replace('\\"', '"')
                            parsed_candidates = _parse_argument_payload(json_str)
                            if parsed_candidates:
                                return parsed_candidates

                        # Fall back to literal_eval when DocETL wraps JSON inside Python-style lists/strings
                        list_expr_match = re.search(
                            r"Invalid output:\s*(\[[\s\S]+?\])", error_msg, re.DOTALL
                        )
                        if list_expr_match:
                            try:
                                parsed_list = ast.literal_eval(list_expr_match.group(1))
                            except Exception:
                                parsed_list = None
                            if isinstance(parsed_list, list):
                                parsed_candidates = _parse_argument_payload(parsed_list)
                                if parsed_candidates:
                                    return parsed_candidates
                        try:
                            if hasattr(response, "choices") and len(response.choices) > 0:
                                choice = response.choices[0]
                                message = choice.message
                                if getattr(message, "tool_calls", None):
                                    parsed_results: List[Dict[str, Any]] = []
                                    for tool_call in message.tool_calls:
                                        if not hasattr(tool_call, "function"):
                                            continue
                                        arguments = getattr(tool_call.function, "arguments", None)
                                        parsed_results.extend(_parse_argument_payload(arguments))
                                    if parsed_results:
                                        return parsed_results
                        except Exception:
                            pass

                        # Try to extract JSON from markdown code blocks in content (for structured_output mode)
                        try:
                            if hasattr(response, "choices") and len(response.choices) > 0:
                                choice = response.choices[0]
                                message = choice.message
                                content = getattr(message, "content", None)
                                if content and isinstance(content, str):
                                    # Check if model is complaining about missing inputs (Qwen issue)
                                    if (
                                        "actual content" in content.lower()
                                        or "missing from your request" in content.lower()
                                        or "provide the necessary details" in content.lower()
                                        or "hypothetical" in content.lower()
                                    ):
                                        # Model didn't see the inputs - this is a prompt/input issue
                                        # Try to extract any JSON that might be in the response anyway
                                        # But also log a warning
                                        logger = logging.getLogger(__name__)
                                        logger.warning(
                                            f"Qwen model indicates it didn't see inputs. Content preview: {content[:200]}..."
                                        )

                                    # Try to extract JSON from markdown code blocks
                                    json_match = re.search(
                                        r"```(?:json)?\s*(\{.*?)\s*```", content, re.DOTALL
                                    )
                                    if not json_match:
                                        # Try without code blocks - find JSON object (may be truncated)
                                        json_match = re.search(r"(\{.*)", content, re.DOTALL)

                                    if json_match:
                                        try:
                                            json_str = json_match.group(1)
                                            # Try to fix truncated JSON
                                            json_str = _sanitize_json_payload(json_str)
                                            parsed = json.loads(json_str)
                                            if parsed is not None and isinstance(parsed, dict):
                                                # Fix common key mismatches (Qwen sometimes uses wrong keys)
                                                if (
                                                    "summary" in parsed
                                                    and "patient_summary" not in parsed
                                                ):
                                                    parsed["patient_summary"] = parsed.pop(
                                                        "summary"
                                                    )
                                                # Check if this is a reduce operation (has consolidated_fields or patient_summary)
                                                is_reduce = (
                                                    "consolidated_fields" in parsed
                                                    or "patient_summary" in parsed
                                                    or "summary" in parsed
                                                )
                                                if is_reduce:
                                                    # For reduce operations, ensure patient_id is set
                                                    if (
                                                        "patient_id" not in parsed
                                                        and "reduce_key" in str(schema)
                                                    ):
                                                        # Try to extract from reduce_key in schema if available
                                                        # This is a fallback - ideally reduce_key should be in the prompt
                                                        pass
                                                # Ensure required fields exist for reduce operations
                                                if is_reduce:
                                                    if "consolidated_fields" not in parsed:
                                                        parsed["consolidated_fields"] = []
                                                    if "patient_summary" not in parsed:
                                                        parsed["patient_summary"] = ""

                                                # Fix truncated extractions array if needed
                                                if "extractions" in parsed:
                                                    extractions = parsed["extractions"]
                                                    if (
                                                        isinstance(extractions, list)
                                                        and extractions
                                                    ):
                                                        # Check if last extraction is incomplete
                                                        last_extraction = extractions[-1]
                                                        if isinstance(last_extraction, dict):
                                                            # Ensure all required fields exist
                                                            required_fields = [
                                                                "field_name",
                                                                "category",
                                                                "data_type",
                                                                "raw_value",
                                                                "normalized_value",
                                                                "units",
                                                                "vocabulary_code",
                                                                "reasoning_excerpt",
                                                                "explanation",
                                                                "confidence_level",
                                                                "confidence_score",
                                                                "inferred",
                                                                "related_entities",
                                                            ]
                                                            for field in required_fields:
                                                                if field not in last_extraction:
                                                                    if field == "related_entities":
                                                                        last_extraction[field] = []
                                                                    elif field in [
                                                                        "confidence_score"
                                                                    ]:
                                                                        last_extraction[field] = 0.5
                                                                    elif field in ["inferred"]:
                                                                        last_extraction[field] = (
                                                                            False
                                                                        )
                                                                    elif field in [
                                                                        "confidence_level"
                                                                    ]:
                                                                        last_extraction[field] = (
                                                                            "medium"
                                                                        )
                                                                    else:
                                                                        last_extraction[field] = ""
                                                        # If last extraction is not a dict (truncated), remove it
                                                        elif not isinstance(last_extraction, dict):
                                                            extractions.pop()

                                                return [parsed]
                                        except json.JSONDecodeError as json_err:
                                            # If JSON is still invalid after fixing, try more aggressive fixes
                                            try:
                                                # Try to extract just the extractions array if it exists
                                                if (
                                                    "extractions" in json_str
                                                    or '"extractions"' in json_str
                                                ):
                                                    # Find the extractions array
                                                    extractions_match = re.search(
                                                        r'"extractions"\s*:\s*\[(.*?)\]',
                                                        json_str,
                                                        re.DOTALL,
                                                    )
                                                    if not extractions_match:
                                                        # Array might be truncated - try to find partial array
                                                        extractions_match = re.search(
                                                            r'"extractions"\s*:\s*\[(.*)',
                                                            json_str,
                                                            re.DOTALL,
                                                        )
                                                        if extractions_match:
                                                            # Try to complete the array
                                                            partial_array = extractions_match.group(
                                                                1
                                                            )
                                                            # Find last complete object
                                                            objects = re.findall(
                                                                r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}",
                                                                partial_array,
                                                            )
                                                            if objects:
                                                                # Use last complete object
                                                                last_obj_str = objects[-1]
                                                                try:
                                                                    last_obj = json.loads(
                                                                        last_obj_str
                                                                    )
                                                                    # Create minimal valid JSON
                                                                    minimal_json = {
                                                                        "extractions": [last_obj]
                                                                    }
                                                                    return [minimal_json]
                                                                except json.JSONDecodeError:
                                                                    pass
                                            except Exception:
                                                pass
                                            # Log the error for debugging
                                            logger = logging.getLogger(__name__)
                                            logger.warning(
                                                f"Failed to parse JSON even after fixing truncation: {json_err}. "
                                                f"Content preview: {content[:500]}..."
                                            )
                        except Exception:
                            pass

                        # Try again with more aggressive parsing
                        try:
                            if hasattr(response, "choices") and len(response.choices) > 0:
                                choice = response.choices[0]
                                message = choice.message

                                if getattr(message, "tool_calls", None):
                                    parsed_results: List[Dict[str, Any]] = []
                                    for tool_call in message.tool_calls:
                                        if not hasattr(tool_call, "function"):
                                            continue
                                        arguments = getattr(tool_call.function, "arguments", None)
                                        parsed_results.extend(_parse_argument_payload(arguments))
                                    if parsed_results:
                                        return parsed_results

                                # Try to extract JSON from markdown code blocks in content
                                content = getattr(message, "content", None)
                                if content:
                                    # Try to extract JSON from markdown code blocks
                                    json_match = re.search(
                                        r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL
                                    )
                                    if json_match:
                                        try:
                                            parsed = json.loads(json_match.group(1))
                                            if parsed is not None and isinstance(parsed, dict):
                                                # Attempt to parse any top-level values that are JSON-encoded strings
                                                for k, v in parsed.items():
                                                    if isinstance(v, str):
                                                        try:
                                                            if v.strip().startswith(
                                                                "{"
                                                            ) or v.strip().startswith("["):
                                                                parsed[k] = json.loads(v)
                                                        except json.JSONDecodeError:
                                                            pass
                                                return [parsed]
                                        except json.JSONDecodeError:
                                            pass
                                    else:
                                        # Try to find JSON object directly (without code blocks)
                                        json_match = re.search(
                                            r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", content, re.DOTALL
                                        )
                                        if json_match:
                                            try:
                                                parsed = json.loads(json_match.group(0))
                                                if parsed is not None and isinstance(parsed, dict):
                                                    # Attempt to parse any top-level values that are JSON-encoded strings
                                                    for k, v in parsed.items():
                                                        if isinstance(v, str):
                                                            try:
                                                                if v.strip().startswith(
                                                                    "{"
                                                                ) or v.strip().startswith("["):
                                                                    parsed[k] = json.loads(v)
                                                            except json.JSONDecodeError:
                                                                pass
                                                    return [parsed]
                                            except json.JSONDecodeError:
                                                pass
                        except Exception:
                            pass
                    # Re-raise original exception if we couldn't fix it
                    raise

            # Apply the patch
            APIWrapper.parse_llm_response = patched_parse_llm_response
        except Exception:
            # If patching fails, continue without patch
            pass

    def _generate_synthetic_extractions(
        self,
        record: Dict[str, Any],
        suffix: str = "missing_extractions",
    ) -> List[Dict[str, Any]]:
        """Create placeholder extraction entries to keep DocETL pipeline flowing."""
        patient_id = record.get("patient_id", "unknown_patient")
        doc_id = record.get("doc_id", "unknown_doc")
        doc_type = record.get("doc_type", "unknown")
        doc_date = record.get("doc_date", "Not Reported")
        field_name = f"synthetic_{suffix}"

        return [
            {
                "field_name": field_name,
                "category": "technical",
                "data_type": "string",
                "raw_value": "",
                "normalized_value": "",
                "units": "",
                "vocabulary_code": "",
                "reasoning_excerpt": (
                    f"Placeholder entry generated for {patient_id}/{doc_id} "
                    f"({doc_type} on {doc_date}) due to upstream decode issue ({suffix})."
                ),
                "explanation": (
                    "Synthetic extraction inserted to keep DocETL pipeline consistent when "
                    "the upstream LLM response fails schema validation or JSON parsing."
                ),
                "confidence_level": "low",
                "confidence_score": 0.0,
                "inferred": True,
                "related_entities": [patient_id, doc_id, suffix],
            }
        ]

    def _log_map_prompts(
        self,
        session_id: str,
        records: List[Dict[str, Any]],
    ) -> None:
        if not self.map_prompt_template or not records:
            return
        # patient_session_id formatından patient_id'yi çıkar (format: {session_id}__{patient_id})
        # Eğer session_id içinde __ varsa, hasta bazlı session'dır
        patient_id = None
        if "__" in session_id:
            parts = session_id.split("__", 1)
            if len(parts) == 2:
                # Ana session_id'yi al (ilk kısım)
                actual_session_id = parts[0]
                patient_id = parts[1]
                session_id = actual_session_id
            else:
                # Fallback: session_id'yi olduğu gibi kullan
                actual_session_id = session_id
        else:
            actual_session_id = session_id

        # Eğer patient_id yoksa, records'dan çıkarmayı dene
        if patient_id is None and records:
            # İlk record'dan patient_id'yi al
            first_record = records[0] if records else {}
            patient_id = first_record.get("patient_id")

        env = Environment(undefined=StrictUndefined)
        template = env.from_string(self.map_prompt_template)
        entries: List[str] = []
        for record in records:
            try:
                prompt_text = template.render(input=record)
            except TemplateError as exc:
                prompt_text = f"[Prompt render error: {exc}]"
            entries.append(f"--- doc_id={record.get('doc_id', 'unknown_doc')} ---\n{prompt_text}")
        self._write_prompt_log(actual_session_id, "stage_extractor", entries, patient_id=patient_id)

    def _log_reduce_prompts(
        self,
        session_id: str,
        resolve_records: Optional[List[Dict[str, Any]]],
    ) -> None:
        if not self.reduce_prompt_template or not resolve_records:
            return
        # patient_session_id formatından patient_id'yi çıkar (format: {session_id}__{patient_id})
        # Eğer session_id içinde __ varsa, hasta bazlı session'dır
        actual_session_id = session_id
        if "__" in session_id:
            parts = session_id.split("__", 1)
            if len(parts) == 2:
                # Ana session_id'yi al (ilk kısım)
                actual_session_id = parts[0]

        env = Environment(undefined=StrictUndefined)
        template = env.from_string(self.reduce_prompt_template)
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for record in resolve_records:
            key = (
                str(record.get("patient_id"))
                if record.get("patient_id")
                else str(record.get("reduce_key", "unknown_patient"))
            )
            grouped.setdefault(key, []).append(record)

        # Her hasta için ayrı log dosyası oluştur
        for patient_id, items in grouped.items():
            entries: List[str] = []
            try:
                prompt_text = template.render(reduce_key=patient_id, inputs=items)
            except TemplateError as exc:
                prompt_text = f"[Prompt render error: {exc}]"
            entries.append(f"--- patient_id={patient_id} ---\n{prompt_text}")
            self._write_prompt_log(
                actual_session_id, "stage_consolidator", entries, patient_id=patient_id
            )

    def _write_prompt_log(
        self,
        session_id: str,
        stage_name: str,
        entries: List[str],
        patient_id: Optional[str] = None,
    ) -> None:
        if not entries:
            return
        # Ana session klasörü oluştur
        log_dir = Path(settings.log_dir) / session_id
        log_dir.mkdir(parents=True, exist_ok=True)

        # Eğer patient_id varsa, hasta bazlı alt klasör oluştur
        if patient_id:
            patient_log_dir = log_dir / str(patient_id)
            patient_log_dir.mkdir(parents=True, exist_ok=True)
            log_path = patient_log_dir / f"{stage_name}_prompts.log"
        else:
            log_path = log_dir / f"{stage_name}_prompts.log"

        with open(log_path, "w", encoding="utf-8") as handle:
            handle.write("\n\n".join(entries))
