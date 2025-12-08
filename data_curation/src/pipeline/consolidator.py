"""Patient-level consolidator powered by DocETL reduce outputs."""

from __future__ import annotations

import json
import logging
import re
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from src.models.schemas import (
    ConsolidatedField,
    ConsolidationResult,
    ExtractionResult,
)
from src.pipeline.docetl_runner import DocETLPipelineArtifacts
from src.pipeline.mcode_template import get_default_mcode_structure
from config.settings import settings
from src.utils.logger import setup_logger
from src.utils.ontology import OntologyLoader


# Category to domain mapping based on output_samples structure
CATEGORY_TO_DOMAIN = {
    "provenance": None,  # file_name goes at top level
    "demographics": "patient",
    "patient": "patient",
    "health_assessment": "health_assessment",
    "vitals": "health_assessment",
    "vital_signs": "health_assessment",
    "diagnosis": "primary_cancers",  # Special handling - array structure
    "cancerd iagnosis": "primary_cancers",
    "primarycancercondition": "primary_cancers",
    "clinical_staging": "cancer_stage",
    "tnmclinicalstagegroup": "cancer_stage",
    "pathological_staging": "cancer_stage",
    "summary_staging": "cancer_stage",
    "staging": "cancer_stage",
    "metastasis": "secondary_cancer_conditions",
    "biomarkers": "biomarkers",
    "treatment": "cancer_treatment",
    "therapy": "cancer_treatment",
    "disease_status": "disease_status",
    "outcome": "outcome_follow_up",
    "performance": "health_assessment",
    "performance_status": "health_assessment",
    "lab_results": "extensions",
    "pathology": "extensions",
    "comorbidities": "extensions",
    "health_conditions": "extensions",
    "technical": None,
}

# Ontology domain names need to be translated into mCODE output domains
ONTOLOGY_DOMAIN_TO_MCODE_DOMAIN = {
    "diagnosis": "primary_cancers",
    "clinical_staging": "cancer_stage",
    "pathological_staging": "cancer_stage",
    "summary_staging": "cancer_stage",
    "performance_status": "health_assessment",
    "patient": "patient",
    "health_assessment": "health_assessment",
    "secondary_cancer_conditions": "secondary_cancer_conditions",
    "biomarkers": "biomarkers",
    "cancer_treatment": "cancer_treatment",
    "disease_status": "disease_status",
    "outcome_follow_up": "outcome_follow_up",
    "extensions": "extensions",
}

# Normalize NAACCR/docetl field identifiers to the canonical mCODE envelope keys.
FIELD_NAME_ALIASES = {
    "naaccr_histology_cd": "histology_morphology",
    "ca_site": "body_site",
    "ca_clinical_t_stage": "tnm_t_clinical",
    "ca_clinical_n_stage": "tnm_n_clinical",
    "ca_clinical_m_stage": "tnm_m_clinical",
    "ca_path_t_stage": "tnm_t_pathologic",
    "ca_path_n_stage": "tnm_n_pathologic",
    "ca_path_m_stage": "tnm_m_pathologic",
    "ca_gen_sum_stage_2": "stage_group",
    "ecog": "performance_status",
    "naaccr_diagnosis_dt": "diagnosis_date",
    "ca_site": "body_site",
    "naaccr_histology_cd": "histology_morphology",
}


class Consolidator:
    """Converts DocETL reduce artifacts into `ConsolidationResult`."""

    def __init__(
        self,
        ontology: Optional[OntologyLoader] = None,
        llm_provider=None,
        provider_label: str = "docetl",
    ):
        self.ontology = ontology or OntologyLoader()
        self.provider_label = provider_label
        self._resolve_records: Optional[List[Dict[str, Any]]] = None  # Cached for multi-cancer detection

    async def consolidate(
        self,
        extraction_result: ExtractionResult,
        pipeline_artifacts: DocETLPipelineArtifacts,
        session_id: str,
    ) -> ConsolidationResult:
        """Transform DocETL reduce output into patient-level schema."""
        if not pipeline_artifacts:
            raise ValueError("DocETL artifacts are required for consolidation.")

        logger = setup_logger(session_id, "stage_consolidator")
        logger.info(
            "Building patient-level objects from %s",
            pipeline_artifacts.patient_output_path,
        )

        start_ts = time.time()
        consolidated_fields = self._build_fields_from_artifacts(
            pipeline_artifacts.patient_records,
            extraction_result,
            pipeline_artifacts.resolve_records,
            logger=logger,
        )
        processing_time = time.time() - start_ts

        result = ConsolidationResult(
            session_id=session_id,
            generated_timestamp=datetime.now(),
            stage="stage_consolidator",
            total_fields_consolidated=len(consolidated_fields),
            consolidated_fields=consolidated_fields,
            processing_time_seconds=processing_time,
        )
        logger.info(
            "Consolidation complete: %d consolidated patients, %.2fs",
            len(consolidated_fields),
            processing_time,
        )
        return result

    def _build_consolidation_summary_text(
        self,
        patient_id: str,
        mcode_extraction: Dict[str, Any],
        doc_count: int,
    ) -> str:
        """Generate a human-readable summary similar to reference outputs."""
        parts: List[str] = []

        doc_part = (
            f"Consolidated patient-level extraction across {doc_count} documents"
            if doc_count
            else "Consolidated patient-level extraction"
        )
        parts.append(doc_part)

        primary_cancers = mcode_extraction.get("primary_cancers") or []
        if primary_cancers:
            cancer_descriptions: List[str] = []
            for cancer in primary_cancers:
                diagnosis = (cancer.get("diagnosis") or {}).get("final_value")
                body_site = (cancer.get("body_site") or {}).get("final_value")
                histology = (cancer.get("histology_morphology") or {}).get("final_value") or ""
                description = diagnosis or body_site or "cancer"
                if histology and histology != "Not Reported":
                    description = f"{description} ({histology})"
                cancer_descriptions.append(description)
            parts.append(
                f"identifying {len(primary_cancers)} primary cancers: "
                f"{'; '.join(cancer_descriptions)}."
            )
        else:
            parts.append("capturing cancer registry domains.")

        disease_status = (mcode_extraction.get("disease_status") or {}).get("final_status", {})
        status_value = (disease_status.get("disease_status") or {}).get("final_value")
        recurrence_indicator = (disease_status.get("recurrence_indicator") or {}).get(
            "final_value"
        ) or ""
        recurrence_date = (disease_status.get("recurrence_date") or {}).get("final_value") or ""
        status_parts: List[str] = []
        if status_value and status_value != "Not Reported":
            status_parts.append(f"overall disease status {status_value}")
        if recurrence_indicator and recurrence_indicator != "Not Reported":
            if recurrence_indicator.lower() in {"yes", "true"} and recurrence_date:
                status_parts.append(f"recurrence noted ({recurrence_date})")
            else:
                status_parts.append(f"recurrence indicator {recurrence_indicator}")
        if status_parts:
            parts.append("Highlights include " + ", ".join(status_parts) + ".")

        biomarkers = mcode_extraction.get("biomarkers") or {}
        tumor_markers = biomarkers.get("tumor_markers", {}).get("final_value")
        positive_markers = biomarkers.get("positive_markers", {}).get("final_value")
        biomarker_snippets: List[str] = []
        if tumor_markers:
            if isinstance(tumor_markers, list):
                biomarker_snippets.append(
                    "tumor markers " + ", ".join(str(val) for val in tumor_markers)
                )
            elif tumor_markers != "Not Reported":
                biomarker_snippets.append(f"tumor markers {tumor_markers}")
        if positive_markers and positive_markers != "Not Reported":
            biomarker_snippets.append(f"positive biomarkers {positive_markers}")
        if biomarker_snippets:
            parts.append("Biomarker findings include " + "; ".join(biomarker_snippets) + ".")

        treatments = mcode_extraction.get("cancer_treatment") or {}
        surgery = treatments.get("surgery_procedures", {}).get("final_value")
        systemic = treatments.get("systemic_therapy", {}).get("final_value")
        treatment_bits: List[str] = []
        if surgery and surgery != "Not Reported":
            treatment_bits.append(f"surgical history ({surgery})")
        if systemic and systemic != "Not Reported":
            treatment_bits.append(f"systemic therapy ({systemic})")
        if treatment_bits:
            parts.append("Treatment summary covers " + " and ".join(treatment_bits) + ".")

        comorbidities = (
            (mcode_extraction.get("extensions") or {}).get("comorbidities", {}).get("final_value")
        )
        if comorbidities and comorbidities != "Not Reported":
            if isinstance(comorbidities, list):
                comorb_text = ", ".join(str(item) for item in comorbidities)
            else:
                comorb_text = str(comorbidities)
            parts.append(f"Documented comorbidities include {comorb_text}.")

        return " ".join(part.strip() for part in parts if part and part.strip())

    def _normalize_all_field_names(self, patient_records: List[Dict[str, Any]]) -> None:
        """Apply FIELD_NAME_ALIASES to all field names in patient records.
        
        This is the SINGLE POINT where NAACCR field names are converted to mCODE format.
        All downstream code can assume field names are already in canonical mCODE format.
        """
        for record in patient_records:
            for field in record.get("consolidated_fields", []):
                if "_original_field_name" not in field:
                    field["_original_field_name"] = field.get("field_name")
                original = field.get("_original_field_name") or field.get("field_name")
                field["field_name"] = FIELD_NAME_ALIASES.get(original, original)

    def _build_fields_from_artifacts(
        self,
        patient_records: List[Dict[str, Any]],
        extraction_result: ExtractionResult,
        resolve_records: Optional[List[Dict[str, Any]]] = None,
        logger: Optional[logging.Logger] = None,
    ) -> List[ConsolidatedField]:
        """Shape DocETL patient outputs to match the sample consolidation artifact."""
        # Cache resolve_records for use in _build_primary_cancers_array
        self._resolve_records = resolve_records
        patient_records = patient_records or []
        if resolve_records and (
            not patient_records
            or all(not record.get("consolidated_fields") for record in patient_records)
            or not self._records_have_ontology_fields(patient_records)
        ):
            patient_records = self._build_patient_records_from_resolve(resolve_records)

        # Normalize all field names to mCODE format in a SINGLE PASS
        # This ensures consistent field names regardless of data source
        self._normalize_all_field_names(patient_records)

        if not patient_records:
            return []

        # Optimized worker count calculation for better parallel processing
        import os

        cpu_count = os.cpu_count() or 4
        # Use more workers: CPU cores * 3 for I/O-bound operations, but respect limits
        optimal_workers = min(
            settings.max_workers,
            max(cpu_count * 3, len(patient_records), 1),
        )
        worker_count = min(optimal_workers, max(1, len(patient_records)))

        active_logger = logger or logging.getLogger(__name__)
        active_logger.debug(f"Using {worker_count} workers for consolidation processing")

        results: List[Optional[ConsolidatedField]] = [None] * len(patient_records)

        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [
                executor.submit(
                    self._build_consolidated_field,
                    idx,
                    record,
                    extraction_result,
                    active_logger,
                )
                for idx, record in enumerate(patient_records)
            ]
            for future in as_completed(futures):
                idx, field = future.result()
                if field:
                    results[idx] = field

        return [field for field in results if field is not None]

    def _build_consolidated_field(
        self,
        idx: int,
        record: Dict[str, Any],
        extraction_result: ExtractionResult,
        logger: logging.Logger,
    ) -> Tuple[int, Optional[ConsolidatedField]]:
        """Construct a single ConsolidatedField entry."""
        try:
            if not record:
                logger.warning("Patient record %s is empty, skipping", idx)
                return idx, None

            mcode_value = self._build_mcode_value(record, extraction_result)
            consolidated_field = ConsolidatedField(
                field_name="mcode_patient_extraction",
                category="mcode_registry",
                consolidated_value=mcode_value["value"],
                all_values=[],
                units=None,
                vocabulary_code=None,
                confidence_score=mcode_value["confidence_score"] or 0.0,
                source_documents=mcode_value["source_documents"],
                consolidation_reasoning=mcode_value["consolidation_reasoning"],
                data_type="object",
            )
            return idx, consolidated_field
        except Exception as exc:
            patient_id = (
                record.get("patient_id", f"patient_{idx}")
                if isinstance(record, dict)
                else f"patient_{idx}"
            )
            logger.error(
                "Failed to build consolidated field for %s: %s",
                patient_id,
                exc,
            )
            return idx, None

    def _build_mcode_value(
        self,
        record: Dict[str, Any],
        extraction_result: ExtractionResult,
    ) -> Dict[str, Any]:
        """Convert DocETL patient record into the canonical JSON envelope matching output_samples."""
        if not record:
            raise ValueError("Empty record provided to _build_mcode_value")
        patient_id = record.get("patient_id", "unknown_patient")
        fields = record.get("consolidated_fields") or []

        # Build domain structure matching output_samples
        mcode_extraction: Dict[str, Any] = {}
        all_source_docs: List[str] = []
        confidences: List[float] = []

        # Group fields by domain
        domain_fields: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        file_name_fields: List[Dict[str, Any]] = []
        diagnosis_fields: List[Dict[str, Any]] = []

        for field_payload in fields:
            if not field_payload or not isinstance(field_payload, dict):
                continue
            self._canonicalize_field_payload(field_payload)
            if self._is_synthetic_field(field_payload):
                continue
            original_name = field_payload.get("_original_field_name") or field_payload.get(
                "field_name", "unknown_field"
            )
            field_name = field_payload.get("field_name", "unknown_field")
            if field_name != "file_name" and not self.ontology.get_field_definition(original_name):
                continue
            category = field_payload.get("category", "general")
            normalized_category = self._normalize_category(category, original_name or field_name)
            field_payload["_domain"] = normalized_category

            # Special handling for file_name (provenance)
            if field_name == "file_name" or category == "provenance":
                file_name_fields.append(field_payload)
                continue

            # Special handling for diagnosis fields (will become primary_cancers array)
            domain_hint = self._determine_domain(field_payload, normalized_category)
            if domain_hint == "primary_cancers" and field_name in {
                "diagnosis",
                "body_site",
                "laterality",
                "histology_morphology",
                "grade",
                "behavior_code",
                "clinical_or_pathologic_indicator",
                "diagnosis_date",
            }:
                diagnosis_fields.append(field_payload)
                continue

            # Map category to domain
            domain_fields[domain_hint].append(field_payload)

        # Build file_name domain (top level)
        if file_name_fields:
            file_name_summary = self._build_field_summary(
                file_name_fields[0], extraction_result, patient_id
            )
            mcode_extraction["file_name"] = {
                "final_value": file_name_summary["final_value"],
                "supporting_evidence": file_name_summary["supporting_evidence"],
                "contradictory_evidence": file_name_summary["contradictory_evidence"],
            }
            source_docs = file_name_summary.get("source_documents") or []
            all_source_docs.extend(source_docs)
            # Only add confidence if it exists from LLM - don't use default
            if "confidence" in file_name_summary and file_name_summary["confidence"] is not None:
                confidences.append(self._clamp_confidence(file_name_summary["confidence"]))

        # Build patient domain
        if "patient" in domain_fields:
            patient_domain: Dict[str, Any] = {}
            for field_payload in domain_fields["patient"]:
                field_name = field_payload.get("field_name", "unknown_field")
                field_summary = self._build_field_summary(
                    field_payload, extraction_result, patient_id
                )
                field_entry: Dict[str, Any] = {
                    "final_value": field_summary["final_value"],
                    "supporting_evidence": field_summary["supporting_evidence"],
                    "contradictory_evidence": field_summary["contradictory_evidence"],
                }
                # Add normalized_value and normalized_code if present
                if field_payload.get("normalized_value"):
                    field_entry["normalized_value"] = field_payload.get("normalized_value")
                if field_payload.get("vocabulary_code"):
                    field_entry["normalized_code"] = field_payload.get("vocabulary_code")
                patient_domain[field_name] = field_entry
                source_docs = field_summary.get("source_documents") or []
                all_source_docs.extend(source_docs)
                # Only add confidence if it exists from LLM - don't use default
                if "confidence" in field_summary and field_summary["confidence"] is not None:
                    confidences.append(self._clamp_confidence(field_summary["confidence"]))
            if patient_domain:
                mcode_extraction["patient"] = patient_domain

        # Build health_assessment domain
        if "health_assessment" in domain_fields:
            health_domain: Dict[str, Any] = {}
            for field_payload in domain_fields["health_assessment"]:
                field_name = field_payload.get("field_name", "unknown_field")
                field_summary = self._build_field_summary(
                    field_payload, extraction_result, patient_id
                )
                field_entry: Dict[str, Any] = {
                    "final_value": field_summary["final_value"],
                    "supporting_evidence": field_summary["supporting_evidence"],
                    "contradictory_evidence": field_summary["contradictory_evidence"],
                }
                if field_payload.get("units"):
                    field_entry["units"] = field_payload.get("units")
                health_domain[field_name] = field_entry
                source_docs = field_summary.get("source_documents") or []
                all_source_docs.extend(source_docs)
                # Only add confidence if it exists from LLM - don't use default
                if "confidence" in field_summary and field_summary["confidence"] is not None:
                    confidences.append(self._clamp_confidence(field_summary["confidence"]))
            if health_domain:
                mcode_extraction["health_assessment"] = health_domain

        # Build primary_cancers array (from diagnosis fields)
        if diagnosis_fields:
            primary_cancers = self._build_primary_cancers_array(
                diagnosis_fields, extraction_result, patient_id,
                primary_cancers_guide=record.get("primary_cancers")
            )
            if primary_cancers:
                mcode_extraction["primary_cancers"] = primary_cancers
                for cancer in primary_cancers:
                    for field_name, field_data in cancer.items():
                        if field_name != "cancer_id" and isinstance(field_data, dict):
                            if "supporting_evidence" in field_data:
                                for evidence in field_data["supporting_evidence"]:
                                    if "source_file" in evidence:
                                        source_file = evidence["source_file"]
                                        # Include all documents (not just summary documents)
                                        all_source_docs.append(source_file)
                                    # Only add confidence if it exists from LLM and is not None
                                    if (
                                        "confidence" in evidence
                                        and evidence.get("confidence") is not None
                                    ):
                                        confidences.append(
                                            self._clamp_confidence(evidence["confidence"])
                                        )

        # Build secondary_cancer_conditions domain
        if "secondary_cancer_conditions" in domain_fields:
            secondary_domain: Dict[str, Any] = {}
            for field_payload in domain_fields["secondary_cancer_conditions"]:
                field_name = field_payload.get("field_name", "unknown_field")
                field_summary = self._build_field_summary(
                    field_payload, extraction_result, patient_id
                )
                secondary_domain[field_name] = {
                    "final_value": field_summary["final_value"],
                    "supporting_evidence": field_summary["supporting_evidence"],
                    "contradictory_evidence": field_summary["contradictory_evidence"],
                }
                source_docs = field_summary.get("source_documents") or []
                all_source_docs.extend(source_docs)
                # Only add confidence if it exists from LLM - don't use default
                if "confidence" in field_summary and field_summary["confidence"] is not None:
                    confidences.append(self._clamp_confidence(field_summary["confidence"]))
            if secondary_domain:
                mcode_extraction["secondary_cancer_conditions"] = secondary_domain

        # Build cancer_stage domain (with timeline structure)
        if "cancer_stage" in domain_fields:
            stage_domain = self._build_cancer_stage_domain(
                domain_fields["cancer_stage"], extraction_result, patient_id
            )
            if stage_domain:
                mcode_extraction["cancer_stage"] = stage_domain

        # Build biomarkers domain
        if "biomarkers" in domain_fields:
            biomarkers_domain: Dict[str, Any] = {}
            for field_payload in domain_fields["biomarkers"]:
                field_name = field_payload.get("field_name", "unknown_field")
                field_summary = self._build_field_summary(
                    field_payload, extraction_result, patient_id
                )
                biomarkers_domain[field_name] = {
                    "final_value": field_summary["final_value"],
                    "supporting_evidence": field_summary["supporting_evidence"],
                    "contradictory_evidence": field_summary["contradictory_evidence"],
                }
                source_docs = field_summary.get("source_documents") or []
                all_source_docs.extend(source_docs)
                # Only add confidence if it exists from LLM - don't use default
                if "confidence" in field_summary and field_summary["confidence"] is not None:
                    confidences.append(self._clamp_confidence(field_summary["confidence"]))
            if biomarkers_domain:
                mcode_extraction["biomarkers"] = biomarkers_domain

        # Build cancer_treatment domain
        if "cancer_treatment" in domain_fields:
            treatment_domain: Dict[str, Any] = {}
            for field_payload in domain_fields["cancer_treatment"]:
                field_name = field_payload.get("field_name", "unknown_field")
                field_summary = self._build_field_summary(
                    field_payload, extraction_result, patient_id
                )
                treatment_domain[field_name] = {
                    "final_value": field_summary["final_value"],
                    "supporting_evidence": field_summary["supporting_evidence"],
                    "contradictory_evidence": field_summary["contradictory_evidence"],
                }
                source_docs = field_summary.get("source_documents") or []
                all_source_docs.extend(source_docs)
                # Only add confidence if it exists from LLM - don't use default
                if "confidence" in field_summary and field_summary["confidence"] is not None:
                    confidences.append(self._clamp_confidence(field_summary["confidence"]))
            if treatment_domain:
                mcode_extraction["cancer_treatment"] = treatment_domain

        # Build disease_status domain (with timeline structure)
        if "disease_status" in domain_fields:
            disease_domain = self._build_disease_status_domain(
                domain_fields["disease_status"], extraction_result, patient_id
            )
            if disease_domain:
                mcode_extraction["disease_status"] = disease_domain

        # Build outcome_follow_up domain
        if "outcome_follow_up" in domain_fields:
            outcome_domain: Dict[str, Any] = {}
            for field_payload in domain_fields["outcome_follow_up"]:
                field_name = field_payload.get("field_name", "unknown_field")
                field_summary = self._build_field_summary(
                    field_payload, extraction_result, patient_id
                )
                outcome_domain[field_name] = {
                    "final_value": field_summary["final_value"],
                    "supporting_evidence": field_summary["supporting_evidence"],
                    "contradictory_evidence": field_summary["contradictory_evidence"],
                }
                source_docs = field_summary.get("source_documents") or []
                all_source_docs.extend(source_docs)
                # Only add confidence if it exists from LLM - don't use default
                if "confidence" in field_summary and field_summary["confidence"] is not None:
                    confidences.append(self._clamp_confidence(field_summary["confidence"]))
            if outcome_domain:
                mcode_extraction["outcome_follow_up"] = outcome_domain

        # Build extensions domain (pathology, comorbidities)
        if "extensions" in domain_fields:
            extensions_domain: Dict[str, Any] = {}
            for field_payload in domain_fields["extensions"]:
                field_name = field_payload.get("field_name", "unknown_field")
                field_summary = self._build_field_summary(
                    field_payload, extraction_result, patient_id
                )
                field_entry: Dict[str, Any] = {
                    "final_value": field_summary["final_value"],
                    "supporting_evidence": field_summary["supporting_evidence"],
                    "contradictory_evidence": field_summary["contradictory_evidence"],
                }
                if field_payload.get("units"):
                    field_entry["units"] = field_payload.get("units")
                extensions_domain[field_name] = field_entry
                source_docs = field_summary.get("source_documents") or []
                all_source_docs.extend(source_docs)
                # Only add confidence if it exists from LLM - don't use default
                if "confidence" in field_summary and field_summary["confidence"] is not None:
                    confidences.append(self._clamp_confidence(field_summary["confidence"]))
            if extensions_domain:
                mcode_extraction["extensions"] = extensions_domain

        # Ensure every column defined in the interview spec exists even if no evidence surfaced.
        mcode_extraction = self._apply_template(mcode_extraction, extraction_result, patient_id)

        # Use weighted harmonic mean for more conservative confidence aggregation
        # If no confidences collected, return None to indicate missing data from LLM
        avg_conf = self._aggregate_confidence_scores(confidences) if confidences else None
        if avg_conf is not None:
            avg_conf = self._clamp_confidence(avg_conf)
        # If avg_conf is still None, it means LLM didn't provide confidence scores
        # We keep it as None rather than defaulting to a fake value
        unique_docs = sorted(set(all_source_docs))
        fallback_docs = self._derive_source_docs_from_extraction(extraction_result)
        if not unique_docs and fallback_docs:
            unique_docs = fallback_docs

        # Include all documents (not just summary documents)
        # No filtering - keep all unique documents

        # Format source document names
        formatted_source_docs = [
            self._format_source_file_name(doc_id, patient_id) for doc_id in unique_docs
        ]
        formatted_source_docs = sorted(set(formatted_source_docs))

        doc_count = len(formatted_source_docs)
        if not doc_count and extraction_result:
            doc_count = extraction_result.total_documents_processed or len(
                extraction_result.document_results or []
            )

        # Build consolidation summary from patient_summary or generate from fields
        patient_summary = record.get("patient_summary", "")
        if not patient_summary:
            patient_summary = self._build_consolidation_summary_text(
                patient_id,
                mcode_extraction,
                doc_count,
            )

        metadata = {
            "total_documents": len(formatted_source_docs),
            "source_files": formatted_source_docs,
            "extraction_date": datetime.now().date().isoformat(),
            "consolidation_summary": patient_summary,
        }

        source_docs = formatted_source_docs

        return {
            "value": {
                "patient_id": patient_id,
                "mcode_extraction": mcode_extraction,
                "metadata": metadata,
            },
            "source_documents": source_docs,
            "confidence_score": avg_conf,
            "consolidation_reasoning": (
                f"mCODE v4.0.0 patient-level extraction with {doc_count} documents (Pydantic validated)"
            ),
        }

    def _build_primary_cancers_array(
        self,
        diagnosis_fields: List[Dict[str, Any]],
        extraction_result: Optional[ExtractionResult] = None,
        patient_id: Optional[str] = None,
        primary_cancers_guide: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """Build primary_cancers array structure from diagnosis fields.
        
        CRITICAL: This method now properly identifies MULTIPLE CANCERS per patient.
        Each unique body_site (cancer location) becomes a separate primary_cancer record.
        
        For example, a patient with:
        - Rectal Cancer (C20.9) diagnosed in 1987
        - Appendiceal Cancer (C18.1) diagnosed in 2005
        - Prostate Cancer (C61.9) diagnosed in 2015
        
        Will generate THREE separate primary_cancer entries, each with its own
        diagnosis_date, histology, staging info, etc.
        """
        if not diagnosis_fields:
            return []
        
        import re
        
        # Helper function to normalize LLM-extracted date values
        def normalize_date_value(date_str: str) -> str:
            """Normalize LLM-extracted date values to YYYY-MM-DD format.
            
            Fixes common LLM issues like:
            - "Diagnosed in1987" → "1987-01-01"
            - "Diagnosed in 1987" → "1987-01-01"
            - "Recently diagnosed in July 2015" → "2015-07-15"
            - "2015-07-15" → "2015-07-15" (unchanged)
            """
            if not date_str:
                return "Not Reported"
            
            # Already in YYYY-MM-DD format
            if re.match(r'^\d{4}-\d{2}-\d{2}$', date_str):
                return date_str
            
            # Fix missing space issue: "in1987" → "in 1987"
            date_str = re.sub(r'in(\d{4})', r'in \1', date_str)
            
            # Month name mappings
            month_names = {
                'january': '01', 'february': '02', 'march': '03', 'april': '04',
                'may': '05', 'june': '06', 'july': '07', 'august': '08',
                'september': '09', 'october': '10', 'november': '11', 'december': '12'
            }
            
            # Try "Month YYYY" format (e.g., "July 2015", "Recently diagnosed in July 2015")
            match = re.search(r'(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{4})', date_str.lower())
            if match:
                month = month_names.get(match.group(1), '01')
                year = match.group(2)
                return f"{year}-{month}-15"
            
            # Try "Diagnosed in YYYY" or just "in YYYY" pattern
            match = re.search(r'in\s+(\d{4})', date_str, re.IGNORECASE)
            if match:
                year = match.group(1)
                return f"{year}-01-01"
            
            # Try just YYYY format
            match = re.search(r'(\d{4})', date_str)
            if match:
                return f"{match.group(1)}-01-01"
            
            return date_str  # Return as-is if no pattern matched
        
        # Step 1: Collect ALL body_site and diagnosis_date values from resolve_records
        # This is the PRIMARY source for multi-cancer detection
        all_site_docs: List[Dict[str, Any]] = []
        all_date_docs: List[Dict[str, Any]] = []
        all_histology_docs: List[Dict[str, Any]] = []
        
        # Use cached resolve_records if available (primary source)
        if self._resolve_records:
            for record in self._resolve_records:
                field_name = record.get("field_name", "")
                raw_value = record.get("resolved_value") or record.get("raw_value") or ""
                doc_id = record.get("doc_id", "")
                doc_date = record.get("doc_date", "")
                
                if field_name == "ca_site" and raw_value and raw_value != "Not Reported":
                    all_site_docs.append({
                        "doc_id": doc_id,
                        "doc_date": doc_date,
                        "raw_value": raw_value,
                        "reasoning": record.get("consolidation_notes", ""),
                    })
                elif field_name == "naaccr_diagnosis_dt" and raw_value and raw_value != "Not Reported":
                    all_date_docs.append({
                        "doc_id": doc_id,
                        "doc_date": doc_date,  # Include doc_date for evidence dating
                        "raw_value": raw_value,
                    })
                elif field_name == "naaccr_histology_cd" and raw_value and raw_value != "Not Reported":
                    all_histology_docs.append({
                        "doc_id": doc_id,
                        "doc_date": doc_date,  # Include doc_date for evidence dating
                        "raw_value": raw_value,
                    })
        
        # Fallback: Also check diagnosis_fields if resolve_records didn't provide enough data
        if not all_site_docs:
            for field_payload in diagnosis_fields:
                if not field_payload or not isinstance(field_payload, dict):
                    continue
                field_name = field_payload.get("field_name", "unknown_field")
                raw_value = (
                    field_payload.get("raw_value") 
                    or field_payload.get("resolved_value") 
                    or field_payload.get("normalized_value")
                )
                
                if field_name == "body_site" and raw_value:
                    # Check supporting_docs first
                    supporting_docs = field_payload.get("supporting_docs") or []
                    if supporting_docs:
                        for doc in supporting_docs:
                            if isinstance(doc, dict):
                                all_site_docs.append({
                                    "doc_id": doc.get("doc_id", ""),
                                    "doc_date": doc.get("doc_date", ""),
                                    "raw_value": doc.get("raw_value") or doc.get("normalized_value") or raw_value,
                                    "reasoning": doc.get("reasoning_excerpt", ""),
                                })
                    else:
                        # Use field payload as single doc
                        all_site_docs.append({
                            "doc_id": field_payload.get("doc_id") or "",
                            "doc_date": field_payload.get("doc_date") or "",
                            "raw_value": raw_value,
                            "reasoning": field_payload.get("consolidation_notes") or "",
                        })
                        
                elif field_name == "diagnosis_date" and raw_value:
                    supporting_docs = field_payload.get("supporting_docs") or []
                    if supporting_docs:
                        for doc in supporting_docs:
                            if isinstance(doc, dict):
                                all_date_docs.append({
                                    "doc_id": doc.get("doc_id", ""),
                                    "raw_value": doc.get("raw_value") or doc.get("normalized_value") or raw_value,
                                })
                    else:
                        all_date_docs.append({
                            "doc_id": field_payload.get("doc_id") or "",
                            "raw_value": raw_value,
                        })
        
        # Step 2: Parse all collected site docs to identify unique cancers
        unique_cancer_sites: Dict[str, Dict[str, Any]] = {}  # site_code -> {docs, diagnosis_dates}
        
        for doc in all_site_docs:
            raw_value = doc.get("raw_value", "")
            if not raw_value:
                continue
            
            # Parse cancer site from the value (e.g., "Colon Cancer (C18.9)")
            # Look for ICD-O-3 site codes like C18.1, C20.9, C61.9
            site_match = re.search(r'\(?(C\d+\.?\d*)\)?', raw_value)
            if site_match:
                site_code = site_match.group(1)
                # Normalize site code (e.g., C61 -> C61.9)
                if '.' not in site_code:
                    site_code = f"{site_code}.9"
                
                if site_code not in unique_cancer_sites:
                    unique_cancer_sites[site_code] = {
                        "site_code": site_code,
                        "site_name": raw_value,
                        "docs": [],
                        "diagnosis_dates": [],
                        "histology_values": [],  # NEW: Track histology per cancer
                    }
                unique_cancer_sites[site_code]["docs"].append(doc)
        
        # Step 3: Associate diagnosis_dates with each cancer site based on doc_id matching
        if all_date_docs and unique_cancer_sites:
            for date_doc in all_date_docs:
                doc_id = date_doc.get("doc_id", "")
                date_value = date_doc.get("raw_value", "")
                doc_date_value = date_doc.get("doc_date", "")  # Get doc_date for evidence
                
                # Find which cancer site this doc belongs to
                for site_code, site_info in unique_cancer_sites.items():
                    for site_doc in site_info["docs"]:
                        if site_doc.get("doc_id") == doc_id:
                            site_info["diagnosis_dates"].append({
                                "date": date_value,
                                "doc_id": doc_id,
                                "doc_date": doc_date_value,  # Include doc_date for evidence
                            })
                            break
        
        # Step 3b: Associate histology_morphology with each cancer site based on doc_id matching
        if all_histology_docs and unique_cancer_sites:
            for hist_doc in all_histology_docs:
                doc_id = hist_doc.get("doc_id", "")
                hist_value = hist_doc.get("raw_value", "")
                
                # Find which cancer site this doc belongs to
                for site_code, site_info in unique_cancer_sites.items():
                    for site_doc in site_info["docs"]:
                        if site_doc.get("doc_id") == doc_id:
                            site_info["histology_values"].append({
                                "value": hist_value,
                                "doc_id": doc_id,
                                "doc_date": hist_doc.get("doc_date", ""),  # Include doc_date for evidence
                            })
                            break
        # Step 4: Build primary_cancer entries for each unique site
        cancers: List[Dict[str, Any]] = []
        cancer_counter = 1
        
        if unique_cancer_sites:
            for site_code, site_info in unique_cancer_sites.items():
                cancer_entry = {
                    "cancer_id": f"cancer_{cancer_counter}",
                    "site_code": site_code,
                }
                cancer_counter += 1
                
                # Build body_site field for this cancer
                site_evidence = []
                for doc in site_info["docs"]:
                    site_evidence.append({
                        "source_file": self._format_source_file_name(doc["doc_id"], patient_id),
                        "snippet": doc["raw_value"],
                        "date": doc["doc_date"],
                        "confidence": 0.95,
                    })
                
                cancer_entry["body_site"] = {
                    "final_value": site_info["site_name"],
                    "supporting_evidence": site_evidence,
                    "contradictory_evidence": [],
                    "supporting_evidence_count": len(site_evidence),
                    "contradictory_evidence_count": 0,
                }
                
                # Build diagnosis_date for this cancer
                guided_date = None
                if primary_cancers_guide:
                    # Find matching guide for this site code
                    for guide in primary_cancers_guide:
                        guide_site = guide.get("site", "")
                        if site_code in guide_site or site_info["site_name"] in guide_site:
                            guided_date = guide.get("diagnosis_date")
                            break
                
                if guided_date and guided_date != "Not Reported":
                    # Use LLM-guided date logic
                    # Find evidence closest to this date
                    matching_dates = []
                    
                    # Normalize guided date for comparison
                    def extract_sortable_date(date_str: str) -> str:
                        """Extract a sortable date string (YYYY-MM-DD) from date text."""
                        import re
                        if not date_str:
                            return "9999-99-99"
                        match = re.search(r'(\d{4})', date_str)
                        return match.group(1) if match else "9999"

                    guide_year = extract_sortable_date(guided_date)
                    
                    for d in site_info["diagnosis_dates"]:
                        doc_year = extract_sortable_date(d.get("date", ""))
                        # Allow fuzzy match (same year)
                        if doc_year == guide_year or guided_date in d.get("date", ""):
                            matching_dates.append(d)
                    
                    if matching_dates:
                        cancer_entry["diagnosis_date"] = {
                            "final_value": guided_date,
                            "supporting_evidence": [{
                                "source_file": self._format_source_file_name(d["doc_id"], patient_id),
                                "snippet": d["date"],  # The extracted diagnosis date value
                                "date": d.get("doc_date") or d["date"],  # Use doc_date if available
                                "confidence": 0.95,
                            } for d in matching_dates],
                            "contradictory_evidence": [],
                            "supporting_evidence_count": len(matching_dates),
                            "contradictory_evidence_count": 0,
                        }
                    else:
                        # Fallback if no evidence matches guide
                         cancer_entry["diagnosis_date"] = {
                            "final_value": guided_date,
                            "supporting_evidence": [], # LLM hallucinated date or evidence lost
                            "contradictory_evidence": [],
                            "supporting_evidence_count": 0,
                            "contradictory_evidence_count": 0,
                        }
                        
                elif site_info["diagnosis_dates"]:
                    # Default logic: Use EARLIEST date
                    # Helper to extract sortable date from various formats
                    def extract_sortable_date(date_str: str) -> str:
                        """Extract a sortable date string (YYYY-MM-DD) from date text."""
                        import re
                        if not date_str:
                            return "9999-99-99"
                        
                        # Try YYYY-MM-DD format first
                        match = re.search(r'(\d{4}-\d{2}-\d{2})', date_str)
                        if match:
                            return match.group(1)
                        
                        # Try "Month DD, YYYY" format (e.g., "January 10, 2019")
                        month_names = {
                            'january': '01', 'february': '02', 'march': '03', 'april': '04',
                            'may': '05', 'june': '06', 'july': '07', 'august': '08',
                            'september': '09', 'october': '10', 'november': '11', 'december': '12'
                        }
                        match = re.search(r'(\w+)\s+(\d{1,2}),?\s+(\d{4})', date_str, re.IGNORECASE)
                        if match:
                            month = month_names.get(match.group(1).lower(), '99')
                            day = match.group(2).zfill(2)
                            year = match.group(3)
                            return f"{year}-{month}-{day}"
                        
                        # Try "Month YYYY" format (e.g., "April 2025")
                        match = re.search(r'(\w+)\s+(\d{4})', date_str, re.IGNORECASE)
                        if match:
                            month = month_names.get(match.group(1).lower(), '99')
                            year = match.group(2)
                            return f"{year}-{month}-01"
                        
                        # Try just YYYY format
                        match = re.search(r'(\d{4})', date_str)
                        if match:
                            return f"{match.group(1)}-01-01"
                        
                        return "9999-99-99"
                    
                    # Sort by extracted date and take earliest
                    dates_sorted = sorted(
                        site_info["diagnosis_dates"],
                        key=lambda x: extract_sortable_date(x.get("date", ""))
                    )
                    earliest = dates_sorted[0] if dates_sorted else None
                    if earliest:
                        # Normalize the date value to YYYY-MM-DD format
                        normalized_date = normalize_date_value(earliest["date"])
                        cancer_entry["diagnosis_date"] = {
                            "final_value": normalized_date,
                            "supporting_evidence": [{
                                "source_file": self._format_source_file_name(d["doc_id"], patient_id),
                                "snippet": d["date"],  # Keep original LLM-extracted snippet for reference
                                "date": d.get("doc_date") or normalize_date_value(d["date"]),  # Use doc_date if available
                                "confidence": 0.95,
                            } for d in site_info["diagnosis_dates"]],
                            "contradictory_evidence": [],
                            "supporting_evidence_count": len(site_info["diagnosis_dates"]),
                            "contradictory_evidence_count": 0,
                        }
                
                # Build histology_morphology for this cancer (use most common value)
                if site_info.get("histology_values"):
                    # Count occurrences and pick most common
                    from collections import Counter
                    value_counts = Counter(h["value"] for h in site_info["histology_values"])
                    most_common = value_counts.most_common(1)[0][0] if value_counts else "Not Reported"
                    
                    cancer_entry["histology_morphology"] = {
                        "final_value": most_common,
                        "supporting_evidence": [{
                            "source_file": self._format_source_file_name(h["doc_id"], patient_id),
                            "snippet": h["value"],
                            "date": h.get("doc_date") or "Not Reported",  # Use doc_date from histology record
                            "confidence": 0.9,
                        } for h in site_info["histology_values"]],
                        "contradictory_evidence": [],
                        "supporting_evidence_count": len(site_info["histology_values"]),
                        "contradictory_evidence_count": 0,
                    }
                
                cancers.append(cancer_entry)
        
        # Fallback: If no unique sites found, use simplified logic
        if not cancers:
            # Fallback logic - single cancer record from first available body_site
            cancer_entry = {"cancer_id": "cancer_1"}
            for field_payload in diagnosis_fields:
                if not field_payload or not isinstance(field_payload, dict):
                    continue
                field_name = field_payload.get("field_name", "unknown_field")
                final_value = (
                    field_payload.get("resolved_value") 
                    or field_payload.get("normalized_value") 
                    or field_payload.get("raw_value")
                    or "Not Reported"
                )
                # Build simple evidence from field_payload
                supporting_docs = field_payload.get("supporting_docs") or []
                supporting_evidence = []
                for doc in supporting_docs:
                    if isinstance(doc, dict):
                        supporting_evidence.append({
                            "source_file": self._format_source_file_name(doc.get("doc_id", ""), patient_id),
                            "snippet": doc.get("raw_value") or final_value,
                            "date": doc.get("doc_date", "Not Reported"),
                            "confidence": 0.9,
                        })
                if not supporting_evidence:
                    # Create minimal entry from field_payload itself
                    supporting_evidence = [{
                        "source_file": "unknown",
                        "snippet": final_value,
                        "date": "Not Reported",
                        "confidence": 0.9,
                    }]
                
                field_entry: Dict[str, Any] = {
                    "final_value": final_value,
                    "supporting_evidence": supporting_evidence,
                    "contradictory_evidence": [],
                    "supporting_evidence_count": len(supporting_evidence),
                    "contradictory_evidence_count": 0,
                }
                if field_payload.get("normalized_value"):
                    field_entry["normalized_value"] = field_payload.get("normalized_value")
                if field_payload.get("vocabulary_code"):
                    field_entry["normalized_code"] = field_payload.get("vocabulary_code")
                cancer_entry[field_name] = field_entry
            cancers.append(cancer_entry)
        
        return cancers

    def _build_cancer_stage_domain(
        self,
        staging_fields: List[Dict[str, Any]],
        extraction_result: Optional[ExtractionResult] = None,
        patient_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Build cancer_stage domain with timeline and final_stage structure."""
        if not staging_fields:
            # Return empty structure - template will fill in required fields with empty evidence arrays
            # Do NOT create default evidence entries - empty arrays are valid
            return {
                "timeline": [],
                "final_stage": {},
                "consolidation_notes": "",
            }
        timeline: List[Dict[str, Any]] = []
        final_stage: Dict[str, Any] = {}

        # Extract dates from supporting_docs to build timeline
        # Include all documents (not just summary documents)
        dates_seen = set()
        for field_payload in staging_fields:
            if not field_payload or not isinstance(field_payload, dict):
                continue
            supporting_docs = field_payload.get("supporting_docs") or []
            for doc in supporting_docs:
                # Ensure doc is a dict, not a string (Qwen tool call responses)
                if not isinstance(doc, dict):
                    # Try to parse if it's a string
                    if isinstance(doc, str):
                        try:
                            import json

                            parsed = json.loads(doc)
                            if isinstance(parsed, dict):
                                doc = parsed
                            else:
                                continue
                        except (json.JSONDecodeError, Exception):
                            continue
                    else:
                        continue

                # Include all documents (no filtering)
                doc_date = doc.get("doc_date")
                if doc_date and doc_date not in dates_seen and doc_date != "Not Reported":
                    dates_seen.add(doc_date)
                    doc_id = doc.get("doc_id", "unknown_doc")
                    timeline.append(
                        {
                            "date": doc_date,
                            "tnm_t": "Not Reported",
                            "tnm_n": "Not Reported",
                            "tnm_m": "Not Reported",
                            "stage_group": "Not Reported",
                            "staging_basis": "Not Reported",
                            "source_file": self._format_source_file_name(doc_id, patient_id),
                            "evidence_snippet": doc.get("reasoning_excerpt", ""),
                        }
                    )

        # Build final_stage structure
        # Field names are already normalized to mCODE format by _normalize_all_field_names()
        stage_field_mapping = {
            "tnm_t_pathologic": "tnm_t_pathologic",
            "tnm_n_pathologic": "tnm_n_pathologic",
            "tnm_m_pathologic": "tnm_m_pathologic",
            "tnm_t_clinical": "tnm_t_clinical",
            "tnm_n_clinical": "tnm_n_clinical",
            "tnm_m_clinical": "tnm_m_clinical",
            "stage_group": "stage_group",
            "staging_system": "staging_system",
        }

        for field_payload in staging_fields:
            field_name = field_payload.get("field_name", "")
            if field_name in stage_field_mapping:
                field_summary = self._build_field_summary(
                    field_payload, extraction_result, patient_id
                )
                final_stage[stage_field_mapping[field_name]] = {
                    "final_value": field_summary["final_value"],
                    "supporting_evidence": field_summary["supporting_evidence"],
                    "contradictory_evidence": field_summary["contradictory_evidence"],
                }

        # Do NOT create default entries if final_stage is empty
        # Template will fill in required fields with empty evidence arrays
        # Empty final_stage is valid when no staging data is available

        return {
            "timeline": sorted(timeline, key=lambda x: x.get("date", "")) if timeline else [],
            "final_stage": final_stage,
            "consolidation_notes": "",
        }

    def _build_disease_status_domain(
        self,
        disease_fields: List[Dict[str, Any]],
        extraction_result: Optional[ExtractionResult] = None,
        patient_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Build disease_status domain with timeline and final_status structure."""
        if not disease_fields:
            # Return empty structure - template will fill in required fields with empty evidence arrays
            # Do NOT create default evidence entries - empty arrays are valid
            return {
                "timeline": [],
                "final_status": {},
            }
        timeline: List[Dict[str, Any]] = []
        final_status: Dict[str, Any] = {}

        # Extract dates from supporting_docs to build timeline
        # Include all documents (not just summary documents)
        dates_seen = set()
        for field_payload in disease_fields:
            if not field_payload or not isinstance(field_payload, dict):
                continue
            supporting_docs = field_payload.get("supporting_docs") or []
            for doc in supporting_docs:
                # Ensure doc is a dict, not a string (Qwen tool call responses)
                if not isinstance(doc, dict):
                    # Try to parse if it's a string
                    if isinstance(doc, str):
                        try:
                            import json

                            parsed = json.loads(doc)
                            if isinstance(parsed, dict):
                                doc = parsed
                            else:
                                continue
                        except (json.JSONDecodeError, Exception):
                            continue
                    else:
                        continue

                # Include all documents (no filtering)
                doc_date = doc.get("doc_date")
                if doc_date and doc_date not in dates_seen and doc_date != "Not Reported":
                    dates_seen.add(doc_date)
                    # Try to extract status from field
                    status_value = (
                        field_payload.get("resolved_value")
                        or field_payload.get("normalized_value")
                        or "Status update"
                    )
                    doc_id = doc.get("doc_id", "unknown_doc")
                    timeline.append(
                        {
                            "date": doc_date,
                            "status": status_value,
                            "source_file": self._format_source_file_name(doc_id, patient_id),
                            "evidence_snippet": doc.get("reasoning_excerpt", ""),
                        }
                    )

        # Build final_status structure
        status_field_mapping = {
            "disease_status": "disease_status",
            "treatment_response": "treatment_response",
            "recurrence_indicator": "recurrence_indicator",
            "recurrence_date": "recurrence_date",
        }

        for field_payload in disease_fields:
            field_name = field_payload.get("field_name", "")
            if field_name in status_field_mapping:
                field_summary = self._build_field_summary(
                    field_payload, extraction_result, patient_id
                )
                final_status[status_field_mapping[field_name]] = {
                    "final_value": field_summary["final_value"],
                    "supporting_evidence": field_summary["supporting_evidence"],
                    "contradictory_evidence": field_summary["contradictory_evidence"],
                }

        # Do NOT create default entries if final_status is empty
        # Template will fill in required fields with empty evidence arrays
        # Empty final_status is valid when no disease status data is available

        return {
            "timeline": sorted(timeline, key=lambda x: x.get("date", "")) if timeline else [],
            "final_status": final_status,
        }

    def _build_patient_records_from_resolve(
        self, resolve_records: Optional[List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """Fallback when DocETL reduce output is empty by synthesizing patient records."""
        if not resolve_records:
            return []
        grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for record in resolve_records:
            if not record or not isinstance(record, dict):
                continue
            patient_id = record.get("patient_id") or "unknown_patient"
            grouped[patient_id].append(record)

        synthetic_records: List[Dict[str, Any]] = []
        for patient_id, items in grouped.items():
            consolidated_fields: List[Dict[str, Any]] = []
            doc_ids = set()
            for entry in items:
                consolidated_fields.append(self._build_consolidated_field_from_resolve(entry))
                if entry.get("doc_id"):
                    doc_ids.add(entry["doc_id"])

            patient_summary = (
                f"Synthesized {len(consolidated_fields)} registry fields across "
                f"{len(doc_ids) or 'N/A'} documents for patient {patient_id}."
            )
            synthetic_records.append(
                {
                    "patient_id": patient_id,
                    "consolidated_fields": consolidated_fields,
                    "patient_summary": patient_summary,
                }
            )
        return synthetic_records

    def _build_consolidated_field_from_resolve(
        self,
        record: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Normalize a resolve record into the structure ReduceOp would emit."""
        extractions_raw = record.get("extractions")
        extraction = {}

        if isinstance(extractions_raw, dict):
            extraction = extractions_raw
        elif isinstance(extractions_raw, str):
            # Try to parse JSON string
            try:
                parsed = json.loads(extractions_raw)
                if isinstance(parsed, dict):
                    extraction = parsed
                elif isinstance(parsed, list) and len(parsed) > 0 and isinstance(parsed[0], dict):
                    # If it's a list, take the first item
                    extraction = parsed[0]
            except (json.JSONDecodeError, Exception):
                extraction = {}
        elif isinstance(extractions_raw, list) and len(extractions_raw) > 0:
            # If it's a list, take the first dict item
            if isinstance(extractions_raw[0], dict):
                extraction = extractions_raw[0]
            elif isinstance(extractions_raw[0], str):
                # Try to parse first item as JSON
                try:
                    parsed = json.loads(extractions_raw[0])
                    if isinstance(parsed, dict):
                        extraction = parsed
                except (json.JSONDecodeError, Exception):
                    extraction = {}

        def first_valid(*values):
            for value in values:
                if value not in (None, "", [], {}):
                    return value
            return None

        field_name = first_valid(
            extraction.get("field_name"), record.get("field_name"), "unknown_field"
        )
        category = first_valid(extraction.get("category"), record.get("category"), "general")
        data_type = first_valid(extraction.get("data_type"), record.get("data_type"), "string")
        normalized_value = first_valid(
            extraction.get("normalized_value"),
            extraction.get("raw_value"),
            record.get("normalized_value"),
            record.get("resolved_value"),
        )
        resolved_value = first_valid(
            extraction.get("normalized_value"),
            extraction.get("raw_value"),
            record.get("resolved_value"),
            record.get("normalized_value"),
        )
        units = first_valid(extraction.get("units"), record.get("units"), "")
        vocabulary_code = first_valid(
            extraction.get("vocabulary_code"),
            record.get("vocabulary_code"),
            "",
        )
        # Get confidence from extraction or record - use actual LLM value, not default
        conf_val = first_valid(
            extraction.get("confidence_score"),
            record.get("confidence_score"),
        )
        confidence = self._clamp_confidence(conf_val) if conf_val is not None else None
        supporting_docs = self._build_supporting_docs_from_resolve(record, extraction)
        consolidation_notes = first_valid(
            record.get("consolidation_notes"),
            extraction.get("explanation"),
            "",
        )

        # NOTE: Field names will be normalized by _normalize_all_field_names() after
        # all records are built, so we keep the original field name here.
        consolidated = {
            "field_name": field_name,  # Will be normalized by _normalize_all_field_names()
            "category": category,
            "data_type": data_type,
            "normalized_value": normalized_value,
            "resolved_value": resolved_value,
            "raw_value": extraction.get("raw_value") or record.get("raw_value"),
            "units": units,
            "vocabulary_code": vocabulary_code,
            "confidence_score": confidence,
            "supporting_docs": supporting_docs,
            "consolidation_notes": consolidation_notes,
        }
        consolidated["_original_field_name"] = field_name
        return consolidated

    def _build_supporting_docs_from_resolve(
        self,
        record: Dict[str, Any],
        extraction: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Create supporting document payloads even when DocETL returns scalars."""
        # Ensure extraction is a dict
        if not isinstance(extraction, dict):
            extraction = {}

        docs = record.get("supporting_docs")

        def base_doc() -> Dict[str, Any]:
            # Get confidence from extraction - use actual LLM value, not default
            conf_val = extraction.get("confidence_score")
            return {
                "doc_id": record.get("doc_id", "unknown_doc"),
                "patient_id": record.get("patient_id", "unknown_patient"),
                "field_name": extraction.get("field_name")
                or record.get("field_name")
                or "unknown_field",
                "raw_value": extraction.get("raw_value"),
                "normalized_value": extraction.get("normalized_value"),
                "reasoning_excerpt": extraction.get("reasoning_excerpt") or "",
                "explanation": extraction.get("explanation") or "",
                "doc_type": record.get("doc_type", "unknown"),
                "doc_date": record.get("doc_date", "Not Reported"),
                "confidence_score": (
                    self._clamp_confidence(conf_val) if conf_val is not None else None
                ),
            }

        if isinstance(docs, list):
            formatted: List[Dict[str, Any]] = []
            for doc in docs:
                if isinstance(doc, dict):
                    # Get confidence from doc or extraction - use actual LLM value
                    conf_val = doc.get("confidence_score")
                    if conf_val is None:
                        conf_val = extraction.get("confidence_score")

                    formatted.append(
                        {
                            "doc_id": doc.get("doc_id", record.get("doc_id", "unknown_doc")),
                            "patient_id": doc.get("patient_id", record.get("patient_id")),
                            "field_name": doc.get("field_name", extraction.get("field_name")),
                            "raw_value": doc.get("raw_value", extraction.get("raw_value")),
                            "normalized_value": doc.get(
                                "normalized_value", extraction.get("normalized_value")
                            ),
                            "reasoning_excerpt": doc.get(
                                "reasoning_excerpt", extraction.get("reasoning_excerpt", "")
                            ),
                            "explanation": doc.get(
                                "explanation", extraction.get("explanation", "")
                            ),
                            "doc_type": doc.get("doc_type", record.get("doc_type", "unknown")),
                            "doc_date": doc.get("doc_date", record.get("doc_date", "Not Reported")),
                            "confidence_score": (
                                self._clamp_confidence(conf_val) if conf_val is not None else None
                            ),
                        }
                    )
                else:
                    new_doc = base_doc()
                    new_doc["reasoning_excerpt"] = str(doc)
                    formatted.append(new_doc)
            return formatted

        if isinstance(docs, str) and docs.strip():
            new_doc = base_doc()
            new_doc["reasoning_excerpt"] = docs
            return [new_doc]

        if extraction:
            return [base_doc()]
        return []

    def _build_field_summary(
        self,
        field_payload: Dict[str, Any],
        extraction_result: Optional[ExtractionResult] = None,
        patient_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Build summary payload for a single resolved field."""
        # Ensure field_payload is a dict
        if not isinstance(field_payload, dict):
            # Try to parse if it's a JSON string
            if isinstance(field_payload, str):
                try:
                    parsed = json.loads(field_payload)
                    if isinstance(parsed, dict):
                        field_payload = parsed
                    else:
                        field_payload = {}
                except (json.JSONDecodeError, Exception):
                    field_payload = {}
            else:
                field_payload = {}

        # Extract final_value - prioritize resolved_value, then normalized_value, then raw_value
        final_value = (
            field_payload.get("resolved_value")
            or field_payload.get("normalized_value")
            or field_payload.get("raw_value")
            or "Not Reported"
        )

        # Clean up final_value - remove empty strings and None values
        if final_value in (None, "", "null", "None"):
            final_value = "Not Reported"
        elif isinstance(final_value, str):
            final_value = final_value.strip()
            if not final_value or final_value.lower() in ("null", "none", "n/a", "na"):
                final_value = "Not Reported"

        supporting_docs = field_payload.get("supporting_docs") or []
        evidence = []
        sanitized_docs: List[Dict[str, Any]] = []
        
        # Get doc_date from field_payload as fallback (may be at record level)
        fallback_doc_date = field_payload.get("doc_date") or field_payload.get("date") or "Not Reported"

        # If supporting_docs is empty, try to build evidence from extraction_result
        if not supporting_docs and extraction_result:
            evidence = self._build_evidence_from_extraction(
                field_payload, extraction_result, patient_id
            )
            # Extract source documents from evidence
            for ev in evidence:
                if ev.get("source_file") and ev["source_file"] != "unknown_doc":
                    sanitized_docs.append({"doc_id": ev["source_file"]})
        else:
            # Use existing supporting_docs
            # Include all documents (not just summary documents)
            for doc in supporting_docs:
                if isinstance(doc, dict):
                    sanitized_docs.append(doc)
                    # Extract reasoning_excerpt for snippet - prioritize reasoning_excerpt
                    reasoning_excerpt = (
                        doc.get("reasoning_excerpt", "") or doc.get("explanation", "") or ""
                    )
                    # Get explanation from doc
                    explanation = (
                        doc.get("explanation", "") or doc.get("consolidation_notes", "") or ""
                    )
                    # If no explanation but we have reasoning_excerpt, use it
                    if not explanation and reasoning_excerpt:
                        explanation = reasoning_excerpt
                    # If still no explanation, create a default one
                    if not explanation:
                        field_name = field_payload.get("field_name", "unknown_field")
                        if final_value and final_value != "Not Reported":
                            explanation = (
                                f"Value '{final_value}' extracted for {field_name} from document."
                            )
                        else:
                            explanation = f"Information for {field_name} extracted from document."

                    # Get doc_date - try doc level, then field_payload fallback, then extraction_result lookup
                    doc_date = doc.get("doc_date") or doc.get("date")
                    if doc_date in (None, "", "null", "None", "Not Reported"):
                        doc_date = fallback_doc_date
                    
                    # If still not found, try extraction_result lookup by doc_id
                    if doc_date in (None, "", "null", "None", "Not Reported") and extraction_result:
                        doc_id_for_lookup = doc.get("doc_id") or doc.get("source_file") or ""
                        if doc_id_for_lookup:
                            extracted_date = self._lookup_doc_date_from_extraction(
                                doc_id_for_lookup, extraction_result
                            )
                            if extracted_date:
                                doc_date = extracted_date

                    # Format source_file properly
                    doc_id = doc.get("doc_id") or doc.get("source_file") or "unknown_doc"
                    source_file = self._format_source_file_name(doc_id, patient_id)

                    # Get confidence from LLM - use actual value, not default
                    confidence_val = doc.get("confidence_score")
                    if confidence_val is None:
                        # If no confidence from LLM, mark as None (will be filtered out)
                        confidence_val = None

                    evidence.append(
                        {
                            "snippet": reasoning_excerpt,
                            "explanation": explanation,
                            "date": doc_date,
                            "source_file": source_file,
                            "confidence": (
                                self._clamp_confidence(confidence_val)
                                if confidence_val is not None
                                else None
                            ),
                            "span": doc.get("span"),
                            "page": doc.get("page"),
                        }
                    )
                elif doc:
                    sanitized_docs.append({"reasoning_excerpt": str(doc)})
                    # Try to get source file from extraction_result
                    source_file = "unknown_doc"
                    if extraction_result and extraction_result.document_results:
                        first_doc = extraction_result.document_results[0]
                        if hasattr(first_doc, "doc_id"):
                            source_file = self._format_source_file_name(
                                first_doc.doc_id, patient_id
                            )
                        elif isinstance(first_doc, dict):
                            source_file = self._format_source_file_name(
                                first_doc.get("doc_id", "unknown_doc"), patient_id
                            )
                    source_file = self._format_source_file_name(source_file, patient_id)

                    # No confidence available for string docs - use None instead of 0.0
                    evidence.append(
                        {
                            "snippet": str(doc),
                            "explanation": f"Field extracted from document: {str(doc)}",
                            "date": "Not Reported",
                            "source_file": source_file,
                            "confidence": None,
                            "span": None,
                            "page": None,
                        }
                    )

        # Always supplement with extraction_result evidence to capture low-confidence traces
        if extraction_result:
            extraction_evidence = self._build_evidence_from_extraction(
                field_payload, extraction_result, patient_id
            )
            if extraction_evidence:
                existing_signatures = {
                    (
                        ev.get("source_file"),
                        ev.get("snippet"),
                        ev.get("explanation"),
                    )
                    for ev in evidence
                }
                for ev in extraction_evidence:
                    signature = (
                        ev.get("source_file"),
                        ev.get("snippet"),
                        ev.get("explanation"),
                    )
                    if signature in existing_signatures:
                        continue
                    evidence.append(ev)
                    existing_signatures.add(signature)
                    source_file = ev.get("source_file")
                    if source_file and source_file != "unknown_doc":
                        sanitized_docs.append({"doc_id": source_file})

        # Do NOT create default supporting_evidence entries
        # Only use evidence that comes from DocETL (supporting_docs) or extraction_result
        # Empty array is valid when there's no supporting evidence available

        supporting_docs = sanitized_docs
        # Only use contradictory_evidence if it actually exists from DocETL resolve/reduce operations
        # Do NOT create default entries - empty array is valid when there's no contradiction
        contradictory_entries = field_payload.get("contradictory_evidence") or []

        if not contradictory_entries and extraction_result and extraction_result.document_results:
            field_name = field_payload.get("_original_field_name") or field_payload.get(
                "field_name", ""
            )
            # Use semantic normalization for comparison to avoid false contradictions
            # This extracts core values like "cT2" from both "cT2" and "Clinical staging revealed cT2"
            raw_baseline = field_payload.get("raw_value") 
            if not raw_baseline:
                raw_baseline = field_payload.get("normalized_value") or field_payload.get("resolved_value")
            
            # Normalize baseline for semantic comparison
            baseline = self._normalize_value_for_comparison(raw_baseline, field_name)
            
            contradictions: List[Dict[str, Any]] = []
            seen_contradictions = set()

            for doc_result in extraction_result.document_results:
                if isinstance(doc_result, dict):
                    extracted_fields = doc_result.get("extracted_fields", [])
                    doc_id = doc_result.get("doc_id", "")
                    doc_date = (
                        doc_result.get("doc_date") or doc_result.get("date") or fallback_doc_date
                    )
                else:
                    extracted_fields = getattr(doc_result, "extracted_fields", [])
                    doc_id = getattr(doc_result, "doc_id", "")
                    doc_date = (
                        getattr(doc_result, "doc_date", None)
                        or getattr(doc_result, "date", None)
                        or fallback_doc_date
                    )

                for extracted_field in extracted_fields:
                    if isinstance(extracted_field, dict):
                        extracted_field_name = extracted_field.get("field_name", "")
                        # Get resolved_value separately for snippet (descriptive text)
                        resolved_val = extracted_field.get("resolved_value") or ""
                        raw_val = extracted_field.get("raw_value") or ""
                        extracted_value = (
                            resolved_val
                            or extracted_field.get("normalized_value")
                            or raw_val
                        )
                        reasoning_excerpt = extracted_field.get("reasoning_excerpt", "")
                        field_evidence = extracted_field.get("field_evidence", {})
                        explanation = (
                            field_evidence.get("explanation", "")
                            if isinstance(field_evidence, dict)
                            else ""
                        )
                        # Use actual confidence from LLM, None if not provided
                        confidence_score = extracted_field.get("confidence_score")
                    else:
                        extracted_field_name = getattr(extracted_field, "field_name", "")
                        # Get resolved_value separately for snippet (descriptive text)
                        resolved_val = getattr(extracted_field, "resolved_value", None) or ""
                        raw_val = getattr(extracted_field, "raw_value", None) or ""
                        extracted_value = (
                            resolved_val
                            or getattr(extracted_field, "normalized_value", None)
                            or raw_val
                        )
                        reasoning_excerpt = getattr(extracted_field, "reasoning_excerpt", "")
                        field_evidence = getattr(extracted_field, "field_evidence", None)
                        explanation = (
                            getattr(field_evidence, "explanation", "") if field_evidence else ""
                        )
                        # Use actual confidence from LLM, None if not provided
                        confidence_score = getattr(extracted_field, "confidence_score", None)

                    if extracted_field_name != field_name:
                        continue

                    if not extracted_value:
                        continue
                    extracted_value_str = str(extracted_value).strip()
                    if not extracted_value_str or extracted_value_str.lower() == "not reported":
                        continue

                    # Use SEMANTIC normalization for comparison
                    # This compares core values like "cT2" regardless of whether it came from 
                    # raw "cT2" or resolved "Clinical staging revealed cT2"
                    extracted_normalized = self._normalize_value_for_comparison(extracted_value_str, field_name)
                    
                    # Also try normalizing the raw value in case resolved contains different info
                    if raw_val:
                        raw_normalized = self._normalize_value_for_comparison(raw_val, field_name)
                        # Use raw_normalized if it's more specific (non-empty and different)
                        if raw_normalized and raw_normalized != extracted_normalized:
                            extracted_normalized = raw_normalized
                    
                    # If baseline is missing, we can't determine contradiction, so skip
                    if not baseline:
                        continue
                    
                    # Semantic comparison: if normalized values match, not a contradiction
                    if extracted_normalized == baseline:
                        continue

                    source_file = self._format_source_file_name(doc_id, patient_id)
                    signature = (source_file, extracted_normalized)
                    if signature in seen_contradictions:
                        continue
                    seen_contradictions.add(signature)

                    if not explanation:
                        if resolved_value and resolved_value != "Not Reported":
                            explanation = (
                                f"Document reports '{extracted_value_str}' which conflicts with "
                                f"consolidated value '{resolved_value}'."
                            )
                        else:
                            explanation = (
                                f"Document reports '{extracted_value_str}' for {field_name}."
                            )

                    # Use actual confidence from LLM, not default
                    conf_val = confidence_score if confidence_score is not None else None
                    
                    # Better snippet selection:
                    # 1. Reasoning Excerpt (best context)
                    # 2. Resolved Value (descriptive)
                    # 3. Extracted Value (might be raw)
                    snippet_text = reasoning_excerpt
                    if not snippet_text:
                        if resolved_val and len(str(resolved_val)) > len(str(raw_val)):
                             snippet_text = str(resolved_val)
                        else:
                             snippet_text = str(extracted_value)

                    contradictions.append(
                        {
                            "snippet": snippet_text,
                            "explanation": explanation,
                            "date": doc_date,
                            "source_file": source_file,
                            "confidence": (
                                self._clamp_confidence(conf_val) if conf_val is not None else None
                            ),
                            "span": None,
                            "page": None,
                        }
                    )

            if contradictions:
                contradictory_entries = contradictions

        source_documents = [
            self._format_source_file_name(doc.get("doc_id", "unknown_doc"), patient_id)
            for doc in supporting_docs
            if isinstance(doc, dict)
        ]
        if not source_documents and extraction_result:
            # Try to get source documents from extraction_result (includes all documents)
            source_documents = self._derive_source_docs_from_extraction(extraction_result)

        # Get confidence from field_payload - use actual LLM value, not default
        conf_score = field_payload.get("confidence_score")
        if conf_score is None:
            # If no confidence in field_payload, try to extract from supporting_docs
            if supporting_docs:
                conf_values = []
                for doc in supporting_docs:
                    if isinstance(doc, dict) and "confidence_score" in doc:
                        doc_conf = doc.get("confidence_score")
                        if doc_conf is not None:
                            conf_values.append(doc_conf)
                if conf_values:
                    # Use average of document confidences
                    conf_score = sum(conf_values) / len(conf_values)

        return {
            "final_value": final_value,
            "supporting_evidence": evidence,
            "contradictory_evidence": contradictory_entries,
            "source_documents": source_documents,
            "confidence": self._clamp_confidence(conf_score) if conf_score is not None else None,
        }

    def _create_default_evidence_entries(
        self,
        field_name: str,
        extraction_result: Optional[ExtractionResult] = None,
        patient_id: Optional[str] = None,
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Return empty evidence arrays - do NOT create default entries.

        This function is kept for backward compatibility but now always returns empty arrays.
        Empty arrays are valid and indicate no evidence is available.
        Only real evidence from DocETL operations should be included.
        """
        # Always return empty arrays - no default entries should be created
        # Empty arrays are valid when there's no evidence available
        return [], []

    def _canonicalize_field_payload(self, field_payload: Dict[str, Any]) -> None:
        """Rename upstream field identifiers so they align with mCODE columns."""
        if "_original_field_name" not in field_payload:
            field_payload["_original_field_name"] = field_payload.get("field_name")
        original_name = field_payload.get("_original_field_name") or field_payload.get("field_name")
        canonical_name = FIELD_NAME_ALIASES.get(original_name, field_payload.get("field_name"))
        if original_name == "ecog":
            value = field_payload.get("resolved_value")
            if value and value not in {"Not Reported", "null"}:
                value_str = str(value)
                if not value_str.lower().startswith("ecog"):
                    field_payload["resolved_value"] = f"ECOG {value_str}"
        field_payload["field_name"] = canonical_name or field_payload.get("field_name")

    def _records_have_ontology_fields(self, patient_records: List[Dict[str, Any]]) -> bool:
        """Check whether DocETL reduce rows contain any ontology-aligned entries."""
        for record in patient_records:
            for field in record.get("consolidated_fields") or []:
                original_name = field.get("_original_field_name") or field.get("field_name")
                canonical_name = FIELD_NAME_ALIASES.get(original_name, field.get("field_name"))
                if canonical_name in FIELD_NAME_ALIASES.values():
                    return True
                if self.ontology.get_field_definition(original_name or ""):
                    return True
        return False

    def _apply_template(
        self,
        payload: Dict[str, Any],
        extraction_result: Optional[ExtractionResult] = None,
        patient_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Backfill missing domains/columns using the canonical mCODE structure."""
        template = get_default_mcode_structure()
        merged = self._deep_merge(template, payload or {})
        # Keep empty evidence arrays as empty - do NOT fill with default entries
        return self._fill_empty_evidence_arrays(merged, extraction_result, patient_id)

    def _fill_empty_evidence_arrays(
        self,
        obj: Any,
        extraction_result: Optional[ExtractionResult] = None,
        patient_id: Optional[str] = None,
    ) -> Any:
        """Recursively process objects but keep empty evidence arrays as empty.

        Do NOT fill empty supporting_evidence or contradictory_evidence arrays with default entries.
        Empty arrays are valid and indicate no evidence is available.
        """
        if isinstance(obj, dict):
            result = {}
            for key, value in obj.items():
                if key == "supporting_evidence" and isinstance(value, list) and len(value) == 0:
                    # Keep supporting_evidence as empty array - do NOT fill with default entries
                    # Empty array is valid and indicates no supporting evidence available
                    result[key] = []
                elif (
                    key == "contradictory_evidence" and isinstance(value, list) and len(value) == 0
                ):
                    # Keep contradictory_evidence as empty array - do NOT fill with default entries
                    # Empty array is valid and indicates no contradictions found
                    result[key] = []
                else:
                    result[key] = self._fill_empty_evidence_arrays(
                        value, extraction_result, patient_id
                    )
            return result
        elif isinstance(obj, list):
            return [
                self._fill_empty_evidence_arrays(item, extraction_result, patient_id)
                for item in obj
            ]
        else:
            return obj

    def _deep_merge(self, template: Any, data: Any) -> Any:
        """Merge dictionaries while preserving defaults for missing leaves."""
        if isinstance(template, dict):
            merged = deepcopy(template)
            if not isinstance(data, dict):
                return deepcopy(data) if data is not None else merged
            for key, value in data.items():
                if key in merged:
                    merged[key] = self._deep_merge(merged[key], value)
                else:
                    merged[key] = deepcopy(value)
            return merged
        if isinstance(template, list):
            if isinstance(data, list) and data:
                return data
            return deepcopy(template)
        if data is None:
            return deepcopy(template)
        return data

    def _build_evidence_from_extraction(
        self,
        field_payload: Dict[str, Any],
        extraction_result: ExtractionResult,
        patient_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Build evidence from extraction_result when supporting_docs is empty.

        Processes all documents (not just summary documents).
        """
        evidence = []
        if not extraction_result or not extraction_result.document_results:
            return evidence

        field_name = field_payload.get("_original_field_name") or field_payload.get(
            "field_name", ""
        )
        resolved_value = field_payload.get("resolved_value") or field_payload.get(
            "normalized_value"
        )

        # Search through extraction results for matching fields
        # Process all documents (no filtering)
        for doc_result in extraction_result.document_results:
            if isinstance(doc_result, dict):
                doc_id = doc_result.get("doc_id", "")
                extracted_fields = doc_result.get("extracted_fields", [])
                doc_date = doc_result.get("doc_date") or doc_result.get("date") or "Not Reported"
            else:
                doc_id = getattr(doc_result, "doc_id", "")
                extracted_fields = getattr(doc_result, "extracted_fields", [])
                doc_date = (
                    getattr(doc_result, "doc_date", None)
                    or getattr(doc_result, "date", None)
                    or "Not Reported"
                )

            for extracted_field in extracted_fields:
                if isinstance(extracted_field, dict):
                    extracted_field_name = extracted_field.get("field_name", "")
                else:
                    extracted_field_name = getattr(extracted_field, "field_name", "")

                # Match by original field name or canonical name
                if extracted_field_name == field_name:
                    if isinstance(extracted_field, dict):
                        raw_value = extracted_field.get("raw_value", "")
                        normalized_value = extracted_field.get("normalized_value", "")
                        field_evidence = extracted_field.get("field_evidence", {})
                        sources = extracted_field.get("sources", [])
                        confidence_score = extracted_field.get("confidence_score", 0.5)
                        reasoning_excerpt_field = extracted_field.get("reasoning_excerpt", "")
                    else:
                        raw_value = getattr(extracted_field, "raw_value", "")
                        normalized_value = getattr(extracted_field, "normalized_value", "")
                        field_evidence = getattr(extracted_field, "field_evidence", {})
                        sources = getattr(extracted_field, "sources", [])
                        confidence_score = getattr(extracted_field, "confidence_score", 0.5)
                        reasoning_excerpt_field = getattr(extracted_field, "reasoning_excerpt", "")

                    # Get reasoning excerpt from sources or field directly
                    reasoning_excerpt = reasoning_excerpt_field
                    if (
                        not reasoning_excerpt
                        and sources
                        and isinstance(sources, list)
                        and len(sources) > 0
                    ):
                        first_source = sources[0]
                        if isinstance(first_source, dict):
                            reasoning_excerpt = first_source.get("reasoning_excerpt", "")
                        else:
                            reasoning_excerpt = getattr(first_source, "reasoning_excerpt", "")

                    explanation = ""
                    if isinstance(field_evidence, dict):
                        explanation = field_evidence.get("explanation", "")
                    else:
                        explanation = getattr(field_evidence, "explanation", "")

                    if not explanation and reasoning_excerpt:
                        explanation = reasoning_excerpt
                    if not explanation:
                        # Use the actual value if available
                        value_to_show = resolved_value or normalized_value or raw_value
                        if value_to_show and value_to_show != "Not Reported":
                            explanation = f"Field {field_name} with value '{value_to_show}' extracted from document."
                        else:
                            explanation = f"Field {field_name} extracted from document."

                    source_file = self._format_source_file_name(doc_id, patient_id)

                    # Use the value from extracted_field if resolved_value doesn't match
                    value_for_evidence = resolved_value
                    if not value_for_evidence or value_for_evidence == "Not Reported":
                        value_for_evidence = normalized_value or raw_value or "Not Reported"

                    # Use actual confidence from LLM, not default
                    conf_val = confidence_score if confidence_score is not None else None
                    evidence.append(
                        {
                            "snippet": reasoning_excerpt or "",
                            "explanation": explanation,
                            "date": doc_date,
                            "source_file": source_file,
                            "confidence": (
                                self._clamp_confidence(conf_val) if conf_val is not None else None
                            ),
                            "span": None,
                            "page": None,
                        }
                    )
                    break  # Found match, move to next document

        return evidence

    def _format_source_file_name(self, doc_id: str, patient_id: Optional[str] = None) -> str:
        """Format doc_id into proper source_file name format.

        Examples:
        - "001" -> "doc_001_jsl_p01_001_summary_doc" (if patient_id is p01)
        - "001" -> "doc_001" (if patient_id is None)
        - "doc_001_jsl_p01_001_summary_doc" -> unchanged
        - "jsl_p01_001_summary_doc" -> "doc_001_jsl_p01_001_summary_doc"
        """
        if not doc_id or doc_id == "unknown_doc" or doc_id.strip() == "":
            # Try to get from patient_id if available
            if patient_id:
                return f"doc_001_jsl_{patient_id}_001_summary_doc"
            return "unknown_doc"

        doc_id = doc_id.strip()

        # If already formatted correctly, return as is
        if doc_id.startswith("doc_") and "_jsl_" in doc_id:
            return doc_id

        # Handle jsl_ prefixed doc_ids
        if doc_id.startswith("jsl_"):
            # Extract parts: jsl_p01_001_summary_doc -> doc_001_jsl_p01_001_summary_doc
            parts = doc_id.split("_")
            if len(parts) >= 4:
                # parts[0] = "jsl", parts[1] = patient_id, parts[2] = doc_number, rest = doc_type
                patient_from_doc = parts[1]
                doc_number = parts[2]
                doc_type = "_".join(parts[3:]) if len(parts) > 3 else "summary_doc"
                return f"doc_{doc_number}_jsl_{patient_from_doc}_{doc_number}_{doc_type}"
            else:
                # Fallback: just add doc_ prefix
                return f"doc_{doc_id}"

        # Try to format: doc_XXX_jsl_PATIENT_XXX_summary_doc
        if patient_id:
            # Extract numeric part from doc_id
            numeric_part = doc_id
            # Remove any non-numeric prefixes
            if not doc_id.isdigit():
                # Try to extract numbers from doc_id
                numbers = re.findall(r"\d+", doc_id)
                if numbers:
                    numeric_part = numbers[0]
                else:
                    # If no numbers found, use the whole doc_id
                    numeric_part = doc_id.replace("_", "").replace("-", "")

            if numeric_part.isdigit():
                numeric_part = numeric_part.zfill(3)  # Pad to 3 digits
                return f"doc_{numeric_part}_jsl_{patient_id}_{numeric_part}_summary_doc"
            else:
                # If not a pure number, try to construct from doc_id
                # Remove common prefixes
                clean_id = doc_id.replace("doc_", "").replace("jsl_", "").strip()
                return f"doc_{clean_id}_jsl_{patient_id}_{clean_id}_summary_doc"

        # Fallback: just prefix with doc_ if not already present
        if not doc_id.startswith("doc_"):
            return f"doc_{doc_id}"
        return doc_id

    def _derive_source_docs_from_extraction(
        self, extraction_result: Optional[ExtractionResult]
    ) -> List[str]:
        """Fallback document provenance derived from extractor outputs.

        Includes all documents (not just summary documents).
        """
        if not extraction_result or not extraction_result.document_results:
            return []
        doc_ids: List[str] = []
        for entry in extraction_result.document_results:
            if isinstance(entry, dict):
                doc_id = entry.get("doc_id")
            else:
                doc_id = getattr(entry, "doc_id", None)

            # Include all documents (no filtering)
            if doc_id:
                doc_id_str = str(doc_id)
                doc_ids.append(doc_id_str)
        return sorted(set(doc_ids))

    def _lookup_doc_date_from_extraction(
        self,
        doc_id: str,
        extraction_result: Optional[ExtractionResult],
    ) -> Optional[str]:
        """Lookup doc_date from extraction_result by doc_id.
        
        This is used as a fallback when supporting_docs from DocETL
        don't contain doc_date (LLM may not always propagate it).
        """
        if not extraction_result or not extraction_result.document_results:
            return None
        
        # Normalize doc_id for comparison (handle various formats)
        normalized_doc_id = doc_id.lower().strip()
        
        for doc_result in extraction_result.document_results:
            if isinstance(doc_result, dict):
                result_doc_id = doc_result.get("doc_id", "")
                result_doc_date = doc_result.get("doc_date")
            else:
                result_doc_id = getattr(doc_result, "doc_id", "")
                result_doc_date = getattr(doc_result, "doc_date", None)
            
            # Compare normalized doc_ids
            if result_doc_id and result_doc_id.lower().strip() == normalized_doc_id:
                if result_doc_date and result_doc_date not in ("", "null", "None", "Not Reported"):
                    return result_doc_date
            
            # Also try partial match (e.g., "doc_001" vs "doc_001_jsl_p01_001_summary_doc")
            if result_doc_id and (
                normalized_doc_id in result_doc_id.lower() or 
                result_doc_id.lower() in normalized_doc_id
            ):
                if result_doc_date and result_doc_date not in ("", "null", "None", "Not Reported"):
                    return result_doc_date
        
        return None

    @staticmethod
    def _clamp_confidence(value: Any, default: Optional[float] = None) -> Optional[float]:
        """
        Clamp arbitrary numeric confidence values into the [0, 1] interval.
        Rounds to 2 decimal places to avoid floating point precision issues.

        Returns None if value is None, ensuring we don't create fake confidence scores.
        The default parameter is only used when value cannot be converted to float.
        """
        # If value is None, return None (no confidence available)
        if value is None:
            return None

        try:
            num = float(value)
        except (TypeError, ValueError):
            # If conversion fails, use default if provided, otherwise return None
            if default is not None:
                return round(max(0.0, min(1.0, default)), 2)
            return None

        # Round to 2 decimals to prevent values like 0.9974999999999999
        clamped = max(0.0, min(1.0, num))
        return round(clamped, 2)

    @staticmethod
    def _aggregate_confidence_scores(confidences: List[float]) -> Optional[float]:
        """
        Aggregate multiple confidence scores using weighted harmonic mean.
        This approach:
        - Penalizes inconsistent evidence (one low score pulls down average)
        - Rewards consistent high-quality evidence
        - More conservative than arithmetic mean

        Returns None if no valid confidences are provided, ensuring we don't
        create fake confidence scores when LLM doesn't provide them.
        """
        if not confidences:
            return None

        # Remove zeros to avoid division errors and clamp to [0, 1]
        valid_confidences = [max(0.0, min(1.0, c)) for c in confidences if c > 0.0]
        if not valid_confidences:
            return None

        # If only one confidence, return it (rounded to 2 decimals)
        if len(valid_confidences) == 1:
            return round(valid_confidences[0], 2)

        # Weighted harmonic mean: gives more weight to lower values (conservative)
        # Formula: n / sum(1/x_i)
        harmonic_mean = len(valid_confidences) / sum(1.0 / c for c in valid_confidences)

        # Apply consistency bonus: if all scores are similar, boost slightly
        std_dev = (
            sum((c - harmonic_mean) ** 2 for c in valid_confidences) / len(valid_confidences)
        ) ** 0.5

        if std_dev < 0.1:  # Very consistent scores
            consistency_bonus = 0.05
        elif std_dev < 0.2:  # Moderately consistent
            consistency_bonus = 0.02
        else:
            consistency_bonus = 0.0

        # Apply bonus but cap at 1.0 BEFORE adding bonus to avoid overflow
        if harmonic_mean + consistency_bonus > 1.0:
            final_score = 1.0
        else:
            final_score = harmonic_mean + consistency_bonus

        # Round to 2 decimal places to avoid floating point precision issues
        # This prevents values like 0.9974999999999999
        final_score = round(final_score, 2)

        return max(0.0, min(1.0, final_score))

    @staticmethod
    def _normalize_value_for_comparison(value: Any, field_name: str = "") -> str:
        """Normalize a value for semantic comparison.
        
        Extracts the core semantic value from both raw values (like "cT2", "1", "2015-07-15")
        and resolved values (like "Clinical staging revealed cT2", "ECOG 1", "Diagnosed in July 2015").
        
        This prevents false contradictions caused by format differences.
        """
        if value is None:
            return ""
        
        val_str = str(value).strip().lower()
        if not val_str or val_str in ("not reported", "null", "none", "n/a"):
            return ""
        
        import re
        
        # Roman numeral to Arabic conversion
        roman_to_arabic = {
            'i': '1', 'ii': '2', 'iii': '3', 'iv': '4', 'v': '5',
            'vi': '6', 'vii': '7', 'viii': '8', 'ix': '9', 'x': '10'
        }
        
        field_lower = field_name.lower()
        
        # Stage group: handle various formats including Roman numerals and TNM combinations
        # Match: "Stage III", "Stage 3", "stage iii", "III", "3", "Summary Stage3", "pT3, pN0, pM0"
        if "stage" in field_lower:
            # First try to find Roman numeral stage
            roman_pattern = r'\b(?:stage\s*)?(i{1,3}|iv|v|vi{0,3}|ix|x)\b'
            roman_match = re.search(roman_pattern, val_str, re.IGNORECASE)
            if roman_match:
                roman_val = roman_match.group(1).lower()
                return roman_to_arabic.get(roman_val, roman_val)
            
            # Then try Arabic numeral (standalone or after "stage")
            arabic_pattern = r'\b(?:stage\s*)?([0-9])\b'
            arabic_match = re.search(arabic_pattern, val_str, re.IGNORECASE)
            if arabic_match:
                return arabic_match.group(1)
            
            # Fallback: extract stage from TNM combination like "pT3, pN0, pM0" or "T3N0M0"
            # Use the T-stage number as the stage group indicator
            tnm_stage_pattern = r'\b[cp]?[tT]([0-4x])(?:[abc])?\b'
            tnm_stage_match = re.search(tnm_stage_pattern, val_str)
            if tnm_stage_match:
                return tnm_stage_match.group(1)
        
        # TNM staging: extract the stage value and handle c/p prefix based on field type
        # For clinical fields: cT3 and T3 (no prefix) are equivalent, but pT3 is DIFFERENT
        # For pathologic fields: pT3 is expected, cT3 and T3 are DIFFERENT
        tnm_pattern = r'\b([cp])?([tnm][0-4x](?:[abc])?)\b'
        tnm_match = re.search(tnm_pattern, val_str, re.IGNORECASE)
        if tnm_match:
            prefix = (tnm_match.group(1) or '').lower()
            stage_value = tnm_match.group(2).lower()
            
            # Check if this is a clinical or pathologic field
            is_clinical_field = 'clinical' in field_lower or '_clinical' in field_lower
            is_pathologic_field = 'pathologic' in field_lower or '_pathologic' in field_lower or 'path' in field_lower
            
            if is_clinical_field:
                # For clinical fields: c prefix or no prefix = "c_<stage>", p prefix = "p_<stage>"
                # This way cT3 and T3 match, but pT3 does not
                if prefix == 'p':
                    return f"p_{stage_value}"  # Will NOT match clinical values
                else:
                    return f"c_{stage_value}"  # cT3 and T3 both become "c_t3"
            elif is_pathologic_field:
                # For pathologic fields: p prefix = "p_<stage>", c prefix or no prefix = "c_<stage>"
                # This way pT3 matches, but cT3 and T3 do not
                if prefix == 'p':
                    return f"p_{stage_value}"  # pT3 becomes "p_t3"
                else:
                    return f"c_{stage_value}"  # Will NOT match pathologic values
            else:
                # Generic TNM field - keep old behavior (strip prefix)
                return stage_value
        
        # ECOG/Performance/KPS status: extract just the number
        if any(x in field_lower for x in ["ecog", "performance", "kps"]):
            # For KPS, values are typically 0-100 in increments of 10
            if "kps" in field_lower:
                kps_pattern = r'\b([1-9]?[0-9]0|100)\b'
                kps_match = re.search(kps_pattern, val_str)
                if kps_match:
                    return kps_match.group(1)
            # For ECOG, values are 0-5
            ecog_pattern = r'\b([0-5])\b'
            ecog_match = re.search(ecog_pattern, val_str)
            if ecog_match:
                return ecog_match.group(1)
        
        # Date: normalize to YYYY-MM format
        date_pattern = r'(\d{4})-(\d{2})-(\d{2})'
        date_match = re.search(date_pattern, val_str)
        if date_match:
            return f"{date_match.group(1)}-{date_match.group(2)}"
        
        # Month year format: "July 2015" -> "2015-07" or "July2015" -> "2015-07"
        month_names = {
            'january': '01', 'february': '02', 'march': '03', 'april': '04',
            'may': '05', 'june': '06', 'july': '07', 'august': '08',
            'september': '09', 'october': '10', 'november': '11', 'december': '12'
        }
        month_year_pattern = r'(january|february|march|april|may|june|july|august|september|october|november|december)\s*(\d{4})'
        month_match = re.search(month_year_pattern, val_str, re.IGNORECASE)
        if month_match:
            month_num = month_names.get(month_match.group(1).lower(), "00")
            return f"{month_match.group(2)}-{month_num}"
        
        # ICD-O codes: extract just the code (C61.9, C34.9, etc.)
        icd_pattern = r'\b([cC]\d{1,2}(?:\.\d)?)\b'
        icd_match = re.search(icd_pattern, val_str)
        if icd_match:
            return icd_match.group(1).lower()
        
        # Histology/Morphology: extract the ICD-O-3 morphology code (8140/3, etc.)
        # or normalize common histology terms to their ICD-O codes
        if "histology" in field_lower or "morphology" in field_lower:
            # First try to find ICD-O-3 morphology code directly
            morphology_pattern = r'\b(\d{4}/\d)\b'
            morph_match = re.search(morphology_pattern, val_str)
            if morph_match:
                return morph_match.group(1)
            
            # Map common histology terms to ICD-O-3 codes
            # This allows "prostatic adenocarcinoma" to match "8140/3 - Adenocarcinoma, NOS"
            histology_mappings = {
                'adenocarcinoma': '8140/3',
                'squamous cell carcinoma': '8070/3',
                'small cell carcinoma': '8041/3',
                'large cell carcinoma': '8012/3',
                'ductal carcinoma': '8500/3',
                'lobular carcinoma': '8520/3',
                'melanoma': '8720/3',
                'sarcoma': '8800/3',
                'lymphoma': '9590/3',
                'leukemia': '9800/3',
            }
            
            for term, code in histology_mappings.items():
                if term in val_str:
                    return code
        
        # Fallback: return cleaned value
        return val_str

    def _normalize_category(self, category: Optional[str], field_name: str) -> str:
        """Normalize raw category strings into ontology-friendly groups."""
        if category:
            normalized = category.strip().lower().replace("-", "_").replace(" ", "_")
            if normalized:
                return normalized
        definition = self.ontology.get_field_definition(field_name) if field_name else None
        if definition and definition.get("category"):
            return str(definition["category"]).lower().replace("-", "_").replace(" ", "_")
        return "general"

    def _determine_domain(self, field_payload: Dict[str, Any], normalized_category: str) -> str:
        """Determine mCODE domain for a field using ontology + fallback mapping."""
        field_name = field_payload.get("_original_field_name") or field_payload.get("field_name")
        definition = self.ontology.get_field_definition(field_name) if field_name else None
        
        # 1. Check if ontology explicitly defines a domain
        if definition and definition.get("domain"):
            domain_value = str(definition["domain"]).lower()
            return ONTOLOGY_DOMAIN_TO_MCODE_DOMAIN.get(
                domain_value, CATEGORY_TO_DOMAIN.get(domain_value, "extensions")
            )

        # 2. Check if normalized_category maps to a known domain
        domain = CATEGORY_TO_DOMAIN.get(normalized_category)
        if domain:
            return domain
            
        # 3. Fallback: check definition category logic if payload category is garbage (e.g. "p01")
        if definition and definition.get("category"):
             def_cat = str(definition["category"]).lower().replace("-", "_").replace(" ", "_")
             domain = CATEGORY_TO_DOMAIN.get(def_cat)
             if domain:
                 return domain

        return "extensions"

    @staticmethod
    def _is_synthetic_field(field_payload: Dict[str, Any]) -> bool:
        """Detect placeholder/synthetic fields injected for technical reasons."""
        field_name = str(field_payload.get("field_name", "")).lower()
        category = str(field_payload.get("category", "")).lower()
        return field_name.startswith("synthetic_") or category == "technical"
