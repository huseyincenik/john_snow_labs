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

    def _build_fields_from_artifacts(
        self,
        patient_records: List[Dict[str, Any]],
        extraction_result: ExtractionResult,
        resolve_records: Optional[List[Dict[str, Any]]] = None,
        logger: Optional[logging.Logger] = None,
    ) -> List[ConsolidatedField]:
        """Shape DocETL patient outputs to match the sample consolidation artifact."""
        patient_records = patient_records or []
        if resolve_records and (
            not patient_records
            or all(not record.get("consolidated_fields") for record in patient_records)
            or not self._records_have_ontology_fields(patient_records)
        ):
            patient_records = self._build_patient_records_from_resolve(resolve_records)

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
                confidence_score=mcode_value["confidence_score"],
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
                diagnosis_fields, extraction_result, patient_id
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
    ) -> List[Dict[str, Any]]:
        """Build primary_cancers array structure from diagnosis fields."""
        if not diagnosis_fields:
            return []
        # Group diagnosis fields by cancer (based on body_site or diagnosis field)
        cancers: Dict[str, Dict[str, Any]] = defaultdict(dict)
        cancer_counter = 1

        for field_payload in diagnosis_fields:
            if not field_payload or not isinstance(field_payload, dict):
                continue
            field_name = field_payload.get("field_name", "unknown_field")
            body_site = None
            diagnosis_value = None

            # Try to identify which cancer this belongs to
            if field_name == "body_site":
                body_site = field_payload.get("resolved_value") or field_payload.get(
                    "normalized_value"
                )
            elif field_name == "diagnosis":
                diagnosis_value = field_payload.get("resolved_value") or field_payload.get(
                    "normalized_value"
                )

            # Use body_site as key, fallback to diagnosis, then counter
            cancer_key = body_site or diagnosis_value or f"cancer_{cancer_counter}"
            if cancer_key not in cancers:
                cancers[cancer_key] = {"cancer_id": f"cancer_{cancer_counter}"}
                cancer_counter += 1

            field_summary = self._build_field_summary(field_payload, extraction_result, patient_id)
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

            cancers[cancer_key][field_name] = field_entry

        return list(cancers.values())

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

        consolidated = {
            "field_name": field_name,
            "category": category,
            "data_type": data_type,
            "normalized_value": normalized_value,
            "resolved_value": resolved_value,
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

                    # Get doc_date - try multiple fields
                    doc_date = doc.get("doc_date") or doc.get("date") or "Not Reported"
                    if doc_date in (None, "", "null", "None"):
                        doc_date = "Not Reported"

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
            resolved_value = (
                field_payload.get("resolved_value")
                or field_payload.get("normalized_value")
                or field_payload.get("raw_value")
            )
            baseline = (
                str(resolved_value).strip().lower()
                if resolved_value and str(resolved_value).strip()
                else ""
            )
            contradictions: List[Dict[str, Any]] = []
            seen_contradictions = set()

            for doc_result in extraction_result.document_results:
                if isinstance(doc_result, dict):
                    extracted_fields = doc_result.get("extracted_fields", [])
                    doc_id = doc_result.get("doc_id", "")
                    doc_date = (
                        doc_result.get("doc_date") or doc_result.get("date") or "Not Reported"
                    )
                else:
                    extracted_fields = getattr(doc_result, "extracted_fields", [])
                    doc_id = getattr(doc_result, "doc_id", "")
                    doc_date = (
                        getattr(doc_result, "doc_date", None)
                        or getattr(doc_result, "date", None)
                        or "Not Reported"
                    )

                for extracted_field in extracted_fields:
                    if isinstance(extracted_field, dict):
                        extracted_field_name = extracted_field.get("field_name", "")
                        extracted_value = (
                            extracted_field.get("resolved_value")
                            or extracted_field.get("normalized_value")
                            or extracted_field.get("raw_value")
                        )
                        reasoning_excerpt = extracted_field.get("reasoning_excerpt", "")
                        field_evidence = extracted_field.get("field_evidence", {})
                        explanation = (
                            field_evidence.get("explanation", "")
                            if isinstance(field_evidence, dict)
                            else ""
                        )
                        confidence_score = extracted_field.get("confidence_score", 0.5)
                    else:
                        extracted_field_name = getattr(extracted_field, "field_name", "")
                        extracted_value = (
                            getattr(extracted_field, "resolved_value", None)
                            or getattr(extracted_field, "normalized_value", None)
                            or getattr(extracted_field, "raw_value", None)
                        )
                        reasoning_excerpt = getattr(extracted_field, "reasoning_excerpt", "")
                        field_evidence = getattr(extracted_field, "field_evidence", None)
                        explanation = (
                            getattr(field_evidence, "explanation", "") if field_evidence else ""
                        )
                        confidence_score = getattr(extracted_field, "confidence_score", 0.5)

                    if extracted_field_name != field_name:
                        continue

                    if not extracted_value:
                        continue
                    extracted_value_str = str(extracted_value).strip()
                    if not extracted_value_str or extracted_value_str.lower() == "not reported":
                        continue

                    extracted_comparable = extracted_value_str.lower()
                    if baseline and extracted_comparable == baseline:
                        continue

                    source_file = self._format_source_file_name(doc_id, patient_id)
                    signature = (source_file, extracted_comparable)
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
                    contradictions.append(
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
        if definition and definition.get("domain"):
            domain_value = str(definition["domain"]).lower()
            return ONTOLOGY_DOMAIN_TO_MCODE_DOMAIN.get(
                domain_value, CATEGORY_TO_DOMAIN.get(domain_value, "extensions")
            )
        return CATEGORY_TO_DOMAIN.get(normalized_category, "extensions")

    @staticmethod
    def _is_synthetic_field(field_payload: Dict[str, Any]) -> bool:
        """Detect placeholder/synthetic fields injected for technical reasons."""
        field_name = str(field_payload.get("field_name", "")).lower()
        category = str(field_payload.get("category", "")).lower()
        return field_name.startswith("synthetic_") or category == "technical"
