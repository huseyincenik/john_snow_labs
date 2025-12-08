#!/usr/bin/env python3
"""
Generate proof files showing field-by-field resolve → reduce transformation.

Usage:
    python scripts/generate_proof.py <session_id>
    
Example:
    python scripts/generate_proof.py beeb44f7-47b8-422d-9950-aec96b891062

Creates:
    proof/<session_id>/
        00_summary.txt           # Overall summary with all primary cancers
        body_site.txt
        histology_morphology.txt
        tnm_t_clinical.txt
        ... (one file per YAML-defined field, using mCODE names)
"""

import json
import shutil
import sys
import yaml
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple


# NAACCR to mCODE field mapping (one-directional)
# NAACCR names -> mCODE canonical names
NAACCR_TO_MCODE = {
    # Diagnosis domain
    "naaccr_diagnosis_dt": "diagnosis_date",
    "ca_site": "body_site",
    "naaccr_histology_cd": "histology_morphology",
    # Clinical staging domain
    "ca_clinical_t_stage": "tnm_t_clinical",
    "ca_clinical_n_stage": "tnm_n_clinical",
    "ca_clinical_m_stage": "tnm_m_clinical",
    # Pathological staging domain
    "ca_path_t_stage": "tnm_t_pathologic",
    "ca_path_n_stage": "tnm_n_pathologic",
    "ca_path_m_stage": "tnm_m_pathologic",
    # Summary staging domain
    "ca_gen_sum_stage_2": "stage_group",
    # Performance status domain
    "ecog": "performance_status",
    "kps": "kps",
}

# Domain structure for organizing proof files
DOMAIN_STRUCTURE = {
    "diagnosis": {
        "title": "DIAGNOSIS DOMAIN",
        "profile": "PrimaryCancerCondition, CancerDiagnosis, DiagnosticReport",
        "fields": ["naaccr_diagnosis_dt", "ca_site", "naaccr_histology_cd"],
    },
    "clinical_staging": {
        "title": "CLINICAL STAGING DOMAIN",
        "profile": "TNMClinicalStageGroup, CancerStage",
        "fields": ["ca_clinical_t_stage", "ca_clinical_n_stage", "ca_clinical_m_stage"],
    },
    "pathological_staging": {
        "title": "PATHOLOGICAL STAGING DOMAIN",
        "profile": "TNMPathologicStageGroup, PathologyReport",
        "fields": ["ca_path_t_stage", "ca_path_n_stage", "ca_path_m_stage"],
    },
    "summary_staging": {
        "title": "SUMMARY STAGING DOMAIN",
        "profile": "CancerStageGroup, SEERSummaryStage",
        "fields": ["ca_gen_sum_stage_2"],
    },
    "performance_status": {
        "title": "PERFORMANCE STATUS DOMAIN",
        "profile": "PerformanceStatus, ECOGStatus, KarnofskyStatus",
        "fields": ["ecog", "kps"],
    },
}


def load_yaml_ontology(yaml_path: Path) -> Tuple[List[str], Dict[str, Dict]]:
    """Load field names and domain structure from cancer_registry_fields.yaml."""
    if not yaml_path.exists():
        return [], {}
    
    with open(yaml_path, "r", encoding="utf-8") as f:
        ontology = yaml.safe_load(f)
    
    fields = []
    domains = {}
    for domain_name, domain_data in ontology.get("domains", {}).items():
        domain_fields = []
        for field in domain_data.get("fields", []):
            if "name" in field:
                fields.append(field["name"])
                domain_fields.append(field["name"])
        domains[domain_name] = {
            "profile": domain_data.get("profile", ""),
            "fields": domain_fields,
        }
    
    return fields, domains


def load_json(filepath: Path) -> Optional[Dict[str, Any]]:
    """Load JSON file if it exists."""
    if not filepath.exists():
        return None
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def find_resolve_file(session_dir: Path) -> Optional[Path]:
    """Find resolve_patient_fields.json in session directory."""
    # Check docetl_intermediate/clinical_registry/
    resolve_path = session_dir / "docetl_intermediate" / "clinical_registry" / "resolve_patient_fields.json"
    if resolve_path.exists():
        return resolve_path
    
    # Check patients/p01/docetl_intermediate/clinical_registry/
    for patient_dir in session_dir.glob("patients/*/docetl_intermediate/clinical_registry"):
        resolve_path = patient_dir / "resolve_patient_fields.json"
        if resolve_path.exists():
            return resolve_path
    
    return None


def find_consolidation_file(session_dir: Path) -> Optional[Path]:
    """Find consolidation JSON file in session directory."""
    # Look for stage_consolidator_*_consolidation.json
    for f in session_dir.glob("*consolidation*.json"):
        return f
    return None


def extract_resolve_fields(resolve_data: List[Dict]) -> Dict[str, List[Dict]]:
    """Group resolve records by field_name (using NAACCR names from source)."""
    fields = defaultdict(list)
    
    if not resolve_data:
        return fields
    
    for record in resolve_data:
        field_name = record.get("field_name")
        if field_name:
            fields[field_name].append(record)
    
    return fields


def extract_primary_cancers(consolidation_data: Dict) -> List[Dict]:
    """Extract primary_cancers array from consolidation output."""
    if not consolidation_data:
        return []
    
    consolidated = consolidation_data.get("consolidated_fields", [])
    
    for field in consolidated:
        if field.get("field_name") == "mcode_patient_extraction":
            consolidated_value = field.get("consolidated_value", {})
            mcode_extraction = consolidated_value.get("mcode_extraction", {})
            return mcode_extraction.get("primary_cancers", [])
    
    return []


def extract_consolidation_fields(consolidation_data: Dict) -> Dict[str, Dict]:
    """Extract field data from consolidation output (uses mCODE names)."""
    fields = {}
    
    if not consolidation_data:
        return fields
    
    consolidated = consolidation_data.get("consolidated_fields", [])
    
    for field in consolidated:
        field_name = field.get("field_name")
        if field_name == "mcode_patient_extraction":
            # Use consolidated_value.mcode_extraction structure (not final_value)
            consolidated_value = field.get("consolidated_value", {})
            mcode_extraction = consolidated_value.get("mcode_extraction", {})
            
            # Extract cancer_stage fields from final_stage (correct path!)
            cancer_stage = mcode_extraction.get("cancer_stage", {})
            final_stage = cancer_stage.get("final_stage", {})
            for stage_field in ["tnm_t_clinical", "tnm_n_clinical", "tnm_m_clinical",
                               "tnm_t_pathologic", "tnm_n_pathologic", "tnm_m_pathologic",
                               "stage_group", "staging_system"]:
                if stage_field in final_stage:
                    field_data = final_stage[stage_field]
                    fields[stage_field] = {
                        "final_value": field_data.get("final_value"),
                        "supporting_evidence_count": len(field_data.get("supporting_evidence", [])),
                        "contradictory_evidence_count": len(field_data.get("contradictory_evidence", [])),
                        "supporting_evidence": field_data.get("supporting_evidence", []),
                        "contradictory_evidence": field_data.get("contradictory_evidence", []),
                    }
            
            # Extract health_assessment fields
            health_assessment = mcode_extraction.get("health_assessment", {})
            for health_field in ["performance_status", "kps", "bmi", "weight", "height"]:
                if health_field in health_assessment:
                    field_data = health_assessment[health_field]
                    fields[health_field] = {
                        "final_value": field_data.get("final_value"),
                        "supporting_evidence_count": len(field_data.get("supporting_evidence", [])),
                        "contradictory_evidence_count": len(field_data.get("contradictory_evidence", [])),
                        "supporting_evidence": field_data.get("supporting_evidence", []),
                        "contradictory_evidence": field_data.get("contradictory_evidence", []),
                    }
            
            # Extract primary_cancers fields (aggregate from ALL items)
            primary_cancers = mcode_extraction.get("primary_cancers", [])
            if primary_cancers and isinstance(primary_cancers, list):
                for cancer_field in ["body_site", "histology_morphology", "diagnosis_date"]:
                    # Aggregate evidence from ALL cancers
                    all_supporting = []
                    all_contradictory = []
                    final_value = None
                    
                    for cancer_item in primary_cancers:
                        if isinstance(cancer_item, dict) and cancer_field in cancer_item:
                            field_data = cancer_item[cancer_field]
                            if isinstance(field_data, dict):
                                # Take final_value from first cancer with this field
                                if final_value is None:
                                    final_value = field_data.get("final_value")
                                # Aggregate evidence from all cancers
                                all_supporting.extend(field_data.get("supporting_evidence", []))
                                all_contradictory.extend(field_data.get("contradictory_evidence", []))
                    
                    if final_value is not None or all_supporting:
                        fields[cancer_field] = {
                            "final_value": final_value,
                            "supporting_evidence_count": len(all_supporting),
                            "contradictory_evidence_count": len(all_contradictory),
                            "supporting_evidence": all_supporting,
                            "contradictory_evidence": all_contradictory,
                        }
            
            # Extract disease_status fields
            disease_status = mcode_extraction.get("disease_status", {})
            for ds_field in ["disease_status", "treatment_response", "recurrence_indicator", "recurrence_date"]:
                if ds_field in disease_status:
                    field_data = disease_status[ds_field]
                    fields[ds_field] = {
                        "final_value": field_data.get("final_value"),
                        "supporting_evidence_count": len(field_data.get("supporting_evidence", [])),
                        "contradictory_evidence_count": len(field_data.get("contradictory_evidence", [])),
                        "supporting_evidence": field_data.get("supporting_evidence", []),
                        "contradictory_evidence": field_data.get("contradictory_evidence", []),
                    }
    
    return fields


def generate_summary_file(
    primary_cancers: List[Dict],
    yaml_domains: Dict[str, Dict],
    output_path: Path
):
    """Generate summary file showing all primary cancers identified."""
    lines = []
    
    # Header
    lines.append("=" * 80)
    lines.append("PATIENT CANCER REGISTRY SUMMARY")
    lines.append("=" * 80)
    lines.append("")
    
    # Primary cancers section
    lines.append("─" * 40)
    lines.append("PRIMARY CANCERS IDENTIFIED")
    lines.append("─" * 40)
    lines.append(f"Total Primary Cancers: {len(primary_cancers)}")
    lines.append("")
    
    for i, cancer in enumerate(primary_cancers, 1):
        cancer_id = cancer.get("cancer_id", f"cancer_{i}")
        site_code = cancer.get("site_code", "Unknown")
        
        # Get body_site info
        body_site = cancer.get("body_site", {})
        site_value = body_site.get("final_value", "Not Reported")
        site_evidence_count = body_site.get("supporting_evidence_count") or len(body_site.get("supporting_evidence", []))
        
        # Get diagnosis_date info
        diagnosis_date = cancer.get("diagnosis_date", {})
        date_value = diagnosis_date.get("final_value", "Not Reported")
        
        lines.append(f"  [{i}] {cancer_id.upper()}")
        lines.append(f"      Site Code: {site_code}")
        lines.append(f"      Body Site: {site_value}")
        lines.append(f"      Diagnosis Date: {date_value}")
        lines.append(f"      Supporting Documents: {site_evidence_count}")
        lines.append("")
    
    # Domain structure section
    lines.append("")
    lines.append("─" * 40)
    lines.append("ONTOLOGY DOMAIN STRUCTURE")
    lines.append("─" * 40)
    lines.append("")
    
    for domain_name, domain_info in DOMAIN_STRUCTURE.items():
        lines.append(f"▶ {domain_info['title']}")
        lines.append(f"  Profile: {domain_info['profile']}")
        lines.append("  Fields:")
        for field in domain_info['fields']:
            mcode_name = NAACCR_TO_MCODE.get(field, field)
            lines.append(f"    • {field} → {mcode_name}")
        lines.append("")
    
    # Write file
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  * {output_path.name}")


def generate_proof_file(
    naaccr_name: str,
    mcode_name: str,
    domain_name: str,
    domain_title: str,
    resolve_records: List[Dict],
    consolidation_data: Optional[Dict],
    primary_cancers: List[Dict],
    output_path: Path
):
    """Generate a single proof file for a field (output in mCODE name)."""
    lines = []
    
    # Header with domain and mapping info
    lines.append("=" * 80)
    lines.append(f"DOMAIN: {domain_title}")
    lines.append(f"FIELD: {mcode_name}")
    lines.append(f"  NAACCR Source: {naaccr_name}")
    lines.append("=" * 80)
    lines.append("")
    
    # === PRIMARY CANCERS SECTION (for diagnosis fields) ===
    if mcode_name in ["body_site", "diagnosis_date", "histology_morphology"] and primary_cancers:
        lines.append("─" * 40)
        lines.append("PRIMARY CANCERS (Multi-Cancer Support)")
        lines.append("─" * 40)
        lines.append(f"Total Primary Cancers: {len(primary_cancers)}")
        lines.append("")
        
        for i, cancer in enumerate(primary_cancers, 1):
            cancer_id = cancer.get("cancer_id", f"cancer_{i}")
            site_code = cancer.get("site_code", "Unknown")
            
            # Get field-specific data for this cancer
            field_data = cancer.get(mcode_name, {})
            final_value = field_data.get("final_value", "Not Reported")
            supporting = field_data.get("supporting_evidence", [])
            
            lines.append(f"  [{i}] {cancer_id.upper()} (Site: {site_code})")
            lines.append(f"      {mcode_name}: {final_value}")
            lines.append(f"      Supporting Documents: {len(supporting)}")
            
            # Show ALL supporting docs
            for j, ev in enumerate(supporting, 1):
                source = ev.get("source_file", "unknown")
                snippet = ev.get("snippet", "")
                date_val = ev.get("date", "Not Reported")
                lines.append(f"        - {source} (Date: {date_val})")
                lines.append(f"          Snippet: {snippet}")
            lines.append("")

    # === RESOLVE SECTION ===
    lines.append("─" * 40)
    lines.append("RESOLVE OUTPUT (Before Reduce)")
    lines.append("─" * 40)
    lines.append(f"Total records: {len(resolve_records)}")
    lines.append("")
    
    # Group by resolved_value
    value_groups = defaultdict(list)
    for r in resolve_records:
        val = r.get("resolved_value") or r.get("raw_value") or "null"
        value_groups[val].append(r.get("doc_id", "unknown"))
    
    lines.append("Unique values:")
    for val, doc_ids in value_groups.items():
        lines.append(f"  - '{val}' (found in {len(doc_ids)} docs)")
        for doc_id in doc_ids:
            lines.append(f"      - {doc_id}")
    lines.append("")
    
    # Sample records (ALL)
    lines.append("Sample records (ALL):")
    for i, r in enumerate(resolve_records, 1):
        lines.append(f"  [{i}] doc_id: {r.get('doc_id')}")
        lines.append(f"      raw_value: {r.get('raw_value')}")
        lines.append(f"      resolved_value: {r.get('resolved_value')}")
        lines.append(f"      confidence: {r.get('confidence_score')}")
        lines.append("")
    
    # === REDUCE SECTION ===
    lines.append("─" * 40)
    lines.append("REDUCE OUTPUT (After Consolidation)")
    lines.append("─" * 40)
    
    # For diagnosis-domain fields (body_site, diagnosis_date, histology_morphology),
    # show each primary cancer separately
    if mcode_name in ["body_site", "diagnosis_date", "histology_morphology"] and primary_cancers:
        lines.append(f"Total Primary Cancers: {len(primary_cancers)}")
        lines.append("")
        
        for i, cancer in enumerate(primary_cancers, 1):
            cancer_id = cancer.get("cancer_id", f"cancer_{i}")
            site_code = cancer.get("site_code", "Unknown")
            
            # Get field-specific data for this cancer
            field_data = cancer.get(mcode_name, {})
            if not isinstance(field_data, dict):
                field_data = {}
            
            final_value = field_data.get("final_value", "Not Reported")
            supporting = field_data.get("supporting_evidence", [])
            contradictory = field_data.get("contradictory_evidence", [])
            
            lines.append(f"▶ PRIMARY CANCER #{i}: {cancer_id.upper()} (Site: {site_code})")
            lines.append(f"  final_value: {final_value}")
            lines.append(f"  supporting_evidence_count: {len(supporting)}")
            lines.append(f"  contradictory_evidence_count: {len(contradictory)}")
            lines.append("")
            
            # Supporting evidence for this cancer
            if supporting:
                lines.append(f"  Supporting evidence:")
                for j, ev in enumerate(supporting, 1):
                    lines.append(f"    [{j}] source: {ev.get('source_file', 'unknown')}")
                    lines.append(f"        snippet: {ev.get('snippet', '')}")
                    lines.append(f"        date: {ev.get('date', 'Not Reported')}")
                    lines.append(f"        confidence: {ev.get('confidence', 'N/A')}")
                    lines.append("")
            else:
                lines.append("  Supporting evidence: None")
                lines.append("")
            
            # Contradictory evidence for this cancer
            if contradictory:
                lines.append(f"  Contradictory evidence:")
                for j, ev in enumerate(contradictory, 1):
                    lines.append(f"    [{j}] source: {ev.get('source_file', 'unknown')}")
                    lines.append(f"        snippet: {ev.get('snippet', '')}")
                    lines.append(f"        date: {ev.get('date', 'Not Reported')}")
                    lines.append(f"        confidence: {ev.get('confidence', 'N/A')}")
                    lines.append("")
            
            lines.append("")
    
    elif consolidation_data:
        # For non-diagnosis fields, use aggregated data
        lines.append(f"final_value: {consolidation_data.get('final_value')}")
        lines.append(f"supporting_evidence_count: {consolidation_data.get('supporting_evidence_count')}")
        lines.append(f"contradictory_evidence_count: {consolidation_data.get('contradictory_evidence_count')}")
        lines.append("")
        
        lines.append("Supporting evidence (ALL):")
        for i, ev in enumerate(consolidation_data.get("supporting_evidence", []), 1):
            lines.append(f"  [{i}] source: {ev.get('source_file')}")
            lines.append(f"      snippet: {ev.get('snippet', '')}")
            lines.append(f"      explanation: {ev.get('explanation', '')}")
            lines.append(f"      date: {ev.get('date', 'Not Reported')}")
            lines.append(f"      confidence: {ev.get('confidence')}")
            lines.append("")
        
        # Contradictory evidence
        contradictory = consolidation_data.get("contradictory_evidence", [])
        if contradictory:
            lines.append("")
            lines.append("Contradictory evidence (ALL):")
            for i, ev in enumerate(contradictory, 1):
                lines.append(f"  [{i}] source: {ev.get('source_file')}")
                lines.append(f"      snippet: {ev.get('snippet', '')}")
                lines.append(f"      explanation: {ev.get('explanation', '')}")
                lines.append(f"      date: {ev.get('date', 'Not Reported')}")
                lines.append(f"      confidence: {ev.get('confidence')}")
                lines.append("")
    else:
        lines.append("[WARN] NO CONSOLIDATION DATA FOUND FOR THIS FIELD")
        lines.append("")
    
    # Write file
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  * {output_path.name}")


def get_domain_for_field(naaccr_name: str) -> Tuple[str, str]:
    """Get domain name and title for a field."""
    for domain_name, domain_info in DOMAIN_STRUCTURE.items():
        if naaccr_name in domain_info["fields"]:
            return domain_name, domain_info["title"]
    return "other", "OTHER"


def main():
    parser = argparse.ArgumentParser(description="Generate proof files")
    parser.add_argument("session_id", nargs="?", help="Session ID")
    parser.add_argument("--input-dir", help="Directory containing result files")
    parser.add_argument("--output-dir", help="Directory to save proof files")
    args = parser.parse_args()

    # Find base directory and paths
    base_dir = Path(__file__).parent.parent
    yaml_path = base_dir / "data" / "ontology" / "cancer_registry_fields.yaml"

    if args.input_dir:
        session_dir = Path(args.input_dir)
        session_id = args.session_id or session_dir.name
    elif args.session_id:
        session_id = args.session_id
        session_dir = base_dir / "data" / "output" / session_id
    else:
        print("Usage: python scripts/generate_proof.py <session_id> [--input-dir <dir>]")
        sys.exit(1)
    
    if not session_dir.exists():
        print(f"[ERROR] Session directory not found: {session_dir}")
        sys.exit(1)
    
    print(f"[DIR] Session: {session_id}")
    print(f"      Path: {session_dir}")
    
    # Load YAML ontology
    yaml_fields, yaml_domains = load_yaml_ontology(yaml_path)
    if not yaml_fields:
        print(f"[WARN] Could not load fields from {yaml_path}")
        print("       Falling back to hardcoded NAACCR_TO_MCODE mapping")
        yaml_fields = list(NAACCR_TO_MCODE.keys())
    else:
        print(f"[YAML] Loaded {len(yaml_fields)} fields from ontology: {yaml_path.name}")
        print(f"       Domains: {', '.join(yaml_domains.keys())}")
    
    # Find and load files
    resolve_path = find_resolve_file(session_dir)
    if not resolve_path:
        print("[WARN] Could not find resolve_patient_fields.json - Resolve section will be empty")
        resolve_data = [] # Empty list if missing
    else:
        print(f"      Resolve file: {resolve_path.name}")
        resolve_data = load_json(resolve_path)
    
    consolidation_path = find_consolidation_file(session_dir)
    if not consolidation_path:
        print("[WARN] Could not find consolidation file")
        consolidation_data = None
    else:
        print(f"      Consolidation file: {consolidation_path.name}")
        consolidation_data = load_json(consolidation_path)
    
    if not resolve_data and not consolidation_data:
        print("[ERROR] No data found to process!")
        sys.exit(1)

    # Create proof directory
    if args.output_dir:
        proof_dir = Path(args.output_dir)
    else:
        proof_dir = base_dir / "proof" / session_id
    
    # Clean previous run
    if proof_dir.exists():
        print(f"[CLEAN] Removing existing proof directory: {proof_dir}")
        shutil.rmtree(proof_dir)
        
    proof_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[OUT] Output directory: {proof_dir}")
    
    # Extract fields and primary cancers
    resolve_fields = extract_resolve_fields(resolve_data) if resolve_data else {}
    consolidation_fields = extract_consolidation_fields(consolidation_data) if consolidation_data else {}
    primary_cancers = extract_primary_cancers(consolidation_data) if consolidation_data else []
    
    print(f"\n[INFO] Found {len(resolve_fields)} fields in resolve output")
    print(f"[INFO] Found {len(consolidation_fields)} fields in consolidation output")
    print(f"[INFO] Found {len(primary_cancers)} primary cancer(s) identified")
    
    # Generate summary file first
    print("\n[WRITE] Generating proof files...")
    summary_path = proof_dir / "00_summary.txt"
    generate_summary_file(primary_cancers, yaml_domains, summary_path)
    
    # Generate proof files - ONLY for YAML-defined fields
    generated_count = 1  # Count summary file
    for naaccr_name in yaml_fields:
        # Get mCODE name for output file (use mapping or same name)
        mcode_name = NAACCR_TO_MCODE.get(naaccr_name, naaccr_name)
        
        # Get domain info
        domain_name, domain_title = get_domain_for_field(naaccr_name)
        
        # Get resolve records (from NAACCR name)
        resolve_records = resolve_fields.get(naaccr_name, [])
        
        # Get consolidation data (from mCODE name)
        consolidation = consolidation_fields.get(mcode_name)
        
        # Skip if no data at all
        if not resolve_records and not consolidation:
            continue
        
        # Output file uses mCODE name
        output_path = proof_dir / f"{mcode_name}.txt"
        generate_proof_file(
            naaccr_name=naaccr_name,
            mcode_name=mcode_name,
            domain_name=domain_name,
            domain_title=domain_title,
            resolve_records=resolve_records,
            consolidation_data=consolidation,
            primary_cancers=primary_cancers,
            output_path=output_path
        )
        generated_count += 1
    
    print(f"\n[DONE] Generated {generated_count} proof files in {proof_dir}")
    print(f"       (Including summary + YAML-defined fields, using mCODE names)")


if __name__ == "__main__":
    main()
