<div align="center">
  <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQkzoKPaZIwIrnqHBYP_if-0vLt-hT6h2h-BQ&s" alt="John Snow Labs logo" width="96">
  <h1>Data Curation Service – DocETL Pipeline</h1>
  <p>LLM-driven oncology extraction pipeline that delivers registry-ready JSON artifacts.</p>
  <img alt="Project views" src="https://komarev.com/ghpvc/?username=huseyincenik&color=orange&label=Data+Curation+Views">
  <p>
    <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/fastapi/fastapi-original-wordmark.svg" alt="FastAPI" height="40">
    &nbsp;
    <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/python/python-original.svg" alt="Python" height="40">
    &nbsp;
    <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/docker/docker-original.svg" alt="Docker" height="40">
    &nbsp;
    <img src="https://upload.wikimedia.org/wikipedia/commons/4/4d/OpenAI_Logo.svg" alt="OpenAI" height="40">
    &nbsp;
    <img src="https://ollama.com/public/ollama.png" alt="Ollama/Qwen" height="40">
  </p>
</div>

This repository contains a complete DocETL-powered **Data Curation Service** that extracts structured oncology data from unstructured documents, normalizes it according to the `cancer_registry_fields.yaml` ontology, and produces patient-level summaries suitable for registry ingestion.

---

## Table of Contents

1. [Project Navigator](#project-navigator)
2. [High-Level Architecture](#high-level-architecture)
3. [DocETL Pipeline Deep Dive](#docetl-pipeline-deep-dive)
   - [Pipeline Flow Schematic](#pipeline-flow-schematic)
   - [Input Format](#input-format)
   - [Operator Details](#operator-details)
4. [Use Case Walkthrough](#use-case-walkthrough-documents-003--004)
5. [Blocking Keys & LLM Call Optimization](#blocking-keys--llm-call-optimization)
6. [Parallel Processing Architecture](#parallel-processing-architecture)
7. [LLM Models & OpenRouter Integration](#llm-models--openrouter-integration)
8. [Configuration Reference](#configuration-reference)
9. [FastAPI Service Architecture](#fastapi-service-architecture)
10. [Getting Started](#getting-started)
11. [Output Artifacts](#output-artifacts)
12. [References](#references)

---

## Project Navigator

- Back to portfolio home → [`../README.md`](../README.md)
- CoNLL generator & custom NER training → [`../generating_conll_files_from_pretrained_models`](../generating_conll_files_from_pretrained_models)
- RAG QA chatbot application → [`../rag_qa_chatbot_application`](../rag_qa_chatbot_application)

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                                      INGESTION LAYER                                        │
│  ┌─────────────────┐    ┌──────────────────┐    ┌─────────────────────────────────────────┐ │
│  │  REST API       │    │  File Upload     │    │  Sample Documents (TXT/PDF)            │ │
│  │  POST /process  │───▶│  POST /upload    │───▶│  input_patient_docs/*.txt              │ │
│  └─────────────────┘    └──────────────────┘    └─────────────────────────────────────────┘ │
└───────────────────────────────────────┬─────────────────────────────────────────────────────┘
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                                    DocETL ENGINE                                            │
│  ┌────────────┐   ┌────────────┐   ┌────────────┐   ┌────────────┐   ┌────────────────────┐ │
│  │  TAGGER    │──▶│    MAP     │──▶│  UNNEST    │──▶│  RESOLVE   │──▶│      REDUCE        │ │
│  │ Chronology │   │ Extraction │   │  Explode   │   │ Deduplicate│   │ Patient-Level      │ │
│  └────────────┘   └────────────┘   └────────────┘   └────────────┘   └────────────────────┘ │
│        │                │                │                │                    │            │
│        ▼                ▼                ▼                ▼                    ▼            │
│   Sorted Docs      Field JSON       Row per Field    Canonical Value    Patient Summary    │
└───────────────────────────────────────┬─────────────────────────────────────────────────────┘
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                              STRUCTURED OUTPUTS & OBSERVABILITY                             │
│  ┌─────────────────────────────┐  ┌─────────────────────┐  ┌──────────────────────────────┐ │
│  │  JSON Artifacts             │  │  Logs               │  │  Demo Artifacts              │ │
│  │  • tagger_result.json       │  │  • stage_*.log      │  │  • extraction_result.json    │ │
│  │  • extraction_result.json   │  │  • prompts.log      │  │  • consolidation_result.json │ │
│  │  • consolidation_result.json│  └─────────────────────┘  └──────────────────────────────┘ │
│  └─────────────────────────────┘                                                            │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## DocETL Pipeline Deep Dive

### Pipeline Flow Schematic

```
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│                           COMPLETE DocETL PIPELINE FLOW                                  │
├──────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                          │
│  ┌─────────────────┐                                                                     │
│  │ INPUT DOCUMENTS │  ← Raw medical documents (clinical notes, pathology, radiology)    │
│  │ (TXT/PDF)       │                                                                     │
│  └────────┬────────┘                                                                     │
│           │                                                                              │
│           ▼                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────────────────┐│
│  │ STAGE 1: TAGGER (Chronological Ordering)                                            ││
│  │ ┌─────────────────────────────────────────────────────────────────────────────────┐ ││
│  │ │ • Parse document metadata (patient_id, doc_id, doc_type, doc_date)              │ ││
│  │ │ • Sort documents chronologically per patient                                     │ ││
│  │ │ • Calculate confidence scores via LLM (type_confidence, date_confidence)        │ ││
│  │ │ • Output: TaggerResult with sorted TaggedDocument list                          │ ││
│  │ └─────────────────────────────────────────────────────────────────────────────────┘ ││
│  └────────┬────────────────────────────────────────────────────────────────────────────┘│
│           │                                                                              │
│           ▼                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────────────────┐│
│  │ STAGE 2: EXTRACTOR (DocETL Map + Normalize + Unnest)                                ││
│  │ ┌───────────────────────────────────────────────────────────────────────────────┐   ││
│  │ │ MAP OPERATOR: extract_clinical_fields                                         │   ││
│  │ │ • LLM extracts ALL NAACCR fields from each document                          │   ││
│  │ │ • Output: {extractions: [{field_name, raw_value, normalized_value, ...}]}    │   ││
│  │ └───────────────────────────────────────────────────────────────────────────────┘   ││
│  │                              ▼                                                       ││
│  │ ┌───────────────────────────────────────────────────────────────────────────────┐   ││
│  │ │ NORMALIZE OPERATOR: normalize_extractions (code_map)                          │   ││
│  │ │ • Ensures 'extractions' is always a valid list                                │   ││
│  │ │ • Handles JSON string parsing from tool calls                                 │   ││
│  │ │ • Fixes malformed LLM outputs                                                 │   ││
│  │ └───────────────────────────────────────────────────────────────────────────────┘   ││
│  │                              ▼                                                       ││
│  │ ┌───────────────────────────────────────────────────────────────────────────────┐   ││
│  │ │ UNNEST OPERATOR: explode_field_records                                        │   ││
│  │ │ • Flattens extractions array into individual rows                             │   ││
│  │ │ • Each row: one field with patient_id, doc_id, evidence                       │   ││
│  │ │ • Enables field-level resolution in next stage                                │   ││
│  │ └───────────────────────────────────────────────────────────────────────────────┘   ││
│  └────────┬────────────────────────────────────────────────────────────────────────────┘│
│           │                                                                              │
│           ▼                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────────────────┐│
│  │ STAGE 3: CONSOLIDATOR (DocETL Resolve + Reduce)                                     ││
│  │ ┌───────────────────────────────────────────────────────────────────────────────┐   ││
│  │ │ RESOLVE OPERATOR: resolve_patient_fields                                      │   ││
│  │ │ • Uses blocking_keys: [patient_id, field_name] to group records               │   ││
│  │ │ • Compares values via comparison_prompt (LLM pairwise matching)               │   ││
│  │ │ • Resolves conflicts via resolution_prompt (picks canonical value)            │   ││
│  │ │ • Output: One resolved value per (patient_id, field_name) pair                │   ││
│  │ └───────────────────────────────────────────────────────────────────────────────┘   ││
│  │                              ▼                                                       ││
│  │ ┌───────────────────────────────────────────────────────────────────────────────┐   ││
│  │ │ REDUCE OPERATOR: reduce_patient_summary                                       │   ││
│  │ │ • Groups all resolved fields by patient_id                                    │   ││
│  │ │ • Generates mCODE-compliant patient registry                                  │   ││
│  │ │ • Creates patient_summary narrative                                           │   ││
│  │ │ • Identifies primary_cancers with site, histology, date                       │   ││
│  │ └───────────────────────────────────────────────────────────────────────────────┘   ││
│  └────────┬────────────────────────────────────────────────────────────────────────────┘│
│           │                                                                              │
│           ▼                                                                              │
│  ┌─────────────────┐                                                                     │
│  │ OUTPUT: mCODE   │  → Patient-level JSON with consolidated_fields, primary_cancers    │
│  │ Patient Record  │                                                                     │
│  └─────────────────┘                                                                     │
│                                                                                          │
└──────────────────────────────────────────────────────────────────────────────────────────┘
```

### Pipeline Stage Details with Examples

| Stage | Source File | Output File | Sample Data |
|-------|-------------|-------------|-------------|
| **INPUT** | `input_patient_docs/*.txt` | - | Raw TXT/PDF documents |
| **TAGGER** | `src/pipeline/tagger.py` | `tagger_result.json` | Sorted documents with confidence |
| **EXTRACTOR** | `src/pipeline/extractor.py` | `extraction_result.json` | NAACCR field extractions |
| **CONSOLIDATOR** | `src/pipeline/consolidator.py` | `consolidation_result.json` | mCODE patient records |

---

#### 📁 INPUT: Raw Documents

**File Path:** `input_patient_docs/jsl_p01_003_radiology_doc.txt`

```text
===== Document 003 =====
Patient Id: p01
Doc Id: 003
Doc Type: radiology
Date: 2015-10-01
Title: NM BONE SCAN, WHOLE BODY
---
CLINICAL STATEMENT: Prostate cancer, Gleason score 7. Staging evaluation.

IMPRESSION: No scintigraphic evidence of osseous metastatic disease.
```

---

#### 📁 STAGE 1: TAGGER Output

**Source:** `src/pipeline/tagger.py` → **Output:** `demo_runs/*/tagger_result.json`

```json
{
  "patient_id": "p01",
  "documents": [
    {
      "doc_id": "doc_003_jsl_p01_003_radiology_doc",
      "doc_type": "radiology",
      "doc_date": "2015-10-01",
      "type_confidence": 0.95,
      "date_confidence": 0.92,
      "sort_order": 1
    },
    {
      "doc_id": "doc_004_jsl_p01_004_clinical_doc",
      "doc_type": "clinical",
      "doc_date": "2015-10-15",
      "type_confidence": 0.97,
      "date_confidence": 0.95,
      "sort_order": 2
    }
  ],
  "total_documents": 2,
  "processing_time_ms": 1250
}
```

---

#### 📁 STAGE 2: EXTRACTOR Output

**Source:** `src/pipeline/extractor.py` + `src/pipeline/docetl_runner.py` → **Output:** `demo_runs/*/extraction_result.json`

```json
{
  "patient_id": "p01",
  "doc_id": "doc_003_jsl_p01_003_radiology_doc",
  "extractions": [
    {
      "field_name": "ca_site",
      "category": "diagnosis",
      "raw_value": "Prostate cancer",
      "normalized_value": "Prostate (C61.9)/Malignant",
      "vocabulary_code": "C61.9",
      "reasoning_excerpt": "CLINICAL STATEMENT: Prostate cancer, Gleason score 7",
      "confidence_score": 0.92,
      "inferred": false
    },
    {
      "field_name": "ca_clinical_m_stage",
      "category": "clinical_staging",
      "raw_value": "No osseous metastatic disease",
      "normalized_value": "cM0",
      "vocabulary_code": "cM0",
      "reasoning_excerpt": "No scintigraphic evidence of osseous metastatic disease",
      "confidence_score": 0.88,
      "inferred": true
    },
    {
      "field_name": "ecog",
      "category": "performance",
      "raw_value": "Not Reported",
      "normalized_value": "Not Reported",
      "reasoning_excerpt": "",
      "confidence_score": 0.35,
      "inferred": false
    }
  ]
}
```

---

#### 📁 STAGE 3: CONSOLIDATOR Output

**Source:** `src/pipeline/consolidator.py` → **Output:** `demo_runs/*/consolidation_result.json`

```json
{
  "patient_id": "p01",
  "consolidated_fields": [
    {
      "field_name": "ca_site",
      "category": "diagnosis",
      "normalized_value": "Prostate (C61.9)/Malignant",
      "resolved_value": "C61.9 - Prostate, primary malignant site",
      "confidence_score": 0.95,
      "supporting_docs": [
        {"doc_id": "doc_003", "doc_type": "radiology", "doc_date": "2015-10-01"},
        {"doc_id": "doc_004", "doc_type": "clinical", "doc_date": "2015-10-15"}
      ],
      "consolidation_notes": "2 agreeing sources (radiology + clinical)"
    },
    {
      "field_name": "naaccr_diagnosis_dt",
      "category": "diagnosis",
      "normalized_value": "2015-07-15",
      "resolved_value": "2015-07-15 - Initial diagnosis date",
      "confidence_score": 0.92
    }
  ],
  "primary_cancers": [
    {
      "site": "C61.9 - Prostate",
      "histology": "8140/3 - Adenocarcinoma, NOS",
      "diagnosis_date": "2015-07-15",
      "staging": "Gleason 3+4=7, unfavorable intermediate-risk"
    }
  ],
  "patient_summary": "60-year-old male with unfavorable intermediate-risk prostate cancer (Gleason 3+4=7, PSA 15.7 ng/mL) diagnosed July 2015. Bone scan negative for metastatic disease. ECOG 1."
}
```

---

### Key File Paths Reference

| Category | File Path | Description |
|----------|-----------|-------------|
| **Ontology** | `cancer_registry_fields.yaml` | NAACCR field definitions |
| **Settings** | `config/settings.py` | Application configuration |
| **Environment** | `config/.env` | API keys and secrets |
| **Tagger** | `src/pipeline/tagger.py` | Document classification & sorting |
| **Extractor** | `src/pipeline/extractor.py` | Field extraction orchestration |
| **DocETL Runner** | `src/pipeline/docetl_runner.py` | Map/Unnest/Resolve/Reduce operators |
| **Consolidator** | `src/pipeline/consolidator.py` | Patient-level aggregation |
| **API Routes** | `src/api/routes.py` | FastAPI endpoints |
| **Main App** | `src/main.py` | FastAPI application entry |
| **Sample Docs** | `input_patient_docs/*.txt` | Raw medical documents |
| **Demo Outputs** | `demo_runs/demo_run_*/` | Pipeline run artifacts |

---

### LLM Prompts Reference

The following prompts are used by DocETL operators to interact with LLM models. These are defined in `src/pipeline/docetl_runner.py`.

#### 🔹 MAP PROMPT (extract_clinical_fields)

Used by the **Map Operator** to extract NAACCR fields from each document:

```text
You are a certified oncology registrar. Extract every NAACCR field exactly as
defined in the ontology below and emit strictly valid JSON.

⚠️ ANTI-HALLUCINATION RULE: ONLY extract values that ACTUALLY APPEAR in the document.
DO NOT invent or guess cancer types, sites, or staging values not mentioned in text.
If a value is not in the document, use "Not Reported" - never make up values.

⚠️ DO NOT COPY EXAMPLES: The examples in this prompt are for illustration ONLY.
NEVER copy these examples into your output unless they exist verbatim in the document.

ONTOLOGY FIELDS TO EXTRACT:
├── naaccr_diagnosis_dt (Date of diagnosis in YYYY-MM-DD)
├── ca_site (Anatomical site with ICD-O-3 code)
├── naaccr_histology_cd (Histology/morphology code)
├── ca_clinical_t_stage (Clinical T stage)
├── ca_clinical_n_stage (Clinical N stage)
├── ca_clinical_m_stage (Clinical M stage)
├── ca_path_t_stage (Pathological T stage)
├── ca_path_n_stage (Pathological N stage)
├── ca_path_m_stage (Pathological M stage)
├── ca_gen_sum_stage_2 (SEER Summary Stage)
├── ecog (ECOG Performance Status 0-5)
└── kps (Karnofsky Performance Score 0-100)

CONFIDENCE SCORE CALIBRATION:
┌────────────────┬────────────────────────────────────────────────┬──────────┐
│ Score Range    │ Evidence Type                                  │ Frequency│
├────────────────┼────────────────────────────────────────────────┼──────────┤
│ 0.92-0.95      │ Explicit & verbatim (exact term in document)   │ RARE     │
│ 0.85-0.91      │ Explicit but interpreted (minor calculation)   │ COMMON   │
│ 0.75-0.84      │ Strong inference from clinical context         │ MEDIUM   │
│ 0.60-0.74      │ Moderate inference from indirect evidence      │ MEDIUM   │
│ 0.30-0.40      │ "Not Reported" or minimal evidence (HARD CAP)  │ LOW      │
└────────────────┴────────────────────────────────────────────────┴──────────┘

OUTPUT SCHEMA:
{
  "extractions": [
    {
      "field_name": "ca_site",
      "category": "diagnosis",
      "raw_value": "Prostate cancer",
      "normalized_value": "Prostate (C61.9)/Malignant",
      "vocabulary_code": "C61.9",
      "reasoning_excerpt": "EXACT quote from document",
      "confidence_score": 0.92,
      "inferred": false
    }
  ]
}
```

---

#### 🔹 COMPARISON PROMPT (resolve_patient_fields)

Used by the **Resolve Operator** for pairwise matching of candidate values:

```text
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

Respond with JSON: {"is_match": true} when both entries represent the same
registry fact after normalization, else {"is_match": false}.
```

**Example Input:**
```json
{
  "input1": {
    "field_name": "ca_site",
    "doc_id": "doc_003",
    "patient_id": "p01",
    "normalized_value": "Prostate (C61.9)/Malignant",
    "reasoning_excerpt": "Prostate cancer, Gleason score 7"
  },
  "input2": {
    "field_name": "ca_site",
    "doc_id": "doc_004",
    "patient_id": "p01",
    "normalized_value": "Prostate (C61.9)/Malignant",
    "reasoning_excerpt": "60-year-old male with prostate cancer"
  }
}
```

**Example Output:**
```json
{"is_match": true}
```

---

#### 🔹 RESOLUTION PROMPT (resolve_patient_fields)

Used by the **Resolve Operator** to pick the canonical value from matched clusters:

```text
You are consolidating oncology registry evidence for patient {{ inputs[0].patient_id }}
and field {{ inputs[0].field_name }}.

Evidence set:
{% for item in inputs %}
---
Document ID: {{ item.doc_id }}
Date: {{ item.doc_date or "Not Reported" }}
Type: {{ item.doc_type }}
Raw Value: "{{ item.raw_value }}"
Normalized Value: "{{ item.normalized_value }}"
Confidence: {{ item.confidence_score }}
Reasoning: "{{ item.reasoning_excerpt }}"
---
{% endfor %}

TASK: Resolve conflicts and determine the most reliable value.

CALIBRATION CHECKLIST:
1. Consistency Audit: Are values identical, compatible, or conflicting?
2. Source Tiering: Pathology > Operative > Imaging > Clinical > Administrative
3. Specificity & Timeliness: Prefer precise dates/codes and recent records
4. Conflict Penalty: Subtract 0.08 for each contradiction you override

CONFIDENCE SCORE ANCHORS:
┌────────────────┬────────────────────────────────────────────────────────────┐
│ 0.95-0.99      │ 3+ agreeing sources OR 2 high-quality identical language  │
│ 0.85-0.94      │ Clear agreement, limited sources                          │
│ 0.75-0.84      │ Majority consensus with minor interpretation              │
│ 0.60-0.74      │ Conflicts exist, one source clearly superior              │
│ 0.45-0.59      │ Significant ambiguity, pick one but warn                  │
│ 0.30-0.44      │ Barely any evidence, mostly inferred                      │
└────────────────┴────────────────────────────────────────────────────────────┘

CRITICAL RULES:
- If some docs have "Not Reported" but others have actual values, ALWAYS use actual values
- Do NOT merge values from different cancer sites
- RESOLVED_VALUE must start with CODE: "C61.9 - Prostate, primary site"

OUTPUT SCHEMA:
{
  "patient_id": "p01",
  "field_name": "ca_site",
  "normalized_value": "Prostate (C61.9)/Malignant",
  "resolved_value": "C61.9 - Prostate, primary malignant site",
  "confidence_score": 0.95,
  "supporting_docs": [
    {"doc_id": "doc_003", "doc_date": "2015-10-01", "doc_type": "radiology"}
  ],
  "consolidation_notes": "2 agreeing sources, boosted confidence"
}
```

---

#### 🔹 REDUCE PROMPT (reduce_patient_summary)

Used by the **Reduce Operator** to generate patient-level mCODE records:

```text
You are a certified oncology registrar consolidating patient-level data.

RESOLVED FIELD EXTRACTIONS FOR PATIENT {{ inputs[0].patient_id }}:

{% for field in inputs %}
Field: {{ field.field_name }} (Category: {{ field.category }})
- Normalized value: {{ field.normalized_value }}
- Resolved value: {{ field.resolved_value }}
- Confidence: {{ field.confidence_score }}
- Supporting docs: {{ field.supporting_docs | length }} documents
{% endfor %}

TASK: Generate patient-level mCODE registry with:
1. consolidated_fields array (all resolved fields)
2. primary_cancers array (one per unique cancer site)
3. patient_summary narrative (2-3 sentences)

MULTI-CANCER RULES:
- Each cancer gets its OWN entry in primary_cancers
- Do NOT mix diagnosis dates from different cancers
- Use site-appropriate histology codes

OUTPUT SCHEMA:
{
  "patient_id": "p01",
  "consolidated_fields": [...],
  "primary_cancers": [
    {
      "site": "C61.9 - Prostate",
      "histology": "8140/3 - Adenocarcinoma, NOS",
      "diagnosis_date": "2015-07-15",
      "staging": "Gleason 3+4=7, unfavorable intermediate-risk"
    }
  ],
  "patient_summary": "60-year-old male with unfavorable intermediate-risk..."
}
```

---

### Input Format

Documents are stored in `input_patient_docs/` with standardized header metadata:

```
===== Document 003 =====
Patient Id: p01
Doc Id: 003
Filename: p01/jsl_p01_003_radiology_doc.txt
Doc Type: radiology
Date: 2015-10-01
Title: NM BONE SCAN, WHOLE BODY
---
[Document content follows...]
```

**Parsed Into:**
```python
DocumentMetadata(
    patient_id="p01",
    doc_id="doc_003_jsl_p01_003_radiology_doc",
    doc_type="radiology",
    doc_date="2015-10-01",
    filename="input_patient_docs/jsl_p01_003_radiology_doc.txt",
    content="..."
)
```

### Operator Details Summary

| Operator | Type | Purpose | Input | Output |
|----------|------|---------|-------|--------|
| **Tagger** | Custom Stage | Chronological ordering + confidence scoring | Raw documents | Sorted TaggedDocument list |
| **Map** | `MapOp` | Extract NAACCR fields via LLM | Document content | `{extractions: [...]}` per doc |
| **Normalize** | `CodeMapOp` | Fix malformed extractions array | Map output | Clean list of extractions |
| **Unnest** | `UnnestOp` | Flatten extractions to rows | Normalized output | One row per field |
| **Resolve** | `ResolveOp` | Deduplicate & select canonical value | Unnested rows | One value per (patient, field) |
| **Reduce** | `ReduceOp` | Aggregate to patient-level record | Resolved values | mCODE patient record |

---

### Operator 1: MAP (extract_clinical_fields)

```
┌──────────────────────────────────────────────────────────────────────────────────────────────┐
│                              MAP OPERATOR: extract_clinical_fields                           │
├──────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                              │
│  PURPOSE: Extract ALL NAACCR oncology fields from each document using LLM                   │
│                                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────────────────────┐ │
│  │                                    INPUT                                               │ │
│  │  ┌──────────────────────────────────────────────────────────────────────────────────┐ │ │
│  │  │ DocumentMetadata {                                                               │ │ │
│  │  │   patient_id: "p01",                                                             │ │ │
│  │  │   doc_id: "doc_003_jsl_p01_003_radiology_doc",                                   │ │ │
│  │  │   doc_type: "radiology",                                                         │ │ │
│  │  │   doc_date: "2015-10-01",                                                        │ │ │
│  │  │   content: "CLINICAL STATEMENT: Prostate cancer, Gleason score 7..."            │ │ │
│  │  │ }                                                                                │ │ │
│  │  └──────────────────────────────────────────────────────────────────────────────────┘ │ │
│  └────────────────────────────────────────────────────────────────────────────────────────┘ │
│                                           │                                                  │
│                                           ▼                                                  │
│  ┌────────────────────────────────────────────────────────────────────────────────────────┐ │
│  │                               LLM PROMPT (Simplified)                                  │ │
│  │  ┌──────────────────────────────────────────────────────────────────────────────────┐ │ │
│  │  │ "You are a certified oncology registrar. Extract every NAACCR field:            │ │ │
│  │  │                                                                                  │ │ │
│  │  │  ONTOLOGY FIELDS TO EXTRACT:                                                     │ │ │
│  │  │  ├── naaccr_diagnosis_dt (Date of diagnosis in YYYY-MM-DD)                       │ │ │
│  │  │  ├── ca_site (Anatomical site with ICD-O-3 code)                                 │ │ │
│  │  │  ├── naaccr_histology_cd (Histology/morphology code)                             │ │ │
│  │  │  ├── ca_clinical_t_stage (Clinical T stage)                                      │ │ │
│  │  │  ├── ca_clinical_n_stage (Clinical N stage)                                      │ │ │
│  │  │  ├── ca_clinical_m_stage (Clinical M stage)                                      │ │ │
│  │  │  ├── ca_path_t_stage (Pathological T stage)                                      │ │ │
│  │  │  ├── ca_path_n_stage (Pathological N stage)                                      │ │ │
│  │  │  ├── ca_path_m_stage (Pathological M stage)                                      │ │ │
│  │  │  ├── ca_gen_sum_stage_2 (SEER Summary Stage)                                     │ │ │
│  │  │  ├── ecog (ECOG Performance Status 0-5)                                          │ │ │
│  │  │  └── kps (Karnofsky Performance Score 0-100)                                     │ │ │
│  │  │                                                                                  │ │ │
│  │  │  For EACH field, return:                                                         │ │ │
│  │  │  - field_name, raw_value, normalized_value                                       │ │ │
│  │  │  - reasoning_excerpt (EXACT quote from document)                                 │ │ │
│  │  │  - confidence_score (0.0-1.0 with calibration rules)                             │ │ │
│  │  │  - inferred (true/false)"                                                        │ │ │
│  │  └──────────────────────────────────────────────────────────────────────────────────┘ │ │
│  └────────────────────────────────────────────────────────────────────────────────────────┘ │
│                                           │                                                  │
│                                           ▼                                                  │
│  ┌────────────────────────────────────────────────────────────────────────────────────────┐ │
│  │                                   OUTPUT                                               │ │
│  │  ┌──────────────────────────────────────────────────────────────────────────────────┐ │ │
│  │  │ {                                                                                │ │ │
│  │  │   "extractions": [                                                               │ │ │
│  │  │     {                                                                            │ │ │
│  │  │       "field_name": "ca_site",                                                   │ │ │
│  │  │       "category": "diagnosis",                                                   │ │ │
│  │  │       "raw_value": "Prostate cancer",                                            │ │ │
│  │  │       "normalized_value": "Prostate (C61.9)/Malignant",                          │ │ │
│  │  │       "reasoning_excerpt": "CLINICAL STATEMENT: Prostate cancer, Gleason...",    │ │ │
│  │  │       "confidence_score": 0.92,                                                  │ │ │
│  │  │       "inferred": false                                                          │ │ │
│  │  │     },                                                                           │ │ │
│  │  │     {                                                                            │ │ │
│  │  │       "field_name": "ca_clinical_m_stage",                                       │ │ │
│  │  │       "raw_value": "No osseous metastatic disease",                              │ │ │
│  │  │       "normalized_value": "cM0",                                                 │ │ │
│  │  │       "reasoning_excerpt": "No scintigraphic evidence of osseous metastases",    │ │ │
│  │  │       "confidence_score": 0.88,                                                  │ │ │
│  │  │       "inferred": true                                                           │ │ │
│  │  │     },                                                                           │ │ │
│  │  │     ... (one entry per ontology field)                                           │ │ │
│  │  │   ]                                                                              │ │ │
│  │  │ }                                                                                │ │ │
│  │  └──────────────────────────────────────────────────────────────────────────────────┘ │ │
│  └────────────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                              │
│  CONFIDENCE SCORE CALIBRATION:                                                               │
│  ┌────────────────────────────────────────────────────────────────────────────────────────┐ │
│  │ 0.92-0.95  │ Explicit & verbatim (exact term appears in document)        │ RARE      │ │
│  │ 0.85-0.91  │ Explicit but interpreted (needs minor calculation)          │ COMMON    │ │
│  │ 0.75-0.84  │ Strong inference from clinical context                      │ MEDIUM    │ │
│  │ 0.60-0.74  │ Moderate inference from indirect evidence                   │ MEDIUM    │ │
│  │ 0.30-0.40  │ "Not Reported" or minimal evidence (HARD CAP: 0.40)         │ LOW       │ │
│  └────────────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                              │
└──────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

### Operator 2: NORMALIZE (normalize_extractions)

```
┌──────────────────────────────────────────────────────────────────────────────────────────────┐
│                          NORMALIZE OPERATOR: normalize_extractions                           │
├──────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                              │
│  PURPOSE: Ensure 'extractions' is ALWAYS a valid list (fix malformed LLM outputs)           │
│  TYPE: CodeMapOp (Python code transformation, no LLM call)                                   │
│                                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────────────────────┐ │
│  │                           PROBLEM CASES HANDLED                                        │ │
│  │                                                                                        │ │
│  │  CASE 1: extractions is a JSON string (from tool calls)                                │ │
│  │  ┌──────────────────────────────────────────────────────────────────────────────────┐ │ │
│  │  │ INPUT:  {"extractions": "[{\"field_name\": \"ca_site\"...}]"}  ← string!         │ │ │
│  │  │ OUTPUT: {"extractions": [{"field_name": "ca_site"...}]}       ← parsed list      │ │ │
│  │  └──────────────────────────────────────────────────────────────────────────────────┘ │ │
│  │                                                                                        │ │
│  │  CASE 2: extractions is None                                                           │ │
│  │  ┌──────────────────────────────────────────────────────────────────────────────────┐ │ │
│  │  │ INPUT:  {"extractions": null}                                                    │ │ │
│  │  │ OUTPUT: {"extractions": []}                                                      │ │ │
│  │  └──────────────────────────────────────────────────────────────────────────────────┘ │ │
│  │                                                                                        │ │
│  │  CASE 3: extractions is a single dict (not array)                                      │ │
│  │  ┌──────────────────────────────────────────────────────────────────────────────────┐ │ │
│  │  │ INPUT:  {"extractions": {"field_name": "ca_site"...}}                            │ │ │
│  │  │ OUTPUT: {"extractions": [{"field_name": "ca_site"...}]}                          │ │ │
│  │  └──────────────────────────────────────────────────────────────────────────────────┘ │ │
│  │                                                                                        │ │
│  │  CASE 4: Nested extractions object                                                     │ │
│  │  ┌──────────────────────────────────────────────────────────────────────────────────┐ │ │
│  │  │ INPUT:  {"extractions": {"extractions": [...]}}  ← double nested                 │ │ │
│  │  │ OUTPUT: {"extractions": [...]}                   ← flattened                     │ │ │
│  │  └──────────────────────────────────────────────────────────────────────────────────┘ │ │
│  └────────────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                              │
│  PYTHON CODE (simplified):                                                                   │
│  ┌────────────────────────────────────────────────────────────────────────────────────────┐ │
│  │ def transform(doc: dict) -> dict:                                                      │ │
│  │     if 'extractions' not in doc:                                                       │ │
│  │         doc['extractions'] = []                                                        │ │
│  │     elif isinstance(doc['extractions'], str):                                          │ │
│  │         doc['extractions'] = json.loads(doc['extractions'])                            │ │
│  │     elif isinstance(doc['extractions'], dict):                                         │ │
│  │         doc['extractions'] = [doc['extractions']]                                      │ │
│  │     return doc                                                                         │ │
│  └────────────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                              │
└──────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

### Operator 3: UNNEST (explode_field_records)

```
┌──────────────────────────────────────────────────────────────────────────────────────────────┐
│                            UNNEST OPERATOR: explode_field_records                            │
├──────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                              │
│  PURPOSE: Flatten the extractions array so each field becomes an individual row             │
│  TYPE: UnnestOp (no LLM call, pure data transformation)                                      │
│                                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────────────────────┐ │
│  │                                 TRANSFORMATION                                         │ │
│  │                                                                                        │ │
│  │  INPUT (1 record with array):                                                          │ │
│  │  ┌──────────────────────────────────────────────────────────────────────────────────┐ │ │
│  │  │ {                                                                                │ │ │
│  │  │   "patient_id": "p01",                                                           │ │ │
│  │  │   "doc_id": "doc_003",                                                           │ │ │
│  │  │   "doc_type": "radiology",                                                       │ │ │
│  │  │   "doc_date": "2015-10-01",                                                      │ │ │
│  │  │   "extractions": [                                                               │ │ │
│  │  │     {"field_name": "ca_site", "normalized_value": "Prostate (C61.9)", ...},      │ │ │
│  │  │     {"field_name": "ca_clinical_m_stage", "normalized_value": "cM0", ...},       │ │ │
│  │  │     {"field_name": "ecog", "normalized_value": "Not Reported", ...}              │ │ │
│  │  │   ]                                                                              │ │ │
│  │  │ }                                                                                │ │ │
│  │  └───────────────────────────────────────┬──────────────────────────────────────────┘ │ │
│  │                                          │                                            │ │
│  │                                          ▼  UNNEST                                    │ │
│  │                                                                                        │ │
│  │  OUTPUT (3 separate records):                                                          │ │
│  │  ┌──────────────────────────────────────────────────────────────────────────────────┐ │ │
│  │  │ ROW 1: {                                                                         │ │ │
│  │  │   "patient_id": "p01", "doc_id": "doc_003", "doc_type": "radiology",             │ │ │
│  │  │   "doc_date": "2015-10-01",                                                      │ │ │
│  │  │   "field_name": "ca_site",                                                       │ │ │
│  │  │   "normalized_value": "Prostate (C61.9)",                                        │ │ │
│  │  │   "confidence_score": 0.92, ...                                                  │ │ │
│  │  │ }                                                                                │ │ │
│  │  ├──────────────────────────────────────────────────────────────────────────────────┤ │ │
│  │  │ ROW 2: {                                                                         │ │ │
│  │  │   "patient_id": "p01", "doc_id": "doc_003", "doc_type": "radiology",             │ │ │
│  │  │   "doc_date": "2015-10-01",                                                      │ │ │
│  │  │   "field_name": "ca_clinical_m_stage",                                           │ │ │
│  │  │   "normalized_value": "cM0",                                                     │ │ │
│  │  │   "confidence_score": 0.88, ...                                                  │ │ │
│  │  │ }                                                                                │ │ │
│  │  ├──────────────────────────────────────────────────────────────────────────────────┤ │ │
│  │  │ ROW 3: {                                                                         │ │ │
│  │  │   "patient_id": "p01", "doc_id": "doc_003", "doc_type": "radiology",             │ │ │
│  │  │   "doc_date": "2015-10-01",                                                      │ │ │
│  │  │   "field_name": "ecog",                                                          │ │ │
│  │  │   "normalized_value": "Not Reported",                                            │ │ │
│  │  │   "confidence_score": 0.35, ...                                                  │ │ │
│  │  │ }                                                                                │ │ │
│  │  └──────────────────────────────────────────────────────────────────────────────────┘ │ │
│  └────────────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                              │
│  CONFIGURATION:                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────────────────────┐ │
│  │ UnnestOp(                                                                              │ │
│  │   name="explode_field_records",                                                        │ │
│  │   unnest_key="extractions",           # Array to flatten                               │ │
│  │   expand_fields=[                     # Fields to promote to top level                 │ │
│  │     "field_name", "category", "data_type", "raw_value",                                │ │
│  │     "normalized_value", "units", "vocabulary_code",                                    │ │
│  │     "reasoning_excerpt", "explanation", "confidence_level",                            │ │
│  │     "confidence_score", "inferred", "related_entities"                                 │ │
│  │   ],                                                                                   │ │
│  │   keep_empty=False,                   # Drop records with empty extractions            │ │
│  │   recursive=True,                     # Handle nested structures                       │ │
│  │   depth=2                             # Max nesting depth                              │ │
│  │ )                                                                                      │ │
│  └────────────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                              │
│  WHY UNNEST?                                                                                 │
│  ┌────────────────────────────────────────────────────────────────────────────────────────┐ │
│  │ Before Unnest: 50 documents × 1 record each = 50 records                               │ │
│  │ After Unnest:  50 documents × 12 fields each = 600 rows                                │ │
│  │                                                                                        │ │
│  │ This enables FIELD-LEVEL resolution in the next stage:                                 │ │
│  │ • Compare ca_site values across all documents                                          │ │
│  │ • Compare diagnosis_date values across all documents                                   │ │
│  │ • Each field resolved independently                                                    │ │
│  └────────────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                              │
└──────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

### Operator 4: RESOLVE (resolve_patient_fields)

```
┌──────────────────────────────────────────────────────────────────────────────────────────────┐
│                          RESOLVE OPERATOR: resolve_patient_fields                            │
├──────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                              │
│  PURPOSE: Deduplicate and select the CANONICAL value for each (patient_id, field_name)      │
│  TYPE: ResolveOp (uses LLM for comparison and resolution)                                    │
│                                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────────────────────┐ │
│  │                              BLOCKING KEYS (Critical!)                                 │ │
│  │                                                                                        │ │
│  │  blocking_keys = ["patient_id", "field_name"]                                          │ │
│  │                                                                                        │ │
│  │  Records are ONLY compared within the same (patient_id, field_name) group:             │ │
│  │                                                                                        │ │
│  │  ALL UNNESTED ROWS (600):                   GROUPED BY BLOCKING KEYS:                  │ │
│  │  ┌─────────────────────────┐               ┌─────────────────────────────────┐        │ │
│  │  │ p01 / ca_site / doc001  │               │ GROUP (p01, ca_site):           │        │ │
│  │  │ p01 / ca_stage / doc001 │     ────►     │   doc001, doc003, doc015, ...   │        │ │
│  │  │ p01 / ca_site / doc003  │               │   Compare ONLY within group     │        │ │
│  │  │ p01 / ecog / doc005     │               └─────────────────────────────────┘        │ │
│  │  │ ...                     │               ┌─────────────────────────────────┐        │ │
│  │  └─────────────────────────┘               │ GROUP (p01, ca_stage):          │        │ │
│  │                                            │   doc001, doc002, ...           │        │ │
│  │  ❌ NO cross-field comparisons             └─────────────────────────────────┘        │ │
│  │  ❌ NO cross-patient comparisons                                                       │ │
│  └────────────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────────────────────┐ │
│  │                         STEP 1: COMPARISON PROMPT (Pairwise)                           │ │
│  │                                                                                        │ │
│  │  For each pair of records in the same group, LLM determines if they match:             │ │
│  │                                                                                        │ │
│  │  ┌──────────────────────────────────────────────────────────────────────────────────┐ │ │
│  │  │ "You are comparing two candidate values for the same oncology registry field.   │ │ │
│  │  │                                                                                  │ │ │
│  │  │  Field 1 (ca_site) from doc_003:                                                 │ │ │
│  │  │  - Value: Prostate (C61.9)/Malignant                                             │ │ │
│  │  │  - Evidence: 'CLINICAL STATEMENT: Prostate cancer, Gleason score 7'              │ │ │
│  │  │                                                                                  │ │ │
│  │  │  Field 2 (ca_site) from doc_004:                                                 │ │ │
│  │  │  - Value: Prostate (C61.9)/Malignant                                             │ │ │
│  │  │  - Evidence: '60-year-old male with prostate cancer'                             │ │ │
│  │  │                                                                                  │ │ │
│  │  │  Respond with JSON: {\"is_match\": true} or {\"is_match\": false}"               │ │ │
│  │  └──────────────────────────────────────────────────────────────────────────────────┘ │ │
│  │                                                                                        │ │
│  │  LLM Response: {"is_match": true}  ✓ Same value, merge into cluster                   │ │
│  └────────────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────────────────────┐ │
│  │                      STEP 2: RESOLUTION PROMPT (Per Cluster)                           │ │
│  │                                                                                        │ │
│  │  For each cluster of matching records, LLM picks the canonical value:                  │ │
│  │                                                                                        │ │
│  │  ┌──────────────────────────────────────────────────────────────────────────────────┐ │ │
│  │  │ "You are consolidating oncology registry evidence for patient p01                │ │ │
│  │  │  and field ca_site.                                                              │ │ │
│  │  │                                                                                  │ │ │
│  │  │  Evidence set:                                                                   │ │ │
│  │  │  ---                                                                             │ │ │
│  │  │  Document ID: doc_003, Date: 2015-10-01, Type: radiology                         │ │ │
│  │  │  Raw Value: 'Prostate cancer', Normalized: 'Prostate (C61.9)/Malignant'          │ │ │
│  │  │  Confidence: 0.92                                                                │ │ │
│  │  │  ---                                                                             │ │ │
│  │  │  Document ID: doc_004, Date: 2015-10-15, Type: clinical                          │ │ │
│  │  │  Raw Value: 'prostate cancer', Normalized: 'Prostate (C61.9)/Malignant'          │ │ │
│  │  │  Confidence: 0.94                                                                │ │ │
│  │  │  ---                                                                             │ │ │
│  │  │                                                                                  │ │ │
│  │  │  TASK: Resolve conflicts and determine the most reliable value.                  │ │ │
│  │  │                                                                                  │ │ │
│  │  │  Source Tiering: Pathology > Operative > Imaging > Clinical > Admin              │ │ │
│  │  │  Confidence Anchors:                                                             │ │ │
│  │  │  - 0.95-0.99: 3+ agreeing sources OR 2 high-quality identical                    │ │ │
│  │  │  - 0.85-0.94: Clear agreement, limited sources                                   │ │ │
│  │  │  - 0.60-0.74: Conflicts exist, one source clearly superior"                      │ │ │
│  │  └──────────────────────────────────────────────────────────────────────────────────┘ │ │
│  └────────────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────────────────────┐ │
│  │                                    OUTPUT                                              │ │
│  │  ┌──────────────────────────────────────────────────────────────────────────────────┐ │ │
│  │  │ {                                                                                │ │ │
│  │  │   "patient_id": "p01",                                                           │ │ │
│  │  │   "field_name": "ca_site",                                                       │ │ │
│  │  │   "normalized_value": "Prostate (C61.9)/Malignant",                              │ │ │
│  │  │   "resolved_value": "C61.9 - Prostate, Malignant primary site",                  │ │ │
│  │  │   "confidence_score": 0.95,                                                      │ │ │
│  │  │   "supporting_docs": [                                                           │ │ │
│  │  │     {"doc_id": "doc_003", "doc_date": "2015-10-01", "doc_type": "radiology"},    │ │ │
│  │  │     {"doc_id": "doc_004", "doc_date": "2015-10-15", "doc_type": "clinical"}      │ │ │
│  │  │   ],                                                                             │ │ │
│  │  │   "consolidation_notes": "2 agreeing sources (radiology + clinical). Boosted    │ │ │
│  │  │                          confidence due to identical normalized values."         │ │ │
│  │  │ }                                                                                │ │ │
│  │  └──────────────────────────────────────────────────────────────────────────────────┘ │ │
│  │                                                                                        │ │
│  │  ONE resolved record per (patient_id, field_name) pair                                 │ │
│  └────────────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                              │
│  CONFLICT RESOLUTION STRATEGIES:                                                             │
│  ┌────────────────────────────────────────────────────────────────────────────────────────┐ │
│  │ Scenario                        │ Resolution Strategy                                  │ │
│  │ ────────────────────────────────┼──────────────────────────────────────────────────── │ │
│  │ Values AGREE                    │ Use highest confidence, boost score                 │ │
│  │ Values CONFLICT                 │ Prefer: Pathology > Operative > Imaging > Clinical │ │
│  │ "Not Reported" vs Actual Value  │ ALWAYS use the actual clinical value               │ │
│  │ Multiple Cancers (different     │ Preserve separate entries per cancer site          │ │
│  │   sites like Colon vs Prostate) │ (do NOT merge different cancers)                   │ │
│  │ Date conflicts                  │ Use EARLIEST valid diagnosis date                  │ │
│  │ Stage conflicts                 │ Prefer HIGHEST stage (pT3 > pT2)                   │ │
│  └────────────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                              │
└──────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

### Operator 5: REDUCE (reduce_patient_summary)

```
┌──────────────────────────────────────────────────────────────────────────────────────────────┐
│                            REDUCE OPERATOR: reduce_patient_summary                           │
├──────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                              │
│  PURPOSE: Aggregate all resolved fields into a PATIENT-LEVEL mCODE registry record          │
│  TYPE: ReduceOp (groups by patient_id, uses LLM for final consolidation)                     │
│                                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────────────────────┐ │
│  │                                  GROUPING                                              │ │
│  │                                                                                        │ │
│  │  reduce_key = "patient_id"                                                             │ │
│  │                                                                                        │ │
│  │  RESOLVED RECORDS (12 per patient):          GROUPED BY patient_id:                    │ │
│  │  ┌───────────────────────────────┐          ┌──────────────────────────────────┐      │ │
│  │  │ p01 / ca_site / resolved      │          │ PATIENT p01:                     │      │ │
│  │  │ p01 / ca_stage / resolved     │   ────►  │   ca_site, ca_stage, diag_dt,   │      │ │
│  │  │ p01 / diagnosis_dt / resolved │          │   histology, ecog, kps, ...     │      │ │
│  │  │ p01 / histology / resolved    │          │   (ALL 12 fields together)       │      │ │
│  │  │ p01 / ecog / resolved         │          └──────────────────────────────────┘      │ │
│  │  │ ...                           │                                                     │ │
│  │  └───────────────────────────────┘                                                     │ │
│  └────────────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────────────────────┐ │
│  │                              LLM PROMPT (Simplified)                                   │ │
│  │  ┌──────────────────────────────────────────────────────────────────────────────────┐ │ │
│  │  │ "You are a certified oncology registrar consolidating patient-level data.       │ │ │
│  │  │                                                                                  │ │ │
│  │  │  RESOLVED FIELD EXTRACTIONS FOR PATIENT p01:                                     │ │ │
│  │  │                                                                                  │ │ │
│  │  │  Field: ca_site (Category: diagnosis)                                            │ │ │
│  │  │  - Normalized value: Prostate (C61.9)/Malignant                                  │ │ │
│  │  │  - Confidence: 0.95                                                              │ │ │
│  │  │  - Supporting docs: doc_003 (radiology), doc_004 (clinical)                      │ │ │
│  │  │                                                                                  │ │ │
│  │  │  Field: naaccr_diagnosis_dt (Category: diagnosis)                                │ │ │
│  │  │  - Normalized value: 2015-07-15                                                  │ │ │
│  │  │  - Confidence: 0.92                                                              │ │ │
│  │  │                                                                                  │ │ │
│  │  │  Field: ecog (Category: performance)                                             │ │ │
│  │  │  - Normalized value: ECOG 1                                                      │ │ │
│  │  │  - Confidence: 0.88                                                              │ │ │
│  │  │  ...                                                                             │ │ │
│  │  │                                                                                  │ │ │
│  │  │  TASK: Generate patient-level mCODE registry with:                               │ │ │
│  │  │  1. consolidated_fields array (all 12 fields)                                    │ │ │
│  │  │  2. primary_cancers array (one per unique cancer site)                           │ │ │
│  │  │  3. patient_summary narrative"                                                   │ │ │
│  │  └──────────────────────────────────────────────────────────────────────────────────┘ │ │
│  └────────────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────────────────────┐ │
│  │                              OUTPUT: mCODE PATIENT RECORD                              │ │
│  │  ┌──────────────────────────────────────────────────────────────────────────────────┐ │ │
│  │  │ {                                                                                │ │ │
│  │  │   "patient_id": "p01",                                                           │ │ │
│  │  │                                                                                  │ │ │
│  │  │   "consolidated_fields": [                                                       │ │ │
│  │  │     {                                                                            │ │ │
│  │  │       "field_name": "ca_site",                                                   │ │ │
│  │  │       "category": "diagnosis",                                                   │ │ │
│  │  │       "normalized_value": "Prostate (C61.9)/Malignant",                          │ │ │
│  │  │       "resolved_value": "C61.9 - Prostate, primary malignant site",              │ │ │
│  │  │       "confidence_score": 0.95,                                                  │ │ │
│  │  │       "consolidation_notes": "2 agreeing sources"                                │ │ │
│  │  │     },                                                                           │ │ │
│  │  │     {                                                                            │ │ │
│  │  │       "field_name": "naaccr_diagnosis_dt",                                       │ │ │
│  │  │       "normalized_value": "2015-07-15",                                          │ │ │
│  │  │       "confidence_score": 0.92                                                   │ │ │
│  │  │     },                                                                           │ │ │
│  │  │     ... (all 12 NAACCR fields)                                                   │ │ │
│  │  │   ],                                                                             │ │ │
│  │  │                                                                                  │ │ │
│  │  │   "primary_cancers": [                                                           │ │ │
│  │  │     {                                                                            │ │ │
│  │  │       "site": "C61.9 - Prostate",                                                │ │ │
│  │  │       "histology": "8140/3 - Adenocarcinoma, NOS",                               │ │ │
│  │  │       "diagnosis_date": "2015-07-15",                                            │ │ │
│  │  │       "staging": "Gleason 3+4=7, unfavorable intermediate-risk"                  │ │ │
│  │  │     },                                                                           │ │ │
│  │  │     {                                                                            │ │ │
│  │  │       "site": "C18.9 - Colon",                                                   │ │ │
│  │  │       "histology": "8480/3 - Mucinous adenocarcinoma",                           │ │ │
│  │  │       "diagnosis_date": "1987-05-12",                                            │ │ │
│  │  │       "staging": "Historical, treated with neoadjuvant chemoradiation"           │ │ │
│  │  │     }                                                                            │ │ │
│  │  │   ],                                                                             │ │ │
│  │  │                                                                                  │ │ │
│  │  │   "patient_summary": "60-year-old male with unfavorable intermediate-risk       │ │ │
│  │  │     prostate cancer (Gleason 3+4=7, PSA 15.7 ng/mL) diagnosed in July 2015.      │ │ │
│  │  │     Prior history of rectal/colon cancer in 1987 treated with neoadjuvant        │ │ │
│  │  │     chemoradiation (30 Gy) and APR. Known Lynch Syndrome. Current treatment       │ │ │
│  │  │     plan: EBRT 75.6 Gy in 42 fractions with 6 months ADT. Bone scan negative     │ │ │
│  │  │     for metastatic disease. ECOG performance status 1."                          │ │ │
│  │  │ }                                                                                │ │ │
│  │  └──────────────────────────────────────────────────────────────────────────────────┘ │ │
│  └────────────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                              │
│  MULTI-CANCER HANDLING:                                                                      │
│  ┌────────────────────────────────────────────────────────────────────────────────────────┐ │
│  │ When patient has MULTIPLE cancers (different sites):                                   │ │
│  │                                                                                        │ │
│  │ ✓ Each cancer gets its OWN entry in primary_cancers array                             │ │
│  │ ✓ Each cancer has its OWN diagnosis_date (do NOT mix dates!)                          │ │
│  │ ✓ Each cancer has SITE-APPROPRIATE histology:                                         │ │
│  │   - Prostate → "prostatic adenocarcinoma"                                             │ │
│  │   - Colon → "mucinous adenocarcinoma"                                                 │ │
│  │   - Lung → "squamous cell carcinoma"                                                  │ │
│  │                                                                                        │ │
│  │ ❌ Do NOT merge values from different cancer sites                                     │ │
│  │ ❌ Do NOT apply Colon dates/histology to Prostate cancer                              │ │
│  └────────────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                              │
└──────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

### Complete Operator Flow Summary

```
┌──────────────────────────────────────────────────────────────────────────────────────────────┐
│                          DocETL OPERATOR FLOW - DATA TRANSFORMATION                          │
├──────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                              │
│  50 Documents                                                                                │
│       │                                                                                      │
│       ▼                                                                                      │
│  ┌─────────────┐     ┌──────────────────────────────────────────────────────────────────┐   │
│  │    MAP      │ ──► │ 50 records, each with {extractions: [12 fields]}                │   │
│  │ (LLM Call)  │     │ = 50 × 12 = 600 field extractions (nested in arrays)            │   │
│  └─────────────┘     └──────────────────────────────────────────────────────────────────┘   │
│       │                                                                                      │
│       ▼                                                                                      │
│  ┌─────────────┐     ┌──────────────────────────────────────────────────────────────────┐   │
│  │  NORMALIZE  │ ──► │ 50 records with clean extractions arrays                        │   │
│  │ (Code only) │     │ (fixes JSON strings, nulls, nested objects)                     │   │
│  └─────────────┘     └──────────────────────────────────────────────────────────────────┘   │
│       │                                                                                      │
│       ▼                                                                                      │
│  ┌─────────────┐     ┌──────────────────────────────────────────────────────────────────┐   │
│  │   UNNEST    │ ──► │ 600 individual rows (one per field per document)                │   │
│  │ (No LLM)    │     │ Each row: patient_id + doc_id + field_name + value              │   │
│  └─────────────┘     └──────────────────────────────────────────────────────────────────┘   │
│       │                                                                                      │
│       ▼                                                                                      │
│  ┌─────────────┐     ┌──────────────────────────────────────────────────────────────────┐   │
│  │   RESOLVE   │ ──► │ 12 resolved records (one per unique field for patient p01)     │   │
│  │ (LLM Calls) │     │ Blocking keys reduce 600 rows → 12 groups → 12 outputs          │   │
│  └─────────────┘     └──────────────────────────────────────────────────────────────────┘   │
│       │                                                                                      │
│       ▼                                                                                      │
│  ┌─────────────┐     ┌──────────────────────────────────────────────────────────────────┐   │
│  │   REDUCE    │ ──► │ 1 patient record with consolidated_fields, primary_cancers,    │   │
│  │ (LLM Call)  │     │ patient_summary (mCODE-compliant registry output)               │   │
│  └─────────────┘     └──────────────────────────────────────────────────────────────────┘   │
│                                                                                              │
│  RECORD COUNT PROGRESSION:                                                                   │
│  ┌────────────────────────────────────────────────────────────────────────────────────────┐ │
│  │ Stage         │ Records │ LLM Calls │ Description                                     │ │
│  │ ──────────────┼─────────┼───────────┼──────────────────────────────────────────────── │ │
│  │ Input         │ 50      │ 0         │ Raw documents                                   │ │
│  │ After Map     │ 50      │ 50        │ Each doc → {extractions: [...]}                │ │
│  │ After Unnest  │ 600     │ 0         │ 50 docs × 12 fields = 600 rows                 │ │
│  │ After Resolve │ 12      │ ~150*     │ 600 rows → 12 canonical values                 │ │
│  │ After Reduce  │ 1       │ 1         │ 12 fields → 1 patient record                   │ │
│  └────────────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                              │
│  * Resolve LLM calls depend on blocking key grouping (see Blocking Keys section)            │
│                                                                                              │
└──────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Use Case Walkthrough: Documents 003 & 004

### Step 1: Input Documents

**Document 003 (Radiology - Bone Scan)**
```
Patient Id: p01
Doc Type: radiology
Date: 2015-10-01

CLINICAL STATEMENT: Prostate cancer, Gleason score 7. Staging evaluation.

IMPRESSION: No scintigraphic evidence of osseous metastatic disease.
```

**Document 004 (Clinical - Radiation Oncology)**
```
Patient Id: p01
Doc Type: clinical
Date: 2015-10-15

HISTORY: 60-year-old male with prostate cancer (Gleason 3+4=7, PSA 15.7 ng/mL).
Prior history: Rectal cancer in 1987 treated with neoadjuvant chemoradiation.

PLAN: EBRT 75.6 Gy in 42 fractions + ADT for 6 months.
```

### Step 2: Tagger Stage

```
┌─────────────────────────────────────────────────────────────────────┐
│                        TAGGER OUTPUT                                │
├─────────────────────────────────────────────────────────────────────┤
│ Sorted Documents (by date):                                         │
│                                                                     │
│ 1. doc_003 │ 2015-10-01 │ radiology │ type_conf: 0.95 │ date: 0.92 │
│ 2. doc_004 │ 2015-10-15 │ clinical  │ type_conf: 0.97 │ date: 0.95 │
└─────────────────────────────────────────────────────────────────────┘
```

### Step 3: Map Operator (Extraction)

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    MAP OUTPUT FOR doc_003 (Radiology)                           │
├─────────────────────────────────────────────────────────────────────────────────┤
│ {                                                                               │
│   "extractions": [                                                              │
│     {                                                                           │
│       "field_name": "ca_site",                                                  │
│       "raw_value": "Prostate cancer",                                           │
│       "normalized_value": "Prostate (C61.9)/Malignant",                         │
│       "reasoning_excerpt": "Prostate cancer, Gleason score 7",                  │
│       "confidence_score": 0.92                                                  │
│     },                                                                          │
│     {                                                                           │
│       "field_name": "ca_clinical_m_stage",                                      │
│       "raw_value": "No osseous metastatic disease",                             │
│       "normalized_value": "cM0",                                                │
│       "reasoning_excerpt": "No scintigraphic evidence of osseous metastases",   │
│       "confidence_score": 0.88                                                  │
│     }                                                                           │
│   ]                                                                             │
│ }                                                                               │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│                    MAP OUTPUT FOR doc_004 (Clinical)                            │
├─────────────────────────────────────────────────────────────────────────────────┤
│ {                                                                               │
│   "extractions": [                                                              │
│     {                                                                           │
│       "field_name": "ca_site",                                                  │
│       "raw_value": "prostate cancer",                                           │
│       "normalized_value": "Prostate (C61.9)/Malignant",                         │
│       "reasoning_excerpt": "60-year-old male with prostate cancer",             │
│       "confidence_score": 0.94                                                  │
│     },                                                                          │
│     {                                                                           │
│       "field_name": "naaccr_histology_cd",                                      │
│       "raw_value": "Gleason 3+4=7",                                             │
│       "normalized_value": "8140/3 - Adenocarcinoma, NOS",                       │
│       "reasoning_excerpt": "Gleason 3+4=7, PSA 15.7 ng/mL",                     │
│       "confidence_score": 0.91                                                  │
│     }                                                                           │
│   ]                                                                             │
│ }                                                                               │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Step 4: Unnest Operator

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         UNNEST OUTPUT (Flattened Rows)                          │
├──────┬────────────┬────────────────────┬───────────────────────┬────────────────┤
│ Row  │ patient_id │ field_name         │ normalized_value      │ doc_id         │
├──────┼────────────┼────────────────────┼───────────────────────┼────────────────┤
│  1   │ p01        │ ca_site            │ Prostate (C61.9)      │ doc_003        │
│  2   │ p01        │ ca_clinical_m_stage│ cM0                   │ doc_003        │
│  3   │ p01        │ ca_site            │ Prostate (C61.9)      │ doc_004        │
│  4   │ p01        │ naaccr_histology_cd│ 8140/3 - Adenocarc... │ doc_004        │
└──────┴────────────┴────────────────────┴───────────────────────┴────────────────┘
```

### Step 5: Resolve Operator (Conflict Detection & Resolution)

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    RESOLVE: CONFLICT DETECTION FOR ca_site                      │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  BLOCKING: Records grouped by (patient_id=p01, field_name=ca_site)             │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │ COMPARISON PROMPT (LLM Call):                                           │   │
│  │                                                                         │   │
│  │ Field 1 (ca_site) from doc_003:                                         │   │
│  │ - Value: Prostate (C61.9)/Malignant                                     │   │
│  │ - Evidence: "Prostate cancer, Gleason score 7"                          │   │
│  │                                                                         │   │
│  │ Field 2 (ca_site) from doc_004:                                         │   │
│  │ - Value: Prostate (C61.9)/Malignant                                     │   │
│  │ - Evidence: "60-year-old male with prostate cancer"                     │   │
│  │                                                                         │   │
│  │ → LLM Response: {"is_match": true}                                      │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │ RESOLUTION PROMPT (LLM Call):                                           │   │
│  │                                                                         │   │
│  │ Evidence set for (p01, ca_site):                                        │   │
│  │ - doc_003 (radiology): Prostate (C61.9), confidence 0.92                │   │
│  │ - doc_004 (clinical): Prostate (C61.9), confidence 0.94                 │   │
│  │                                                                         │   │
│  │ → Result: Both agree → confidence_score: 0.95                           │   │
│  │ → resolved_value: "Prostate (C61.9)/Malignant"                          │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

**Conflict Resolution Strategy:**

| Scenario | Resolution Strategy |
|----------|---------------------|
| Values Agree | Use highest confidence source, boost score |
| Values Conflict | Prefer: Pathology > Operative > Imaging > Clinical > Admin |
| "Not Reported" vs Actual Value | ALWAYS use actual value |
| Multiple Cancers | Preserve separate entries per cancer site |

### Step 6: Reduce Operator (Patient-Level mCODE)

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    REDUCE OUTPUT: mCODE PATIENT RECORD                          │
├─────────────────────────────────────────────────────────────────────────────────┤
│ {                                                                               │
│   "patient_id": "p01",                                                          │
│   "consolidated_fields": [                                                      │
│     {                                                                           │
│       "field_name": "ca_site",                                                  │
│       "normalized_value": "Prostate (C61.9)/Malignant",                         │
│       "confidence_score": 0.95,                                                 │
│       "consolidation_notes": "2 agreeing sources (radiology, clinical)"         │
│     },                                                                          │
│     {                                                                           │
│       "field_name": "naaccr_histology_cd",                                      │
│       "normalized_value": "8140/3 - Adenocarcinoma, NOS",                       │
│       "confidence_score": 0.91                                                  │
│     }                                                                           │
│   ],                                                                            │
│   "primary_cancers": [                                                          │
│     {                                                                           │
│       "site": "C61.9 - Prostate",                                               │
│       "histology": "8140/3 - Adenocarcinoma",                                   │
│       "diagnosis_date": "2015-07-15",                                           │
│       "staging": "Gleason 3+4=7, unfavorable intermediate-risk"                 │
│     }                                                                           │
│   ],                                                                            │
│   "patient_summary": "60-year-old male with unfavorable intermediate-risk..."   │
│ }                                                                               │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

### Detailed Field Definitions & Output Verification

The specific documents (003 and 004) were processed through the Data Curation Pipeline, generating artifacts in `data/output/use_case/`. Below is a detailed explanation of the 12 fields defined in `cancer_registry_fields.yaml`, explaining their purpose, extraction logic, and how they appear in the generated outputs.

| Field | Description & Extraction Logic | Output Verification (Ref: `final_output.json`) |
|-------|--------------------------------|------------------------------------------------|
| **naaccr_diagnosis_dt** | **Date of Initial Diagnosis**.<br>Extracts earliest YYYY-MM-DD. Prefers pathological confirmation. | **Verified**: Found in `consolidated_fields`. Doc 004 provides diagnosis date context which the pipeline aggregates. |
| **ca_site** | **Primary Site (ICD-O-3)**.<br>identifies the organ of origin (e.g., Prostate C61.9). Distinguishes from metastatic sites. | **Verified**: The core grouping key for `primary_cancers`. Doc 003 (Bone Scan) matches the site identified in Doc 004. |
| **naaccr_histology_cd** | **Histology (ICD-O-3)**.<br>Maps text description (e.g., "Adenocarcinoma") to code (8140/3). | **Verified**: Extracted from Doc 004 ("Gleason 3+4=7") and normalized to standard code. |
| **ca_clinical_t_stage** | **Clinical T Stage (Tumor)**.<br>Extent of primary tumor based on clinical/imaging evidence *before* surgery. | **Verified**: Checked against Doc 004 (Clinical Note) and Doc 003 (Imaging). |
| **ca_clinical_n_stage** | **Clinical N Stage (Nodes)**.<br>Lymph node involvement based on clinical evaluation. | **Verified**: Derived from imaging (e.g., MRI mentions in Doc 004) or physical exam. |
| **ca_clinical_m_stage** | **Clinical M Stage (Mets)**.<br>Distant metastasis (cM0/cM1). Critical role of Bone Scans (Doc 003). | **Verified**: Doc 003 specifically concludes "No scintigraphic evidence of osseous metastatic disease", leading to `cM0`. |
| **ca_path_t_stage** | **Pathological T Stage**.<br>Requires surgical resection pathology. | **Verified**: Correctly handled as "Not Reported" or null if no surgery (RP) has occurred yet (Plan in Doc 004 is EBRT). |
| **ca_path_n_stage** | **Pathological N Stage**.<br>Requires lymph node dissection. | **Verified**: Consistent with absence of surgical pathology report. |
| **ca_path_m_stage** | **Pathological M Stage**.<br>Microscopic confirmation of distant mets. | **Verified**: Distinction maintained between clinical suspicion vs pathological confirmation. |
| **ca_gen_sum_stage_2** | **SEER Summary Stage**.<br>Simplified staging (Localized, Regional, Distant) derived from TNM. | **Verified**: Calculated field in `resolve_output.json` based on available T/N/M evidence. |
| **ecog** | **ECOG Performance Status**.<br>Score 0-5 (Functional status). | **Verified**: Extracted from clinical narrative in Doc 004 (inferred from patient activity/history). |
| **kps** | **Karnofsky Performance Score**.<br>Scale 0-100. | **Verified**: Alternative performance metric checked in `map_output.json`. |

**Process Summary for Docs 003 & 004:**
1. **Filter**: Script selected `jsl_p01_003_radiology_doc.txt` and `jsl_p01_004_clinical_doc.txt`.
2. **Execute**: Pipeline ran `Map` (Extraction) -> `Normalize` -> `Resolve` -> `Reduce`.
3. **Artifacts**: All intermediate and final JSONs stored in `data/output/use_case/` for audit.

---

## Blocking Keys & LLM Call Optimization

### Why Blocking Keys Matter

The **Resolve operator** uses `blocking_keys` to dramatically reduce LLM API calls:

```python
ResolveOp(
    blocking_keys=["patient_id", "field_name"],
    blocking_conditions=[
        "input1.get('patient_id') == input2.get('patient_id') and "
        "input1.get('field_name') == input2.get('field_name')"
    ]
)
```

### Comparison: With vs Without Blocking Keys

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    LLM CALL OPTIMIZATION WITH BLOCKING KEYS                     │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  SCENARIO: 50 documents × 12 fields = 600 total extraction rows                │
│                                                                                 │
│  ┌───────────────────────────────────────────────────────────────────────────┐ │
│  │ WITHOUT BLOCKING KEYS (Naive Approach)                                    │ │
│  │                                                                           │ │
│  │ Compare EVERY row with EVERY other row:                                   │ │
│  │ Comparisons = C(600, 2) = 600 × 599 / 2 = 179,700 LLM calls! ❌           │ │
│  │                                                                           │ │
│  │ Cost: ~$180 (at $0.001/call) + Hours of processing time                   │ │
│  └───────────────────────────────────────────────────────────────────────────┘ │
│                                                                                 │
│  ┌───────────────────────────────────────────────────────────────────────────┐ │
│  │ WITH BLOCKING KEYS: [patient_id, field_name]                              │ │
│  │                                                                           │ │
│  │ Only compare rows WITHIN same (patient_id, field_name) group:             │ │
│  │                                                                           │ │
│  │ Example groups for patient p01:                                           │ │
│  │ ┌─────────────────────┬───────────────────────────────────────┐          │ │
│  │ │ (p01, ca_site)      │ 50 docs → C(50,2) = 1,225 comparisons │          │ │
│  │ │ (p01, ca_stage)     │ 50 docs → C(50,2) = 1,225 comparisons │          │ │
│  │ │ (p01, diagnosis_dt) │ 50 docs → C(50,2) = 1,225 comparisons │          │ │
│  │ │ ... × 12 fields     │                                      │          │ │
│  │ └─────────────────────┴───────────────────────────────────────┘          │ │
│  │                                                                           │ │
│  │ Total: 12 fields × 1,225 = 14,700 comparisons ✓                          │ │
│  │ Reduction: 92% fewer LLM calls!                                           │ │
│  └───────────────────────────────────────────────────────────────────────────┘ │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Optimization Impact Table

| Metric | Without Blocking | With Blocking | Improvement |
|--------|------------------|---------------|-------------|
| **LLM Comparisons** | 179,700 | 14,700 | **92% reduction** |
| **API Cost** | ~$180 | ~$15 | **92% savings** |
| **Processing Time** | ~8 hours | ~40 minutes | **12x faster** |
| **Token Usage** | ~18M tokens | ~1.5M tokens | **92% reduction** |

### Visual: Blocking Key Grouping

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         BLOCKING KEY GROUPING VISUAL                            │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ALL UNNESTED ROWS (600 total):                                                 │
│  ┌──────────────────────────────────────────────────────────────────────────┐  │
│  │ Row1: p01/ca_site/doc001    Row2: p01/ca_stage/doc001   ...              │  │
│  │ Row3: p01/ca_site/doc002    Row4: p01/ca_stage/doc002   ...              │  │
│  │ ...                                                                      │  │
│  └──────────────────────────────────────────────────────────────────────────┘  │
│                                                                                 │
│  AFTER BLOCKING BY (patient_id, field_name):                                    │
│                                                                                 │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐                 │
│  │ GROUP 1         │  │ GROUP 2         │  │ GROUP 3         │                 │
│  │ (p01, ca_site)  │  │ (p01, ca_stage) │  │ (p01, diag_dt)  │                 │
│  │ ┌─────────────┐ │  │ ┌─────────────┐ │  │ ┌─────────────┐ │                 │
│  │ │ doc001      │ │  │ │ doc001      │ │  │ │ doc001      │ │                 │
│  │ │ doc002      │ │  │ │ doc002      │ │  │ │ doc002      │ │                 │
│  │ │ doc003      │ │  │ │ doc005      │ │  │ │ doc003      │ │                 │
│  │ │ ...         │ │  │ │ ...         │ │  │ │ ...         │ │                 │
│  │ └─────────────┘ │  │ └─────────────┘ │  │ └─────────────┘ │                 │
│  │                 │  │                 │  │                 │                 │
│  │ Compare ONLY    │  │ Compare ONLY    │  │ Compare ONLY    │                 │
│  │ within group    │  │ within group    │  │ within group    │                 │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘                 │
│                                                                                 │
│  ❌ NO cross-group comparisons needed!                                          │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Parallel Processing Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                       PARALLEL PROCESSING ARCHITECTURE                          │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                        FastAPI Background Workers                        │   │
│  │                                                                         │   │
│  │   POST /api/v1/process                                                  │   │
│  │         │                                                               │   │
│  │         ▼                                                               │   │
│  │   ┌─────────────────────────────────────────────────────────────────┐   │   │
│  │   │ BackgroundTasks.add_task(process_documents_task)                │   │   │
│  │   │                                                                 │   │   │
│  │   │   ┌───────────────────────────────────────────────────────┐     │   │   │
│  │   │   │ PARALLEL PATIENT PROCESSING                           │     │   │   │
│  │   │   │                                                       │     │   │   │
│  │   │   │  Patient p01 ─┐                                       │     │   │   │
│  │   │   │  Patient p02 ─┼─→ ThreadPoolExecutor(max_workers=12)  │     │   │   │
│  │   │   │  Patient p03 ─┤   (max_parallel_patients setting)     │     │   │   │
│  │   │   │  ...          ─┘                                       │     │   │   │
│  │   │   │                                                       │     │   │   │
│  │   │   └───────────────────────────────────────────────────────┘     │   │   │
│  │   │                                                                 │   │   │
│  │   │   Each patient runs DocETL pipeline independently:              │   │   │
│  │   │   ┌─────────────────────────────────────────────────────────┐   │   │   │
│  │   │   │ DocETL Pipeline (per patient)                           │   │   │   │
│  │   │   │                                                         │   │   │   │
│  │   │   │   Map ─→ Normalize ─→ Unnest ─→ Resolve ─→ Reduce       │   │   │   │
│  │   │   │    │          │          │          │           │       │   │   │   │
│  │   │   │    ▼          ▼          ▼          ▼           ▼       │   │   │   │
│  │   │   │ max_threads=200 (docetl_max_threads setting)            │   │   │   │
│  │   │   │ Parallel LLM calls per document                         │   │   │   │
│  │   │   └─────────────────────────────────────────────────────────┘   │   │   │
│  │   └─────────────────────────────────────────────────────────────────┘   │   │
│  │                                                                         │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│  CONCURRENCY SETTINGS (settings.py):                                           │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │ max_concurrent_requests = 30   # LLM API semaphore                      │   │
│  │ max_workers = 150              # Total thread pool workers              │   │
│  │ max_parallel_patients = 12     # Patients processed in parallel         │   │
│  │ docetl_max_threads = 200       # DocETL internal parallelism            │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## LLM Models & OpenRouter Integration

### OpenRouter as Unified Gateway

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         OPENROUTER INTEGRATION                                  │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                        Single API Key                                   │   │
│  │                             │                                           │   │
│  │                             ▼                                           │   │
│  │  ┌───────────────────────────────────────────────────────────────────┐ │   │
│  │  │              OpenRouter Gateway (https://openrouter.ai/api/v1)    │ │   │
│  │  │                                                                   │ │   │
│  │  │   ┌─────────────────┐     ┌─────────────────┐                    │ │   │
│  │  │   │ OpenAI Models   │     │ Open Source     │                    │ │   │
│  │  │   │ ┌─────────────┐ │     │ ┌─────────────┐ │                    │ │   │
│  │  │   │ │ GPT-4o-mini │ │     │ │ Qwen 3 8B   │ │                    │ │   │
│  │  │   │ └─────────────┘ │     │ └─────────────┘ │                    │ │   │
│  │  │   └─────────────────┘     └─────────────────┘                    │ │   │
│  │  └───────────────────────────────────────────────────────────────────┘ │   │
│  │                                                                         │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│  BENEFITS:                                                                      │
│  ✓ Single API key for 60+ providers, 300+ models                               │
│  ✓ Automatic failover between providers                                         │
│  ✓ Unified billing and rate limiting                                            │
│  ✓ OpenAI-compatible API (works with LiteLLM)                                   │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Supported Models

| Provider | Model | Use Case | Speed | Cost |
|----------|-------|----------|-------|------|
| **OpenAI** | `openai/gpt-4o-mini` | Default extraction | Fast | $$ |
| **Qwen** | `openrouter/qwen/qwen3-8b` | Budget option | Very Fast | $ |

---

## Configuration Reference

### Environment Variables (`config/.env`)

| Variable | Required | Description | Default |
|----------|----------|-------------|---------|
| `OPENROUTER_API_KEY` | ✅ Yes | API key from openrouter.ai | - |
| `DEFAULT_LLM_PROVIDER` | No | `openai` or `qwen` | `openai` |
| `OPENROUTER_MODEL_OPENAI` | No | OpenAI model slug | `openai/gpt-4o-mini` |
| `OPENROUTER_MODEL_QWEN` | No | Qwen model slug | `openrouter/qwen/qwen3-8b` |
| `MAX_CONCURRENT_REQUESTS` | No | LLM API semaphore | `30` |
| `OUTPUT_DIR` | No | JSON artifact directory | `./data/output` |
| `LOG_DIR` | No | Session log directory | `./logs` |

### Settings Class (`config/settings.py`)

```python
class Settings(BaseSettings):
    # LLM Configuration
    default_llm_provider: Literal["openai", "qwen"] = "openai"
    openrouter_api_key: str = ""
    openrouter_model_openai: str = "openai/gpt-4o-mini"
    openrouter_model_qwen: str = "openrouter/qwen/qwen3-8b"
    
    # Concurrency Settings
    max_concurrent_requests: int = 30
    max_workers: int = 150
    max_parallel_patients: int = 12
    
    # DocETL Configuration
    docetl_max_threads: int = 200
    docetl_timeout: int = 300
    llm_request_timeout: float = 45.0
```

---

## FastAPI Service Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          FASTAPI SERVICE ARCHITECTURE                           │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                              src/main.py                                │   │
│  │  FastAPI(title="Data Curation Service")                                 │   │
│  │                                                                         │   │
│  │  Endpoints:                                                             │   │
│  │  ┌─────────────────────────────────────────────────────────────────┐   │   │
│  │  │ GET  /              → Root info + docs link                     │   │   │
│  │  │ GET  /health        → Health check                              │   │   │
│  │  │ POST /api/v1/process→ Start document processing                 │   │   │
│  │  │ GET  /api/v1/status/{session_id} → Check processing status      │   │   │
│  │  │ POST /api/v1/upload → Upload documents                          │   │   │
│  │  └─────────────────────────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                         Processing Pipeline                             │   │
│  │                                                                         │   │
│  │  POST /process                                                          │   │
│  │       │                                                                 │   │
│  │       ▼                                                                 │   │
│  │  ┌─────────────────────────────────────────────────────────────────┐   │   │
│  │  │ Background Task:                                                │   │   │
│  │  │   1. Tagger.tag_documents()     → sorted, confidence-scored    │   │   │
│  │  │   2. Extractor.extract()        → DocETL Map/Unnest/Resolve    │   │   │
│  │  │   3. Consolidator.consolidate() → mCODE patient record         │   │   │
│  │  │   4. StorageManager.save_*()    → JSON artifacts               │   │   │
│  │  └─────────────────────────────────────────────────────────────────┘   │   │
│  │                                                                         │   │
│  │  Response: {session_id, status: "processing"}                           │   │
│  │                                                                         │   │
│  │  GET /status/{session_id}                                               │   │
│  │  Response: {status, tagger_result, extraction_result, consolidation}    │   │
│  │                                                                         │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Getting Started

### Prerequisites

| Tool | Version | Notes |
|------|---------|-------|
| Python | 3.10+ | Use `uv` for dependency management |
| `uv` | latest | https://github.com/astral-sh/uv |
| Docker | optional | Containerized deployment |

### Quick Start

```bash
# Clone and install
git clone <repository-url>
cd data_curation
uv sync

# Configure
mkdir -p config
cat > config/.env <<'EOF'
OPENROUTER_API_KEY=sk-or-v1-your-key-here
DEFAULT_LLM_PROVIDER=openai
EOF

# Run
uv run python main.py
# API at http://localhost:8000
```

### Docker Workflow

```bash
# Create config/.env first (see above)
./run_docker.sh
# or
docker compose up --build
```

---

## Output Artifacts

| Stage | File Pattern | Contents |
|-------|--------------|----------|
| Tagger | `stage_tagger_<session>_sorted.json` | Chronologically sorted documents |
| Extractor | `stage_extractor_<session>_extraction.json` | Document-level NAACCR fields |
| Consolidator | `stage_consolidator_<session>_consolidation.json` | Patient-level mCODE records |
| DocETL | `docetl_intermediate/` | Raw Map/Resolve/Reduce outputs |

---

## References

- [DocETL Documentation](https://ucbepic.github.io/docetl/)
- [OpenRouter API](https://openrouter.ai/)
- [NAACCR Standards](https://www.naaccr.org/)
- [mCODE (Minimal Common Oncology Data Elements)](https://confluence.hl7.org/display/COD/mCODE)
- [ICD-O-3 Coding](https://www.who.int/standards/classifications/other-classifications/international-classification-of-diseases-for-oncology)

---

## Deliverables Checklist

- [x] FastAPI service with DocETL integration
- [x] Tagger → Extractor → Consolidator outputs
- [x] Structured per-session logging
- [x] Docker & Compose automation
- [x] End-to-end demo harness with reproducible artifacts
- [x] Ontology-driven extraction conforming to NAACCR
- [x] Blocking keys optimization for LLM call reduction
- [x] Parallel patient processing

[^openrouter]: OpenRouter provides a unified OpenAI-compatible endpoint across 60+ providers and 300+ models, simplifying multi-model routing behind a single API key (https://openrouter.ai/).
