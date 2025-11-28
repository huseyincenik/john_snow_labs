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

## Project Navigator

- Back to portfolio home → [`../README.md`](../README.md)
- CoNLL generator & custom NER training → [`../generating_conll_files_from_pretrained_models`](../generating_conll_files_from_pretrained_models)
- RAG QA chatbot application → [`../rag_qa_chatbot_application`](../rag_qa_chatbot_application)

---

## High-Level Architecture

```
┌───────────────────────────────────────────────────────────────────────┐
│                           Ingestion Layer                             │
│    Upload REST API + sample docs (TXT/PDF) written to input storage    │
└───────────────┬───────────────────────────────────────────────────────┘
                ▼
┌───────────────────────────────────────────────────────────────────────┐
│                            DocETL Engine                              │
│   1. Tagger (chronology) → 2. Extractor (NAACCR fields) →              │
│   3. Consolidator (patient-level Resolve + Reduce)                     │
└───────────────┬───────────────────────────────────────────────────────┘
                ▼
┌───────────────────────────────────────────────────────────────────────┐
│                    Structured Outputs & Observability                  │
│  JSON artifacts + logs + demo artifacts → feed downstream pipelines    │
└───────────────────────────────────────────────────────────────────────┘
```

DocETL outputs can be reused by the **CoNLL generator** to bootstrap annotations or ingested by the **RAG chatbot** to expand its knowledge base, ensuring consistent clinical narratives across the portfolio.

---

## 1. System Overview

### 1.1 Key Capabilities

- **DocETL Python pipeline** (Tagger → Map → Unnest → Resolve → Reduce) implemented with FastAPI background workers.
- **Multi-provider LLM support**: OpenRouter-backed routing that sequentially exercises OpenAI GPT-4o-mini and Qwen 2.5 7B Instruct models.[^openrouter]
- **Parallel ingestion & demos**: asynchronous document loading in FastAPI plus concurrent OpenAI/Qwen benchmark runs in the demo harness keep wall-clock time low.
- **Canonical outputs**: document-level extractions and patient-level consolidations aligned with NAACCR fields.
- **Auditable processing**: per-session logs, prompt/response capture, JSON artifacts per stage.
- **Turnkey demo flow**: a Python script that exercises every API endpoint end-to-end.

### 1.2 Pipeline Schematic

```
┌────────────────┐    ┌──────────────┐    ┌──────────────┐    ┌────────────────────┐
│ Input documents │ -> │ Tagger       │ -> │ Extractor    │ -> │ Consolidator        │
│ (txt / pdf)     │    │ (chronology) │    │ (DocETL Map) │    │ (Resolve + Reduce) │
└────────────────┘    └──────────────┘    └──────────────┘    └────────────────────┘
        │                     │                    │                    │
        │                     │                    │                    └─► `stage_stage_consolidator_<session>.json`
        │                     │                    └─► `stage_stage_extractor_<session>.json`
        │                     └─► `stage_stage_tagger_<session>_sorted.json`
        └─► Logs (`logs/<session_id>/stage_*.log`)
```

### 1.3 DocETL Operators

| Step | DocETL Operator | Purpose |
| ---- | --------------- | ------- |
| Extractor | `MapOp` `extract_clinical_fields` | Builds a fully typed JSON schema from `cancer_registry_fields.yaml` and enforces OpenAI-style reasoning metadata per field. |
| Extractor | `UnnestOp` `explode_field_records` | Flattens the map payload so each ontology field becomes an individual row with `patient_id`, `doc_id`, and evidence. |
| Consolidator | `ResolveOp` `resolve_patient_fields` | Performs canonical value selection per `(patient_id, field_name)` cluster with blocking keys + provenance capture. |
| Consolidator | `ReduceOp` `reduce_patient_summary` | Aggregates resolved entries into patient-level registries and emits `consolidated_fields` + a narrative summary. |

DocETL runs as a **memory dataset** so no temporary CSV/JSON artifacts are required, yet every stage checkpoint is persisted to `data/output/<session_id>/docetl_intermediate/<step>/<op>.json` for debugging.

### 1.4 Repository Layout

```
data_curation/
├── src/
│   ├── api/               # FastAPI routers & background workers
│   ├── pipeline/          # Tagger, Extractor, Consolidator
│   ├── models/            # Pydantic v2 schemas
│   ├── utils/             # Config, storage, logging, LLM adapters
│   └── main.py            # Uvicorn entry point
├── data/
│   ├── input/             # Sample patient docs
│   ├── output/<session>/  # Stage artifacts (tagger/extractor/consolidator)
│   └── ontology/          # cancer_registry_fields.yaml
├── config/
│   ├── .env.example
│   └── settings.py
├── scripts/
│   └── e2e_demo.py        # Automated demo harness
├── docker-compose.yml
├── run_docker.sh
└── README.md
```

---

## 2. Getting Started

### 2.1 Prerequisites

| Tool                    | Version  | Notes                                      |
| ----------------------- | -------- | ------------------------------------------ |
| Python                  | 3.10+    | Use `uv` for dependency/workflow tooling   |
| `uv`                    | latest   | <https://github.com/astral-sh/uv>          |
| Docker & Docker Compose | optional | Containerized deployment of FastAPI stack  |

### 2.2 Local (bare-metal) Setup

#### Step 1: Clone and Install Dependencies

```bash
git clone <repository-url>
cd data_curation

# Install dependencies with uv (no pip install)
uv sync
```

#### Step 2: Get OpenRouter API Key

1. Go to [OpenRouter.ai](https://openrouter.ai/)
2. Sign up or log in to your account
3. Navigate to [API Keys](https://openrouter.ai/keys) section
4. Create a new API key (it will look like `sk-or-v1-...`)
5. Copy your API key

#### Step 3: Create Configuration File

Create a `.env` file in the `config/` directory:

**Option A: Copy from example (if available)**
```bash
# Create config directory if it doesn't exist
mkdir -p config

# Copy example file and edit with your API key
cp config/.env.example config/.env
# Then edit config/.env and replace the placeholder with your actual API key
```

**Option B: Create manually**
```bash
# Create config directory if it doesn't exist
mkdir -p config

# Create .env file
cat > config/.env <<'EOF'
# REQUIRED: OpenRouter API Key
# Get your key from: https://openrouter.ai/keys
OPENROUTER_API_KEY=sk-or-v1-your-actual-api-key-here

# Optional: LLM Provider Settings
DEFAULT_LLM_PROVIDER=openai
OPENROUTER_MODEL_OPENAI=openai/gpt-4o-mini
OPENROUTER_MODEL_QWEN=openrouter/qwen/qwen3-8b
OPENROUTER_REFERER=https://local.data-curation
OPENROUTER_APP_TITLE=Data Curation Service
EOF
```

**Important:** 
- Replace `sk-or-v1-your-actual-api-key-here` with your actual OpenRouter API key from Step 2
- The `.env` file must be in the `config/` directory (not the root directory)
- Never commit your `.env` file to version control (it should be in `.gitignore`)

#### Step 4: Launch the Service

```bash
# Launch API (development)
uv run python main.py
# or
uv run uvicorn src.main:app --reload

# API will be available at http://localhost:8000
```

**Note:** If you see an error about missing `OPENROUTER_API_KEY`, make sure:
- The `.env` file is in the `config/` directory (not the root directory)
- The API key is correctly set (no extra spaces or quotes)
- You've saved the file after editing

### 2.3 Docker Workflow

#### Step 1: Create Configuration File

Before running Docker, you need to create the `.env` file with your OpenRouter API key:

```bash
# Create config directory if it doesn't exist
mkdir -p config

# Create .env file with your OpenRouter API key
cat > config/.env <<'EOF'
OPENROUTER_API_KEY=sk-or-v1-your-actual-api-key-here
DEFAULT_LLM_PROVIDER=openai
OPENROUTER_MODEL_OPENAI=openai/gpt-4o-mini
OPENROUTER_MODEL_QWEN=openrouter/qwen/qwen3-8b
EOF
```

**Important:** Replace `sk-or-v1-your-actual-api-key-here` with your actual OpenRouter API key. See [Section 3.2](#32-getting-your-openrouter-api-key) for instructions on how to get one.

#### Step 2: Build and Run

```bash
# Build & run
./run_docker.sh
# or
docker compose up --build
```

**Note:** `docker-compose.yml` only runs the FastAPI container. All LLM calls are proxied through OpenRouter, so no local models are downloaded. The `.env` file from `config/.env` will be automatically mounted into the container.

---

## 3. Configuration Reference

### 3.1 Environment Variables

All configuration is done through environment variables loaded from `config/.env`. The application will automatically load this file on startup.

| Variable                  | Required | Description                                                                 | Example                      |
| ------------------------- | -------- | --------------------------------------------------------------------------- | ---------------------------- |
| `OPENROUTER_API_KEY`      | ✅ **Yes** | API key from [OpenRouter](https://openrouter.ai/keys)                       | `sk-or-v1-...`               |
| `DEFAULT_LLM_PROVIDER`    | No       | Initial provider (`openai` or `qwen`)                                      | `openai`                     |
| `OPENROUTER_MODEL_OPENAI` | No       | Model slug used when `llm_provider=openai`                                 | `openai/gpt-4o-mini`         |
| `OPENROUTER_MODEL_QWEN`   | No       | Model slug used when `llm_provider=qwen`                                    | `openrouter/qwen/qwen3-8b`   |
| `OPENROUTER_REFERER`      | No       | URL identifying your app (used to satisfy OpenRouter headers)               | `https://local.data-curation` |
| `OPENROUTER_APP_TITLE`    | No       | Human-readable title sent to OpenRouter                                     | `Data Curation Service`      |
| `OUTPUT_DIR`              | No       | JSON artifact root                                                          | `./data/output`              |
| `LOG_DIR`                 | No       | Session log root                                                            | `./logs`                     |
| `MAX_CONCURRENT_REQUESTS` | No       | Extractor semaphore                                                         | `30`                         |

### 3.2 Getting Your OpenRouter API Key

1. **Visit OpenRouter**: Go to [https://openrouter.ai/](https://openrouter.ai/)
2. **Sign Up/Login**: Create an account or log in
3. **Navigate to Keys**: Click on "Keys" in the navigation menu or go directly to [https://openrouter.ai/keys](https://openrouter.ai/keys)
4. **Create API Key**: Click "Create Key" button
5. **Copy Key**: Your API key will start with `sk-or-v1-`. Copy it immediately (you won't be able to see it again)
6. **Add to .env**: Paste it into your `config/.env` file as `OPENROUTER_API_KEY=sk-or-v1-your-key-here`

### 3.3 Configuration File Location

The application looks for `.env` file in the following locations (in order):
1. `config/.env` (recommended)
2. `.env` (root directory)

**Best Practice**: Always use `config/.env` to keep configuration organized.

### 3.4 OpenRouter Integration

> OpenRouter acts as a unified OpenAI-compatible gateway across many labs, so the backend just needs one API key and one base URL to reach both OpenAI GPT-4o-mini and Qwen models.[^openrouter] These values are automatically copied into the standard `OPENAI_API_KEY` / `OPENAI_API_BASE` environment variables at startup so DocETL/LiteLLM can authenticate without additional configuration. **If the key is missing, the app will fail fast with a clear error message.**

---

## 4. API Usage

### 4.1 Process Documents

```bash
curl -X POST "http://localhost:8000/api/v1/process" \
  -H "Content-Type: application/json" \
  -d '{
        "patient_ids": ["p01"],
        "process_all": false,
        "llm_provider": "openai",
        "llm_model": "openai/gpt-4o-mini"
      }'
```
Switch `"llm_provider"` to `"qwen"` and `"llm_model"` to `"openrouter/qwen/qwen-2.5-7b-instruct"` to run the same request with Qwen.

Response:

```json
{
  "session_id": "238de31e-7e02-493e-9f1f-63815234c063",
  "status": "processing",
  "message": "Processing started"
}
```

### 4.2 Check Status

```bash
curl "http://localhost:8000/api/v1/status/238de31e-7e02-493e-9f1f-63815234c063"
```

Returns the latest status plus any available `tagger_result`, `extraction_result`, and `consolidation_result`.

### 4.3 Upload Files (optional)

```bash
curl -X POST "http://localhost:8000/api/v1/upload" \
  -F "files=@input_patient_docs/jsl_p01_001_summary_doc.txt" \
  -F "files=@input_patient_docs/jsl_p01_050_summary_doc.txt"
```

---

## 5. Automated Demo Flow

`scripts/e2e_demo.py` drives the full workflow (upload → process → poll status → archive results).

```bash
uv run python scripts/e2e_demo.py \
  --base-url http://localhost:8000 \
  --patient-ids p01 \
  --upload \
  --num-docs 5 \
  --llm-providers openai qwen \
  --provider-models openai=openai/gpt-4o-mini \
  --provider-models qwen=openrouter/qwen/qwen-2.5-7b-instruct \
  --poll-timeout 900
```

Use additional `--provider-models PROVIDER=MODEL` flags to test alternative OpenRouter slugs per provider (e.g., `anthropic/claude-3.7-sonnet`). The script processes OpenAI first and then Qwen, storing artifacts under `demo_runs/<timestamp>/<idx>_<provider>/`.
Providers run concurrently via asyncio, so overall runtime roughly matches the slowest provider rather than the sum of both runs.

1. _(optional)_ Upload sample docs from `input_patient_docs/`.
2. Trigger `/process` with the selected provider/model.
3. Poll `/status/{session}` until `completed` (handles transient `httpx` errors with retries).
4. Save:
   - `01_upload_response.json`
   - `02_process_response.json`
   - `03_status_final.json`
   - `tagger_result.json`
   - `extraction_result.json`
   - `consolidation_result.json`
5. Aggregate log stream into `demo.log`.

Artifacts live under `demo_runs/demo_run_<timestamp>/<idx>_<provider>/`.

---

## 6. Output Artifacts

| Stage        | File Pattern                                                     | Highlights                                                          |
| ------------ | ---------------------------------------------------------------- | ------------------------------------------------------------------- |
| Tagger       | `data/output/<session>/stage_stage_tagger_<session>_sorted.json` | Chronologically sorted documents, confidence scores, split metadata |
| Extractor    | `.../stage_stage_extractor_<session>_extraction.json`            | Document-level NAACCR fields with evidence, timestamps, confidence  |
| DocETL raw   | `.../docetl_patient_results.json` + `docetl_intermediate/`       | Full Map/Resolve/Reduce payloads for replay + auditing              |
| Consolidator | `.../stage_stage_consolidator_<session>_consolidation.json`      | Patient-level resolved values, provenance, consolidated reasoning   |

### 6.1 Sample Outputs

```1:34:output_samples/stage_extractor_48a759f7-2a79-4b98-86ff-dd8a4094e97b_extraction.json
{
  "session_id": "48a759f7-2a79-4b98-86ff-dd8a4094e97b",
  "generated_timestamp": "2025-10-17T01:32:30.907927",
  "stage": "stage_extractor",
  "total_documents_processed": 2,
  "document_results": [
    {
      "doc_id": "doc_001_jsl_p01_001_summary_doc",
      "extracted_fields": [
        {
          "field_name": "file_name",
          "category": "provenance",
          "raw_value": "p01/jsl_p01_001_summary_doc.txt",
          "normalized_value": "p01/jsl_p01_001_summary_doc.txt",
          "...": "..."
        }
      ]
    }
  ]
}
```

```1:38:output_samples/stage_consolidator_48a759f7-2a79-4b98-86ff-dd8a4094e97b_consolidation.json
{
  "session_id": "48a759f7-2a79-4b98-86ff-dd8a4094e97b",
  "generated_timestamp": "2025-10-17T01:35:01.830776",
  "stage": "stage_consolidator",
  "consolidated_fields": [
    {
      "field_name": "mcode_patient_extraction",
      "category": "mcode_registry",
      "...": "..."
    }
  ]
}
```

Each session also records:

- `logs/<session>/stage_extractor.log`
- `logs/<session>/stage_consolidator.log`

---

## 7. Logging & Observability

- Structured logger with timestamped lines per stage.
- Full stack traces when LLM calls fail (e.g., connectivity, schema validation).
- LLM prompts/responses saved in session logs for audit trails.
- Automatic retry/backoff for `APIConnectionError`, `APITimeoutError`, `RateLimitError`.
- Prompt captures:
  - `logs/<session>/stage_extractor_prompts.log` → rendered DocETL Map prompts per document.
  - `logs/<session>/stage_consolidator_prompts.log` → rendered Reduce prompts per patient (post-resolve inputs).

---

## 8. Ontology & DocETL Integration

### 8.1 DocETL Pipeline Architecture

The service uses **DocETL** (https://github.com/ucbepic/docetl) to orchestrate the ETL pipeline:

```
Documents → Map (extract fields) → Unnest (explode arrays) → Resolve (deduplicate) → Reduce (patient-level)
```

**Operators Used:**
- **Map**: Extracts all NAACCR fields from each document using LLM
- **Unnest**: Expands the `extractions` array into individual field records
- **Resolve**: Deduplicates and merges field values across documents for the same patient+field
- **Reduce**: Aggregates resolved fields into patient-level summaries

### 8.2 DocETL Configuration

- **Source**: DocETL is installed from PyPI (`docetl>=0.2.5`)
- **Dependency Management**: Declared in `pyproject.toml`, installed via `uv sync` / Docker build
- **Pipeline Definition**: `src/pipeline/docetl_runner.py` builds the pipeline programmatically
- **Outputs**: 
  - Intermediate results: `data/output/<session>/docetl_intermediate/`
  - Final results: `data/output/<session>/docetl_patient_results.json`

### 8.3 Ontology Alignment

- Ontology file: `data/ontology/cancer_registry_fields.yaml`
- Extractor prompt dynamically enumerates every required NAACCR field
- Consolidator merges doc-level entries into final patient summaries using DocETL Resolve + Reduce

---

## 9. References & Further Reading

- [DocETL documentation](https://ucbepic.github.io/docetl/)
- [DocWrangler blog](https://data-people-group.github.io/blogs/2025/01/13/docwrangler/)
- [NAACCR standards](https://www.naaccr.org/)
- [ICD-O-3](https://www.who.int/standards/classifications/other-classifications/international-classification-of-diseases-for-oncology)

---

## 10. Deliverables Checklist

- [x] FastAPI service with DocETL integration
- [x] Tagger → Extractor → Consolidator outputs
- [x] Structured per-session logging
- [x] Docker & Compose automation (including local Qwen)
- [x] End-to-end demo harness with reproducible artifacts
- [x] Ontology-driven extraction conforming to NAACCR

[^openrouter]: OpenRouter provides a unified OpenAI-compatible endpoint across 60+ providers and 300+ models, simplifying multi-model routing behind a single API key (<https://openrouter.ai/>).
