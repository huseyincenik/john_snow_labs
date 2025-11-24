<div align="center">
  <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQkzoKPaZIwIrnqHBYP_if-0vLt-hT6h2h-BQ&s" alt="John Snow Labs logo" width="96">
  <h1>Data Curation Service – DocETL Pipeline</h1>
  <p>LLM-driven oncology extraction pipeline that delivers registry-ready JSON artifacts.</p>
  <img alt="Project views" src="https://komarev.com/ghpvc/?username=huseyincenik&color=orange&label=Data+Curation+Views">
  <p>
    <img src="https://fastapi.tiangolo.com/img/logo-margin/logo-teal.png" alt="FastAPI" height="40">
    &nbsp;
    <img src="https://www.python.org/static/community_logos/python-logo.png" alt="Python" height="40">
    &nbsp;
    <img src="https://www.docker.com/wp-content/uploads/2022/03/Moby-logo.png" alt="Docker" height="40">
    &nbsp;
    <img src="https://seeklogo.com/images/O/openai-logo-8B9BFEDC26-seeklogo.com.png" alt="OpenAI" height="40">
    &nbsp;
    <img src="https://ollama.ai/public/icon-192.png" alt="Ollama/Qwen" height="40">
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

- **DocETL pipeline** (Tagger → Extractor → Consolidator) implemented with FastAPI background workers.
- **Multi-provider LLM support**: OpenAI Responses API plus OpenAI-compatible local endpoints (Qwen/Gemma).
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

### 1.3 Repository Layout

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
| Docker & Docker Compose | optional | Required for turnkey local Qwen deployment |

### 2.2 Local (bare-metal) Setup

```bash
git clone <repository-url>
cd data_curation

# Install dependencies
uv pip install -e .

# Configure environment
cp config/.env.example config/.env
# edit config/.env and add OPENAI_API_KEY, QWEN_* etc.

# Launch API (development)
uv run python src/main.py            # or: uvicorn src.main:app --reload
# API available at http://localhost:8000
```

### 2.3 Docker Workflow

```bash
# Build & run (API + optional Qwen service)
./run_docker.sh                      # cleans old containers, pulls Qwen model, runs docker compose
# or
docker compose up --build
```

`docker-compose.yml` includes:

- `api`: FastAPI service (reads `config/.env`)
- `qwen`: `ollama/ollama` container exposing `http://qwen:11434/v1`

To test Qwen locally without `run_docker.sh`:

```bash
docker compose up --build
docker exec -it qwen ollama pull qwen2.5:0.5b-instruct   # first run only
```

---

## 3. Configuration Reference

| Variable                  | Description                                         | Example                    |
| ------------------------- | --------------------------------------------------- | -------------------------- |
| `DEFAULT_LLM_PROVIDER`    | `openai`, `qwen`, or `local`                        | `qwen`                     |
| `OPENAI_API_KEY`          | OpenAI key                                          | `sk-...`                   |
| `OPENAI_MODEL`            | Default OpenAI model                                | `gpt-4o-mini`              |
| `OPENAI_BASE_URL`         | (optional) override; leave blank for api.openai.com |                            |
| `QWEN_API_BASE`           | URL of Qwen/Ollama service                          | `http://qwen:11434/v1`     |
| `QWEN_MODEL`              | Model name to pull                                  | `qwen2.5:0.5b-instruct`    |
| `LOCAL_API_BASE`          | Custom OpenAI-compatible endpoint                   | `http://localhost:1234/v1` |
| `OUTPUT_DIR`              | JSON artifact root                                  | `./data/output`            |
| `LOG_DIR`                 | Session log root                                    | `./logs`                   |
| `MAX_CONCURRENT_REQUESTS` | Extractor semaphore                                 | `5`                        |

> By default the stack runs **only Qwen** so everything works offline. If you want OpenAI as well, set `DEFAULT_LLM_PROVIDER=openai` (or pass `"llm_provider": "openai"` in your API request / CLI flags) and make sure `OPENAI_API_KEY` is filled in.
>
> Tip: Keep `OPENAI_BASE_URL` unset instead of an empty string. The client automatically falls back to `https://api.openai.com/v1`.

---

## 4. API Usage

### 4.1 Process Documents

```bash
curl -X POST "http://localhost:8000/api/v1/process" \
  -H "Content-Type: application/json" \
  -d '{
        "patient_ids": ["p01"],
        "process_all": false,
        "llm_provider": "qwen",
        "llm_model": "qwen2.5:0.5b-instruct"
      }'
```

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
  --llm-providers qwen \
  --provider-models qwen=qwen2.5:0.5b-instruct \
  --poll-timeout 900 \
  --qwen-timeout 1800
```

> Need to run OpenAI as well? Append `--llm-providers openai qwen --provider-models openai=gpt-4o-mini` (and optionally tweak `--poll-timeout` for OpenAI). The script will execute providers in the supplied order.
> `--qwen-timeout` defines the minimum ceiling for Qwen; if the session still hasn’t completed, the script automatically extends the timeout first to 3600s, then 4800s, and only fails after exhausting that plan. This keeps long-running local inference jobs from flaking out.

Steps executed per provider:

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
| Consolidator | `.../stage_stage_consolidator_<session>_consolidation.json`      | Patient-level resolved values, provenance, consolidated reasoning   |

Each session also records:

- `logs/<session>/stage_extractor.log`
- `logs/<session>/stage_consolidator.log`

---

## 7. Logging & Observability

- Structured logger with timestamped lines per stage.
- Full stack traces when LLM calls fail (e.g., connectivity, schema validation).
- LLM prompts/responses saved in session logs for audit trails.
- Automatic retry/backoff for `APIConnectionError`, `APITimeoutError`, `RateLimitError`.

---

## 8. Ontology & DocETL Alignment

- Ontology file: `data/ontology/cancer_registry_fields.yaml`
- Extractor prompt dynamically enumerates every required NAACCR field.
- Consolidator merges doc-level entries into final patient summaries (DocETL Resolve + Reduce).

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
