# 🧠 Data Curation Engineer – Technical Task

## Objective

Develop a straightforward **Data Curation Service** utilizing **DocETL** to efficiently extract and consolidate complex documents.

You will design a modular ETL pipeline capable of:

1. Extracting structured medical entities and evidence from multiple clinical documents.
2. Consolidating and normalizing those extractions at **patient level**.
3. Supporting both **cloud (OpenAI API)** and **local (e.g., Gemma-3 / Qwen-3)** LLM back-ends.
4. Logging all prompts and outputs per session for provenance and auditability.

---

## 1️⃣ Overall Flow

```
Input  →  DocETL-based Extraction  →  DocETL Consolidation/Normalization  →  Output (Storage & Logs)
```

### Stages

| Stage            | Purpose                                                   | Required | Notes                                |
| :--------------- | :-------------------------------------------------------- | :------- | :----------------------------------- |
| **Tagger**       | (Optional) classify and order documents chronologically   | ❌        | Use metadata if available            |
| **Extractor**    | field-level extraction with clinical reasoning & evidence | ✅        | Implement with DocETL map operators  |
| **Consolidator** | patient-level aggregation & normalization                 | ✅        | Implement with DocETL resolve/reduce |

---

## 2️⃣ Functional Requirements

### Inputs

* Multiple unstructured documents (`.txt` / `.pdf` / `.json`) from the provided dataset.
* Each document includes metadata: `patient_id`, `doc_type`, `doc_date`, `content`.

### Processing

* Use Python API to define a **DocETL pipeline** (map → resolve → reduce).
* Extraction fields and normalization schema **must follow the provided `cancer_registry_fields.yaml` ontology file**, which defines the field list and metadata structure to be used throughout extraction and consolidation.
  This ontology aligns with **NAACCR Cancer Registry core elements** (diagnosis, TNM staging, performance status, etc.).
* Each field must include:

  * `raw_value`
  * `normalized_value`
  * `reasoning_excerpt` (where in the document)
  * `explanation` (why chosen)
* Design Pydantic v2 models dynamically to ensure type-safety.
* Support concurrent processing (async + thread pool, semaphore-based).
* Maintain OpenAI-style structured output validation.

### Outputs

1. **Document-level extraction results** → `doc_level_results`
2. **Patient-level consolidated results** → `patient_level_results`

Store outputs in:

* PostgreSQL (tables `doc_extractions`, `patient_consolidations`) **or**
* Local JSON files under `data/output/{session_id}/`.

### Logging

* Every request/session creates a unique `session_id`.
* Log full LLM prompt & response context to a file:
  `logs/{session_id}/stage_extractor.log`, `stage_consolidator.log`.

---

## 3️⃣ Technical Requirements

| Category             | Specification                                                      |
| :------------------- | :----------------------------------------------------------------- |
| **Language**         | Python 3.10 +                                                      |
| **Framework**        | FastAPI                                                            |
| **Schema**           | Pydantic v2                                                        |
| **Environment**      | `uv` package manager                                               |
| **LLM Integration**  | OpenAI Responses API + local Gemma-3/Qwen-3 model via transformers |
| **ETL Engine**       | [DocETL](https://ucbepic.github.io/docetl/#project-origin)         |
| **Storage**          | PostgreSQL or local JSON                                           |
| **Containerization** | Docker + Docker Compose                                            |
| **Concurrency**      | asyncio / ThreadPoolExecutor with configurable limit               |
| **Configuration**    | `.env` for keys & settings                                         |
| **Validation**       | automatic retry / validation via DocETL’s `validate` block         |

---

## 4️⃣ Expected Project Structure

```
data-curation-docetl/
├── src/
│   ├── api/                 # FastAPI endpoints
│   ├── pipeline/            # DocETL + orchestration
│   ├── models/              # Pydantic v2 schemas
│   ├── utils/               # Logging, config, db connectors
│   └── main.py              # Entry point
├── data/
│   ├── input/               # Dataset provided
│   ├── output/{session_id}/
│   └── ontology/
│       └── cancer_registry_fields.yaml   # Field definitions for extraction & normalization
├── config/
│   ├── .env.example
│   └── settings.py
├── docker-compose.yml
├── Dockerfile
└── README.md
```

---

## 5️⃣ README (Deliverable)

Your `README.md` must clearly describe:

1. **Overview** of the pipeline and purpose.
2. **How to run** locally with `uv run python main.py`.
3. **How to run** via Docker Compose (`docker compose up`).
4. **API usage** examples:

   * File upload endpoint
   * Patient ID list input
5. **DocETL integration** details (operators used, YAML structure).
6. **Sample outputs** (doc + patient level).
7. **Logging structure** and sample session log.

---

## 6️⃣ Evaluation Criteria

| Aspect                                               | Weight |
| :--------------------------------------------------- | :----: |
| Correctness of DocETL integration                    |  25 %  |
| Pipeline architecture & modularity                   |  20 %  |
| Provenance & logging implementation                  |  15 %  |
| Code quality (readability, type safety, async usage) |  15 %  |
| Dockerization & README clarity                       |  10 %  |
| Robustness of outputs (doc/patient level)            |  15 %  |

---

## 7️⃣ DocETL Reference Materials

* 📘 [Interactive LLM-Powered Data Processing with DocWrangler](https://data-people-group.github.io/blogs/2025/01/13/docwrangler/)
* 📄 [DocETL – Agentic Query Rewriting and Evaluation for Complex Document Processing](https://arxiv.org/pdf/2410.12189)
* 📄 [Steering Semantic Data Processing With DocWrangler](https://arxiv.org/abs/2504.14764)
* 🎥 [YouTube Talk – DocETL for Complex Document Processing](https://youtu.be/ytAsNoTZfhw)

---

## 8️⃣ Submission

* GitHub repository link or compressed archive (`.zip`).
* Must include Docker setup and sample output files.
* Deadline & dataset will be provided separately.

---

### ✅ Deliverables Checklist

* [ ] FastAPI service with DocETL integration
* [ ] Two output levels (doc + patient)
* [ ] Structured logging per session
* [ ] Docker & Compose runnable
* [ ] Comprehensive README
* [ ] `cancer_registry_fields.yaml` ontology included and used in pipeline

---

**End of Task Document**
*(All materials and examples must be original; do not reuse internal/external repositories.)*

---
