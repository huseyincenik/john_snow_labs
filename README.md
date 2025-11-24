# John Snow Labs – Interview Task Portfolio

![Repo views](https://komarev.com/ghpvc/?username=huseyincenik&color=orange&label=Portfolio+Views)

Unified workspace that hosts three deliverables prepared for the John Snow Labs interview process. Each sub-project is fully documented, dockerized where applicable, and wired together through the navigation below.

## 🔗 Project Index

| Project                        | Logo                                                                                                                                | Description                                                                                                                | Key Links                                                                                                                                    |
| ------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| Data Curation Service          | <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQkzoKPaZIwIrnqHBYP_if-0vLt-hT6h2h-BQ&s" alt="JSL logo" width="48"/> | DocETL-powered oncology extraction pipeline that normalizes clinical notes into registry-ready JSON artifacts.             | [`README`](data_curation/README.md) · [`Docker Compose`](data_curation/docker-compose.yml)                                                   |
| CoNLL Generator & NER Training | <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQkzoKPaZIwIrnqHBYP_if-0vLt-hT6h2h-BQ&s" alt="JSL logo" width="48"/> | Spark NLP Healthcare workflow that converts pretrained model outputs into CoNLL datasets and fine-tunes custom NER models. | [`README`](generating_conll_files_from_pretrained_models/README.md) · [`Notebooks`](generating_conll_files_from_pretrained_models/notebooks) |
| RAG QA Chatbot                 | <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQkzoKPaZIwIrnqHBYP_if-0vLt-hT6h2h-BQ&s" alt="JSL logo" width="48"/> | Streamlit-based Retrieval-Augmented Generation assistant with dual LLM support, semantic caching, and observability.       | [`README`](rag_qa_chatbot_application/README.md) · [`Start Scripts`](rag_qa_chatbot_application/start.sh)                                    |

> All README files are in English for easy sharing across international review panels.

## 🧭 Quick Navigation

- `data_curation/` → Oncology DocETL extraction APIs
- `generating_conll_files_from_pretrained_models/` → Spark NLP Healthcare notebooks + training utilities
- `rag_qa_chatbot_application/` → Streamlit RAG app (OpenAI + Ollama)

## 🧰 Tech Stack Logos

- **Data Curation Service**  
  <img src="https://fastapi.tiangolo.com/img/logo-margin/logo-teal.png" alt="FastAPI" height="36">
  &nbsp;
  <img src="https://www.python.org/static/community_logos/python-logo.png" alt="Python" height="36">
  &nbsp;
  <img src="https://www.docker.com/wp-content/uploads/2022/03/Moby-logo.png" alt="Docker" height="36">
  &nbsp;
  <img src="https://seeklogo.com/images/O/openai-logo-8B9BFEDC26-seeklogo.com.png" alt="OpenAI" height="36">
  &nbsp;
  <img src="https://ollama.ai/public/icon-192.png" alt="Ollama/Qwen" height="36">

- **CoNLL Generator & NER Training**  
  <img src="https://upload.wikimedia.org/wikipedia/commons/f/f3/Apache_Spark_logo.svg" alt="Apache Spark" height="36">
  &nbsp;
  <img src="https://nlp.johnsnowlabs.com/assets/images/logo.svg" alt="Spark NLP" height="36">
  &nbsp;
  <img src="https://www.python.org/static/community_logos/python-logo.png" alt="Python" height="36">
  &nbsp;
  <img src="https://upload.wikimedia.org/wikipedia/commons/3/38/Jupyter_logo.svg" alt="Jupyter" height="36">

- **RAG QA Chatbot**  
  <img src="https://streamlit.io/images/brand/streamlit-logo-primary-colormark-darktext.png" alt="Streamlit" height="36">
  &nbsp;
  <img src="https://avatars.githubusercontent.com/u/126733545?s=200&v=4" alt="LangChain" height="36">
  &nbsp;
  <img src="https://seeklogo.com/images/O/openai-logo-8B9BFEDC26-seeklogo.com.png" alt="OpenAI" height="36">
  &nbsp;
  <img src="https://www.docker.com/wp-content/uploads/2022/03/Moby-logo.png" alt="Docker" height="36">
  &nbsp;
  <img src="https://upload.wikimedia.org/wikipedia/commons/a/a7/Faiss_logo.svg" alt="FAISS" height="36">

## 🧱 Cohesive Architecture Snapshot

```
┌──────────────────────────────────────────────────────────────────────────┐
│                          Data Ingestion & Curation                       │
│        (data_curation – DocETL Tagger → Extractor → Consolidator)        │
└───────────────┬──────────────────────────────────────────────────────────┘
                │ Structured oncology JSON + audit logs
                ▼
┌──────────────────────────────────────────────────────────────────────────┐
│           CoNLL Generation & Custom NER Training (Spark NLP)             │
│  (generating_conll_files_from_pretrained_models – notebooks + src)       │
└───────────────┬──────────────────────────────────────────────────────────┘
                │ Trained NER models + labeled datasets
                ▼
┌──────────────────────────────────────────────────────────────────────────┐
│        Retrieval-Augmented Generation QA Chatbot (Streamlit RAG)         │
│      (rag_qa_chatbot_application – dual LLM + semantic cache)            │
└──────────────────────────────────────────────────────────────────────────┘
```

Pipeline hand-offs:

1. Data Curation outputs NAACCR-aligned JSON that can seed additional labeling.
2. CoNLL generator refines labels/models that can enrich the RAG knowledge base.
3. RAG chatbot exposes curated knowledge through conversational interfaces.

## 🚀 How to Run Each Project

| Project       | Local Start                     | Docker Start                                     |
| ------------- | ------------------------------- | ------------------------------------------------ |
| Data Curation | `uv run python src/main.py`     | `./run_docker.sh`                                |
| CoNLL + NER   | Jupyter via `notebooks/*.ipynb` | Configure Spark cluster / Databricks             |
| RAG Chatbot   | `streamlit run enhanced_app.py` | `./start.sh` (Linux/Mac) / `start.bat` (Windows) |

> See the corresponding README inside each folder for environment variables, credentials, and troubleshooting details.

## 🧰 Tooling & Conventions

- **Python 3.10+** across all projects
- **uv / pip** for dependency management
- **Docker & Docker Compose** for reproducible deployments
- **VS Code + Cursor** recommended for navigating the repo

## 📬 Support

Please open issues or reach out during the interview process if any of the demos require additional credentials or clarifications.
