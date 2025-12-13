<div align="center">
  <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQkzoKPaZIwIrnqHBYP_if-0vLt-hT6h2h-BQ&s" alt="John Snow Labs logo" width="96">
  <h1>RAG QA Chatbot Application</h1>
  <p>Streamlit-based assistant with dual LLM support, semantic caching, and observability dashboards.</p>
  <img alt="Project views" src="https://komarev.com/ghpvc/?username=huseyincenik&color=orange&label=RAG+Chatbot+Views">
  <p>
    <img src="https://streamlit.io/images/brand/streamlit-logo-primary-colormark-darktext.png" alt="Streamlit" height="40">
    &nbsp;
    <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/6/60/LangChain_Logo.svg/2560px-LangChain_Logo.svg.png" alt="LangChain" height="40">
    &nbsp;
    <img src="https://upload.wikimedia.org/wikipedia/commons/4/4d/OpenAI_Logo.svg" alt="OpenAI" height="40">
    &nbsp;
    <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/docker/docker-original.svg" alt="Docker" height="40">
    &nbsp;
    <img src="https://daxg39y63pxwu.cloudfront.net/images/blog/faiss-vector-database/FAISS_Vector_Database.webp" alt="FAISS" height="40">
  </p>
</div>

---

## Project Navigator

- Portfolio overview → [`../README.md`](../README.md)
- DocETL data curation service → [`../data_curation`](../data_curation)
- Spark NLP CoNLL generator → [`../generating_conll_files_from_pretrained_models`](../generating_conll_files_from_pretrained_models)

> Feed curated oncology JSON or custom NER outputs from the sister projects into this app's `current_db/` or upload workflow to deliver an end-to-end demo.

---

## 📋 Table of Contents

- [Features](#features)
- [System Architecture](#system-architecture-deep-dive)
- [LLM Provider Comparison: OpenAI vs Qwen](#llm-provider-comparison-openai-vs-qwen)
- [Core Components & Code References](#core-components--code-references)
- [Data Flow & Pipeline](#data-flow--pipeline)
- [FAISS Vector Database System](#faiss-vector-database-system)
- [Semantic Caching System](#semantic-caching-system)
- [Document Processing & Metadata Extraction](#document-processing--metadata-extraction)
- [PDF Highlighting System](#pdf-highlighting-system)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Quick Start with Docker](#quick-start-with-docker)
- [Configuration](#configuration)
- [Usage Guide](#usage-guide)
- [Troubleshooting](#troubleshooting)

---

## ✨ Features

### Core Functionality

- **📄 Multi-Format Document Upload**: Support for PDF, DOCX, and TXT files
- **🤖 Dual LLM Support**:
  - OpenAI API (GPT-4o)
  - Local LLM via Ollama (Qwen2.5:7b)
- **🗄️ Dual Database System**:
  - **Current DB**: Pre-loaded PubMed database (persistent, comes with Docker)
  - **New DB**: Temporary in-memory database for uploaded documents
  - **Flexible Search Modes**: Choose to search in current DB, new DB, or both combined
- **🔍 Advanced Semantic Search**: FAISS vector store with cosine similarity
- **💬 Conversational Interface**: Real-time chat with document-based responses
- **📊 Source Attribution**: Complete transparency with source documents, accuracy scores, and JSON metadata

### Advanced Features

- **🎯 Contextual Compression & LLM Filtering** (OpenAI mode)
- **💾 Intelligent Semantic Caching System**
- **⚙️ Customizable Parameters** (chunk size, overlap, k, temperature)
- **📄 PDF Highlighting** for source visualization
- **📈 Real-Time Monitoring** (logs, cache stats, vector store info)
- **🗑️ Chunk Management** (delete chunks, clear cache)

---

## 🏗️ System Architecture Deep Dive

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                                    USER INTERFACE (Streamlit)                                │
│                                       enhanced_app.py                                        │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐        │
│  │  Model Selection │  │ Document Upload │  │   Chat Input    │  │  System Info   │        │
│  │  OpenAI / Qwen  │  │  PDF/DOCX/TXT   │  │   User Query    │  │  Logs/Cache    │        │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘  └────────────────┘        │
└───────────┼────────────────────┼────────────────────┼────────────────────────────────────────┘
            │                    │                    │
            ▼                    ▼                    ▼
┌───────────────────────────────────────────────────────────────────────────────────────────────┐
│                                    SERVICE LAYER (src/services/)                              │
│                                                                                               │
│  ┌─────────────────────────┐    ┌─────────────────────────┐    ┌─────────────────────────┐   │
│  │    LLM Service          │    │   Document Processor    │    │   Vector Store Manager  │   │
│  │    llm_service.py       │    │   document_processor.py │    │   vector_store.py       │   │
│  │                         │    │                         │    │                         │   │
│  │  • Initialize OpenAI    │    │  • Extract text (PDF,   │    │  • FAISS index ops      │   │
│  │  • Initialize Qwen      │    │    DOCX, TXT)           │    │  • Similarity search    │   │
│  │  • Validate connection  │    │  • Page-by-page extract │    │  • Contextual compress  │   │
│  │  • Generate answers     │    │  • Chunk creation       │    │  • Database merging     │   │
│  └─────────────────────────┘    │  • Metadata extraction  │    │  • Score calculation    │   │
│                                 └─────────────────────────┘    └─────────────────────────┘   │
│                                                                                               │
│  ┌─────────────────────────┐    ┌─────────────────────────┐    ┌─────────────────────────┐   │
│  │    Cache Manager        │    │   New DB Manager        │    │   PubMed Initializer    │   │
│  │    cache_manager.py     │    │   new_db_manager.py     │    │   pubmed_db_init...py   │   │
│  │                         │    │                         │    │                         │   │
│  │  • Semantic similarity  │    │  • Temporary stores     │    │  • Load pre-built index │   │
│  │  • Cosine matching      │    │  • In-memory FAISS      │    │  • PubMed embeddings    │   │
│  │  • TTL expiration       │    │  • Provider-specific    │    │  • OpenAI/Qwen indices  │   │
│  │  • LRU eviction         │    │  • Merge with current   │    │                         │   │
│  └─────────────────────────┘    └─────────────────────────┘    └─────────────────────────┘   │
└───────────────────────────────────────────────────────────────────────────────────────────────┘
            │                              │                                │
            ▼                              ▼                                ▼
┌───────────────────────────────────────────────────────────────────────────────────────────────┐
│                                    STORAGE LAYER                                              │
│                                                                                               │
│  ┌─────────────────────────────────────┐    ┌─────────────────────────────────────────────┐  │
│  │          FAISS Vector Store          │    │              Cache Storage                  │  │
│  │                                      │    │                                            │  │
│  │  current_db/                         │    │  data/cache/                               │  │
│  │  ├── openai/                         │    │  └── qa_cache.pkl                          │  │
│  │  │   └── faiss_index/                │    │      • Question embeddings                 │  │
│  │  │       ├── index.faiss ← vectors   │    │      • Answer pairs                        │  │
│  │  │       └── index.pkl   ← metadata  │    │      • Timestamps                          │  │
│  │  └── qwen/                           │    │      • Hit counts                          │  │
│  │      └── faiss_index/                │    │                                            │  │
│  │          ├── index.faiss             │    │                                            │  │
│  │          └── index.pkl               │    │                                            │  │
│  └─────────────────────────────────────┘    └─────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────────────────────────────────────┘
```

### Request Processing Flow

```
User Question: "What is diabetes?"
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ STEP 1: CACHE CHECK                                                             │
│ File: src/services/cache_manager.py (Lines 280-347)                             │
│                                                                                 │
│   question_embedding = embeddings.embed_query("What is diabetes?")              │
│   for cached_entry in cache:                                                    │
│       similarity = cosine_similarity(question_embedding, cached_embedding)      │
│       if similarity >= 0.85:  # 85% threshold                                  │
│           return cached_response  ✓ CACHE HIT                                   │
│                                                                                 │
│   → If no match found: Continue to RAG pipeline                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
         │
         ▼ (Cache Miss)
┌─────────────────────────────────────────────────────────────────────────────────┐
│ STEP 2: DATABASE SELECTION                                                      │
│ File: src/services/vector_store.py (Lines 503-710)                              │
│                                                                                 │
│   db_mode = "current" | "new" | "current+new"                                   │
│                                                                                 │
│   if db_mode == "current":                                                      │
│       vector_store = load(current_db/openai/faiss_index)                        │
│   elif db_mode == "new":                                                        │
│       vector_store = session_state.new_vector_store                             │
│   elif db_mode == "current+new":                                                │
│       # MERGE: Extract all docs, create combined store                          │
│       current_docs = current_db.docstore._dict.values()                         │
│       new_docs = new_db.docstore._dict.values()                                 │
│       merged_store = FAISS.from_texts(all_docs, embeddings)                     │
└─────────────────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ STEP 3: RETRIEVAL (Different per LLM provider)                                  │
│                                                                                 │
│ ┌─ OpenAI Mode (Lines 839-895) ─────────────────────────────────────────────┐   │
│ │                                                                           │   │
│ │  # Advanced MMR retrieval with Contextual Compression                     │   │
│ │  base_retriever = vector_store.as_retriever(                              │   │
│ │      search_type="mmr",                                                   │   │
│ │      search_kwargs={"k": k*4, "fetch_k": k*10, "lambda_mult": 0.5}        │   │
│ │  )                                                                        │   │
│ │                                                                           │   │
│ │  # LLM filters irrelevant content (SLOW but HIGH QUALITY)                 │   │
│ │  compressor = LLMChainExtractor.from_llm(llm)                             │   │
│ │  compression_retriever = ContextualCompressionRetriever(                  │   │
│ │      base_compressor=compressor,                                          │   │
│ │      base_retriever=base_retriever                                        │   │
│ │  )                                                                        │   │
│ └───────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│ ┌─ Qwen Mode (Lines 850-872) ───────────────────────────────────────────────┐   │
│ │                                                                           │   │
│ │  # COMPRESSION DISABLED for Qwen (CPU performance)                        │   │
│ │  use_compression = False  # Too slow on CPU                               │   │
│ │                                                                           │   │
│ │  # Use basic MMR retriever without LLM filtering                          │   │
│ │  retriever_to_use = base_retriever  # Skip compression                    │   │
│ │                                                                           │   │
│ │  ⚠️ Reason: Qwen runs locally via Ollama (CPU-bound)                      │   │
│ │     Contextual Compression requires multiple LLM calls                    │   │
│ │     Each call takes 5-15 seconds on CPU vs <1 second on OpenAI            │   │
│ └───────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ STEP 4: ANSWER GENERATION                                                       │
│ File: src/services/vector_store.py (Lines 874-895)                              │
│                                                                                 │
│   conversation_chain = ConversationalRetrievalChain.from_llm(                   │
│       llm=llm,                                                                  │
│       retriever=retriever_to_use,                                               │
│       memory=memory,                                                            │
│       return_source_documents=True,                                             │
│       combine_docs_chain_kwargs={"prompt": qa_prompt}                           │
│   )                                                                             │
│                                                                                 │
│   response = conversation_chain({"question": query, "chat_history": []})        │
└─────────────────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ STEP 5: ACCURACY SCORING                                                        │
│ File: src/services/vector_store.py (Lines 1153-1230)                            │
│                                                                                 │
│   # Calculate ANSWER-CONTENT similarity (not query-content!)                    │
│   answer_embedding = embeddings.embed_query(answer_text)                        │
│                                                                                 │
│   for source in sources:                                                        │
│       content_embedding = embeddings.embed_query(source.content)                │
│       similarity = cosine_similarity(answer_embedding, content_embedding)       │
│       source.accuracy_score = similarity  # 0.0 to 1.0                          │
│                                                                                 │
│   # Sort by accuracy: highest contribution first                                │
│   sources.sort(key=lambda x: x.accuracy_score, reverse=True)                    │
└─────────────────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ STEP 6: CACHE STORAGE                                                           │
│ File: src/services/cache_manager.py (Lines 349-401)                             │
│                                                                                 │
│   cache_entry = CacheEntry(                                                     │
│       question=query,                                                           │
│       answer=result,                                                            │
│       embedding=question_embedding  # For future semantic matching              │
│   )                                                                             │
│   cache.put(query, result, k=k)                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ RESPONSE TO USER                                                                │
│                                                                                 │
│   {                                                                             │
│     "response": "Diabetes is a chronic disease...",                             │
│     "sources": [                                                                │
│       {                                                                         │
│         "content": "...",                                                       │
│         "structured_metadata": {                                                │
│           "Document_Id": "PMC7047764",                                          │
│           "Document_Name": "diabetes_study.pdf",                                │
│           "Chunk_Index": 5,                                                     │
│           "Start_Char": 1024,                                                   │
│           "End_Char": 1824,                                                     │
│           "Is_New_Doc": false,                                                  │
│           "File_Type": "pdf",                                                   │
│           "Chunk_Size": 800,                                                    │
│           "Model_Provider": "OpenAI (API)"                                      │
│         },                                                                      │
│         "accuracy_score": 0.94                                                  │
│       }                                                                         │
│     ],                                                                          │
│     "metadata": {"cached": false, "model": "OpenAI gpt-4o"}                     │
│   }                                                                             │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 🔄 LLM Provider Comparison: OpenAI vs Qwen

### Comparison Table

| Feature | OpenAI (API) | Local LLM (Qwen) |
|---------|--------------|------------------|
| **Model** | GPT-4o | Qwen2.5:7b |
| **Embeddings** | text-embedding-ada-002 | all-minilm |
| **Response Time** | 2-5 seconds | 10-60 seconds |
| **Contextual Compression** | ✅ Enabled | ❌ Disabled (too slow) |
| **Cost** | Pay per token | Free (local) |
| **Internet Required** | Yes | No (after download) |
| **Default Temperature** | 0.7 | 0.3 |
| **Timeout** | Default | 120 seconds |

### Code Implementation Differences

#### 1. LLM Initialization (`src/services/llm_service.py`, Lines 29-78)

```python
# OpenAI Initialization
if model_provider == "OpenAI (API)":
    self.llm = ChatOpenAI(
        model=config.model.openai_model,           # gpt-4o
        temperature=config.model.openai_temperature, # 0.7
        max_tokens=config.model.openai_max_tokens,
        openai_api_key=api_key
    )

# Qwen/Ollama Initialization
elif model_provider == "Local LLM (Qwen)":
    self.llm = ChatOpenAI(
        model=config.model.local_model,            # qwen2.5:7b
        temperature=config.model.local_temperature, # 0.3 (more deterministic)
        max_tokens=config.model.local_max_tokens,
        openai_api_key='ollama',                   # Placeholder
        openai_api_base=config.model.ollama_base_url, # http://ollama:11434/v1
        request_timeout=120                         # 2 min timeout for slow CPU
    )
```

#### 2. Embedding Initialization (`src/services/vector_store.py`, Lines 194-236)

```python
# OpenAI Embeddings
if model_provider == "OpenAI (API)":
    self.embeddings = OpenAIEmbeddings(
        model=config.model.openai_embedding_model,  # text-embedding-ada-002
        openai_api_key=api_key
    )

# Qwen/Ollama Embeddings
elif model_provider == "Local LLM (Qwen)":
    self.embeddings = OllamaEmbeddings(
        model=config.model.local_embedding_model,   # all-minilm
        base_url=config.model.ollama_base_url.replace("/v1", ""),
        num_ctx=2048  # Increased context window for embeddings
    )
```

#### 3. Retrieval Strategy (`src/services/vector_store.py`, Lines 850-872)

```python
# CRITICAL DIFFERENCE: Compression disabled for Qwen
use_compression = self.current_model_provider != "Local LLM (Qwen)"

if use_compression:
    # OpenAI: Use LLM Compression for better quality
    compressor = LLMChainExtractor.from_llm(llm)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever
    )
    retriever_to_use = compression_retriever
else:
    # Qwen: Skip compression (too slow on CPU)
    # Each compression call = 5-15 seconds on CPU
    # With k=5 sources, that's 25-75 seconds extra latency!
    retriever_to_use = base_retriever
```

### 🔴 CRITICAL: Why Contextual Compression is Disabled for Qwen

#### What is Contextual Compression?

Contextual Compression is a LangChain feature that uses the LLM to **filter and extract only relevant parts** from retrieved documents before generating the final answer. It improves answer quality by removing irrelevant content.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    CONTEXTUAL COMPRESSION EXPLAINED                             │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  WITHOUT Compression (Basic Retrieval):                                         │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │  Query: "What causes diabetes?"                                          │   │
│  │                    │                                                     │   │
│  │                    ▼                                                     │   │
│  │  ┌──────────────────────────────────────────────────────────────────┐   │   │
│  │  │ Retrieved Chunk (800 chars):                                      │   │   │
│  │  │ "Diabetes mellitus is a metabolic disease. The pancreas produces  │   │   │
│  │  │ insulin which regulates blood sugar. Type 2 diabetes is caused    │   │   │
│  │  │ by insulin resistance and beta-cell dysfunction. Risk factors     │   │   │
│  │  │ include obesity, sedentary lifestyle, genetics... [irrelevant     │   │   │
│  │  │ hospital statistics and study methodology details...]"            │   │   │
│  │  └──────────────────────────────────────────────────────────────────┘   │   │
│  │                    │                                                     │   │
│  │                    ▼ (All 800 chars sent to LLM)                         │   │
│  │             ┌─────────────┐                                              │   │
│  │             │  Generate   │                                              │   │
│  │             │   Answer    │                                              │   │
│  │             └─────────────┘                                              │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│  WITH Compression (LLM Filtering):                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │  Query: "What causes diabetes?"                                          │   │
│  │                    │                                                     │   │
│  │                    ▼                                                     │   │
│  │  ┌──────────────────────────────────────────────────────────────────┐   │   │
│  │  │ Retrieved Chunk (800 chars): [same as above]                      │   │   │
│  │  └──────────────────────────────────────────────────────────────────┘   │   │
│  │                    │                                                     │   │
│  │                    ▼ (LLM call #1: Compression)                          │   │
│  │  ┌──────────────────────────────────────────────────────────────────┐   │   │
│  │  │ LLMChainExtractor prompt:                                         │   │   │
│  │  │ "Given the question 'What causes diabetes?', extract ONLY the     │   │   │
│  │  │ relevant parts from this document..."                             │   │   │
│  │  │                                                                   │   │   │
│  │  │ Output (200 chars):                                               │   │   │
│  │  │ "Type 2 diabetes is caused by insulin resistance and beta-cell   │   │   │
│  │  │ dysfunction. Risk factors include obesity, sedentary lifestyle,  │   │   │
│  │  │ genetics."                                                        │   │   │
│  │  └──────────────────────────────────────────────────────────────────┘   │   │
│  │                    │                                                     │   │
│  │                    ▼ (Only 200 chars sent to LLM)                        │   │
│  │             ┌─────────────┐                                              │   │
│  │             │  Generate   │ (LLM call #2: Answer)                       │   │
│  │             │   Answer    │                                              │   │
│  │             └─────────────┘                                              │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────┘
```

#### The Problem with Qwen + Compression

For each retrieved document, the LLMChainExtractor makes **a separate LLM call** to compress it. With `k=5` sources:

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│              COMPRESSION OVERHEAD: OpenAI vs Qwen                               │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  OpenAI (k=5 sources):                                                          │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │  Compress Doc 1   Compress Doc 2   Compress Doc 3   ...   Compress Doc 5 │   │
│  │      0.3 sec    +     0.3 sec    +     0.3 sec    + ... +     0.3 sec    │   │
│  │  ────────────────────────────────────────────────────────────────────── │   │
│  │                     Total: ~1.5 seconds                                  │   │
│  │                         (Parallel on GPU)                                │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│  Qwen on CPU (k=5 sources):                                                     │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │  Compress Doc 1 → Compress Doc 2 → Compress Doc 3 → ... → Compress Doc 5 │   │
│  │      10 sec     +     12 sec     +     8 sec      + ... +     15 sec     │   │
│  │  ────────────────────────────────────────────────────────────────────── │   │
│  │                     Total: ~55 seconds                                   │   │
│  │              (Sequential on CPU, memory-bound)                           │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│  ⚠️ REASON FOR SLOWNESS:                                                        │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │ 1. CPU Inference: 7B parameters loaded into RAM (~4GB)                   │   │
│  │ 2. No GPU Acceleration: Each token generated sequentially                │   │
│  │ 3. Memory Bandwidth: CPU RAM slower than GPU VRAM                        │   │
│  │ 4. Model Loading: Qwen model kept in memory but inference is slow        │   │
│  │ 5. Each Compression = Full LLM Forward Pass (~5-15 seconds)              │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────┘
```

#### Code Implementation Detail

**File: `src/services/vector_store.py`, Lines 850-872**

```python
# TEMPORARY: Disable Contextual Compression for Ollama (too slow on CPU)
# Use basic retriever for faster responses
use_compression = self.current_model_provider != "Local LLM (Qwen)"

if use_compression:
    # OpenAI Mode: Full compression pipeline
    # Step 1: Create LLM Chain Extractor for contextual compression
    compressor = LLMChainExtractor.from_llm(llm)
    
    # Step 2: Wrap the base retriever with compression
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,    # Uses LLM to filter content
        base_retriever=base_retriever   # MMR retriever with k*4 candidates
    )
    
    self.logger.info(
        f"Using Contextual Compression Retriever with LLM filtering for query: '{query[:50]}...'"
    )
    retriever_to_use = compression_retriever
else:
    # Qwen Mode: Skip compression for performance
    self.logger.info(
        f"Using basic retriever (Compression disabled for Ollama/CPU) for query: '{query[:50]}...'"
    )
    retriever_to_use = base_retriever  # Direct MMR retrieval, no LLM filtering
```

#### Quality Trade-off

| Aspect | With Compression (OpenAI) | Without Compression (Qwen) |
|--------|---------------------------|----------------------------|
| Answer Precision | Higher (filtered context) | Lower (full chunks) |
| Irrelevant Content | Removed by LLM | May be included |
| Response Time | +1-2 seconds | No overhead |
| Token Usage | Lower (compressed) | Higher (full chunks) |

### Why Qwen is Slower: Technical Explanation

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        OPENAI vs QWEN PERFORMANCE                               │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  OpenAI (Cloud GPU):                                                            │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐               │
│  │Embedding│→ │ Search  │→ │Compress │→ │ Filter  │→ │ Answer  │= 3-5 sec      │
│  │ 100ms   │  │ 200ms   │  │1-2 sec  │  │ 500ms   │  │1-2 sec  │               │
│  └─────────┘  └─────────┘  └─────────┘  └─────────┘  └─────────┘               │
│                                                                                 │
│  Qwen (Local CPU):                                                              │
│  ┌─────────┐  ┌─────────┐  ┌───────────────────────────────────┐               │
│  │Embedding│→ │ Search  │→ │           Answer                  │= 10-60 sec    │
│  │2-5 sec  │  │ 500ms   │  │          10-50 sec                │               │
│  └─────────┘  └─────────┘  └───────────────────────────────────┘               │
│                            ↑                                                    │
│                            │ Compression SKIPPED                                │
│                                                                                 │
│  ⚠️ If compression was enabled for Qwen:                                        │
│  ┌─────────┐  ┌─────────┐  ┌─────────────────────────────────────────────────┐ │
│  │Embedding│→ │ Search  │→ │     Compress (5-15 sec × k sources)             │ │
│  │2-5 sec  │  │ 500ms   │  │     = 25-75 extra seconds!                      │ │
│  └─────────┘  └─────────┘  └─────────────────────────────────────────────────┘ │
│                                                                                 │
│  Total with compression: 60-150 seconds (unacceptable UX)                       │
│  Total without compression: 10-60 seconds (acceptable)                          │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 🗄️ FAISS Vector Database System

### Database Structure

```
current_db/
├── openai/
│   └── faiss_index/
│       ├── index.faiss     ← Binary vector data (embeddings)
│       └── index.pkl       ← Document metadata (pickled Python dict)
└── qwen/
    └── faiss_index/
        ├── index.faiss     ← Binary vector data (different dimension!)
        └── index.pkl       ← Document metadata
```

### Why Separate Indices per Provider?

OpenAI and Qwen use different embedding models with **different vector dimensions**:

| Provider | Embedding Model | Vector Dimension |
|----------|-----------------|------------------|
| OpenAI | text-embedding-ada-002 | 1536 |
| Qwen | all-minilm | 384 |

**You cannot mix vectors of different dimensions in the same FAISS index!**

### FAISS Index Internal Structure

#### index.faiss - Binary Vector Data

The `index.faiss` file contains the actual embedding vectors stored in a highly optimized binary format:

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         index.faiss INTERNAL STRUCTURE                          │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  File Header:                                                                   │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │  Index Type: IndexFlatIP (Inner Product / Cosine Similarity)             │   │
│  │  Dimension: 1536 (OpenAI) or 384 (Qwen)                                  │   │
│  │  Total Vectors: 2480 (PubMed chunks)                                     │   │
│  │  Normalized: Yes (L2 normalization for cosine similarity)                │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│  Vector Storage (for OpenAI, 1536 dimensions):                                  │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │  Vector 0: [0.0123, -0.0456, 0.0789, ..., 0.0321]  ← 1536 float32 values │   │
│  │  Vector 1: [-0.0234, 0.0567, -0.0890, ..., -0.0432]                      │   │
│  │  Vector 2: [0.0345, -0.0678, 0.0901, ..., 0.0543]                        │   │
│  │  ...                                                                     │   │
│  │  Vector 2479: [0.0456, 0.0789, -0.0123, ..., -0.0654]                    │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│  Memory Layout:                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │  Total Size = num_vectors × dimension × sizeof(float32)                  │   │
│  │             = 2480 × 1536 × 4 bytes                                      │   │
│  │             = ~15.3 MB (OpenAI index)                                    │   │
│  │                                                                          │   │
│  │  For Qwen:  = 2480 × 384 × 4 bytes = ~3.8 MB                             │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────┘
```

#### index.pkl - Document Metadata (Pickled Python Objects)

The `index.pkl` file contains the document store mapping vector IDs to their content and metadata:

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         index.pkl INTERNAL STRUCTURE                            │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  Serialized Python Object (pickle format):                                      │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                                                                          │   │
│  │  {                                                                       │   │
│  │    "docstore": InMemoryDocstore({                                        │   │
│  │                                                                          │   │
│  │      "abc123-uuid-1": Document(                                          │   │
│  │        page_content="Diabetes mellitus is a metabolic disease...",       │   │
│  │        metadata={                                                        │   │
│  │          "pubmed_id": "PMC7047764",                                      │   │
│  │          "document_name": "PubMed_PMC7047764",                           │   │
│  │          "chunk_id": "abc123-uuid-1",                                    │   │
│  │          "chunk_index": 0,                                               │   │
│  │          "page": 1,                                                      │   │
│  │          "start_char": 0,                                                │   │
│  │          "end_char": 800,                                                │   │
│  │          "start_char_in_page": 0,                                        │   │
│  │          "end_char_in_page": 800,                                        │   │
│  │          "file_type": "pdf",                                             │   │
│  │          "source": "PubMed_PMC7047764"                                   │   │
│  │        }                                                                 │   │
│  │      ),                                                                  │   │
│  │                                                                          │   │
│  │      "def456-uuid-2": Document(                                          │   │
│  │        page_content="Type 2 diabetes is characterized by...",            │   │
│  │        metadata={                                                        │   │
│  │          "pubmed_id": "PMC7047764",                                      │   │
│  │          "document_name": "PubMed_PMC7047764",                           │   │
│  │          "chunk_id": "def456-uuid-2",                                    │   │
│  │          "chunk_index": 1,                                               │   │
│  │          "page": 1,                                                      │   │
│  │          "start_char": 700,                                              │   │
│  │          "end_char": 1500,                                               │   │
│  │          ...                                                             │   │
│  │        }                                                                 │   │
│  │      ),                                                                  │   │
│  │                                                                          │   │
│  │      ... (2478 more Document objects)                                    │   │
│  │                                                                          │   │
│  │    }),                                                                   │   │
│  │                                                                          │   │
│  │    "index_to_docstore_id": {                                             │   │
│  │      0: "abc123-uuid-1",   ← Maps FAISS vector index to document UUID    │   │
│  │      1: "def456-uuid-2",                                                 │   │
│  │      2: "ghi789-uuid-3",                                                 │   │
│  │      ...                                                                 │   │
│  │      2479: "xyz999-uuid-2480"                                            │   │
│  │    }                                                                     │   │
│  │  }                                                                       │   │
│  │                                                                          │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### FAISS Search Process

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         FAISS SIMILARITY SEARCH                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  Query: "What causes diabetes?"                                                 │
│         │                                                                       │
│         ▼                                                                       │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │ STEP 1: Embed Query                                                      │   │
│  │                                                                          │   │
│  │   query_embedding = embeddings.embed_query("What causes diabetes?")      │   │
│  │   # Result: [0.0234, -0.0567, 0.0890, ..., 0.0123]  (1536 dims)          │   │
│  └──────────────────────────────────┬──────────────────────────────────────┘   │
│                                     ▼                                           │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │ STEP 2: Compute Similarity with ALL Vectors                              │   │
│  │                                                                          │   │
│  │   FAISS uses Inner Product (cosine similarity for normalized vectors):   │   │
│  │                                                                          │   │
│  │   similarity(query, vector_i) = Σ(query[j] × vector_i[j]) for j=0..1535  │   │
│  │                                                                          │   │
│  │   Results:                                                               │   │
│  │   ┌──────────┬────────────┐                                              │   │
│  │   │ Vector 0 │ sim: 0.82  │ ← "Diabetes mellitus is a metabolic..."     │   │
│  │   │ Vector 1 │ sim: 0.91  │ ← "Type 2 diabetes is caused by..."  ⭐     │   │
│  │   │ Vector 2 │ sim: 0.45  │                                              │   │
│  │   │ ...      │ ...        │                                              │   │
│  │   │ Vector 547│ sim: 0.88 │ ← "Risk factors for diabetes include..."    │   │
│  │   │ ...      │ ...        │                                              │   │
│  │   └──────────┴────────────┘                                              │   │
│  └──────────────────────────────────┬──────────────────────────────────────┘   │
│                                     ▼                                           │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │ STEP 3: Return Top-k Results (k=5)                                       │   │
│  │                                                                          │   │
│  │   # FAISS returns vector indices sorted by similarity                    │   │
│  │   top_k_indices = [1, 547, 823, 1245, 0]  # Vector indices               │   │
│  │   top_k_scores  = [0.91, 0.88, 0.85, 0.84, 0.82]  # Similarity scores   │   │
│  │                                                                          │   │
│  │   # Map indices to documents using index.pkl                             │   │
│  │   for idx in top_k_indices:                                              │   │
│  │       doc_id = index_to_docstore_id[idx]  # e.g., "def456-uuid-2"       │   │
│  │       document = docstore[doc_id]          # Get Document object         │   │
│  │       yield document.page_content, document.metadata                     │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### MMR (Maximal Marginal Relevance) Retrieval

The application uses MMR to balance relevance and diversity:

```python
# File: src/services/vector_store.py, Lines 825-838
base_retriever = vector_store_to_use.as_retriever(
    search_type="mmr",  # Maximal Marginal Relevance
    search_kwargs={
        "k": k * 4,          # Fetch 4x candidates (e.g., 20 for k=5)
        "fetch_k": k * 10,   # Initial pool (e.g., 50 for k=5)
        "lambda_mult": 0.5   # Balance: 0=diversity, 1=relevance
    }
)
```

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    MMR: RELEVANCE vs DIVERSITY                                  │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  Without MMR (pure similarity):                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │  Top 5 results might all be from same document section:                  │   │
│  │  1. "Diabetes is caused by insulin resistance..." (sim: 0.91)           │   │
│  │  2. "Diabetes is caused by genetic factors..." (sim: 0.90)  ← Similar   │   │
│  │  3. "Diabetes causes include lifestyle..." (sim: 0.89)      ← Similar   │   │
│  │  4. "Causes of diabetes are multifactorial..." (sim: 0.88) ← Similar   │   │
│  │  5. "Diabetes causation involves..." (sim: 0.87)           ← Similar   │   │
│  │                                                                          │   │
│  │  ⚠️ Redundant information, limited perspective                           │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│  With MMR (lambda=0.5):                                                         │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │  Top 5 results balanced for relevance AND diversity:                     │   │
│  │  1. "Diabetes is caused by insulin resistance..." (sim: 0.91)           │   │
│  │  2. "Treatment options for diabetes include..." (sim: 0.75)  ← Diverse  │   │
│  │  3. "Risk factors: obesity, age, genetics..." (sim: 0.82)   ← Diverse  │   │
│  │  4. "Type 1 vs Type 2 diabetes differences..." (sim: 0.78) ← Diverse  │   │
│  │  5. "Prevention through diet and exercise..." (sim: 0.73)  ← Diverse  │   │
│  │                                                                          │   │
│  │  ✅ Comprehensive coverage of the topic                                   │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Database Modes (`src/services/vector_store.py`, Lines 503-710)

```python
# Mode 1: Current DB Only
if self.db_mode == "current":
    current_db_path = self._get_current_db_path()  # current_db/openai/faiss_index
    self._load_vector_store(custom_path=current_db_path)
    vector_store_to_use = self.vector_store

# Mode 2: New DB Only (uploaded documents)
elif self.db_mode == "new":
    vector_store_to_use = new_vector_store  # In-memory, temporary

# Mode 3: Current + New (Merged)
elif self.db_mode == "current+new":
    # Extract all documents from both stores
    current_docs = list(current_db_vector_store.docstore._dict.values())
    new_docs = list(new_db_vector_store.docstore._dict.values())
    
    # Extract texts and metadata
    all_texts = [doc.page_content for doc in current_docs + new_docs]
    all_metadatas = [doc.metadata for doc in current_docs + new_docs]
    
    # Create NEW merged store (original stores unchanged!)
    vector_store_to_use = FAISS.from_texts(
        texts=all_texts,
        embedding=self.embeddings,
        metadatas=all_metadatas,
        normalize_L2=True
    )
```

### Database Merge Visualization

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    CURRENT + NEW DATABASE MERGE                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  current_db (PubMed)          new_db (Uploaded)                                 │
│  ┌─────────────────┐          ┌─────────────────┐                              │
│  │ Doc 1: Cancer   │          │ Doc A: Report   │                              │
│  │ Doc 2: Diabetes │          │ Doc B: Study    │                              │
│  │ Doc 3: Heart    │          │                 │                              │
│  │ (2480 chunks)   │          │ (50 chunks)     │                              │
│  └────────┬────────┘          └────────┬────────┘                              │
│           │                            │                                        │
│           └──────────┬─────────────────┘                                        │
│                      ▼                                                          │
│           ┌─────────────────────┐                                               │
│           │   MERGE OPERATION    │                                               │
│           │                      │                                               │
│           │ • Extract all docs   │ ← Lines 609-656                              │
│           │ • Combine texts      │                                               │
│           │ • Create new FAISS   │ ← Lines 668-677                              │
│           │ • Original unchanged │                                               │
│           └──────────┬──────────┘                                               │
│                      ▼                                                          │
│           ┌─────────────────────┐                                               │
│           │  MERGED STORE        │                                               │
│           │  (In-memory only)    │                                               │
│           │  2530 total chunks   │                                               │
│           │                      │                                               │
│           │  ⚠️ NOT SAVED!       │                                               │
│           │  Only for this query │                                               │
│           └─────────────────────┘                                               │
│                                                                                 │
│  File: src/services/vector_store.py, Lines 598-693                              │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 💾 Semantic Caching System

### How Caching Works

The cache system uses **semantic similarity** rather than exact string matching. This means:
- "What is diabetes?" and "Tell me about diabetes" may hit the same cache entry
- Different phrasing of the same question gets cached responses
- 85% similarity threshold prevents false matches between related but different topics

### Cache Architecture (`src/services/cache_manager.py`)

```python
class CacheEntry:
    """Cache entry with semantic embedding"""
    def __init__(self, question: str, answer: Dict, embedding: np.ndarray):
        self.question = question
        self.answer = answer
        self.embedding = embedding      # For semantic similarity matching
        self.timestamp = datetime.now()
        self.hit_count = 0
        self.last_accessed = datetime.now()
```

### Similarity Matching Algorithm (Lines 121-159)

```python
def _compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """Compute cosine similarity between two semantic embeddings"""
    
    # Normalize to unit vectors
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    
    # Cosine similarity: dot product of normalized vectors
    similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
    
    return max(0.0, min(1.0, similarity))  # Clamp to [0, 1]
```

### Cache Lookup Process (Lines 280-347)

```
User Query: "What causes diabetes?"
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ STEP 1: Generate Query Embedding                                                │
│                                                                                 │
│   query_embedding = embeddings.embed_query("What causes diabetes?")             │
│   # Returns: np.array([0.12, -0.34, 0.56, ...]) # 1536 dimensions for OpenAI    │
└─────────────────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ STEP 2: Search Cache for Similar Questions                                      │
│                                                                                 │
│   for cache_key, entry in self.cache.items():                                   │
│       if entry.is_expired(ttl=3600):  # Skip if older than 1 hour              │
│           continue                                                              │
│                                                                                 │
│       similarity = cosine_similarity(query_embedding, entry.embedding)          │
│                                                                                 │
│       if similarity >= 0.85:  # 85% threshold                                   │
│           return entry.answer  # ✓ CACHE HIT!                                   │
│                                                                                 │
│   Cached: "What is diabetes?" → similarity = 0.72 → ✗ Below threshold           │
│   Cached: "Diabetes causes?" → similarity = 0.91 → ✓ Match found!               │
└─────────────────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ STEP 3: Return Cached Response with Metadata                                    │
│                                                                                 │
│   cached_answer["metadata"]["cached"] = True                                    │
│   cached_answer["metadata"]["cache_similarity"] = 0.91                          │
│   cached_answer["metadata"]["cache_hit_count"] = 3                              │
│   cached_answer["metadata"]["original_question"] = "Diabetes causes?"           │
│                                                                                 │
│   return cached_answer                                                          │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Cache Statistics

```python
# Get cache statistics (Line 431-452)
stats = {
    "enabled": True,
    "total_entries": 25,
    "max_size": 100,
    "hits": 150,
    "misses": 50,
    "total_queries": 200,
    "hit_rate": 75.0,  # 75% of queries served from cache
    "cache_saves": 50,
    "ttl_seconds": 3600  # 1 hour TTL
}
```

---

## 📄 Document Processing & Metadata Extraction

### Document Processing Flow (`src/services/document_processor.py`)

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                      DOCUMENT PROCESSING PIPELINE                                │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  Upload: report.pdf                                                             │
│         │                                                                       │
│         ▼                                                                       │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │ STEP 1: VALIDATION (Lines 145-165)                                       │   │
│  │                                                                          │   │
│  │   • Check file extension (.pdf, .docx, .txt)                             │   │
│  │   • Check file size (< max_file_size_mb)                                 │   │
│  │   • Check if empty                                                       │   │
│  └──────────────────────────────────┬──────────────────────────────────────┘   │
│                                     ▼                                           │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │ STEP 2: TEXT EXTRACTION (Lines 225-328)                                  │   │
│  │                                                                          │   │
│  │   PDF: _extract_pdf_text_by_pages() using PyMuPDF (fitz)                 │   │
│  │   DOCX: _extract_docx_text_by_pages() with paragraph splitting           │   │
│  │   TXT: _extract_txt_text_by_pages() with line-based pagination           │   │
│  │                                                                          │   │
│  │   Output: [(page_1_text, 1), (page_2_text, 2), ...]                      │   │
│  └──────────────────────────────────┬──────────────────────────────────────┘   │
│                                     ▼                                           │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │ STEP 3: CHUNKING WITH POSITION TRACKING (Lines 468-606)                  │   │
│  │                                                                          │   │
│  │   text_splitter = RecursiveCharacterTextSplitter(                        │   │
│  │       chunk_size=800,                                                    │   │
│  │       chunk_overlap=100                                                  │   │
│  │   )                                                                      │   │
│  │                                                                          │   │
│  │   for page_text, page_number in pages_content:                           │   │
│  │       chunks = text_splitter.split_text(page_text)                       │   │
│  │       for chunk in chunks:                                               │   │
│  │           # Track position in page for highlighting                      │   │
│  │           chunk_start_in_page = page_text.find(chunk)                    │   │
│  │           chunk_end_in_page = chunk_start_in_page + len(chunk)           │   │
│  └──────────────────────────────────┬──────────────────────────────────────┘   │
│                                     ▼                                           │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │ STEP 4: METADATA CREATION (Lines 583-600)                                │   │
│  │                                                                          │   │
│  │   chunk = DocumentChunk(                                                 │   │
│  │       id=str(uuid.uuid4()),                                              │   │
│  │       document_id=document.id,                                           │   │
│  │       content=chunk_text,                                                │   │
│  │       chunk_index=chunk_index,                                           │   │
│  │       start_char=chunk_start_in_page,     # ← For highlighting           │   │
│  │       end_char=chunk_end_in_page,         # ← For highlighting           │   │
│  │       metadata={                                                         │   │
│  │           'page': page_number,            # Real page number             │   │
│  │           'document_name': document.name,                                │   │
│  │           'file_type': 'pdf',                                            │   │
│  │           'chunk_size': len(chunk_text),                                 │   │
│  │           'model_provider': 'OpenAI (API)',                              │   │
│  │           'start_char_in_page': chunk_start_in_page,  # ← Key for PDF    │   │
│  │           'end_char_in_page': chunk_end_in_page,      # ← Key for PDF    │   │
│  │           'page_text_length': len(page_text)                             │   │
│  │       }                                                                  │   │
│  │   )                                                                      │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Metadata JSON Structure

When a response is generated, each source includes structured metadata:

```json
{
  "Document_Id": "PMC7047764",
  "Document_Name": "diabetes_research.pdf",
  "Chunk_Index": 5,
  "Start_Char": 1024,
  "End_Char": 1824,
  "Is_New_Doc": false,
  "File_Type": "pdf",
  "Chunk_Size": 800,
  "Model_Provider": "OpenAI (API)"
}
```

**Code Reference:** `src/services/vector_store.py`, Lines 1118-1130

### Context JSON Structure

The Context expander shows all source content:

```json
[
  {
    "source_index": 1,
    "file_name": "diabetes_research.pdf",
    "page": "5",
    "content": "Diabetes mellitus is a chronic disease characterized by..."
  },
  {
    "source_index": 2,
    "file_name": "diabetes_research.pdf",
    "page": "12",
    "content": "Type 2 diabetes is most commonly caused by..."
  }
]
```

**Code Reference:** `enhanced_app.py`, Lines 2006-2043

---

## 🔍 PDF Highlighting System

### How PDF Highlighting Works (`enhanced_app.py`, Lines 2244-2642)

The system highlights the exact text chunks used to generate the answer:

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         PDF HIGHLIGHTING FLOW                                    │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  User selects: "diabetes_research.pdf Page: 5"                                  │
│         │                                                                       │
│         ▼                                                                       │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │ STEP 1: GET PDF CONTENT (Lines 2103-2152)                                │   │
│  │                                                                          │   │
│  │   pdf_contents = st.session_state.get("uploaded_pdf_contents", {})       │   │
│  │   pdf_content = pdf_contents["diabetes_research.pdf"]  # bytes           │   │
│  └──────────────────────────────────┬──────────────────────────────────────┘   │
│                                     ▼                                           │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │ STEP 2: FIND MATCHING SOURCES (Lines 2165-2176)                          │   │
│  │                                                                          │   │
│  │   matching_sources = []                                                  │   │
│  │   for source in sources:                                                 │   │
│  │       if source.file == selected_file and source.page == selected_page:  │   │
│  │           matching_sources.append(source)                                │   │
│  │   # Multiple chunks can be on same page!                                 │   │
│  └──────────────────────────────────┬──────────────────────────────────────┘   │
│                                     ▼                                           │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │ STEP 3: OPEN PDF WITH PyMuPDF (Lines 2264-2274)                          │   │
│  │                                                                          │   │
│  │   import fitz  # PyMuPDF                                                 │   │
│  │   pdf_document = fitz.open(stream=pdf_content, filetype="pdf")           │   │
│  │   page = pdf_document[page_number - 1]  # 0-indexed                      │   │
│  │   page_text = page.get_text()                                            │   │
│  └──────────────────────────────────┬──────────────────────────────────────┘   │
│                                     ▼                                           │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │ STEP 4: POSITION-BASED HIGHLIGHTING (Lines 2296-2329)                    │   │
│  │                                                                          │   │
│  │   # Strategy 1: Use character positions (most accurate)                  │   │
│  │   start_char = source.get("start_char_in_page")                          │   │
│  │   end_char = source.get("end_char_in_page")                              │   │
│  │                                                                          │   │
│  │   if start_char and end_char:                                            │   │
│  │       exact_text = page_text[start_char:end_char]                        │   │
│  │       instances = page.search_for(exact_text.strip())                    │   │
│  │       for inst in instances:                                             │   │
│  │           highlight = page.add_highlight_annot(inst)                     │   │
│  │           highlight.set_colors(stroke=[1, 1, 0])  # Yellow               │   │
│  │           highlight.update()                                             │   │
│  └──────────────────────────────────┬──────────────────────────────────────┘   │
│                                     ▼                                           │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │ STEP 5: FALLBACK STRATEGIES (Lines 2331-2365)                            │   │
│  │                                                                          │   │
│  │   # Strategy 2: Content-based search                                     │   │
│  │   if not text_instances:                                                 │   │
│  │       highlight_text = source.get("content", "")                         │   │
│  │       instances = page.search_for(highlight_text_clean)                  │   │
│  │                                                                          │   │
│  │   # Strategy 3: Sentence splitting                                       │   │
│  │   if not instances:                                                      │   │
│  │       sentences = re.split(r'[.!?]\s+', highlight_text_clean)            │   │
│  │       for sentence in sentences:                                         │   │
│  │           instances.extend(page.search_for(sentence))                    │   │
│  │                                                                          │   │
│  │   # Strategy 4: Word chunks                                              │   │
│  │   if not instances:                                                      │   │
│  │       words = highlight_text_clean.split()                               │   │
│  │       for i in range(0, len(words), 15):  # 15 word chunks               │   │
│  │           chunk = ' '.join(words[i:i+15])                                │   │
│  │           instances.extend(page.search_for(chunk))                       │   │
│  └──────────────────────────────────┬──────────────────────────────────────┘   │
│                                     ▼                                           │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │ STEP 6: RENDER TO IMAGE (Lines 2397-2419)                                │   │
│  │                                                                          │   │
│  │   from pdf2image import convert_from_bytes                               │   │
│  │                                                                          │   │
│  │   # Create new PDF with just the highlighted page                        │   │
│  │   new_pdf = fitz.open()                                                  │   │
│  │   new_pdf.insert_pdf(pdf_document, from_page=page_number-1,              │   │
│  │                      to_page=page_number-1)                              │   │
│  │   page_pdf_bytes = new_pdf.tobytes()                                     │   │
│  │                                                                          │   │
│  │   images = convert_from_bytes(page_pdf_bytes, dpi=150)                   │   │
│  │   st.image(images[0], use_container_width=True)                          │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Text Extraction Consistency

**Important:** Both text extraction (for chunking) and highlighting use PyMuPDF (`fitz`) to ensure consistency:

```python
# Document Processing (src/services/document_processor.py, Lines 280-300)
import fitz
pdf_document = fitz.open(stream=file_content, filetype="pdf")
page = pdf_document[page_num]
page_text = page.get_text()  # Same method as highlighting

# PDF Highlighting (enhanced_app.py, Lines 2264-2283)
import fitz
pdf_document = fitz.open(stream=pdf_content, filetype="pdf")
page = pdf_document[page_number - 1]
page_text = page.get_text()  # Identical extraction!
```

---

## 🛠️ Technology Stack

### Frontend
- **Streamlit**: Interactive web interface
- **Custom CSS**: Modern, responsive UI design

### Backend
- **LangChain**: RAG pipeline orchestration
- **FAISS**: Vector similarity search
- **OpenAI Embeddings**: text-embedding-ada-002
- **Ollama Embeddings**: all-minilm

### LLM Models
- **OpenAI**: GPT-4o (API-based)
- **Qwen**: Qwen2.5:7b (local via Ollama)

### Document Processing
- **PyPDF2**: PDF parsing
- **PyMuPDF (fitz)**: PDF text extraction and highlighting
- **python-docx**: DOCX parsing
- **LangChain Text Splitters**: Intelligent chunking

### Storage
- **FAISS**: Vector database (in-memory/persistent)
- **Pickle**: Metadata and cache storage

### Containerization
- **Docker**: Application containerization
- **Docker Compose**: Multi-container orchestration

---

## 📁 Project Structure

```
rag_qa_chatbot_application/
│
├── enhanced_app.py              # Main Streamlit application (2813 lines)
│   ├── ChatbotApp class         # Main application class
│   ├── _display_pdf_page()      # PDF highlighting (Lines 2244-2642)
│   ├── _render_sidebar()        # UI configuration
│   └── _process_user_query()    # Query processing
│
├── src/                         # Source code modules
│   ├── config/
│   │   └── settings.py          # Application settings & constants
│   │
│   ├── models/
│   │   └── data_models.py       # Pydantic models
│   │
│   ├── services/
│   │   ├── document_processor.py    # Document parsing & chunking (615 lines)
│   │   │   ├── process_uploaded_files()
│   │   │   ├── _extract_pdf_text_by_pages()  ← PyMuPDF
│   │   │   └── create_chunks()               ← Position tracking
│   │   │
│   │   ├── vector_store.py          # FAISS vector store + RAG (2103 lines)
│   │   │   ├── initialize_embeddings()       # OpenAI vs Qwen
│   │   │   ├── search_documents_for_openai() # With compression
│   │   │   ├── _get_current_db_path()        # Database selection
│   │   │   └── set_db_mode()                 # current/new/current+new
│   │   │
│   │   ├── llm_service.py           # LLM initialization & management (362 lines)
│   │   │   ├── initialize_llm()     # OpenAI vs Qwen setup
│   │   │   └── validate_api_connection()
│   │   │
│   │   ├── cache_manager.py         # Semantic caching system (511 lines)
│   │   │   ├── _compute_similarity()         # Cosine similarity
│   │   │   ├── _find_similar_question()      # 85% threshold
│   │   │   ├── get()                         # Cache lookup
│   │   │   └── put()                         # Cache storage
│   │   │
│   │   ├── new_db_manager.py        # Temporary document databases (247 lines)
│   │   │   ├── create_new_vector_store()
│   │   │   └── clear_new_vector_store()
│   │   │
│   │   └── pubmed_db_initializer.py # Pre-built PubMed index loader
│   │
│   └── utils/
│       ├── logger.py            # Logging configuration
│       └── helpers.py           # Helper functions
│
├── current_db/                  # Persistent PubMed database
│   ├── openai/
│   │   └── faiss_index/
│   │       ├── index.faiss      # OpenAI embedding vectors
│   │       └── index.pkl        # Document metadata
│   └── qwen/
│       └── faiss_index/
│           ├── index.faiss      # Qwen embedding vectors
│           └── index.pkl        # Document metadata
│
├── data/                        # Runtime data storage
│   ├── vectorstore/             # New document FAISS index
│   └── cache/
│       └── qa_cache.pkl         # Question-answer cache
│
├── logs/
│   └── app.log                  # Application logs
│
├── Dockerfile                   # Docker container configuration
├── docker-compose.yml           # Multi-container setup
├── requirements.txt             # Python dependencies
├── start.bat / start.ps1 / start.sh  # Startup scripts
└── stop.bat / stop.ps1 / stop.sh     # Shutdown scripts
```

---

## 🚀 Quick Start with Docker

### Prerequisites

- Docker Desktop (v20.10+)
- Docker Compose (v1.29+)
- 8GB RAM minimum (16GB recommended for Ollama)
- 20GB free disk space

### Installation Steps

#### 🚀 Quick Start (Automatic - RECOMMENDED)

Use the provided startup scripts that automatically open the browser:

**Windows (Command Prompt):**
```bash
git clone <repository-url>
cd rag_qa_chatbot_application
start.bat
```

**Windows (PowerShell):**
```powershell
git clone <repository-url>
cd rag_qa_chatbot_application
.\start.ps1
```

**Linux/Mac:**
```bash
git clone <repository-url>
cd rag_qa_chatbot_application
chmod +x start.sh
./start.sh
```

The script will:
- ✅ Start Docker containers
- ✅ Wait for application to be ready
- ✅ Automatically open browser at http://localhost:8501
- ✅ Show status and useful commands

#### 📋 Manual Start (Advanced)

**1. Clone the Repository**
```bash
git clone <repository-url>
cd rag_qa_chatbot_application
```

**2. Start the Application**
```bash
docker-compose up -d
```

This single command will:
- ✅ Pull the Ollama image
- ✅ Download Qwen2.5:7b model (~4.7GB)
- ✅ Download all-minilm embedding model
- ✅ Build the chatbot application
- ✅ Start all services
- ✅ Configure networking

**3. Monitor Startup Progress**
```bash
docker-compose logs -f
```

Wait for:
```
ollama  | Pulling Qwen2.5:7b model...
ollama  | Pulling all-minilm embedding model...
ollama  | Models ready!
rag-chatbot | You can now view your Streamlit app in your browser.
```

**4. Access the Application**
- Manually open browser: http://localhost:8501
- The UI will be ready to use!

### First-Time Setup (in UI)

1. **Select Model Provider**
   - Option A: **Local LLM (Qwen)** - No API key needed!
   - Option B: **OpenAI (API)** - Enter your OpenAI API key

2. **Initialize LLM**
   - Click "🔧 Initialize LLM" button
   - Wait for success message
   - The system automatically loads the pre-configured PubMed database

3. **Select Database Mode** (in Sidebar)
   - **Current DB**: Search in pre-loaded PubMed database (default)
   - **New DB**: Process and search only in uploaded documents (temporary)
   - **Current + New DB**: Combine both databases for comprehensive search (recommended)

4. **Upload Documents** (Optional)
   - Click "📁 Document Upload"
   - Select PDF, DOCX, or TXT files
   - Adjust chunking parameters (optional)
   - Click "🚀 Process Documents"
   - Documents are added to "New DB" (temporary) - won't persist after container restart

5. **Adjust Parameters**
   - **Chunk Size**: 200-2000 characters (default: 800)
   - **Chunk Overlap**: 0-500 characters (default: 100)
   - **Number of Sources (k)**: 1-10 documents (default: 5)
   - **Temperature**: 0.0-1.0 (default: 0.7 OpenAI, 0.3 Qwen)

6. **Start Asking Questions!**
   - Type your question in the chat input
   - Get answers with source citations, accuracy scores, and JSON-structured metadata
   - The system searches based on your selected database mode

### Stopping the Application

When you're done using the application:

#### 🛑 Quick Stop (Automatic - RECOMMENDED)

**Windows (Command Prompt):**
```bash
stop.bat
```

**Windows (PowerShell):**
```powershell
.\stop.ps1
```

**Linux/Mac:**
```bash
chmod +x stop.sh
./stop.sh
```

The script will:
- ✅ Stop all Docker containers
- ✅ Clean up resources
- ✅ Preserve your data (documents, vector store, cache)
- ✅ Show restart instructions

#### 📋 Manual Stop (Advanced)

**Stop containers (data preserved):**
```bash
docker-compose down
```

**Stop and remove ALL data (complete cleanup):**
```bash
docker-compose down -v
```

> ⚠️ **Note:** Using `-v` flag will delete:
> - ❌ Uploaded documents
> - ❌ Vector store index
> - ❌ Cached questions/answers
> - ❌ Ollama models (will need to re-download)

---

## ⚙️ Configuration

### Environment Variables

Create a `.env` file in the project root:

```bash
# Ollama Configuration (Default)
OLLAMA_BASE_URL=http://ollama:11434

# Cache Settings
CACHE_ENABLED=true
CACHE_TTL_SECONDS=3600
CACHE_MAX_SIZE=100
```

---

## 🐛 Troubleshooting

### Common Issues

#### 1. Ollama Models Not Loading
```bash
docker-compose logs ollama
docker exec -it rag-ollama ollama pull qwen2.5:7b
docker exec -it rag-ollama ollama pull all-minilm
```

#### 2. Out of Memory
```bash
# Increase Docker memory: Docker Desktop → Settings → Resources → Memory: 8GB+
# Or use OpenAI API instead (less resource-intensive)
```

#### 3. Slow Response Times (Qwen)
- This is expected for local CPU inference
- Enable caching for repeated queries (10x faster)
- Reduce number of sources (k)
- Use OpenAI for faster responses

#### 4. PDF Highlighting Not Working
- Ensure PyMuPDF (fitz) is installed
- Check that start_char_in_page/end_char_in_page are in metadata
- Verify PDF was processed with same extraction method

### Logging

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f rag-chatbot
docker-compose logs -f ollama

# Application logs (inside container)
docker exec -it rag-qa-chatbot cat logs/app.log
```

---

## 📊 Performance Metrics

### Typical Performance

| Metric | OpenAI | Qwen |
|--------|--------|------|
| Document Processing | 1-5 seconds | 5-15 seconds |
| First Query | 3-10 seconds | 15-60 seconds |
| Cached Query | <1 second | <1 second |
| Memory Usage | 2-4 GB | 6-10 GB |

### Optimization Tips

1. **Use Caching**: Enable for 10x faster repeated queries
2. **Adjust Chunk Size**: Smaller chunks = faster but less context
3. **Number of Sources**: Fewer sources = faster responses
4. **Model Choice**: Qwen is slower but free, OpenAI is faster but paid

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📝 License

This project is developed as part of an interview task for John Snow Labs.

---

## 🙏 Acknowledgments

- **LangChain**: For the excellent RAG framework
- **Ollama**: For making local LLMs accessible
- **Streamlit**: For the intuitive UI framework
- **FAISS**: For efficient vector similarity search
- **OpenAI**: For powerful embeddings and LLM API
- **PyMuPDF**: For PDF processing and highlighting

---

**Built with ❤️ using LangChain, FAISS, and Streamlit**
