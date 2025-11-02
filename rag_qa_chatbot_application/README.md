# RAG QA Chatbot Application

A production-ready **Retrieval-Augmented Generation (RAG)** chatbot application with advanced document processing, semantic search, and intelligent question-answering capabilities.

## 📋 Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Quick Start with Docker](#quick-start-with-docker)
- [Configuration](#configuration)
- [Usage Guide](#usage-guide)
- [Advanced Features](#advanced-features)
- [API Documentation](#api-documentation)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

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

#### 🎯 Contextual Compression & LLM Filtering

Our application implements state-of-the-art retrieval techniques:

1. **Contextual Compression Retriever**

   - Uses LLM to extract only query-relevant content from documents
   - Filters out irrelevant information before answer generation
   - Ensures high-quality, focused responses

2. **Two-Stage Filtering Process**

   ```
   User Query → Vector Search (k*3 candidates)
              → LLM Compression (relevance filtering)
              → Top k Results
   ```

3. **Benefits**
   - Eliminates irrelevant content through LLM filtering
   - Provides semantically accurate answers
   - Reduces hallucinations and off-topic responses

#### 💾 Intelligent Caching System

- **Semantic Query Matching**: Finds similar previous queries using embeddings
- **95% Similarity Threshold**: Smart cache hits for similar questions
- **TTL-based Expiration**: Automatic cache cleanup (default: 1 hour)
- **Performance Boost**: Up to 10x faster for repeated queries
- **Cache Statistics**: Real-time monitoring of hits, misses, and efficiency

#### ⚙️ Customizable Parameters

- **Chunk Size**: 200-2000 characters (default: 800)
- **Chunk Overlap**: 0-500 characters (default: 100)
- **Number of Sources (k)**: 1-10 documents (default: 5)
- **Model Temperature**: 0.0-1.0 (default: 0.7 for OpenAI, 0.3 for Qwen)

#### 🗄️ Dual Database Architecture

The application supports two vector databases:

1. **Current DB (Persistent)**
   - Pre-loaded PubMed database shipped with Docker
   - Persistent across container restarts
   - Located in `current_db/` directory
   - Separate indices for OpenAI and Qwen embeddings
   - Read-only during runtime (changes don't persist when Docker stops)

2. **New DB (Temporary)**
   - In-memory database for user-uploaded documents
   - Created when documents are processed
   - Cleared when container stops
   - Allows testing and experimentation without affecting persistent data

**Database Selection Modes:**
- **Current DB**: Search only in pre-loaded PubMed database
- **New DB**: Search only in newly uploaded documents (temporary)
- **Current + New DB**: Search in both databases combined (recommended for comprehensive results)

#### 📈 Real-Time Monitoring

- Vector store statistics (documents, chunks, index size)
- Cache performance metrics
- Response accuracy scores
- Source document tracking with JSON-structured metadata

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         User Interface                           │
│                    (Streamlit Web App)                           │
└──────────────────────────┬──────────────────────────────────────┘
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
        ▼                  ▼                  ▼
┌───────────────┐  ┌──────────────┐  ┌──────────────────┐
│   Document    │  │   Vector     │  │   LLM Service    │
│   Processor   │  │   Store      │  │   (OpenAI /      │
│               │  │   Manager    │  │    Ollama)       │
└───────┬───────┘  └──────┬───────┘  └────────┬─────────┘
        │                  │                   │
        ▼                  ▼                   ▼
┌───────────────────────────────────────────────────────┐
│              Advanced RAG Pipeline                     │
│                                                        │
│  1. Document Chunking & Embedding                     │
│  2. FAISS Vector Storage                              │
│  3. Contextual Compression Retrieval                  │
│  4. LLM-based Relevance Filtering                     │
│  5. Answer Generation with Sources                    │
│  6. Answer Generation with Sources                    │
│  7. Semantic Cache Layer                              │
└────────────────────────────────────────────────────────┘
```

### RAG Pipeline Flow

```
Document Upload
    ↓
Text Extraction (PDF/DOCX/TXT)
    ↓
Intelligent Chunking (with overlap)
    ↓
Embedding Generation (OpenAI/all-minilm)
    ↓
FAISS Vector Store (L2 distance)
    ↓
User Query
    ↓
Semantic Cache Check (95% similarity)
    ↓ (cache miss)
Vector Similarity Search (k*3 candidates)
    ↓
Contextual Compression (LLM filters irrelevant content)
    ↓
Top k Selection (user-defined count)
    ↓
Answer Generation (LLM with context)
    ↓
Source Attribution & Accuracy Scores
    ↓
Cache Storage (for future queries)
    ↓
Response to User
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
├── enhanced_app.py              # Main Streamlit application
├── requirements.txt             # Python dependencies
├── Dockerfile                   # Docker container configuration
├── docker-compose.yml           # Multi-container setup (app + Ollama)
├── .dockerignore               # Docker build exclusions
├── .gitignore                  # Git exclusions
├── expectations.txt            # Project requirements
│
├── src/                        # Source code modules
│   ├── __init__.py
│   │
│   ├── config/                 # Configuration management
│   │   ├── __init__.py
│   │   └── settings.py         # Application settings & constants
│   │
│   ├── models/                 # Data models & schemas
│   │   ├── __init__.py
│   │   └── data_models.py      # Pydantic models
│   │
│   ├── services/               # Core business logic
│   │   ├── __init__.py
│   │   ├── document_processor.py    # Document parsing & chunking
│   │   ├── vector_store.py          # FAISS vector store + RAG
│   │   ├── llm_service.py           # LLM initialization & management
│   │   └── cache_manager.py         # Semantic caching system
│   │
│   └── utils/                  # Utility functions
│       ├── __init__.py
│       ├── logger.py           # Logging configuration
│       └── helpers.py          # Helper functions
│
├── current_db/                 # Persistent PubMed database (shipped with Docker)
│   ├── openai/
│   │   └── faiss_index/        # OpenAI embeddings index
│   │       ├── index.faiss     # Vector index
│   │       └── index.pkl       # Document mappings
│   └── qwen/
│       └── faiss_index/        # Qwen embeddings index
│           ├── index.faiss     # Vector index
│           └── index.pkl       # Document mappings
│
├── data/                       # Runtime data storage
│   ├── vectorstore/            # Legacy/new document FAISS index files (optional)
│   │   ├── faiss_index/
│   │   │   ├── index.faiss     # Vector index
│   │   │   └── index.pkl       # Document mappings
│   │   └── document_metadata.pkl   # Document metadata
│   │
│   └── cache/                  # Cache storage
│       └── qa_cache.pkl        # Question-answer cache
│
└── logs/                       # Application logs
    └── app.log                 # Runtime logs
```

### Key Components

#### `enhanced_app.py`

Main Streamlit application with:

- UI rendering and user interactions
- File upload handling
- Chat interface management
- Real-time metrics display

#### `src/services/vector_store.py`

Advanced RAG implementation:

- FAISS vector store management
- **Contextual Compression Retriever** integration
- **LLM-based relevance filtering**
- Similarity threshold application
- Semantic search with accuracy scoring

#### `src/services/cache_manager.py`

Intelligent caching system:

- Semantic query similarity matching
- TTL-based expiration
- Pickle-based persistence
- Performance monitoring

#### `src/services/document_processor.py`

Document handling:

- Multi-format parsing (PDF, DOCX, TXT)
- Intelligent text chunking
- Metadata extraction

#### `src/services/llm_service.py`

LLM management:

- Multi-provider support (OpenAI, Ollama)
- API connection validation
- Error handling

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
- ✅ **Automatically open browser** at http://localhost:8501
- ✅ Show status and useful commands

#### 📋 Manual Start (Advanced)

1. **Clone the Repository**

   ```bash
   git clone <repository-url>
   cd rag_qa_chatbot_application
   ```

2. **Start the Application**

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

3. **Monitor Startup Progress**

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

4. **Access the Application**
   - Manually open browser: **http://localhost:8501**
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

5. **Start Asking Questions!**
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
- ✅ **Preserve your data** (documents, vector store, cache)
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

**Note**: Using `-v` flag will delete:

- ❌ Uploaded documents
- ❌ Vector store index
- ❌ Cached questions/answers
- ❌ Ollama models (will need to re-download)

---

## ⚙️ Configuration

### Environment Variables

Create a `.env` file in the project root (optional):

```bash
# OpenAI API (Optional)
OPENAI_API_KEY=sk-your-api-key-here

# Ollama Configuration (Default)
OLLAMA_BASE_URL=http://ollama:11434

# Cache Settings
CACHE_ENABLED=true
CACHE_TTL_SECONDS=3600
CACHE_MAX_SIZE=100
```

### Docker Compose Options

#### Use OpenAI Only (without Ollama)

```yaml
# In docker-compose.yml, comment out the ollama service
services:
  # ollama:
  #   ...

  rag-chatbot:
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
```

#### Adjust Resource Limits

```yaml
services:
  ollama:
    deploy:
      resources:
        limits:
          memory: 8G
        reservations:
          memory: 4G
```

---

## 📖 Usage Guide

### Document Upload

1. **Supported Formats**

   - PDF (`.pdf`)
   - Microsoft Word (`.docx`)
   - Plain Text (`.txt`)

2. **Processing Parameters**

   - **Chunk Size** (800): Smaller = more precise, larger = more context
   - **Chunk Overlap** (100): Prevents information loss at chunk boundaries

3. **Database Selection**

   - Use the sidebar radio buttons to select database mode:
     - **Current DB**: Use pre-loaded PubMed database
     - **New DB**: Process documents into temporary database
     - **Current + New DB**: Combine both for comprehensive search
   - Documents processed in "New DB" mode are temporary and won't persist after container restart
   - "Current DB" remains unchanged when Docker is stopped

4. **Best Practices**
   - Upload related documents together
   - Use descriptive filenames
   - Keep documents focused on specific topics
   - Use "Current + New DB" mode for best search coverage

### Asking Questions

1. **Retrieval Parameters**

   - **Number of Sources (k=5)**: How many document chunks to retrieve

2. **Model Parameters**

   - **Temperature (0.7)**:
     - 0.0-0.3: Focused, deterministic (factual Q&A)
     - 0.4-0.7: Balanced (recommended)
     - 0.8-1.0: Creative, varied

3. **Question Types**
   - **Factual**: "What is X?"
   - **Comparative**: "Compare X and Y"
   - **Analytical**: "Why does X happen?"
   - **Summarization**: "Summarize the key points about X"

### Understanding Response Information

Each response includes:

```
📊 Response Information
├─ Model: OpenAI gpt-4o (or Qwen2.5:7b)
├─ Sources Used: 3 documents
├─ Method: Contextual Compression + LLM Filtering
├─ Database Mode: Current + New DB
├─ ⚡ Cached Response (if applicable)
│
└─ 📚 View Sources (expandable)
    ├─ Source 1
    │   ├─ File: document.pdf
    │   ├─ Page: 5
    │   ├─ Accuracy: 94.2%
    │   ├─ Content Preview: [full chunk text]
    │   └─ Metadata (JSON):
    │       ├─ Document_Id: uuid
    │       ├─ Document_Name: document.pdf
    │       ├─ Chunk_Index: 5
    │       ├─ Start_Char: 1024
    │       ├─ End_Char: 1824
    │       ├─ Is_New_Doc: true/false
    │       ├─ File_Type: pdf
    │       ├─ Chunk_Size: 800
    │       └─ Model_Provider: OpenAI (API)
    │
    └─ Source 2
        └─ ...
```

**Note**: Metadata is now provided in structured JSON format, and Chunk_Id has been removed from the metadata structure.

### Cache Management

- **View Cache Stats**: Check sidebar "💾 Cache Info"
- **Clear Cache**: Click "🧹 Clear Cache" to reset
- **Cache Behavior**:
  - Exact query match: Instant response
  - Similar query (95%+): Cached response with similarity score
  - New query: Full RAG pipeline execution

---

## 🔬 Advanced Features

### Contextual Compression Explained

Our application uses LangChain's **Contextual Compression Retriever** to ensure only relevant content is used:

```python
# 1. Base retriever gets k*3 candidates
base_retriever = vector_store.as_retriever(k=15)

# 2. LLM compressor filters irrelevant content
compressor = LLMChainExtractor.from_llm(llm)

# 3. Compression retriever combines both
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=base_retriever
)

# Result: Only query-relevant content reaches answer generation
```

### Document Retrieval Process

We use **similarity_search_with_score** to retrieve documents with their similarity scores:

```python
# Get documents with similarity scores
docs_with_scores = vector_store.similarity_search_with_score(query, k=k)

# The system uses all retrieved sources for comprehensive answers
# LLM determines relevance and filters content appropriately
```

### Semantic Caching Algorithm

```python
# 1. Generate embedding for new query
query_embedding = embeddings.embed_query(user_query)

# 2. Compare with cached queries
for cached_query in cache:
    similarity = cosine_similarity(query_embedding, cached_embedding)
    if similarity >= 0.90:  # 90% threshold for cache matching
        return cached_response

# 3. If no match, execute RAG and cache result
```

### Metadata Structure (JSON Format)

The application now provides metadata in a structured JSON format:

```json
{
  "Document_Id": "uuid-string",
  "Document_Name": "document.pdf",
  "Chunk_Index": 5,
  "Start_Char": 1024,
  "End_Char": 1824,
  "Is_New_Doc": true,
  "File_Type": "pdf",
  "Chunk_Size": 800,
  "Model_Provider": "OpenAI (API)"
}
```

**Key Changes:**
- Metadata is now provided as structured JSON in the `structured_metadata` field
- `Chunk_Id` has been **removed** from the metadata structure
- All metadata fields are consistently formatted
- Source information includes both original metadata (for backward compatibility) and new structured metadata

---

## 🐛 Troubleshooting

### Common Issues

#### 1. Ollama Models Not Loading

```bash
# Check Ollama status
docker-compose logs ollama

# Manually pull models
docker exec -it rag-ollama ollama pull qwen2.5:7b
docker exec -it rag-ollama ollama pull all-minilm
```

#### 2. Out of Memory

```bash
# Increase Docker memory limit
# Docker Desktop → Settings → Resources → Memory: 8GB+

# Or use OpenAI API instead (less resource-intensive)
```

#### 3. Port Already in Use

```bash
# Change port in docker-compose.yml
ports:
  - "8502:8501"  # Use 8502 instead
```

#### 4. Vector Store Not Loading

```bash
# Clear and rebuild
docker-compose down -v
rm -rf data/vectorstore/*
docker-compose up -d
```

#### 5. Slow Response Times

- **Solution 1**: Enable caching (check sidebar)
- **Solution 2**: Reduce number of sources (k)
- **Solution 3**: Increase similarity threshold
- **Solution 4**: Use smaller chunk sizes

### Logging

View detailed logs:

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

## 🔧 API Documentation

### Docker Commands

```bash
# Start all services
docker-compose up -d

# Stop all services
docker-compose down

# Rebuild after code changes
docker-compose up -d --build

# View logs
docker-compose logs -f [service-name]

# Restart specific service
docker-compose restart rag-chatbot

# Remove all data
docker-compose down -v
```

### Service URLs

- **Chatbot UI**: http://localhost:8501
- **Ollama API**: http://localhost:11434
- **Ollama Models**: http://localhost:11434/api/tags

### Health Checks

```bash
# Check chatbot health
curl http://localhost:8501/_stcore/health

# Check Ollama health
curl http://localhost:11434/api/tags
```

---

## 📊 Performance Metrics

### Typical Performance

- **Document Processing**: 1-5 seconds per document
- **First Query**: 3-10 seconds (with compression)
- **Cached Query**: <1 second
- **Memory Usage**: 2-4GB (app) + 4-8GB (Ollama)

### Optimization Tips

1. **Use Caching**: Enable for 10x faster repeated queries
2. **Adjust Chunk Size**: Smaller chunks = faster but less context
3. **Number of Sources**: Fewer sources = faster responses
4. **Model Choice**: Qwen is slower but free, OpenAI is faster but paid

---

## 🤝 Contributing

We welcome contributions! Please follow these guidelines:

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

---

## 📞 Support

For issues, questions, or suggestions:

- Open an issue on GitHub
- Check the [Troubleshooting](#troubleshooting) section
- Review logs in `logs/app.log`

---

**Built with ❤️ using LangChain, FAISS, and Streamlit**
