# Quick Start Guide - RAG QA Chatbot

Get up and running in **5 minutes**!

## Prerequisites

- Docker Desktop installed
- 8GB RAM minimum
- 20GB free disk space

## 🚀 Super Quick Start (RECOMMENDED)

Use our startup script that **automatically opens the browser**!

### Windows (Command Prompt):
```bash
start.bat
```

### Windows (PowerShell):
```powershell
.\start.ps1
```

### Linux/Mac:
```bash
chmod +x start.sh
./start.sh
```

**That's it!** The script will:
1. Start Docker containers
2. Wait for application to be ready
3. **Automatically open your browser** at http://localhost:8501

## 📋 Manual Start (Alternative)

If you prefer manual control:

```bash
docker-compose up -d
```

This will automatically:
- ✅ Pull Ollama Docker image
- ✅ Download Qwen2.5:7b model (~4.7GB)
- ✅ Download all-minilm embedding model
- ✅ Build the chatbot application
- ✅ Start all services

### Monitor Progress

```bash
docker-compose logs -f
```

Wait for these messages:
```
ollama  | Models ready!
rag-chatbot | You can now view your Streamlit app in your browser.
```

### Access the App

Manually open your browser: **http://localhost:8501**

## First Use

### Step 1: Initialize LLM

1. Select "**Local LLM (Qwen)**" (no API key needed!)
2. Click "**🔧 Initialize LLM**"
3. Wait for "✅ LLM initialized successfully!"
4. The system automatically loads the pre-configured PubMed database

### Step 2: Select Database Mode (in Sidebar)

Choose how you want to search:

- **Current DB**: Search in pre-loaded PubMed database (default, persistent)
- **New DB**: Process and search only in uploaded documents (temporary)
- **Current + New DB**: Combine both databases (recommended for comprehensive results)

### Step 3: Upload Documents (Optional)

1. Click "**📁 Document Upload**"
2. Select PDF, DOCX, or TXT files
3. Click "**🚀 Process Documents**"
4. Wait for "✅ Successfully created knowledge base!"
5. **Note**: Documents are added to "New DB" (temporary) and won't persist after container restart

### Step 4: Ask Questions

Type your question in the chat input and get instant answers with:
- Source citations
- Accuracy scores
- JSON-structured metadata (Chunk_Id removed)
- Search results from your selected database mode

## Using OpenAI Instead

If you prefer OpenAI (faster but requires API key):

1. Get API key from: https://platform.openai.com/api-keys
2. In UI, select "**OpenAI (API)**"
3. Enter your API key
4. Click "**🔧 Initialize LLM**"

## Stopping the Application

When you're done using the chatbot:

### 🛑 Quick Stop (Recommended):

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
./stop.sh
```

This will stop containers and **preserve your data**.

### Manual Stop:

**Stop containers (keep data):**
```bash
docker-compose down
```

**Complete cleanup (removes ALL data):**
```bash
docker-compose down -v
```

**Warning**: `-v` flag deletes documents, vector store, cache, and Ollama models!

---

## Troubleshooting

### Ollama Models Not Loading?
```bash
# Check status
docker-compose logs ollama

# Manually pull models
docker exec -it rag-ollama ollama pull qwen2.5:7b
```

### Port 8501 Already in Use?
```bash
# Edit docker-compose.yml
ports:
  - "8502:8501"  # Change to 8502
```

### Need More Help?
See the [Full README](README.md) for detailed documentation.

---

**That's it! You're ready to use your RAG Chatbot! 🚀**

