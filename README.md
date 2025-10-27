# John Snow Labs - Interview Task

This repository contains the complete implementation of a **RAG (Retrieval-Augmented Generation) QA Chatbot Application**.

## 📁 Repository Structure

```
john_snow_labs/
└── rag_qa_chatbot_application/    # Main RAG chatbot application
    ├── enhanced_app.py             # Streamlit application
    ├── Dockerfile                  # Docker configuration
    ├── docker-compose.yml          # Multi-container setup
    ├── requirements.txt            # Python dependencies
    ├── README.md                   # Detailed documentation
    └── src/                        # Source code modules
```

## 🚀 Quick Start

Navigate to the application directory and follow the instructions:

```bash
cd rag_qa_chatbot_application
```

Then follow the **[Complete Documentation](rag_qa_chatbot_application/README.md)** for:
- Docker setup and deployment
- Feature descriptions
- Usage guide
- Troubleshooting

## ✨ Key Features

- **Advanced RAG Pipeline** with Contextual Compression
- **Dual LLM Support**: OpenAI API + Local Ollama (Qwen2.5:7b)
- **Semantic Caching** for 10x faster repeated queries
- **Multi-format Document Support**: PDF, DOCX, TXT
- **Real-time Source Attribution** with accuracy scores
- **Fully Dockerized** with automatic model setup

## 📖 Full Documentation

See **[rag_qa_chatbot_application/README.md](rag_qa_chatbot_application/README.md)** for complete documentation including:

- Detailed architecture and flow diagrams
- Technology stack overview
- Project structure explanation
- Docker configuration guide
- Advanced features documentation
- Troubleshooting guide

## 🏃 One-Click Deployment

### Windows:
```bash
cd rag_qa_chatbot_application
start.bat
```

### Linux/Mac:
```bash
cd rag_qa_chatbot_application
./start.sh
```

**The browser will open automatically** at **http://localhost:8501** ✨

### Or Manual Start:
```bash
cd rag_qa_chatbot_application && docker-compose up -d
```

Then manually open: **http://localhost:8501**

## 🛑 Stopping the Application

### Windows:
```bash
stop.bat
```

### Linux/Mac:
```bash
./stop.sh
```

**Preserves**: Documents, vector store, and cache  
**Removes**: Running containers only

For complete cleanup (removes all data):
```bash
docker-compose down -v
```

---

**Developed for John Snow Labs Interview Task**
