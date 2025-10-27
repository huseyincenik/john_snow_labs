"""
Configuration settings for RAG QA Chatbot Application
"""
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

# Base paths
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data"
VECTORSTORE_DIR = DATA_DIR / "vectorstore"
LOGS_DIR = BASE_DIR / "logs"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
VECTORSTORE_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)


@dataclass
class ModelConfig:
    """Model configuration settings"""
    # OpenAI API settings
    openai_model: str = "gpt-4o-mini"
    openai_temperature: float = 0.7
    openai_max_tokens: int = 1000

    # Local LLM (Qwen via Ollama) settings
    local_model: str = "qwen2.5:3b"
    local_temperature: float = 0.3
    local_max_tokens: int = 1000
    ollama_base_url: str = "http://localhost:11434/v1"

    # Embedding settings
    openai_embedding_model: str = "text-embedding-ada-002"
    local_embedding_model: str = "all-minilm"  # Ollama embedding model


@dataclass
class DocumentConfig:
    """Document processing configuration"""
    chunk_size: int = 1000
    chunk_overlap: int = 200
    max_file_size_mb: int = 50
    supported_extensions: tuple = (".pdf", ".docx", ".txt")

    # Text splitting strategies
    openai_chunk_size: int = 1000
    local_chunk_size: int = 1000  # Keep same as OpenAI for consistency
    local_chunk_overlap: int = 200


@dataclass
class VectorStoreConfig:
    """Vector store configuration"""
    faiss_index_type: str = "HNSW"
    similarity_threshold: float = 0.7
    max_retrieved_docs: int = 5
    index_path: str = str(VECTORSTORE_DIR / "faiss_index")


@dataclass
class UIConfig:
    """UI configuration"""
    page_title: str = "RAG QA Chatbot"
    page_icon: str = "🤖"
    sidebar_width: int = 300
    max_conversation_history: int = 50


@dataclass
class LoggingConfig:
    """Logging configuration"""
    log_level: str = "DEBUG"  # Changed to DEBUG for detailed logging
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_file: str = str(LOGS_DIR / "app.log")
    max_bytes: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5


@dataclass
class CacheConfig:
    """Cache configuration"""
    enable_cache: bool = True
    cache_ttl: int = 3600  # 1 hour in seconds
    max_cache_size: int = 100


@dataclass
class SecurityConfig:
    """Security configuration"""
    enable_auth: bool = False
    session_timeout: int = 3600  # 1 hour
    max_file_uploads: int = 10


class AppConfig:
    """Main application configuration"""

    def __init__(self):
        self.model = ModelConfig()
        self.document = DocumentConfig()
        self.vectorstore = VectorStoreConfig()
        self.ui = UIConfig()
        self.logging = LoggingConfig()
        self.cache = CacheConfig()
        self.security = SecurityConfig()

        # Environment variables
        self.openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")

    def validate_api_keys(self, model_name: str) -> bool:
        """Validate required API keys based on selected model"""
        if model_name == "OpenAI" and not self.openai_api_key:
            return False
        return True


# Global config instance
config = AppConfig()

# Constants
# Supported model providers
SUPPORTED_MODELS = ["OpenAI (API)", "Local LLM (Qwen)"]
DEFAULT_PROMPT_TEMPLATE = """
Answer the question as detailed as possible from the provided context. 
Make sure to provide all the details. If the answer is not in the provided context, 
just say "answer is not available in the context", don't provide the wrong answer.

Context:
{context}

Question: 
{question}

Answer:
"""
