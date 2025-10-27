"""
Data models for RAG QA Chatbot Application
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any
from enum import Enum


class ModelProvider(Enum):
    """Supported model providers"""
    OPENAI = "OpenAI (API)"
    LOCAL_LLM = "Local LLM (Qwen)"


class DocumentType(Enum):
    """Supported document types"""
    PDF = "pdf"
    DOCX = "docx"
    TXT = "txt"


class MessageRole(Enum):
    """Chat message roles"""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


@dataclass
class Document:
    """Document model"""
    id: str
    name: str
    content: str
    file_type: DocumentType
    file_size: int
    file_hash: str
    upload_timestamp: datetime = field(default_factory=datetime.now)
    chunk_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Post initialization processing"""
        if isinstance(self.file_type, str):
            self.file_type = DocumentType(self.file_type)


@dataclass
class DocumentChunk:
    """Document chunk model"""
    id: str
    document_id: str
    content: str
    chunk_index: int
    start_char: int
    end_char: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ChatMessage:
    """Chat message model"""
    id: str
    role: MessageRole
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    model_provider: Optional[ModelProvider] = None
    tokens_used: Optional[int] = None
    response_time: Optional[float] = None
    source_documents: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Post initialization processing"""
        if isinstance(self.role, str):
            self.role = MessageRole(self.role)
        if isinstance(self.model_provider, str):
            self.model_provider = ModelProvider(self.model_provider)


@dataclass
class ConversationSession:
    """Conversation session model"""
    id: str
    title: str
    messages: List[ChatMessage] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    documents: List[str] = field(default_factory=list)  # Document IDs
    model_provider: Optional[ModelProvider] = None
    total_tokens: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_message(self, message: ChatMessage) -> None:
        """Add message to conversation"""
        self.messages.append(message)
        self.updated_at = datetime.now()
        if message.tokens_used:
            self.total_tokens += message.tokens_used

    def get_recent_messages(self, count: int = 10) -> List[ChatMessage]:
        """Get recent messages"""
        return self.messages[-count:] if self.messages else []


@dataclass
class VectorStoreInfo:
    """Vector store information"""
    index_path: str
    total_documents: int
    total_chunks: int
    embedding_model: str
    last_updated: datetime = field(default_factory=datetime.now)
    index_size_bytes: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QueryResult:
    """Query result model"""
    query: str
    answer: str
    source_documents: List[DocumentChunk]
    model_provider: ModelProvider
    response_time: float
    tokens_used: Optional[int] = None
    confidence_score: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UserSession:
    """User session model"""
    id: str
    user_id: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)
    conversations: List[str] = field(default_factory=list)  # Conversation IDs
    uploaded_documents: List[str] = field(default_factory=list)  # Document IDs
    preferences: Dict[str, Any] = field(default_factory=dict)

    def update_activity(self) -> None:
        """Update last activity timestamp"""
        self.last_activity = datetime.now()


@dataclass
class SystemMetrics:
    """System metrics model"""
    timestamp: datetime = field(default_factory=datetime.now)
    active_sessions: int = 0
    total_documents: int = 0
    total_conversations: int = 0
    total_queries: int = 0
    average_response_time: float = 0.0
    memory_usage_mb: float = 0.0
    disk_usage_mb: float = 0.0
    error_count: int = 0


@dataclass
class APIUsageStats:
    """API usage statistics"""
    provider: ModelProvider
    total_requests: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0
    average_response_time: float = 0.0
    error_count: int = 0
    last_request: Optional[datetime] = None

    def add_request(self, tokens: int, response_time: float, cost: float = 0.0) -> None:
        """Add API request statistics"""
        self.total_requests += 1
        self.total_tokens += tokens
        self.total_cost += cost

        # Update average response time
        if self.total_requests == 1:
            self.average_response_time = response_time
        else:
            self.average_response_time = (
                (self.average_response_time *
                 (self.total_requests - 1) + response_time)
                / self.total_requests
            )

        self.last_request = datetime.now()


__all__ = [
    "ModelProvider",
    "DocumentType",
    "MessageRole",
    "Document",
    "DocumentChunk",
    "ChatMessage",
    "ConversationSession",
    "VectorStoreInfo",
    "QueryResult",
    "UserSession",
    "SystemMetrics",
    "APIUsageStats"
]
