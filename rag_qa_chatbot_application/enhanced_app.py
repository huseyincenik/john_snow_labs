"""
Enhanced RAG QA Chatbot Application with Streamlit
"""

import streamlit as st
import uuid
import pandas as pd
import base64
import os
import json
import io
import re
from datetime import datetime
from typing import List, Optional, Dict
import traceback
import PyPDF2

# Import our modules
from src.config import (
    config,
    SUPPORTED_MODELS,
    CURRENT_DB_OPENAI_DIR,
    CURRENT_DB_QWEN_DIR,
)
from src.models import (
    ModelProvider,
    ConversationSession,
    ChatMessage,
    MessageRole,
    Document,
    UserSession,
)
from src.services import (
    DocumentProcessor,
    VectorStoreManager,
    LLMService,
    NewDBManager,
    PubMedDBInitializer,
)
from src.utils import app_logger, format_file_size, truncate_text
from pathlib import Path
from langchain_community.vectorstores import FAISS


class ChatbotApp:
    """Main Chatbot Application Class"""

    def __init__(self):
        self.logger = app_logger
        self.document_processor = DocumentProcessor()
        self.vector_store_manager = VectorStoreManager()
        self.llm_service = LLMService()
        self.new_db_manager = NewDBManager()
        self.pubmed_db_initializer = PubMedDBInitializer()

        # Initialize session state
        self._initialize_session_state()

    def _initialize_session_state(self):
        """Initialize Streamlit session state"""
        if "user_session" not in st.session_state:
            st.session_state.user_session = UserSession(
                id=str(uuid.uuid4()), preferences={"theme": "light", "language": "en"}
            )

        if "current_conversation" not in st.session_state:
            st.session_state.current_conversation = ConversationSession(
                id=str(uuid.uuid4()), title="New Conversation"
            )

        if "documents" not in st.session_state:
            st.session_state.documents = []

        # Default parameters
        if "chunk_size" not in st.session_state:
            st.session_state.chunk_size = 800

        if "chunk_overlap" not in st.session_state:
            st.session_state.chunk_overlap = 100

        if "search_k" not in st.session_state:
            st.session_state.search_k = 7

        if "model_temperature" not in st.session_state:
            st.session_state.model_temperature = 0.7

        if "vector_store_initialized" not in st.session_state:
            st.session_state.vector_store_initialized = False

        if "llm_initialized" not in st.session_state:
            st.session_state.llm_initialized = False

        if "processing_status" not in st.session_state:
            st.session_state.processing_status = None

        if "db_mode" not in st.session_state:
            st.session_state.db_mode = "current"  # "current", "new", "current+new"

        if "new_vector_store" not in st.session_state:
            st.session_state.new_vector_store = None

    def run(self):
        """Main application entry point"""
        self._setup_page_config()
        self._render_header()

        # Sidebar for configuration
        self._render_sidebar()

        # Main chat interface
        self._render_main_interface()

        # Footer
        self._render_footer()

    def _setup_page_config(self):
        """Configure Streamlit page"""
        st.set_page_config(
            page_title=config.ui.page_title,
            page_icon=config.ui.page_icon,
            layout="wide",
            initial_sidebar_state="expanded",
        )

        # Custom CSS
        st.markdown(
            """
        <style>
        .main {
            padding-top: 2rem;
        }
        .stChatMessage {
            padding: 1rem;
            margin: 0.5rem 0;
            border-radius: 0.5rem;
        }
        .user-message {
            background-color: #e3f2fd;
            margin-left: 20%;
        }
        .assistant-message {
            background-color: #f5f5f5;
            margin-right: 20%;
        }
        .sidebar .sidebar-content {
            padding-top: 2rem;
        }
        .metric-container {
            background-color: #f8f9fa;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 0.5rem 0;
        }
        .status-success {
            color: #28a745;
            font-weight: bold;
        }
        .status-error {
            color: #dc3545;
            font-weight: bold;
        }
        .status-warning {
            color: #ffc107;
            font-weight: bold;
        }
        /* Word wrap for JSON code blocks */
        .stCodeBlock {
            word-wrap: break-word;
            white-space: pre-wrap;
            overflow-wrap: break-word;
        }
        .stCodeBlock pre {
            word-wrap: break-word;
            white-space: pre-wrap;
            overflow-wrap: break-word;
        }
        </style>
        """,
            unsafe_allow_html=True,
        )

    def _is_vector_store_ready(self) -> bool:
        """Check if vector store is ready based on db_mode"""
        db_mode = st.session_state.get("db_mode", "current")

        # For "new" mode, use existing vector_store_initialized flag
        if db_mode == "new":
            return st.session_state.vector_store_initialized

        # For "current" mode, check if current_db exists
        if db_mode == "current":
            # Check if vector_store_initialized (for uploaded docs) OR current_db exists
            if st.session_state.vector_store_initialized:
                return True

            # Check if current_db exists for the current model provider
            current_model_provider = st.session_state.get("current_model_provider")
            if current_model_provider == "OpenAI (API)":
                faiss_index_path = CURRENT_DB_OPENAI_DIR / "faiss_index"
                faiss_file = faiss_index_path / "index.faiss"
                pkl_file = faiss_index_path / "index.pkl"
                return faiss_file.exists() and pkl_file.exists()
            elif current_model_provider == "Local LLM (Qwen)":
                faiss_index_path = CURRENT_DB_QWEN_DIR / "faiss_index"
                faiss_file = faiss_index_path / "index.faiss"
                pkl_file = faiss_index_path / "index.pkl"
                return faiss_file.exists() and pkl_file.exists()
            else:
                # Provider not set, check both
                openai_path = CURRENT_DB_OPENAI_DIR / "faiss_index"
                qwen_path = CURRENT_DB_QWEN_DIR / "faiss_index"

                openai_exists = (openai_path / "index.faiss").exists() and (
                    openai_path / "index.pkl"
                ).exists()
                qwen_exists = (qwen_path / "index.faiss").exists() and (
                    qwen_path / "index.pkl"
                ).exists()
                return openai_exists or qwen_exists

        # For "current+new" mode, check if current_db OR new_vector_store exists
        if db_mode == "current+new":
            # Check if new_db exists
            if st.session_state.vector_store_initialized:
                return True

            new_vector_store = st.session_state.get("new_vector_store", None)
            if new_vector_store:
                return True

            # Check if current_db exists
            current_model_provider = st.session_state.get("current_model_provider")
            if current_model_provider == "OpenAI (API)":
                faiss_index_path = CURRENT_DB_OPENAI_DIR / "faiss_index"
                faiss_file = faiss_index_path / "index.faiss"
                pkl_file = faiss_index_path / "index.pkl"
                return faiss_file.exists() and pkl_file.exists()
            elif current_model_provider == "Local LLM (Qwen)":
                faiss_index_path = CURRENT_DB_QWEN_DIR / "faiss_index"
                faiss_file = faiss_index_path / "index.faiss"
                pkl_file = faiss_index_path / "index.pkl"
                return faiss_file.exists() and pkl_file.exists()
            else:
                # Provider not set, check both
                openai_path = CURRENT_DB_OPENAI_DIR / "faiss_index"
                qwen_path = CURRENT_DB_QWEN_DIR / "faiss_index"

                openai_exists = (openai_path / "index.faiss").exists() and (
                    openai_path / "index.pkl"
                ).exists()
                qwen_exists = (qwen_path / "index.faiss").exists() and (
                    qwen_path / "index.pkl"
                ).exists()
                return openai_exists or qwen_exists

        # Default fallback
        return st.session_state.vector_store_initialized

    def _render_header(self):
        """Render application header"""
        st.title("🤖 Advanced RAG QA Chatbot")
        st.markdown("---")

        # Display current status
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            docs_count = len(st.session_state.documents)
            st.metric("📄 Documents", docs_count)

        with col2:
            messages_count = len(st.session_state.current_conversation.messages)
            st.metric("💬 Messages", messages_count)

        with col3:
            if self._is_vector_store_ready():
                st.markdown(
                    '<p class="status-success">✅ Vector Store Ready</p>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<p class="status-warning">⏳ Vector Store Not Ready</p>',
                    unsafe_allow_html=True,
                )

        with col4:
            if st.session_state.llm_initialized:
                st.markdown(
                    '<p class="status-success">✅ LLM Ready</p>', unsafe_allow_html=True
                )
            else:
                st.markdown(
                    '<p class="status-warning">⏳ LLM Not Ready</p>',
                    unsafe_allow_html=True,
                )

    def _render_sidebar(self):
        """Render sidebar configuration"""
        with st.sidebar:
            st.title("⚙️ Configuration")

            # Model selection
            model_provider = st.selectbox(
                "Select Model Provider:",
                SUPPORTED_MODELS,
                help="Choose the AI model provider for generating responses",
            )

            # API Key input
            api_key = self._render_api_key_input(model_provider)

            # Initialize LLM button
            if st.button("🔧 Initialize LLM", type="primary", use_container_width=True):
                if api_key:
                    self._initialize_llm(model_provider, api_key)
                else:
                    st.error("Please provide an API key")

            st.markdown("---")

            # Database mode selection
            self._render_db_mode_section()

            st.markdown("---")

            # File upload section
            self._render_file_upload_section()

            st.markdown("---")

            # Chat management
            self._render_chat_management_section()

            st.markdown("---")

            # System information
            self._render_system_info_section()

            # Social links
            self._render_social_links()

    def _render_api_key_input(self, model_provider: str) -> Optional[str]:
        """Render API key input based on selected provider"""
        if model_provider == "OpenAI (API)":
            api_key = st.text_input(
                "OpenAI API Key:", type="password", help="Enter your OpenAI API key"
            )
            if not api_key:
                st.info(
                    "💡 Get your API key from [OpenAI](https://platform.openai.com/api-keys)"
                )

        elif model_provider == "Local LLM (Qwen)":
            # No API key needed for local Ollama
            api_key = "ollama"  # Placeholder for local LLM
            st.info("🖥️ Using local Qwen2.5:7b model via Ollama")
            st.info("💡 Make sure Ollama is running: `ollama run qwen2.5:7b`")

        else:
            api_key = None
            st.warning(f"API key input not implemented for {model_provider}")

        return api_key

    def _render_db_mode_section(self):
        """Render database mode selection"""
        st.subheader("🗄️ Database Mode")

        db_mode = st.radio(
            "Select Database Mode:",
            ["current", "new", "current+new"],
            index=(
                ["current", "new", "current+new"].index(st.session_state.db_mode)
                if st.session_state.db_mode in ["current", "new", "current+new"]
                else 0
            ),
            help=(
                "**Current DB**: Search in pre-loaded PubMed database\n"
                "**New DB**: Search only in newly uploaded documents (temporary)\n"
                "**Current + New DB**: Search in both databases combined"
            ),
        )

        # Update session state and vector store manager
        # CRITICAL: Always sync db_mode to ensure consistency
        if db_mode != st.session_state.db_mode:
            st.session_state.db_mode = db_mode
        # Always set db_mode to ensure it's synced (even if session state already has it)
        self.vector_store_manager.set_db_mode(db_mode)

        # Show mode description
        if db_mode == "current":
            st.info("🔍 Searching in PubMed database (pre-loaded, persistent)")
        elif db_mode == "new":
            st.info(
                "🔍 Searching only in newly uploaded documents (temporary, cleared on restart)"
            )
        else:
            st.info("🔍 Searching in both PubMed database and newly uploaded documents")

    def _ensure_empty_faiss_index(self, model_provider: str, api_key: str):
        """
        Ensure that FAISS index exists for the current model provider.
        If index doesn't exist for OpenAI, creates it with PubMed data using OpenAI embeddings.
        Similar to Qwen behavior where PubMed data is automatically loaded.

        Args:
            model_provider: The model provider name
            api_key: API key for embeddings (if needed)
        """
        try:
            if model_provider == "OpenAI (API)":
                faiss_index_path = CURRENT_DB_OPENAI_DIR / "faiss_index"
                faiss_file = faiss_index_path / "index.faiss"
                pkl_file = faiss_index_path / "index.pkl"

                # Check if index already exists
                if faiss_file.exists() and pkl_file.exists():
                    self.logger.info(
                        f"OpenAI FAISS index already exists at {faiss_index_path}"
                    )
                    return

                # Index doesn't exist - initialize with PubMed data using OpenAI embeddings
                self.logger.info(
                    f"OpenAI FAISS index not found at {faiss_index_path}. "
                    f"Initializing with PubMed data using OpenAI embeddings..."
                )

                # Ensure embeddings are initialized
                if not self.vector_store_manager.embeddings:
                    self.vector_store_manager.initialize_embeddings(
                        model_provider, api_key
                    )

                # Initialize PubMed database with OpenAI embeddings
                # This will create the index with actual PubMed data (1000 documents)
                results = self.pubmed_db_initializer.initialize_databases(
                    openai_api_key=api_key if api_key and api_key != "ollama" else None
                )

                if results.get("openai"):
                    self.logger.info(
                        f"Successfully created OpenAI FAISS index with PubMed data at {faiss_index_path}"
                    )
                else:
                    self.logger.warning(
                        f"Failed to create OpenAI FAISS index with PubMed data. "
                        f"Index may not be available."
                    )

        except Exception as e:
            self.logger.warning(f"Failed to create OpenAI FAISS index: {str(e)}")
            # Don't fail initialization if index creation fails

    def _initialize_pubmed_databases(self, openai_api_key: Optional[str] = None):
        """Initialize PubMed databases (called after LLM initialization)"""
        try:
            if "pubmed_db_initialized" in st.session_state:
                return  # Already initialized

            st.session_state.pubmed_db_initialized = True

            # Initialize databases
            results = self.pubmed_db_initializer.initialize_databases(
                openai_api_key=(
                    openai_api_key
                    if openai_api_key and openai_api_key != "ollama"
                    else None
                )
            )

            if results.get("openai") or results.get("qwen"):
                status_msg = []
                if results.get("openai"):
                    status_msg.append("✅ OpenAI database ready")
                if results.get("qwen"):
                    status_msg.append("✅ Qwen database ready")
                self.logger.info(" | ".join(status_msg))
            else:
                self.logger.info("PubMed databases initialization skipped or failed")

        except Exception as e:
            self.logger.error(f"Failed to initialize PubMed databases: {str(e)}")
            st.session_state.pubmed_db_initialized = (
                True  # Mark as attempted to avoid retry loop
            )

    def _render_file_upload_section(self):
        """Render file upload section"""
        st.subheader("📁 Document Upload")

        uploaded_files = st.file_uploader(
            "Upload Documents",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True,
            help="Upload PDF, DOCX, or TXT files to create or update the knowledge base",
        )

        if uploaded_files:
            # Remove duplicate files (same filename) - keep only the first occurrence
            seen_filenames = {}
            unique_files = []
            duplicate_files = []
            
            for file in uploaded_files:
                filename = file.name
                if filename not in seen_filenames:
                    seen_filenames[filename] = True
                    unique_files.append(file)
                else:
                    duplicate_files.append(filename)
            
            # Show warning if duplicates were found
            if duplicate_files:
                st.warning(
                    f"⚠️ **{len(duplicate_files)} duplicate file(s) removed:**\n\n" +
                    "\n".join([f"• {name}" for name in duplicate_files])
                )
            
            # Use unique files for processing
            uploaded_files = unique_files
            
            st.write(f"📋 {len(uploaded_files)} unique file(s) selected:")
            for file in uploaded_files:
                file_size = file.size if hasattr(file, "size") else len(file.getvalue())
                st.write(f"• {file.name} ({format_file_size(file_size)})")

            # Document processing parameters
            with st.expander("⚙️ Processing Parameters"):
                chunk_size = st.slider(
                    "Chunk Size",
                    min_value=200,
                    max_value=2000,
                    value=800,
                    step=100,
                    help="Size of text chunks for processing",
                )
                chunk_overlap = st.slider(
                    "Chunk Overlap",
                    min_value=0,
                    max_value=500,
                    value=100,
                    step=50,
                    help="Overlap between consecutive chunks",
                )
                st.session_state.chunk_size = chunk_size
                st.session_state.chunk_overlap = chunk_overlap

                # Check if parameters changed and warn user
                if st.session_state.vector_store_initialized:
                    previous_chunk_size = getattr(
                        st.session_state, "previous_chunk_size", 800
                    )
                    previous_chunk_overlap = getattr(
                        st.session_state, "previous_chunk_overlap", 100
                    )

                    if (
                        chunk_size != previous_chunk_size
                        or chunk_overlap != previous_chunk_overlap
                    ):
                        st.warning(
                            "⚠️ Chunk parameters have changed! To apply new settings, you need to reprocess your documents. "
                            + "The existing vector store will be updated with new chunks."
                        )

                # Store current parameters for comparison
                st.session_state.previous_chunk_size = chunk_size
                st.session_state.previous_chunk_overlap = chunk_overlap

            if st.button(
                "🚀 Process Documents", type="primary", use_container_width=True
            ):
                self._process_documents(uploaded_files)

        # Retrieval parameters section
        with st.expander("🔍 Retrieval Parameters"):
            search_k = st.slider(
                "Number of Sources (k)",
                min_value=1,
                max_value=15,
                value=7,
                step=1,
                help="Number of document chunks to retrieve (more = better coverage but slower)",
            )
            st.session_state.search_k = search_k

            st.info(
                "💡 **Retrieval Information:**\n\n"
                + "The system uses similarity search with scores to find the most relevant documents. "
                + "All retrieved sources will be used by the AI model to generate comprehensive answers. "
                + "The accuracy scores shown are calculated based on how well each source contributed to the final answer."
            )

        # Model parameters section
        with st.expander("🤖 Model Parameters"):
            # Get current model provider to show relevant parameters
            current_provider = st.session_state.get(
                "current_model_provider", "OpenAI (API)"
            )

            temperature = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=1.0,
                value=0.7 if current_provider == "OpenAI (API)" else 0.3,
                step=0.1,
                help="Controls randomness in responses. Lower values (0.1-0.3) are more focused and deterministic, higher values (0.7-1.0) are more creative and varied.",
            )

            # Store temperature in session state
            st.session_state.model_temperature = temperature

            st.info(
                f"💡 **Temperature Guide:**\n"
                + f"- **0.0-0.3**: Highly focused, deterministic (good for factual Q&A)\n"
                + f"- **0.4-0.7**: Balanced creativity and focus (recommended)\n"
                + f"- **0.8-1.0**: Very creative, more varied responses\n\n"
                + f"**Current Provider**: {current_provider}\n\n"
                + f"**Note**: Max tokens are automatically set by the model for optimal performance."
            )

    def _render_chat_management_section(self):
        """Render chat management section"""
        st.subheader("💭 Chat Management")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                self._clear_current_conversation()

        with col2:
            # Export conversation as CSV
            if st.session_state.current_conversation.messages:
                if st.button("📥 Export as CSV", use_container_width=True):
                    self._export_conversation_csv()
            else:
                st.button(
                    "📥 Export as CSV",
                    use_container_width=True,
                    disabled=True,
                    help="No conversation to export",
                )

    def _render_system_info_section(self):
        """Render system information section"""

        # Helper function to get chunk count from vector store
        def get_chunk_count(store):
            """Get chunk count from a FAISS vector store"""
            if not store:
                return 0
            try:
                if hasattr(store, "docstore") and store.docstore:
                    if hasattr(store.docstore, "_dict"):
                        return len(store.docstore._dict)
                if hasattr(store, "index") and hasattr(store.index, "ntotal"):
                    return store.index.ntotal
            except Exception:
                pass
            return 0

        # Helper function to get document count from new vector store
        def get_new_doc_count(store):
            """Get unique document count from new DB vector store"""
            if not store:
                return 0
            try:
                if hasattr(store, "docstore") and store.docstore:
                    if hasattr(store.docstore, "_dict"):
                        unique_doc_sources = set()
                        for doc_metadata in store.docstore._dict.values():
                            if hasattr(doc_metadata, "metadata"):
                                metadata = doc_metadata.metadata
                                doc_source = (
                                    metadata.get("source")
                                    or metadata.get("document_name")
                                    or metadata.get("filename")
                                    or metadata.get("document_id")
                                )
                                if doc_source:
                                    unique_doc_sources.add(doc_source)
                        return len(unique_doc_sources) if unique_doc_sources else 0
            except Exception as e:
                self.logger.debug(f"Could not get new doc count: {e}")
            return 0

        # Helper function to calculate in-memory FAISS index size
        def get_in_memory_index_size(store):
            """Calculate approximate in-memory size of FAISS index"""
            if not store:
                return 0
            try:
                total_size = 0

                # Get FAISS index size (vectors)
                if hasattr(store, "index") and store.index:
                    if hasattr(store.index, "ntotal") and hasattr(store.index, "d"):
                        num_vectors = store.index.ntotal
                        dimension = store.index.d
                        # FAISS uses float32 (4 bytes per float) for vectors
                        vector_size = num_vectors * dimension * 4
                        total_size += vector_size

                # Estimate docstore size (documents + metadata)
                if hasattr(store, "docstore") and store.docstore:
                    if hasattr(store.docstore, "_dict"):
                        docstore_dict = store.docstore._dict
                        # Rough estimate: each document with metadata ~500-2000 bytes
                        # We'll calculate more accurately
                        for doc in docstore_dict.values():
                            if hasattr(doc, "page_content"):
                                # Page content size
                                total_size += len(doc.page_content.encode("utf-8"))
                            if hasattr(doc, "metadata"):
                                # Metadata size (rough estimate)
                                metadata = doc.metadata
                                if isinstance(metadata, dict):
                                    for key, value in metadata.items():
                                        total_size += len(str(key).encode("utf-8"))
                                        total_size += len(str(value).encode("utf-8"))

                return total_size
            except Exception as e:
                self.logger.debug(f"Could not calculate in-memory index size: {e}")
                return 0

        # Create container for stats (Visually at top)
        stats_container = st.container()

        # System management buttons (Defined here but stats will render above due to container)
        st.markdown("---")
        st.subheader("🔧 System Management")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("📤 Export Logs", use_container_width=True):
                self._export_logs()

        with col2:
            if st.button("🗑️ Clear Logs", use_container_width=True):
                self._clear_logs()

        col3, col4 = st.columns(2)

        with col3:
            if st.button("🗑️ Delete Chunks", use_container_width=True):
                self._delete_chunks()

        with col4:
            if st.button("🧹 Clear Cache", use_container_width=True):
                self._clear_cache()

        # Render stats into container (runs AFTER buttons to reflect new state)
        with stats_container:
            st.subheader("📊 System Info")
            
            # Fetch current state (inside with block to get latest values)
            db_mode = st.session_state.get("db_mode", "current")
            new_vector_store = st.session_state.get("new_vector_store", None)

            # Render based on DB mode
            if db_mode == "current":
                # Show only current DB
                vector_info = self.vector_store_manager.get_store_info()
                if vector_info:
                    # Use cached value if available, otherwise use vector_info
                    total_docs = (
                        self.vector_store_manager._cached_total_documents
                        if self.vector_store_manager._cached_total_documents is not None
                        else vector_info.total_documents
                    )
                    st.write(f"📚 Total Documents: {total_docs}")

                    # Get chunk count
                    actual_chunks = get_chunk_count(self.vector_store_manager.vector_store)
                    if actual_chunks == 0:
                        actual_chunks = vector_info.total_chunks

                    # Cache chunk count
                    if actual_chunks > 0:
                        if (
                            "total_chunks_cached" not in st.session_state
                            or actual_chunks >= st.session_state.total_chunks_cached
                        ):
                            st.session_state.total_chunks_cached = actual_chunks

                    st.write(
                        f"🧩 Total Chunks: {st.session_state.get('total_chunks_cached', actual_chunks)}"
                    )
                    st.write(
                        f"💾 Index Size: {format_file_size(vector_info.index_size_bytes)}"
                    )
                else:
                    st.write("📭 No vector store found")

            elif db_mode == "new":
                # Show only new DB
                if new_vector_store:
                    new_chunks = get_chunk_count(new_vector_store)
                    new_docs = get_new_doc_count(new_vector_store)
                    new_index_size = get_in_memory_index_size(new_vector_store)

                    st.write(f"📚 Total Documents: {new_docs}")
                    st.write(f"🧩 Total Chunks: {new_chunks}")
                    st.write(
                        f"💾 Index Size: {format_file_size(new_index_size)} (in-memory)"
                    )
                else:
                    st.write("📭 No vector store found")

            elif db_mode == "current+new":
                # Show combined stats
                vector_info = self.vector_store_manager.get_store_info()

                current_docs = 0
                current_chunks = 0
                current_index_size = 0

                if vector_info:
                    current_docs = (
                        self.vector_store_manager._cached_total_documents
                        if self.vector_store_manager._cached_total_documents is not None
                        else vector_info.total_documents
                    )
                    current_chunks = get_chunk_count(self.vector_store_manager.vector_store)
                    if current_chunks == 0:
                        current_chunks = vector_info.total_chunks
                    current_index_size = vector_info.index_size_bytes

                new_docs = 0
                new_chunks = 0
                new_index_size = 0

                if new_vector_store:
                    new_chunks = get_chunk_count(new_vector_store)
                    new_docs = get_new_doc_count(new_vector_store)
                    new_index_size = get_in_memory_index_size(new_vector_store)

                total_docs = current_docs + new_docs
                total_chunks = current_chunks + new_chunks
                total_index_size = current_index_size + new_index_size

                if total_docs > 0 or total_chunks > 0:
                    st.write(
                        f"📚 Total Documents: {total_docs} (Current: {current_docs}, New: {new_docs})"
                    )
                    st.write(
                        f"🧩 Total Chunks: {total_chunks} (Current: {current_chunks}, New: {new_chunks})"
                    )
                    if total_index_size > 0:
                        size_info = f"💾 Index Size: {format_file_size(total_index_size)}"
                        if current_index_size > 0 and new_index_size > 0:
                            size_info += f" (Current: {format_file_size(current_index_size)}, New: {format_file_size(new_index_size)} in-memory)"
                        elif current_index_size > 0:
                            size_info += f" (Current: {format_file_size(current_index_size)}, New: {format_file_size(new_index_size)} in-memory)"
                        elif new_index_size > 0:
                            size_info += " (in-memory only)"
                        st.write(size_info)
                    elif current_index_size > 0:
                        st.write(
                            f"💾 Index Size: {format_file_size(current_index_size)} (Current DB only)"
                        )
                else:
                    st.write("📭 No vector store found")

            # Cache statistics
            st.markdown("---")
            st.subheader("💾 Cache Info")
            cache_stats = self.vector_store_manager.get_cache_stats()
            if cache_stats["enabled"]:
                st.write(f"✅ Cache: Enabled")
                st.write(
                    f"📦 Cached Entries: {cache_stats['total_entries']}/{cache_stats['max_size']}"
                )
                st.write(f"🎯 Cache Hits: {cache_stats['hits']}")
                st.write(f"❌ Cache Misses: {cache_stats['misses']}")
                if cache_stats["total_queries"] > 0:
                    st.write(f"📈 Hit Rate: {cache_stats['hit_rate']:.1f}%")
                st.write(f"⏱️ TTL: {cache_stats['ttl_seconds']}s")
            else:
                st.write("⚠️ Cache: Disabled")


    def _render_social_links(self):
        """Render social media links"""
        st.markdown("---")
        st.subheader("🔗 Connect")

        linkedin_url = "https://www.linkedin.com/in/huseyincenik/"
        kaggle_url = "https://www.kaggle.com/huseyincenik/"
        github_url = "https://github.com/huseyincenik/"

        st.markdown(
            f"""
        [![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)]({linkedin_url})
        [![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)]({kaggle_url})
        [![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)]({github_url})
        """
        )

    def _render_main_interface(self):
        """Render main chat interface"""
        # Display conversation
        self._render_conversation()

        # Chat input
        self._render_chat_input()

    def _render_conversation(self):
        """Render conversation messages"""
        if not st.session_state.current_conversation.messages:
            st.info("👋 Welcome! Upload some documents and start asking questions.")
            return

        # Display messages in pairs (user + assistant)
        messages = st.session_state.current_conversation.messages
        i = 0
        while i < len(messages):
            if i < len(messages) and messages[i].role == MessageRole.USER:
                user_message = messages[i]
                assistant_message = None

                # Check if there's a corresponding assistant message
                if (
                    i + 1 < len(messages)
                    and messages[i + 1].role == MessageRole.ASSISTANT
                ):
                    assistant_message = messages[i + 1]

                # Render user message with edit option
                with st.chat_message("user"):
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        st.write(user_message.content)
                        if user_message.timestamp:
                            st.caption(
                                f"🕒 {user_message.timestamp.strftime('%H:%M:%S')}"
                            )

                    with col2:
                        # Edit butonu ve timestamp aynı hizada olsun diye boş alan bırakıyoruz
                        st.write("")  # Timestamp hizası için boşluk
                        if st.button(
                            "✏️ Edit",
                            key=f"edit_history_{user_message.id}",
                            help="Edit and regenerate",
                        ):
                            # Store edit info in session state
                            st.session_state.edit_message_id = user_message.id
                            st.session_state.edit_query = user_message.content
                            st.session_state.edit_assistant_id = (
                                assistant_message.id if assistant_message else None
                            )
                            st.rerun()

                # Render assistant message if exists
                if assistant_message:
                    self._render_assistant_message_with_actions(
                        assistant_message, user_message.content
                    )
                    i += 2  # Skip both user and assistant message
                else:
                    i += 1  # Only user message, move to next
            else:
                # Orphaned assistant message (shouldn't happen normally)
                if messages[i].role == MessageRole.ASSISTANT:
                    with st.chat_message("assistant"):
                        # Format content: replace literal \n with actual line breaks (blank lines)
                        formatted_content = messages[i].content.replace('\\n', '\n\n') if isinstance(messages[i].content, str) else str(messages[i].content)
                        st.markdown(formatted_content)
                i += 1

    def _render_assistant_message_with_actions(
        self, assistant_message: ChatMessage, user_query: str
    ):
        """Render assistant message with action buttons"""
        with st.chat_message("assistant"):
            # Display response in main area (no columns)
            # Format content: replace literal \n with actual line breaks (blank lines)
            formatted_content = assistant_message.content.replace('\\n', '\n\n') if isinstance(assistant_message.content, str) else str(assistant_message.content)
            st.markdown(formatted_content)
            
            # Check if LLM indicated insufficient information
            insufficient_info = (
                assistant_message.metadata.get("insufficient_info", False)
                if assistant_message.metadata
                else False
            )
            source_count = (
                assistant_message.metadata.get("source_count", 0)
                if assistant_message.metadata
                else 0
            )

            # Only show sources if we have valid sources and LLM didn't say "no info"
            if (
                assistant_message.metadata
                and assistant_message.metadata.get("sources")
                and source_count > 0
                and not insufficient_info
            ):
                sources = assistant_message.metadata["sources"]
                
                # Create two columns: left for metadata+context, right for PDF preview
                col_metadata, col_pdf = st.columns([1.2, 1])
                
                with col_metadata:
                    # Metadata expander
                    with st.expander("📊 Metadata", expanded=False):
                        # Create JSON array in the format shown in image
                        source_json_array = []
                        for source in sources:
                            source_file = source.get("source", "Unknown")
                            page = source.get("page", "Unknown")
                            
                            # Convert page to list if it's a single value
                            if isinstance(page, (int, str)):
                                page_numbers = [str(page)]
                            elif isinstance(page, list):
                                page_numbers = [str(p) for p in page]
                            else:
                                page_numbers = ["Unknown"]
                            
                            source_json = {
                                "file_name": source_file,
                                "page_numbers": page_numbers,
                                "source_type": "internal"
                            }
                            source_json_array.append(source_json)
                        
                        # Display JSON with word wrap
                        source_json_str = json.dumps(source_json_array, indent=2, ensure_ascii=False)
                        # Use st.text_area with word wrap instead of st.code for better wrapping
                        st.text_area(
                            "Metadata JSON",
                            source_json_str,
                            height=200,
                            key=f"metadata_json_hist_{assistant_message.id}_{uuid.uuid4()}",
                            disabled=True,
                            label_visibility="collapsed"
                        )
                    
                    # Context expander
                    with st.expander("📝 Context", expanded=False):
                        # Create context JSON array with content from sources
                        context_json_array = []
                        for i, source in enumerate(sources, 1):
                            source_file = source.get("source", "Unknown")
                            page = source.get("page", "Unknown")
                            content = source.get("content", "")
                            
                            # Format content: replace literal \n with actual line breaks
                            if isinstance(content, str):
                                # Replace literal \n (escaped newline) with actual newlines
                                formatted_content = content.replace('\\n', '\n')
                            else:
                                formatted_content = str(content)
                            
                            context_json = {
                                "source_index": i,
                                "file_name": source_file,
                                "page": str(page),
                                "content": formatted_content
                            }
                            context_json_array.append(context_json)
                        
                        # Display context JSON with word wrap
                        context_json_str = json.dumps(context_json_array, indent=2, ensure_ascii=False)
                        # Replace escaped newlines (\\n) in JSON string with actual newlines (blank lines) for better readability
                        # This makes the JSON more readable in the text area - each \n becomes a blank line
                        context_json_str = context_json_str.replace('\\n', '\n\n')
                        # Use st.text_area with word wrap instead of st.code for better wrapping
                        st.text_area(
                            "Context JSON",
                            context_json_str,
                            height=400,
                            key=f"context_json_hist_{assistant_message.id}_{uuid.uuid4()}",
                            disabled=True,
                            label_visibility="collapsed"
                        )
                
                with col_pdf:
                    st.markdown("### 📄 See PDF Documents")
                    
                    # Get unique files from sources
                    unique_files = {}
                    for source in sources:
                        source_file = source.get("source", "Unknown")
                        page = source.get("page", "Unknown")
                        
                        # Convert page to int if possible
                        try:
                            if isinstance(page, str) and page.isdigit():
                                page_num = int(page)
                            elif isinstance(page, int):
                                page_num = page
                            else:
                                page_num = 1
                        except:
                            page_num = 1
                        
                        if source_file not in unique_files:
                            unique_files[source_file] = []
                        if page_num not in unique_files[source_file]:
                            unique_files[source_file].append(page_num)
                    
                    # Create dropdown for file selection
                    if unique_files:
                        file_options = []
                        for file_name, pages in unique_files.items():
                            for page in sorted(pages):
                                file_options.append(f"{file_name} Page: {page}")
                        
                        # Use Streamlit's built-in key mechanism to preserve selection
                        selectbox_key = f"pdf_select_{assistant_message.id}"
                        
                        # Ensure current selection is valid, if not set to first option
                        if selectbox_key not in st.session_state or st.session_state[selectbox_key] not in file_options:
                            if file_options:
                                st.session_state[selectbox_key] = file_options[0]
                        
                        # Use selectbox with key - Streamlit handles state automatically
                        # When selection changes, it triggers rerun automatically
                        # Get the selected value directly from the widget (always returns current value)
                        selected_file_page = st.selectbox(
                            "Choose PDF File",
                            file_options,
                            key=selectbox_key
                        )
                        
                        # Extract filename and page from selection
                        if selected_file_page:
                            try:
                                # Parse "filename.pdf Page: X"
                                parts = selected_file_page.rsplit(" Page: ", 1)
                                if len(parts) == 2:
                                    selected_file = parts[0]
                                    selected_page = int(parts[1])
                                    
                                    # Get PDF content from session state
                                    pdf_contents = st.session_state.get("uploaded_pdf_contents", {})
                                    
                                    # Try to find the file (check exact match and variations)
                                    pdf_content = None
                                    from pathlib import Path
                                    selected_file_path = Path(selected_file)
                                    
                                    # Try multiple matching strategies
                                    # 1. Exact match
                                    if selected_file in pdf_contents:
                                        pdf_content = pdf_contents[selected_file]
                                    
                                    # 2. Try with .pdf extension (for DOCX files converted to PDF)
                                    if not pdf_content:
                                        pdf_variant = selected_file_path.with_suffix('.pdf')
                                        if str(pdf_variant) in pdf_contents:
                                            pdf_content = pdf_contents[str(pdf_variant)]
                                    
                                    # 3. Try with .docx extension (for files that might be stored with original extension)
                                    if not pdf_content:
                                        docx_variant = selected_file_path.with_suffix('.docx')
                                        if str(docx_variant) in pdf_contents:
                                            pdf_content = pdf_contents[str(docx_variant)]
                                    
                                    # 4. Try case-insensitive matching
                                    if not pdf_content:
                                        selected_lower = selected_file.lower()
                                        for stored_name, content in pdf_contents.items():
                                            if stored_name.lower() == selected_lower:
                                                pdf_content = content
                                                break
                                    
                                    # 5. Try stem-based matching (filename without extension)
                                    if not pdf_content:
                                        selected_stem = selected_file_path.stem.lower()
                                        for stored_name, content in pdf_contents.items():
                                            stored_stem = Path(stored_name).stem.lower()
                                            if stored_stem == selected_stem:
                                                pdf_content = content
                                                break
                                    
                                    # 6. Try partial matching (as last resort)
                                    if not pdf_content:
                                        selected_stem = selected_file_path.stem.lower()
                                        for stored_name, content in pdf_contents.items():
                                            stored_stem = Path(stored_name).stem.lower()
                                            if selected_stem in stored_stem or stored_stem in selected_stem:
                                                pdf_content = content
                                                break
                                    
                                    if pdf_content:
                                        # Create a unique key for PDF display based on selection
                                        # Use hash of selection to ensure uniqueness and force cache-busting
                                        import hashlib
                                        selection_hash = hashlib.md5(f"{selected_file}_{selected_page}".encode()).hexdigest()[:8]
                                        pdf_display_key = f"pdf_display_{assistant_message.id}_{selected_file}_{selected_page}_{selection_hash}"
                                        
                                        # Track last selection to detect changes
                                        last_selection_key = f"last_pdf_selection_{assistant_message.id}"
                                        current_selection = f"{selected_file}_{selected_page}"
                                        
                                        # Find ALL matching sources for this file and page (ID-based)
                                        # There can be multiple chunks on the same page that need to be highlighted
                                        matching_sources = []
                                        for source in sources:
                                            source_file = source.get("source", "Unknown")
                                            source_page = source.get("page", "Unknown")
                                            try:
                                                source_page_num = int(source_page) if isinstance(source_page, (int, str)) and str(source_page).isdigit() else None
                                                if source_file == selected_file and source_page_num == selected_page:
                                                    matching_sources.append(source)
                                            except:
                                                pass
                                        
                                        # Use container to force re-render when selection changes
                                        # Container key includes selection to force update
                                        container_key = f"pdf_container_{assistant_message.id}_{selected_file}_{selected_page}"
                                        with st.container():
                                            # Display PDF page with highlights
                                            # Unique display_key ensures immediate update when page changes
                                            self._display_pdf_page(pdf_content, selected_page, matching_sources, display_key=pdf_display_key)
                                        
                                        # Update last selection for change detection
                                        st.session_state[last_selection_key] = current_selection
                                    else:
                                        # Debug: show available keys if PDF not found
                                        available_keys = list(pdf_contents.keys())
                                        if available_keys:
                                            st.warning(
                                                f"⚠️ PDF content not found for: **{selected_file}**\n\n"
                                                f"Available files in cache: {', '.join(available_keys[:5])}"
                                                + (f" (and {len(available_keys) - 5} more)" if len(available_keys) > 5 else "")
                                            )
                                        else:
                                            st.info(f"PDF content not found for: {selected_file}. No PDF files available in cache.")
                            except Exception as e:
                                st.error(f"Error displaying PDF: {str(e)}")
                    else:
                        st.info("No PDF files available for preview")
            elif insufficient_info:
                # LLM indicated insufficient information
                st.markdown("### ℹ️ Insufficient Information")
                st.info(
                    "The AI model indicated that the provided documents do not contain "
                    "enough information to answer this question reliably."
                )
            else:
                # No sources at all
                st.markdown("### ℹ️ No Sources Available")
                st.write(
                    "No source information could be extracted for this response."
                )

            # Action buttons below the response
            st.markdown("---")
            col1, col2, col3 = st.columns([1, 1, 2])

            with col1:
                if st.button(
                    "🔄 Retry",
                    key=f"retry_{assistant_message.id}",
                    help="Regenerate response with current parameters",
                ):
                    # Store the current query and assistant ID for retry
                    st.session_state.retry_query = user_query
                    st.session_state.retry_assistant_id = assistant_message.id
                    st.rerun()

            with col2:
                if st.button(
                    "📋 Copy",
                    key=f"copy_{assistant_message.id}",
                    help="Copy response to clipboard",
                ):
                    # Use JavaScript to copy to clipboard
                    copy_text = assistant_message.content.replace('"', '\\"').replace(
                        "\n", "\\n"
                    )
                    st.markdown(
                        f"""
                    <script>
                    navigator.clipboard.writeText("{copy_text}").then(function() {{
                        console.log('Text copied to clipboard');
                    }});
                    </script>
                    """,
                        unsafe_allow_html=True,
                    )
                    st.success("📋 Response copied to clipboard!", icon="✅")

            if assistant_message.timestamp:
                st.caption(f"🕒 {assistant_message.timestamp.strftime('%H:%M:%S')}")

    def _render_message(self, message: ChatMessage):
        """Render a single message (legacy method, now replaced by _render_conversation)"""
        if message.role == MessageRole.USER:
            with st.chat_message("user"):
                st.write(message.content)
                if message.timestamp:
                    st.caption(f"🕒 {message.timestamp.strftime('%H:%M:%S')}")

        elif message.role == MessageRole.ASSISTANT:
            with st.chat_message("assistant"):
                st.write(message.content)

                # Show metadata if available
                if message.metadata:
                    with st.expander("ℹ️ Response Details"):
                        if message.response_time:
                            st.write(f"⚡ Response Time: {message.response_time:.2f}s")
                        if message.model_provider:
                            st.write(f"🤖 Model: {message.model_provider.value}")
                        if message.source_documents:
                            st.write(
                                f"📚 Sources: {len(message.source_documents)} documents"
                            )

                if message.timestamp:
                    st.caption(f"🕒 {message.timestamp.strftime('%H:%M:%S')}")

    def _render_chat_input(self):
        """Render chat input section"""
        # Check if there's an edit request
        if "edit_message_id" in st.session_state:
            st.markdown("### ✏️ Edit your question:")

            # Create a form for editing
            with st.form("edit_form"):
                edited_query = st.text_area(
                    "Edit your question:",
                    value=st.session_state.edit_query,
                    height=100,
                    placeholder="Type your edited question here...",
                )

                col1, col2 = st.columns(2)
                with col1:
                    submit_edit = st.form_submit_button(
                        "🔄 Update & Regenerate", type="primary"
                    )
                with col2:
                    cancel_edit = st.form_submit_button("❌ Cancel")

                if submit_edit and edited_query.strip():
                    # Update the user message
                    messages = st.session_state.current_conversation.messages
                    for msg in messages:
                        if msg.id == st.session_state.edit_message_id:
                            msg.content = edited_query.strip()
                            break

                    # Clear edit state
                    edit_assistant_id = st.session_state.get("edit_assistant_id")
                    del st.session_state.edit_message_id
                    del st.session_state.edit_query
                    if "edit_assistant_id" in st.session_state:
                        del st.session_state.edit_assistant_id

                    # Process the edited query (with replace mode for assistant message)
                    if edit_assistant_id:
                        st.session_state.replace_assistant_id = edit_assistant_id

                    self._process_user_query(edited_query.strip(), is_edit=True)
                    return

                if cancel_edit:
                    # Clear edit state
                    if "edit_message_id" in st.session_state:
                        del st.session_state.edit_message_id
                    if "edit_query" in st.session_state:
                        del st.session_state.edit_query
                    if "edit_assistant_id" in st.session_state:
                        del st.session_state.edit_assistant_id
                    st.rerun()
            return

        # Check if there's a retry query
        if "retry_query" in st.session_state:
            retry_query = st.session_state.retry_query
            retry_assistant_id = st.session_state.get("retry_assistant_id")
            del st.session_state.retry_query  # Clear it after use
            if "retry_assistant_id" in st.session_state:
                del st.session_state.retry_assistant_id

            if not st.session_state.vector_store_initialized:
                st.error("❌ Please upload and process documents first!")
                return

            if not st.session_state.llm_initialized:
                st.error("❌ Please initialize the LLM first!")
                return

            # Set replace mode for retry
            if retry_assistant_id:
                st.session_state.replace_assistant_id = retry_assistant_id

            self._process_user_query(retry_query, is_retry=True)
            return

        user_input = st.chat_input("Ask a question about your documents...")

        if user_input:
            # Check DB mode - "current" mode can work without uploaded docs
            db_mode = st.session_state.get("db_mode", "current")
            if db_mode == "new" and not st.session_state.vector_store_initialized:
                st.error("❌ Please upload and process documents first!")
                return
            elif db_mode == "current" and not st.session_state.llm_initialized:
                st.error("❌ Please initialize the LLM first!")
                return
            elif db_mode == "current+new" and not st.session_state.llm_initialized:
                st.error("❌ Please initialize the LLM first!")
                return

            if not st.session_state.llm_initialized:
                st.error("❌ Please initialize the LLM first!")
                return

            self._process_user_query(user_input)

    def _render_footer(self):
        """Render application footer"""
        st.markdown("---")
        st.markdown(
            "<div style='text-align: center; color: #666;'>"
            "Built with ❤️ using Streamlit, LangChain, and FAISS"
            "</div>",
            unsafe_allow_html=True,
        )

    def _initialize_llm(self, model_provider: str, api_key: str):
        """Initialize LLM service"""
        try:
            with st.spinner("🔧 Initializing LLM..."):
                # Initialize embeddings for vector store
                self.vector_store_manager.initialize_embeddings(model_provider, api_key)

                # Initialize LLM
                self.llm_service.initialize_llm(model_provider, api_key)

                # Validate connection
                if self.llm_service.validate_api_connection():
                    st.session_state.llm_initialized = True
                    st.session_state.current_model_provider = model_provider
                    st.session_state.api_key = api_key  # Store API key for later use

                    # Ensure empty FAISS index exists for OpenAI (like Qwen)
                    if model_provider == "OpenAI (API)":
                        self._ensure_empty_faiss_index(model_provider, api_key)

                    # Initialize PubMed databases after LLM is ready
                    self._initialize_pubmed_databases(
                        openai_api_key=(
                            api_key if model_provider == "OpenAI (API)" else None
                        )
                    )

                    st.success(f"✅ {model_provider} LLM initialized successfully!")
                    self.logger.info(f"LLM initialized: {model_provider}")
                else:
                    if model_provider == "Local LLM (Qwen)":
                        st.error(
                            "❌ Failed to connect to Ollama service.\n\n"
                            "**Possible reasons:**\n"
                            "- Ollama container is still starting up (wait 1-2 minutes)\n"
                            "- Qwen2.5:7b model is still being downloaded (can take 5-10 minutes)\n"
                            "- Ollama service is not running\n\n"
                            "**To check status:**\n"
                            "Run: `docker-compose logs ollama`"
                        )
                    else:
                        st.error(
                            "❌ Failed to validate API connection. Please check your API key."
                        )

        except Exception as e:
            st.error(f"❌ Failed to initialize LLM: {str(e)}")
            self.logger.error(f"LLM initialization failed: {str(e)}")

    def _process_documents(self, uploaded_files):
        """Process uploaded documents"""
        try:
            # Check if LLM is initialized first
            if not st.session_state.get("llm_initialized", False):
                st.error("❌ Please initialize LLM first before processing documents!")
                st.info(
                    "💡 Select a model provider, enter your API key, and click 'Initialize LLM'"
                )
                return

            # Check for duplicate files before processing
            from src.utils import get_file_hash
            
            # Get existing document hashes from session state
            existing_hashes = set()
            existing_documents = st.session_state.get("documents", [])
            for doc in existing_documents:
                if doc.file_hash:
                    existing_hashes.add(doc.file_hash)
            
            # Filter out duplicate files
            filtered_files = []
            duplicate_files = []
            
            for uploaded_file in uploaded_files:
                # Read file content to calculate hash
                file_content = uploaded_file.read()
                uploaded_file.seek(0)  # Reset file pointer
                
                # Calculate hash of original file content (before any conversion)
                file_hash = get_file_hash(file_content)
                
                # Check if this file was already processed
                if file_hash in existing_hashes:
                    # Find the existing document name for better error message
                    existing_doc = next((d for d in existing_documents if d.file_hash == file_hash), None)
                    existing_name = existing_doc.name if existing_doc else uploaded_file.name
                    duplicate_files.append((uploaded_file.name, existing_name))
                else:
                    filtered_files.append(uploaded_file)
            
            # Show warnings for duplicate files
            if duplicate_files:
                duplicate_names = [name for name, _ in duplicate_files]
                existing_names = [existing for _, existing in duplicate_files]
                st.warning(
                    f"⚠️ **{len(duplicate_files)} file(s) already processed and skipped:**\n\n" +
                    "\n".join([f"• **{name}** (already processed as: {existing})" 
                               for name, existing in zip(duplicate_names, existing_names)])
                )
            
            if not filtered_files:
                st.error("❌ All files have already been processed. Please upload new files.")
                return

            with st.spinner("📄 Processing documents..."):
                # Process only non-duplicate documents
                documents = self.document_processor.process_uploaded_files(
                    filtered_files
                )

                if not documents:
                    st.error("❌ No documents were successfully processed")
                    return

                # Create chunks
                all_chunks = []
                current_model_provider = st.session_state.get(
                    "current_model_provider", "OpenAI"
                )

                for doc in documents:
                    chunks = self.document_processor.create_chunks(
                        doc,
                        current_model_provider,
                        chunk_size=st.session_state.chunk_size,
                        chunk_overlap=st.session_state.chunk_overlap,
                    )
                    all_chunks.extend(chunks)

                # Ensure embeddings are initialized for the current model provider
                # (This should already be done in _initialize_llm, but double-check)
                if not self.vector_store_manager.embeddings:
                    # Re-initialize embeddings if missing
                    try:
                        api_key = st.session_state.get("api_key", "")
                        if api_key:
                            self.vector_store_manager.initialize_embeddings(
                                current_model_provider, api_key
                            )
                        else:
                            st.error("❌ API key not found. Please re-initialize LLM.")
                            return
                    except Exception as e:
                        st.error(f"❌ Failed to initialize embeddings: {str(e)}")
                        return

                # Handle different DB modes
                db_mode = st.session_state.get("db_mode", "current")

                if db_mode == "new" or db_mode == "current+new":
                    # Check if there's an existing new vector store
                    existing_new_vector_store = st.session_state.get(
                        "new_vector_store", None
                    ) or self.new_db_manager.get_new_vector_store(
                        current_model_provider
                    )

                    if existing_new_vector_store:
                        # Add to existing new vector store instead of replacing it
                        self.new_db_manager.add_to_new_vector_store(
                            vector_store=existing_new_vector_store,
                            documents=documents,
                            chunks=all_chunks,
                        )
                        new_vector_store = existing_new_vector_store
                        action = "added to new database"
                    else:
                        # Create new vector store for uploaded documents
                        new_vector_store = self.new_db_manager.create_new_vector_store(
                            documents=documents,
                            chunks=all_chunks,
                            embeddings=self.vector_store_manager.embeddings,
                            provider_name=current_model_provider,
                        )
                        action = "created in new database"

                    st.session_state.new_vector_store = new_vector_store
                    vector_info = None  # New DB doesn't have metadata yet
                else:
                    # Update or create regular vector store
                    if st.session_state.vector_store_initialized:
                        vector_info = self.vector_store_manager.update_vector_store(
                            documents, all_chunks
                        )
                        action = "updated"
                    else:
                        vector_info = self.vector_store_manager.create_vector_store(
                            documents, all_chunks
                        )
                        st.session_state.vector_store_initialized = True
                        action = "created"

                # Update session state
                st.session_state.documents.extend(documents)

                # Store PDF content in session state for preview
                if "uploaded_pdf_contents" not in st.session_state:
                    st.session_state.uploaded_pdf_contents = {}
                
                for doc in documents:
                    # Store PDF content (for DOCX files, content is already converted to PDF)
                    if doc.file_type.value == "pdf" or doc.metadata.get("converted_from_docx", False):
                        # Use document name as key, store PDF bytes
                        st.session_state.uploaded_pdf_contents[doc.name] = doc.content
                        
                        # For DOCX files converted to PDF, also store with .pdf extension for easier lookup
                        if doc.metadata.get("converted_from_docx", False):
                            from pathlib import Path
                            # Create PDF version of filename (replace .docx with .pdf)
                            pdf_name = str(Path(doc.name).with_suffix('.pdf'))
                            st.session_state.uploaded_pdf_contents[pdf_name] = doc.content
                            # Also store with original filename from metadata
                            original_name = doc.metadata.get("original_name", doc.name)
                            if original_name != doc.name:
                                st.session_state.uploaded_pdf_contents[original_name] = doc.content
                                # And PDF version of original name
                                original_pdf_name = str(Path(original_name).with_suffix('.pdf'))
                                st.session_state.uploaded_pdf_contents[original_pdf_name] = doc.content

                # If using new or current+new mode, mark as initialized
                if db_mode in ["new", "current+new"]:
                    st.session_state.vector_store_initialized = True

                # Show success message
                if vector_info:
                    st.success(
                        f"""
                    ✅ Successfully {action} knowledge base!
                    - 📄 Documents: {len(documents)}
                    - 🧩 Chunks: {len(all_chunks)}
                    - 💾 Total Size: {format_file_size(vector_info.index_size_bytes)}
                    """
                    )
                else:
                    st.success(
                        f"""
                    ✅ Successfully {action}!
                    - 📄 Documents: {len(documents)}
                    - 🧩 Chunks: {len(all_chunks)}
                    - 🗄️ Database Mode: {db_mode}
                    """
                    )

                self.logger.info(
                    f"Processed {len(documents)} documents with {len(all_chunks)} chunks"
                )

        except Exception as e:
            st.error(f"❌ Document processing failed: {str(e)}")
            self.logger.error(
                f"Document processing failed: {str(e)}\n{traceback.format_exc()}"
            )

    def _process_user_query(
        self, user_input: str, is_edit: bool = False, is_retry: bool = False
    ):
        """Process user query and generate response"""
        try:
            # Check if we need to replace an existing assistant message
            replace_assistant_id = st.session_state.get("replace_assistant_id")
            if replace_assistant_id:
                del st.session_state.replace_assistant_id

            # For new queries (not edit/retry), add user message to conversation
            if not is_edit and not is_retry:
                user_message = ChatMessage(
                    id=str(uuid.uuid4()), role=MessageRole.USER, content=user_input
                )
                st.session_state.current_conversation.add_message(user_message)

                # Display user message immediately with edit button
                with st.chat_message("user"):
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        st.write(user_input)
                    with col2:
                        st.write("")  # Timestamp hizası için boşluk
                        if st.button(
                            "✏️ Edit",
                            key=f"edit_active_{user_message.id}",
                            help="Edit and regenerate",
                        ):
                            # Store edit info in session state
                            st.session_state.edit_message_id = user_message.id
                            st.session_state.edit_query = user_input
                            # Assistant mesajı henüz oluşturulmadı, o yüzden None
                            st.session_state.edit_assistant_id = None
                            st.rerun()

            # Generate response
            if not replace_assistant_id:
                # For new messages, use chat_message context
                with st.chat_message("assistant"):
                    with st.spinner("🤔 Thinking..."):
                        # Ensure embeddings are available for search
                        current_model_provider = st.session_state.get(
                            "current_model_provider", "OpenAI"
                        )
                        api_key = st.session_state.get("api_key", "")

                        if not self.vector_store_manager.embeddings and api_key:
                            try:
                                self.vector_store_manager.initialize_embeddings(
                                    current_model_provider, api_key
                                )
                            except Exception as e:
                                st.error(
                                    f"❌ Failed to initialize embeddings for search: {str(e)}"
                                )
                                return

                        # Get search parameters
                        search_k = st.session_state.get("search_k", 5)

                        # Get model parameters
                        temperature = st.session_state.get("model_temperature", 0.7)

                        # Get new_vector_store from session state if available
                        new_vector_store = st.session_state.get(
                            "new_vector_store", None
                        )

                        # CRITICAL: Ensure db_mode is synced before search
                        db_mode = st.session_state.get("db_mode", "current")
                        self.vector_store_manager.set_db_mode(db_mode)

                        if current_model_provider == "OpenAI (API)":
                            search_result = (
                                self.vector_store_manager.search_documents_for_openai(
                                    user_input,
                                    api_key,
                                    k=search_k,
                                    temperature=temperature,
                                    new_vector_store=new_vector_store,
                                )
                            )
                        elif current_model_provider == "Local LLM (Qwen)":
                            # Use the same OpenAI search method but with Ollama configuration
                            search_result = (
                                self.vector_store_manager.search_documents_for_openai(
                                    user_input,
                                    api_key,
                                    k=search_k,
                                    temperature=temperature,
                                    new_vector_store=new_vector_store,
                                )
                            )
                        else:
                            search_result = {"response": "Unsupported model provider"}

                        if not search_result or "Error" in search_result.get(
                            "response", ""
                        ):
                            response_text = search_result.get(
                                "response",
                                "I couldn't find relevant information in the uploaded documents to answer your question.",
                            )
                            response_time = 0.0
                            assistant_message = ChatMessage(
                                id=str(uuid.uuid4()),
                                role=MessageRole.ASSISTANT,
                                content=response_text,
                                model_provider=ModelProvider(
                                    st.session_state.current_model_provider
                                ),
                                response_time=response_time,
                            )
                        else:
                            # Create assistant message with the response and source information
                            sources = search_result.get("sources", [])
                            metadata = search_result.get("metadata", {})

                            assistant_message = ChatMessage(
                                id=str(uuid.uuid4()),
                                role=MessageRole.ASSISTANT,
                                content=search_result.get(
                                    "response", "No response generated"
                                ),
                                model_provider=ModelProvider(current_model_provider),
                                response_time=0.0,
                                tokens_used=0,
                                source_documents=[
                                    s.get("source", "Unknown") for s in sources
                                ],
                                metadata={
                                    "source_count": metadata.get("source_count", 0),
                                    "model": metadata.get("model", "Unknown"),
                                    "sources": sources,
                                    "retrieval_method": metadata.get(
                                        "retrieval_method", "Unknown"
                                    ),
                                },
                            )

                            # Store response time for metrics
                            st.session_state.last_response_time = 0.0

                        # Add new assistant message to conversation
                        st.session_state.current_conversation.add_message(
                            assistant_message
                        )

                        # Display response immediately
                        self._display_assistant_response(assistant_message, user_input)
            else:
                # For replacement, process without chat context and rerun
                with st.spinner("🤔 Regenerating..."):
                    # Ensure embeddings are available for search
                    current_model_provider = st.session_state.get(
                        "current_model_provider", "OpenAI"
                    )
                    api_key = st.session_state.get("api_key", "")

                    if not self.vector_store_manager.embeddings and api_key:
                        try:
                            self.vector_store_manager.initialize_embeddings(
                                current_model_provider, api_key
                            )
                        except Exception as e:
                            st.error(
                                f"❌ Failed to initialize embeddings for search: {str(e)}"
                            )
                            return

                    # Get search parameters
                    search_k = st.session_state.get("search_k", 5)

                    # Get model parameters
                    temperature = st.session_state.get("model_temperature", 0.7)

                    # Get new_vector_store from session state if available
                    new_vector_store = st.session_state.get("new_vector_store", None)

                    # CRITICAL: Ensure db_mode is synced before search
                    db_mode = st.session_state.get("db_mode", "current")
                    self.vector_store_manager.set_db_mode(db_mode)

                    if current_model_provider == "OpenAI (API)":
                        search_result = (
                            self.vector_store_manager.search_documents_for_openai(
                                user_input,
                                api_key,
                                k=search_k,
                                temperature=temperature,
                                new_vector_store=new_vector_store,
                            )
                        )
                    elif current_model_provider == "Local LLM (Qwen)":
                        # Use the same OpenAI search method but with Ollama configuration
                        search_result = (
                            self.vector_store_manager.search_documents_for_openai(
                                user_input,
                                api_key,
                                k=search_k,
                                temperature=temperature,
                                new_vector_store=new_vector_store,
                            )
                        )
                    else:
                        search_result = {"response": "Unsupported model provider"}

                    if not search_result or "Error" in search_result.get(
                        "response", ""
                    ):
                        response_text = search_result.get(
                            "response",
                            "I couldn't find relevant information in the uploaded documents to answer your question.",
                        )
                        response_time = 0.0
                        assistant_message = ChatMessage(
                            id=replace_assistant_id,
                            role=MessageRole.ASSISTANT,
                            content=response_text,
                            model_provider=ModelProvider(
                                st.session_state.current_model_provider
                            ),
                            response_time=response_time,
                        )
                    else:
                        # Create assistant message with the response and source information
                        sources = search_result.get("sources", [])
                        metadata = search_result.get("metadata", {})

                        assistant_message = ChatMessage(
                            id=replace_assistant_id,
                            role=MessageRole.ASSISTANT,
                            content=search_result.get(
                                "response", "No response generated"
                            ),
                            model_provider=ModelProvider(current_model_provider),
                            response_time=0.0,
                            tokens_used=0,
                            source_documents=[
                                s.get("source", "Unknown") for s in sources
                            ],
                            metadata={
                                "source_count": metadata.get("source_count", 0),
                                "model": metadata.get("model", "Unknown"),
                                "sources": sources,
                                "retrieval_method": metadata.get(
                                    "retrieval_method", "Unknown"
                                ),
                            },
                        )

                        # Store response time for metrics
                        st.session_state.last_response_time = 0.0

                    # Update the existing message in conversation
                    messages = st.session_state.current_conversation.messages
                    for i, msg in enumerate(messages):
                        if msg.id == replace_assistant_id:
                            messages[i] = assistant_message
                            break

                    # Rerun to refresh the display
                    st.rerun()

        except Exception as e:
            st.error(f"❌ Failed to process query: {str(e)}")
            self.logger.error(
                f"Query processing failed: {str(e)}\n{traceback.format_exc()}"
            )

    def _display_assistant_response(
        self, assistant_message: ChatMessage, user_query: str
    ):
        """Display assistant response with source information and action buttons"""
        # Display response in main area (no columns)
        # Format content: replace literal \n with actual line breaks (blank lines)
        formatted_content = assistant_message.content.replace('\\n', '\n\n') if isinstance(assistant_message.content, str) else str(assistant_message.content)
        st.markdown(formatted_content)
        
        # Create layout below response: metadata+context on left, PDF preview on right
        if assistant_message.metadata and assistant_message.metadata.get("sources"):
            sources = assistant_message.metadata["sources"]
            
            # Create two columns: left for metadata+context, right for PDF preview
            col_metadata, col_pdf = st.columns([1.2, 1])
            
            with col_metadata:
                # Metadata expander
                with st.expander("📊 Metadata", expanded=False):
                    # Create JSON array in the format shown in image
                    source_json_array = []
                    for source in sources:
                        source_file = source.get("source", "Unknown")
                        page = source.get("page", "Unknown")
                        
                        # Convert page to list if it's a single value
                        if isinstance(page, (int, str)):
                            page_numbers = [str(page)]
                        elif isinstance(page, list):
                            page_numbers = [str(p) for p in page]
                        else:
                            page_numbers = ["Unknown"]
                        
                        source_json = {
                            "file_name": source_file,
                            "page_numbers": page_numbers,
                            "source_type": "internal"
                        }
                        source_json_array.append(source_json)
                    
                    # Display JSON with word wrap
                    source_json_str = json.dumps(source_json_array, indent=2, ensure_ascii=False)
                    # Use st.text_area with word wrap instead of st.code for better wrapping
                    st.text_area(
                        "Metadata JSON",
                        source_json_str,
                        height=200,
                        key=f"metadata_json_new_{assistant_message.id}_{uuid.uuid4()}",
                        disabled=True,
                        label_visibility="collapsed"
                    )
                
                # Context expander
                with st.expander("📝 Context", expanded=False):
                    # Create context JSON array with content from sources
                    context_json_array = []
                    for i, source in enumerate(sources, 1):
                        source_file = source.get("source", "Unknown")
                        page = source.get("page", "Unknown")
                        content = source.get("content", "")
                        
                        # Format content: replace literal \n with actual line breaks
                        if isinstance(content, str):
                            # Replace literal \n (escaped newline) with actual newlines
                            formatted_content = content.replace('\\n', '\n')
                        else:
                            formatted_content = str(content)
                        
                        context_json = {
                            "source_index": i,
                            "file_name": source_file,
                            "page": str(page),
                            "content": formatted_content
                        }
                        context_json_array.append(context_json)
                    
                    # Display context JSON with word wrap
                    context_json_str = json.dumps(context_json_array, indent=2, ensure_ascii=False)
                    # Replace escaped newlines (\\n) in JSON string with actual newlines (blank lines) for better readability
                    # This makes the JSON more readable in the text area - each \n becomes a blank line
                    context_json_str = context_json_str.replace('\\n', '\n\n')
                    # Use st.text_area with word wrap instead of st.code for better wrapping
                    st.text_area(
                        "Context JSON",
                        context_json_str,
                        height=400,
                        key=f"context_json_new_{assistant_message.id}_{uuid.uuid4()}",
                        disabled=True,
                        label_visibility="collapsed"
                    )
                
                with col_pdf:
                    st.markdown("### 📄 See PDF Documents")
                    
                    # Get unique files from sources
                    unique_files = {}
                    for source in sources:
                        source_file = source.get("source", "Unknown")
                        page = source.get("page", "Unknown")
                        
                        # Convert page to int if possible
                        try:
                            if isinstance(page, str) and page.isdigit():
                                page_num = int(page)
                            elif isinstance(page, int):
                                page_num = page
                            else:
                                page_num = 1
                        except:
                            page_num = 1
                        
                        if source_file not in unique_files:
                            unique_files[source_file] = []
                        if page_num not in unique_files[source_file]:
                            unique_files[source_file].append(page_num)
                    
                    # Create dropdown for file selection
                    if unique_files:
                        file_options = []
                        for file_name, pages in unique_files.items():
                            for page in sorted(pages):
                                file_options.append(f"{file_name} Page: {page}")
                        
                        # Use Streamlit's built-in key mechanism to preserve selection
                        selectbox_key = f"pdf_select_{assistant_message.id}"
                        
                        # Ensure current selection is valid, if not set to first option
                        if selectbox_key not in st.session_state or st.session_state[selectbox_key] not in file_options:
                            if file_options:
                                st.session_state[selectbox_key] = file_options[0]
                        
                        # Use selectbox with key - Streamlit handles state automatically
                        # When selection changes, it triggers rerun automatically
                        # Get the selected value directly from the widget (always returns current value)
                        selected_file_page = st.selectbox(
                            "Choose PDF File",
                            file_options,
                            key=selectbox_key
                        )
                        
                        # Extract filename and page from selection
                        if selected_file_page:
                            try:
                                # Parse "filename.pdf Page: X"
                                parts = selected_file_page.rsplit(" Page: ", 1)
                                if len(parts) == 2:
                                    selected_file = parts[0]
                                    selected_page = int(parts[1])
                                    
                                    # Get PDF content from session state
                                    pdf_contents = st.session_state.get("uploaded_pdf_contents", {})
                                    
                                    # Try to find the file (check exact match and variations)
                                    pdf_content = None
                                    from pathlib import Path
                                    selected_file_path = Path(selected_file)
                                    
                                    # Try multiple matching strategies
                                    # 1. Exact match
                                    if selected_file in pdf_contents:
                                        pdf_content = pdf_contents[selected_file]
                                    
                                    # 2. Try with .pdf extension (for DOCX files converted to PDF)
                                    if not pdf_content:
                                        pdf_variant = selected_file_path.with_suffix('.pdf')
                                        if str(pdf_variant) in pdf_contents:
                                            pdf_content = pdf_contents[str(pdf_variant)]
                                    
                                    # 3. Try with .docx extension (for files that might be stored with original extension)
                                    if not pdf_content:
                                        docx_variant = selected_file_path.with_suffix('.docx')
                                        if str(docx_variant) in pdf_contents:
                                            pdf_content = pdf_contents[str(docx_variant)]
                                    
                                    # 4. Try case-insensitive matching
                                    if not pdf_content:
                                        selected_lower = selected_file.lower()
                                        for stored_name, content in pdf_contents.items():
                                            if stored_name.lower() == selected_lower:
                                                pdf_content = content
                                                break
                                    
                                    # 5. Try stem-based matching (filename without extension)
                                    if not pdf_content:
                                        selected_stem = selected_file_path.stem.lower()
                                        for stored_name, content in pdf_contents.items():
                                            stored_stem = Path(stored_name).stem.lower()
                                            if stored_stem == selected_stem:
                                                pdf_content = content
                                                break
                                    
                                    # 6. Try partial matching (as last resort)
                                    if not pdf_content:
                                        selected_stem = selected_file_path.stem.lower()
                                        for stored_name, content in pdf_contents.items():
                                            stored_stem = Path(stored_name).stem.lower()
                                            if selected_stem in stored_stem or stored_stem in selected_stem:
                                                pdf_content = content
                                                break
                                    
                                    if pdf_content:
                                        # Create a unique key for PDF display based on selection
                                        # Use hash of selection to ensure uniqueness and force cache-busting
                                        import hashlib
                                        selection_hash = hashlib.md5(f"{selected_file}_{selected_page}".encode()).hexdigest()[:8]
                                        pdf_display_key = f"pdf_display_{assistant_message.id}_{selected_file}_{selected_page}_{selection_hash}"
                                        
                                        # Track last selection to detect changes
                                        last_selection_key = f"last_pdf_selection_{assistant_message.id}"
                                        current_selection = f"{selected_file}_{selected_page}"
                                        
                                        # Find ALL matching sources for this file and page (ID-based)
                                        # There can be multiple chunks on the same page that need to be highlighted
                                        matching_sources = []
                                        for source in sources:
                                            source_file = source.get("source", "Unknown")
                                            source_page = source.get("page", "Unknown")
                                            try:
                                                source_page_num = int(source_page) if isinstance(source_page, (int, str)) and str(source_page).isdigit() else None
                                                if source_file == selected_file and source_page_num == selected_page:
                                                    matching_sources.append(source)
                                            except:
                                                pass
                                        
                                        # Use container to force re-render when selection changes
                                        # Container key includes selection to force update
                                        container_key = f"pdf_container_{assistant_message.id}_{selected_file}_{selected_page}"
                                        with st.container():
                                            # Display PDF page with highlights
                                            # Unique display_key ensures immediate update when page changes
                                            self._display_pdf_page(pdf_content, selected_page, matching_sources, display_key=pdf_display_key)
                                        
                                        # Update last selection for change detection
                                        st.session_state[last_selection_key] = current_selection
                                    else:
                                        # Debug: show available keys if PDF not found
                                        available_keys = list(pdf_contents.keys())
                                        if available_keys:
                                            st.warning(
                                                f"⚠️ PDF content not found for: **{selected_file}**\n\n"
                                                f"Available files in cache: {', '.join(available_keys[:5])}"
                                                + (f" (and {len(available_keys) - 5} more)" if len(available_keys) > 5 else "")
                                            )
                                        else:
                                            st.info(f"PDF content not found for: {selected_file}. No PDF files available in cache.")
                            except Exception as e:
                                st.error(f"Error displaying PDF: {str(e)}")
                        else:
                            st.info("No PDF files available for preview")
        else:
            st.info("No source information available")

        # Action buttons below the response (only show if not in edit mode)
        if "edit_message_id" not in st.session_state:
            st.markdown("---")
            col1, col2, col3 = st.columns([1, 1, 2])

            with col1:
                if st.button(
                    "🔄 Retry",
                    key=f"retry_new_{assistant_message.id}",
                    help="Regenerate response with current parameters",
                ):
                    # Store the current query and assistant ID for retry
                    st.session_state.retry_query = user_query
                    st.session_state.retry_assistant_id = assistant_message.id
                    st.rerun()

            with col2:
                if st.button(
                    "📋 Copy",
                    key=f"copy_new_{assistant_message.id}",
                    help="Copy response to clipboard",
                ):
                    # Use JavaScript to copy to clipboard
                    copy_text = assistant_message.content.replace('"', '\\"').replace(
                        "\n", "\\n"
                    )
                    st.markdown(
                        f"""
                    <script>
                    navigator.clipboard.writeText("{copy_text}").then(function() {{
                        console.log('Text copied to clipboard');
                    }});
                    </script>
                    """,
                        unsafe_allow_html=True,
                    )
                    st.success("✅ Response copied to clipboard!", icon="📋")

    def _display_pdf_page(self, pdf_content: bytes, page_number: int, highlight_sources: Optional[List[Dict]] = None, display_key: Optional[str] = None):
        """
        Display a specific page from a PDF document with optional text highlighting
        
        Args:
            pdf_content: PDF file content as bytes
            page_number: Page number to display (1-based)
            highlight_sources: List of source dictionaries containing chunk information for highlighting.
                             Each source can contain:
                             - content: Text content to highlight
                             - start_char_in_page: Start character position in page text
                             - end_char_in_page: End character position in page text
                             - chunk_id: Chunk ID for reference
                             Can also be a single Dict for backward compatibility
        """
        try:
            # Try to use PyMuPDF for highlighting if available
            try:
                import fitz  # PyMuPDF
                
                # Open PDF document
                pdf_document = fitz.open(stream=pdf_content, filetype="pdf")
                
                # Check if page number is valid (1-based to 0-based conversion)
                if page_number < 1 or page_number > len(pdf_document):
                    st.warning(f"Page {page_number} not found. PDF has {len(pdf_document)} pages.")
                    pdf_document.close()
                    return
                
                # Get the specific page (convert to 0-based index)
                page = pdf_document[page_number - 1]
                
                # Add highlights if highlight_sources is provided
                if highlight_sources:
                    # Normalize to list if single dict provided (backward compatibility)
                    if isinstance(highlight_sources, dict):
                        highlight_sources = [highlight_sources]
                    
                    # Extract page text once using PyMuPDF (same as chunk extraction)
                    page_text = page.get_text()
                    
                    # Collect all text instances to highlight from all chunks
                    all_text_instances = []
                    
                    # Process each source/chunk (ID-based highlighting)
                    for highlight_source in highlight_sources:
                        if not highlight_source:
                            continue
                        
                        chunk_id = highlight_source.get("chunk_id", "")
                        self.logger.debug(f"Processing chunk ID: {chunk_id} for highlighting")
                        
                        # Try to use position-based highlighting if available
                        start_char_in_page = highlight_source.get("start_char_in_page")
                        end_char_in_page = highlight_source.get("end_char_in_page")
                        
                        text_instances = []
                        
                        # Strategy: Use position-based extraction (PyMuPDF consistent)
                        if start_char_in_page is not None and end_char_in_page is not None:
                            try:
                                # Use direct indexing since we're using PyMuPDF for both extraction and highlighting
                                if len(page_text) >= end_char_in_page:
                                    # Extract the exact text from the page using position
                                    exact_text = page_text[start_char_in_page:end_char_in_page]
                                    if exact_text.strip():
                                        # Search for this exact text in the PDF
                                        instances = page.search_for(exact_text.strip())
                                        text_instances.extend(instances)
                                        self.logger.debug(f"Chunk {chunk_id}: Position-based highlight pos {start_char_in_page}-{end_char_in_page}, {len(exact_text)} chars, found {len(instances)} instances")
                                else:
                                    # Page text length mismatch - try to find content in page text
                                    content_text = highlight_source.get("content", "").strip()
                                    if content_text:
                                        # Find the content in the page text
                                        content_pos = page_text.find(content_text[:100])  # Try first 100 chars
                                        if content_pos >= 0:
                                            # Found it, use the found position
                                            end_pos = min(content_pos + len(content_text), len(page_text))
                                            exact_text = page_text[content_pos:end_pos]
                                            if exact_text.strip():
                                                instances = page.search_for(exact_text.strip())
                                                text_instances.extend(instances)
                                                self.logger.debug(f"Chunk {chunk_id}: Content-based position found at pos {content_pos}, {len(instances)} instances")
                            except Exception as pos_error:
                                self.logger.debug(f"Chunk {chunk_id}: Position-based highlight failed: {str(pos_error)}, falling back to content")
                        
                        # Fallback: Use content-based highlighting
                        if not text_instances:
                            highlight_text = highlight_source.get("content", "")
                            if highlight_text:
                                try:
                                    import re
                                    # Clean and normalize the highlight text
                                    highlight_text_clean = highlight_text.replace('\n', ' ').replace('\r', ' ')
                                    highlight_text_clean = re.sub(r'\s+', ' ', highlight_text_clean).strip()
                                    
                                    # Strategy 1: Try to find the entire text first
                                    instances = page.search_for(highlight_text_clean)
                                    text_instances.extend(instances)
                                    
                                    # Strategy 2: If not found, split into sentences
                                    if not text_instances:
                                        sentences = re.split(r'[.!?]\s+', highlight_text_clean)
                                        sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
                                        for sentence in sentences:
                                            if len(sentence.strip()) > 5:
                                                instances = page.search_for(sentence.strip())
                                                text_instances.extend(instances)
                                    
                                    # Strategy 3: Try smaller chunks if needed
                                    if not text_instances:
                                        words = highlight_text_clean.split()
                                        chunk_size = 15
                                        for i in range(0, len(words), chunk_size):
                                            chunk = ' '.join(words[i:i+chunk_size])
                                            if len(chunk.strip()) > 5:
                                                instances = page.search_for(chunk.strip())
                                                text_instances.extend(instances)
                                    
                                except Exception as highlight_error:
                                    self.logger.warning(f"Chunk {chunk_id}: Could not highlight text: {str(highlight_error)}")
                        
                        # Add found instances to the collection (avoid duplicates)
                        for inst in text_instances:
                            # Check if this instance overlaps significantly with existing ones
                            is_duplicate = False
                            for existing in all_text_instances:
                                # Check if rectangles overlap significantly (within 10 pixels)
                                if (abs(inst.x0 - existing.x0) < 10 and 
                                    abs(inst.y0 - existing.y0) < 10 and
                                    abs(inst.x1 - existing.x1) < 10 and
                                    abs(inst.y1 - existing.y1) < 10):
                                    is_duplicate = True
                                    break
                            if not is_duplicate:
                                all_text_instances.append(inst)
                    
                    # Highlight all found instances with yellow color
                    for inst in all_text_instances:
                        try:
                            # Add highlight annotation (yellow color)
                            highlight = page.add_highlight_annot(inst)
                            highlight.set_colors(stroke=[1, 1, 0])  # Yellow RGB
                            highlight.update()
                        except Exception as annot_error:
                            self.logger.debug(f"Could not add highlight annotation: {str(annot_error)}")
                    
                    if all_text_instances:
                        self.logger.debug(f"Highlighted {len(all_text_instances)} text instances from {len(highlight_sources)} chunks on page {page_number}")
                    else:
                        self.logger.debug(f"No text instances found to highlight on page {page_number} from {len(highlight_sources)} chunks")
                
                # Convert highlighted page to image
                try:
                    from pdf2image import convert_from_bytes
                    from PIL import Image
                    
                    # Create a new PDF with just this highlighted page before closing
                    new_pdf = fitz.open()
                    new_pdf.insert_pdf(pdf_document, from_page=page_number - 1, to_page=page_number - 1)
                    
                    # Get PDF bytes from the new document
                    page_pdf_bytes = new_pdf.tobytes()
                    
                    # Close both documents
                    new_pdf.close()
                    pdf_document.close()
                    
                    # Convert to image
                    images = convert_from_bytes(page_pdf_bytes, first_page=1, last_page=1, dpi=150)
                    
                    if images:
                        # Display the highlighted image with cache-busting key
                        cache_buster = display_key if display_key else f"page_{page_number}_{hash(str(page_number))}"
                        st.image(images[0], use_container_width=True, key=f"pdf_image_{cache_buster}")
                    else:
                        raise Exception("Image conversion failed")
                
                except ImportError:
                    # Fallback: Save highlighted page as PDF and display in iframe
                    # Create a new PDF with just this page
                    new_pdf = fitz.open()
                    new_pdf.insert_pdf(pdf_document, from_page=page_number - 1, to_page=page_number - 1)
                    
                    # Get PDF bytes before closing
                    page_pdf_bytes = new_pdf.tobytes()
                    
                    # Close documents
                    new_pdf.close()
                    pdf_document.close()
                    
                    # Encode to base64
                    pdf_base64 = base64.b64encode(page_pdf_bytes).decode('utf-8')
                    
                    # Display in iframe with cache-busting
                    # Use display_key for cache-busting to ensure immediate update on selection change
                    cache_buster = display_key if display_key else f"page_{page_number}_{hash(pdf_base64[:100])}"
                    # Use fragment with cache-buster and timestamp to force browser to reload
                    import time
                    timestamp = int(time.time() * 1000)
                    pdf_display = f'''
                    <iframe id="pdf_iframe_{cache_buster}_{timestamp}" 
                            src="data:application/pdf;base64,{pdf_base64}#page={page_number}&v={cache_buster}&t={timestamp}" 
                            width="100%" 
                            height="600px" 
                            style="border: 1px solid #ccc;">
                    </iframe>
                    <script>
                        // Force iframe reload by removing and re-adding
                        var iframe = document.getElementById('pdf_iframe_{cache_buster}_{timestamp}');
                        if (iframe) {{
                            iframe.src = iframe.src;
                        }}
                    </script>
                    '''
                    st.markdown(pdf_display, unsafe_allow_html=True)
                
                except Exception as img_error:
                    # Fallback to base64 if image conversion fails
                    # Check if document is still open
                    if not pdf_document.is_closed:
                        try:
                            new_pdf = fitz.open()
                            new_pdf.insert_pdf(pdf_document, from_page=page_number - 1, to_page=page_number - 1)
                            page_pdf_bytes = new_pdf.tobytes()
                            new_pdf.close()
                            pdf_document.close()
                            
                            pdf_base64 = base64.b64encode(page_pdf_bytes).decode('utf-8')
                            cache_buster = display_key if display_key else f"page_{page_number}_{hash(pdf_base64[:100])}"
                            # Use fragment with cache-buster and timestamp to force browser to reload
                            import time
                            timestamp = int(time.time() * 1000)
                            pdf_display = f'''
                            <iframe id="pdf_iframe_{cache_buster}_{timestamp}" 
                                    src="data:application/pdf;base64,{pdf_base64}#page={page_number}&v={cache_buster}&t={timestamp}" 
                                    width="100%" 
                                    height="600px" 
                                    style="border: 1px solid #ccc;">
                            </iframe>
                            <script>
                                // Force iframe reload by removing and re-adding
                                var iframe = document.getElementById('pdf_iframe_{cache_buster}_{timestamp}');
                                if (iframe) {{
                                    iframe.src = iframe.src;
                                }}
                            </script>
                            '''
                            st.markdown(pdf_display, unsafe_allow_html=True)
                        except Exception as fallback_error:
                            if not pdf_document.is_closed:
                                pdf_document.close()
                            st.error(f"Error displaying PDF: {str(fallback_error)}")
                            self.logger.error(f"PDF display error: {str(fallback_error)}\n{traceback.format_exc()}")
                    else:
                        # Document already closed, try to recreate from original content
                        try:
                            pdf_document_reopen = fitz.open(stream=pdf_content, filetype="pdf")
                            page_reopen = pdf_document_reopen[page_number - 1]
                            
                            # Try to highlight again if needed (use comprehensive strategy)
                            # Note: highlight_text is no longer available in this scope, use highlight_sources instead
                            if highlight_sources:
                                # Re-process highlights using the same logic as above
                                # Normalize to list if single dict provided
                                if isinstance(highlight_sources, dict):
                                    highlight_sources = [highlight_sources]
                                
                                page_text_reopen = page_reopen.get_text()
                                all_text_instances_reopen = []
                                
                                for highlight_source in highlight_sources:
                                    if not highlight_source:
                                        continue
                                    
                                    chunk_id = highlight_source.get("chunk_id", "")
                                    start_char_in_page = highlight_source.get("start_char_in_page")
                                    end_char_in_page = highlight_source.get("end_char_in_page")
                                    text_instances_reopen = []
                                    
                                    # Use position-based highlighting
                                    if start_char_in_page is not None and end_char_in_page is not None and len(page_text_reopen) >= end_char_in_page:
                                        try:
                                            exact_text = page_text_reopen[start_char_in_page:end_char_in_page]
                                            if exact_text.strip():
                                                instances = page_reopen.search_for(exact_text.strip())
                                                text_instances_reopen.extend(instances)
                                        except:
                                            pass
                                    
                                    # Fallback to content-based
                                    if not text_instances_reopen:
                                        highlight_text = highlight_source.get("content", "")
                                        if highlight_text:
                                            try:
                                                import re
                                                highlight_text_clean = highlight_text.replace('\n', ' ').replace('\r', ' ')
                                                highlight_text_clean = re.sub(r'\s+', ' ', highlight_text_clean).strip()
                                                instances = page_reopen.search_for(highlight_text_clean)
                                                text_instances_reopen.extend(instances)
                                            except:
                                                pass
                                    
                                    # Add to collection
                                    for inst in text_instances_reopen:
                                        is_duplicate = False
                                        for existing in all_text_instances_reopen:
                                            if (abs(inst.x0 - existing.x0) < 10 and 
                                                abs(inst.y0 - existing.y0) < 10 and
                                                abs(inst.x1 - existing.x1) < 10 and
                                                abs(inst.y1 - existing.y1) < 10):
                                                is_duplicate = True
                                                break
                                        if not is_duplicate:
                                            all_text_instances_reopen.append(inst)
                                
                                # Apply highlights
                                for inst in all_text_instances_reopen:
                                    try:
                                        highlight = page_reopen.add_highlight_annot(inst)
                                        highlight.set_colors(stroke=[1, 1, 0])
                                        highlight.update()
                                    except:
                                        pass
                            
                            # Create PDF and display
                            new_pdf = fitz.open()
                            new_pdf.insert_pdf(pdf_document_reopen, from_page=page_number - 1, to_page=page_number - 1)
                            page_pdf_bytes = new_pdf.tobytes()
                            new_pdf.close()
                            pdf_document_reopen.close()
                            
                            pdf_base64 = base64.b64encode(page_pdf_bytes).decode('utf-8')
                            cache_buster = display_key if display_key else f"page_{page_number}_{hash(pdf_base64[:100])}"
                            # Use fragment with cache-buster and timestamp to force browser to reload
                            import time
                            timestamp = int(time.time() * 1000)
                            pdf_display = f'''
                            <iframe id="pdf_iframe_{cache_buster}_{timestamp}" 
                                    src="data:application/pdf;base64,{pdf_base64}#page={page_number}&v={cache_buster}&t={timestamp}" 
                                    width="100%" 
                                    height="600px" 
                                    style="border: 1px solid #ccc;">
                            </iframe>
                            <script>
                                // Force iframe reload by removing and re-adding
                                var iframe = document.getElementById('pdf_iframe_{cache_buster}_{timestamp}');
                                if (iframe) {{
                                    iframe.src = iframe.src;
                                }}
                            </script>
                            '''
                            st.markdown(pdf_display, unsafe_allow_html=True)
                        except Exception as reopen_error:
                            st.error(f"Error displaying PDF: {str(reopen_error)}")
                            self.logger.error(f"PDF display error: {str(reopen_error)}\n{traceback.format_exc()}")
            
            except ImportError:
                # PyMuPDF not available, use PyPDF2 fallback (no highlighting)
                pdf_file = io.BytesIO(pdf_content)
                pdf_reader = PyPDF2.PdfReader(pdf_file)
                
                if page_number < 1 or page_number > len(pdf_reader.pages):
                    st.warning(f"Page {page_number} not found. PDF has {len(pdf_reader.pages)} pages.")
                    return
                
                page = pdf_reader.pages[page_number - 1]
                
                # Try pdf2image
                try:
                    from pdf2image import convert_from_bytes
                    images = convert_from_bytes(pdf_content, first_page=page_number, last_page=page_number, dpi=150)
                    if images:
                        cache_buster = display_key if display_key else f"page_{page_number}_{hash(str(page_number))}"
                        st.image(images[0], use_container_width=True, key=f"pdf_image_pypdf2_{cache_buster}")
                    else:
                        raise Exception("Image conversion failed")
                except (ImportError, Exception):
                    # Fallback to base64
                    pdf_writer = PyPDF2.PdfWriter()
                    pdf_writer.add_page(page)
                    output_pdf = io.BytesIO()
                    pdf_writer.write(output_pdf)
                    output_pdf.seek(0)
                    pdf_base64 = base64.b64encode(output_pdf.read()).decode('utf-8')
                    cache_buster = display_key if display_key else f"page_{page_number}_{hash(pdf_base64[:100])}"
                    pdf_display = f'''
                    <iframe src="data:application/pdf;base64,{pdf_base64}#{cache_buster}" 
                            width="100%" 
                            height="600px" 
                            style="border: 1px solid #ccc;">
                    </iframe>
                    '''
                    st.markdown(pdf_display, unsafe_allow_html=True, key=f"pdf_iframe_pypdf2_{cache_buster}")
            
        except Exception as e:
            st.error(f"Error displaying PDF page: {str(e)}")
            self.logger.error(f"PDF display error: {str(e)}\n{traceback.format_exc()}")

    def _clear_current_conversation(self):
        """Clear current conversation"""
        st.session_state.current_conversation = ConversationSession(
            id=str(uuid.uuid4()), title="New Conversation"
        )
        self.llm_service.clear_conversation_memory()
        st.success("✅ Conversation cleared!")
        self.logger.info("Conversation cleared")

    def _save_conversation(self):
        """Save current conversation"""
        # This is a placeholder - in a real app, you'd save to a database
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"conversation_{timestamp}.json"
        st.success(f"✅ Conversation saved as {filename}")
        self.logger.info(f"Conversation saved: {filename}")

    def _export_logs(self):
        """Export application logs"""
        import os
        from pathlib import Path

        log_file = Path(config.logging.log_file)

        if not log_file.exists():
            st.warning("⚠️ No log file found")
            return

        try:
            with open(log_file, "r", encoding="utf-8") as f:
                log_content = f.read()

            # Create download link
            b64 = base64.b64encode(log_content.encode("utf-8")).decode()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"app_logs_{timestamp}.txt"

            href = f'<a href="data:file/txt;base64,{b64}" download="{filename}">📤 Download Logs</a>'
            st.markdown(href, unsafe_allow_html=True)
            st.success("✅ Log export ready!")

        except Exception as e:
            st.error(f"❌ Error exporting logs: {str(e)}")

    def _clear_logs(self):
        """Clear application logs"""
        import os
        from pathlib import Path

        log_file = Path(config.logging.log_file)

        if not log_file.exists():
            st.warning("⚠️ No log file found")
            return

        try:
            # Clear log file content
            with open(log_file, "w", encoding="utf-8") as f:
                f.write("")
            st.success("✅ Logs cleared successfully!")

        except Exception as e:
            st.error(f"❌ Error clearing logs: {str(e)}")

    def _delete_chunks(self):
        """Delete only new_db chunks and clear related system info"""
        try:
            # Clear new_db vector stores
            current_model_provider = st.session_state.get(
                "current_model_provider", None
            )
            if current_model_provider:
                # Clear for the current provider
                self.new_db_manager.clear_new_vector_store(current_model_provider)
            else:
                # Clear all new vector stores if no provider is set
                self.new_db_manager.clear_new_vector_store()

            # Clear new_vector_store from session state
            if "new_vector_store" in st.session_state:
                st.session_state.new_vector_store = None

            # Clear only new_db documents from session state
            # Keep documents that were not in new_db (if any exist)
            # For simplicity, we'll clear all documents since new_db documents are temporary
            # If you want to preserve current_db documents, you'd need to track which documents belong to which DB
            st.session_state.documents = []

            # Reset vector_store_initialized flag only if we're in new_db mode
            db_mode = st.session_state.get("db_mode", "current")
            if db_mode in ["new", "current+new"]:
                # Only reset if there are no more new_db chunks
                if not st.session_state.get("new_vector_store"):
                    # If in "new" mode and we cleared all, reset the flag
                    if db_mode == "new":
                        st.session_state.vector_store_initialized = False

            st.success(
                "✅ New DB chunks deleted successfully! Current DB remains intact."
            )

        except Exception as e:
            st.error(f"❌ Error deleting chunks: {str(e)}")

    def _clear_cache(self):
        """Clear question-answer cache"""
        try:
            # Get cache stats before clearing
            cache_stats = self.vector_store_manager.get_cache_stats()
            entries_count = cache_stats.get("total_entries", 0)

            # Clear cache
            self.vector_store_manager.clear_cache()

            st.success(
                f"✅ Cache cleared successfully! Removed {entries_count} cached entries."
            )

        except Exception as e:
            st.error(f"❌ Error clearing cache: {str(e)}")

    def _export_conversation_csv(self):
        """Export conversation as CSV"""
        if not st.session_state.current_conversation.messages:
            st.warning("⚠️ No messages to export")
            return

        # Create DataFrame
        data = []
        for msg in st.session_state.current_conversation.messages:
            data.append(
                {
                    "Timestamp": msg.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                    "Role": msg.role.value,
                    "Content": msg.content,
                    "Model": msg.model_provider.value if msg.model_provider else "",
                    "Response Time": msg.response_time if msg.response_time else "",
                    "Tokens Used": msg.tokens_used if msg.tokens_used else "",
                }
            )

        df = pd.DataFrame(data)
        csv = df.to_csv(index=False)

        # Create download link
        b64 = base64.b64encode(csv.encode()).decode()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"conversation_{timestamp}.csv"

        href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">📥 Download CSV</a>'
        st.markdown(href, unsafe_allow_html=True)

        st.success("✅ CSV export ready!")


def main():
    """Main application entry point"""
    app = ChatbotApp()
    app.run()


if __name__ == "__main__":
    main()
