"""
Enhanced RAG QA Chatbot Application with Streamlit
"""
import streamlit as st
import uuid
import pandas as pd
import base64
from datetime import datetime
from typing import List, Optional
import traceback

# Import our modules
from src.config import config, SUPPORTED_MODELS
from src.models import (
    ModelProvider, ConversationSession, ChatMessage,
    MessageRole, Document, UserSession
)
from src.services import DocumentProcessor, VectorStoreManager, LLMService
from src.utils import app_logger, format_file_size, truncate_text


class ChatbotApp:
    """Main Chatbot Application Class"""

    def __init__(self):
        self.logger = app_logger
        self.document_processor = DocumentProcessor()
        self.vector_store_manager = VectorStoreManager()
        self.llm_service = LLMService()

        # Initialize session state
        self._initialize_session_state()

    def _initialize_session_state(self):
        """Initialize Streamlit session state"""
        if 'user_session' not in st.session_state:
            st.session_state.user_session = UserSession(
                id=str(uuid.uuid4()),
                preferences={'theme': 'light', 'language': 'en'}
            )

        if 'current_conversation' not in st.session_state:
            st.session_state.current_conversation = ConversationSession(
                id=str(uuid.uuid4()),
                title="New Conversation"
            )

        if 'documents' not in st.session_state:
            st.session_state.documents = []

        # Default parameters
        if 'chunk_size' not in st.session_state:
            st.session_state.chunk_size = 800

        if 'chunk_overlap' not in st.session_state:
            st.session_state.chunk_overlap = 100

        if 'search_k' not in st.session_state:
            st.session_state.search_k = 5

        if 'similarity_threshold' not in st.session_state:
            st.session_state.similarity_threshold = 0.75

        if 'model_temperature' not in st.session_state:
            st.session_state.model_temperature = 0.7

        if 'vector_store_initialized' not in st.session_state:
            st.session_state.vector_store_initialized = False

        if 'llm_initialized' not in st.session_state:
            st.session_state.llm_initialized = False

        if 'processing_status' not in st.session_state:
            st.session_state.processing_status = None

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
            initial_sidebar_state="expanded"
        )

        # Custom CSS
        st.markdown("""
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
        </style>
        """, unsafe_allow_html=True)

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
            messages_count = len(
                st.session_state.current_conversation.messages)
            st.metric("💬 Messages", messages_count)

        with col3:
            if st.session_state.vector_store_initialized:
                st.markdown(
                    '<p class="status-success">✅ Vector Store Ready</p>', unsafe_allow_html=True)
            else:
                st.markdown(
                    '<p class="status-warning">⏳ Vector Store Not Ready</p>', unsafe_allow_html=True)

        with col4:
            if st.session_state.llm_initialized:
                st.markdown(
                    '<p class="status-success">✅ LLM Ready</p>', unsafe_allow_html=True)
            else:
                st.markdown(
                    '<p class="status-warning">⏳ LLM Not Ready</p>', unsafe_allow_html=True)

    def _render_sidebar(self):
        """Render sidebar configuration"""
        with st.sidebar:
            st.title("⚙️ Configuration")

            # Model selection
            model_provider = st.selectbox(
                "Select Model Provider:",
                SUPPORTED_MODELS,
                help="Choose the AI model provider for generating responses"
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
                "OpenAI API Key:",
                type="password",
                help="Enter your OpenAI API key"
            )
            if not api_key:
                st.info(
                    "💡 Get your API key from [OpenAI](https://platform.openai.com/api-keys)")

        elif model_provider == "Local LLM (Qwen)":
            # No API key needed for local Ollama
            api_key = "ollama"  # Placeholder for local LLM
            st.info("🖥️ Using local Qwen2.5:7b model via Ollama")
            st.info("💡 Make sure Ollama is running: `ollama run qwen2.5:7b`")

        else:
            api_key = None
            st.warning(f"API key input not implemented for {model_provider}")

        return api_key

    def _render_file_upload_section(self):
        """Render file upload section"""
        st.subheader("📁 Document Upload")

        uploaded_files = st.file_uploader(
            "Upload Documents",
            type=['pdf', 'docx', 'txt'],
            accept_multiple_files=True,
            help="Upload PDF, DOCX, or TXT files to create or update the knowledge base"
        )

        if uploaded_files:
            st.write(f"📋 {len(uploaded_files)} file(s) selected:")
            for file in uploaded_files:
                file_size = file.size if hasattr(
                    file, 'size') else len(file.getvalue())
                st.write(f"• {file.name} ({format_file_size(file_size)})")

            # Document processing parameters
            with st.expander("⚙️ Processing Parameters"):
                chunk_size = st.slider(
                    "Chunk Size",
                    min_value=200,
                    max_value=2000,
                    value=800,
                    step=100,
                    help="Size of text chunks for processing"
                )
                chunk_overlap = st.slider(
                    "Chunk Overlap",
                    min_value=0,
                    max_value=500,
                    value=100,
                    step=50,
                    help="Overlap between consecutive chunks"
                )
                st.session_state.chunk_size = chunk_size
                st.session_state.chunk_overlap = chunk_overlap

                # Check if parameters changed and warn user
                if st.session_state.vector_store_initialized:
                    previous_chunk_size = getattr(
                        st.session_state, 'previous_chunk_size', 800)
                    previous_chunk_overlap = getattr(
                        st.session_state, 'previous_chunk_overlap', 100)

                    if chunk_size != previous_chunk_size or chunk_overlap != previous_chunk_overlap:
                        st.warning("⚠️ Chunk parameters have changed! To apply new settings, you need to reprocess your documents. " +
                                   "The existing vector store will be updated with new chunks.")

                # Store current parameters for comparison
                st.session_state.previous_chunk_size = chunk_size
                st.session_state.previous_chunk_overlap = chunk_overlap

            if st.button("🚀 Process Documents", type="primary", use_container_width=True):
                self._process_documents(uploaded_files)

        # Retrieval parameters section
        with st.expander("🔍 Retrieval Parameters"):
            search_k = st.slider(
                "Number of Sources (k)",
                min_value=1,
                max_value=10,
                value=5,
                step=1,
                help="Number of document chunks to retrieve for each query"
            )
            similarity_threshold = st.slider(
                "Similarity Threshold (%)",
                min_value=0.0,
                max_value=1.0,
                value=0.75,  # Default to 0.75 (75%) for high-quality results
                step=0.05,
                help="Minimum cosine similarity score for source inclusion. " +
                     "Scores are based on true cosine similarity between embeddings. " +
                     "Recommended: 0.70-0.80 for precision, 0.60-0.70 for balanced results."
            )
            st.session_state.search_k = search_k
            st.session_state.similarity_threshold = similarity_threshold

            st.info("💡 Cosine Similarity Scores (mathematically accurate):\n" +
                    "- 0.90-1.00 = Nearly identical content (excellent)\n" +
                    "- 0.75-0.90 = Highly relevant (recommended threshold)\n" +
                    "- 0.60-0.75 = Moderately relevant (good for broad searches)\n" +
                    "- 0.50-0.60 = Somewhat related (may include tangential content)\n" +
                    "- Below 0.50 = Low relevance (likely unrelated)")

        # Model parameters section
        with st.expander("🤖 Model Parameters"):
            # Get current model provider to show relevant parameters
            current_provider = st.session_state.get(
                'current_model_provider', 'OpenAI (API)')

            temperature = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=1.0,
                value=0.7 if current_provider == "OpenAI (API)" else 0.3,
                step=0.1,
                help="Controls randomness in responses. Lower values (0.1-0.3) are more focused and deterministic, higher values (0.7-1.0) are more creative and varied."
            )

            # Store temperature in session state
            st.session_state.model_temperature = temperature

            st.info(f"💡 **Temperature Guide:**\n" +
                    f"- **0.0-0.3**: Highly focused, deterministic (good for factual Q&A)\n" +
                    f"- **0.4-0.7**: Balanced creativity and focus (recommended)\n" +
                    f"- **0.8-1.0**: Very creative, more varied responses\n\n" +
                    f"**Current Provider**: {current_provider}\n\n" +
                    f"**Note**: Max tokens are automatically set by the model for optimal performance.")

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
                st.button("📥 Export as CSV", use_container_width=True,
                          disabled=True, help="No conversation to export")

    def _render_system_info_section(self):
        """Render system information section"""
        st.subheader("📊 System Info")

        # Vector store info
        vector_info = self.vector_store_manager.get_store_info()
        if vector_info:
            st.write(f"📚 Total Documents: {vector_info.total_documents}")
            # Get actual chunk count from vector store with better error handling
            try:
                if hasattr(self.vector_store_manager, 'vector_store') and self.vector_store_manager.vector_store:
                    # Try multiple methods to get chunk count
                    actual_chunks = 0

                    if hasattr(self.vector_store_manager.vector_store, 'docstore'):
                        if self.vector_store_manager.vector_store.docstore:
                            if hasattr(self.vector_store_manager.vector_store.docstore, '_dict'):
                                actual_chunks = len(
                                    self.vector_store_manager.vector_store.docstore._dict)

                    if actual_chunks == 0 and hasattr(self.vector_store_manager.vector_store, 'index'):
                        if hasattr(self.vector_store_manager.vector_store.index, 'ntotal'):
                            actual_chunks = self.vector_store_manager.vector_store.index.ntotal

                    # Fallback to vector_info if still 0
                    if actual_chunks == 0:
                        actual_chunks = vector_info.total_chunks

                    # If we got a valid chunk count, save it to session state
                    if actual_chunks > 0:
                        # Only update if we got a new value (don't overwrite existing with 0)
                        if 'total_chunks_cached' not in st.session_state or actual_chunks >= st.session_state.total_chunks_cached:
                            st.session_state.total_chunks_cached = actual_chunks

                    st.write(
                        f"🧩 Total Chunks: {st.session_state.get('total_chunks_cached', actual_chunks)}")
                else:
                    # Use cached value if available, otherwise vector_info
                    cached_chunks = st.session_state.get(
                        'total_chunks_cached', vector_info.total_chunks)
                    st.write(f"🧩 Total Chunks: {cached_chunks}")
            except Exception as e:
                self.logger.warning(f"Could not get chunk count: {e}")
                # Use cached value if available, otherwise vector_info
                cached_chunks = st.session_state.get(
                    'total_chunks_cached', vector_info.total_chunks)
                st.write(f"🧩 Total Chunks: {cached_chunks}")
            st.write(
                f"� Index Size: {format_file_size(vector_info.index_size_bytes)}")
        else:
            st.write("📭 No vector store found")

        # Cache statistics
        st.markdown("---")
        st.subheader("💾 Cache Info")
        cache_stats = self.vector_store_manager.get_cache_stats()
        if cache_stats['enabled']:
            st.write(f"✅ Cache: Enabled")
            st.write(
                f"📦 Cached Entries: {cache_stats['total_entries']}/{cache_stats['max_size']}")
            st.write(f"🎯 Cache Hits: {cache_stats['hits']}")
            st.write(f"❌ Cache Misses: {cache_stats['misses']}")
            if cache_stats['total_queries'] > 0:
                st.write(f"📈 Hit Rate: {cache_stats['hit_rate']:.1f}%")
            st.write(f"⏱️ TTL: {cache_stats['ttl_seconds']}s")
        else:
            st.write("⚠️ Cache: Disabled")

        # System management buttons
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

    def _render_social_links(self):
        """Render social media links"""
        st.markdown("---")
        st.subheader("🔗 Connect")

        linkedin_url = "https://www.linkedin.com/in/huseyincenik/"
        kaggle_url = "https://www.kaggle.com/huseyincenik/"
        github_url = "https://github.com/huseyincenik/"

        st.markdown(f"""
        [![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)]({linkedin_url})
        [![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)]({kaggle_url})
        [![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)]({github_url})
        """)

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
                if i + 1 < len(messages) and messages[i + 1].role == MessageRole.ASSISTANT:
                    assistant_message = messages[i + 1]

                # Render user message with edit option
                with st.chat_message("user"):
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        st.write(user_message.content)
                        if user_message.timestamp:
                            st.caption(
                                f"🕒 {user_message.timestamp.strftime('%H:%M:%S')}")

                    with col2:
                        # Edit butonu ve timestamp aynı hizada olsun diye boş alan bırakıyoruz
                        st.write("")  # Timestamp hizası için boşluk
                        if st.button("✏️ Edit", key=f"edit_history_{user_message.id}", help="Edit and regenerate"):
                            # Store edit info in session state
                            st.session_state.edit_message_id = user_message.id
                            st.session_state.edit_query = user_message.content
                            st.session_state.edit_assistant_id = assistant_message.id if assistant_message else None
                            st.rerun()

                # Render assistant message if exists
                if assistant_message:
                    self._render_assistant_message_with_actions(
                        assistant_message, user_message.content)
                    i += 2  # Skip both user and assistant message
                else:
                    i += 1  # Only user message, move to next
            else:
                # Orphaned assistant message (shouldn't happen normally)
                if messages[i].role == MessageRole.ASSISTANT:
                    with st.chat_message("assistant"):
                        st.write(messages[i].content)
                i += 1

    def _render_assistant_message_with_actions(self, assistant_message: ChatMessage, user_query: str):
        """Render assistant message with action buttons"""
        with st.chat_message("assistant"):
            # Display response in main area
            col1, col2 = st.columns([2, 1])

            with col1:
                st.write(assistant_message.content)

            # Show source information in right column
            with col2:
                # Check if LLM indicated insufficient information
                insufficient_info = assistant_message.metadata.get(
                    'insufficient_info', False) if assistant_message.metadata else False
                source_count = assistant_message.metadata.get(
                    'source_count', 0) if assistant_message.metadata else 0

                # Only show sources if we have valid sources and LLM didn't say "no info"
                if assistant_message.metadata and assistant_message.metadata.get('sources') and source_count > 0 and not insufficient_info:
                    sources = assistant_message.metadata['sources']

                    st.markdown("### 📊 Response Information")

                    # Model and source info
                    model_name = assistant_message.metadata.get(
                        'model', 'Unknown')
                    retrieval_method = assistant_message.metadata.get(
                        'retrieval_method', 'Unknown')

                    # Check if response came from cache
                    is_cached = assistant_message.metadata.get('cached', False)
                    cache_similarity = assistant_message.metadata.get(
                        'cache_similarity', None)

                    st.metric("Model", model_name)
                    st.metric("Sources Used", source_count)

                    # Show cache indicator if applicable
                    if is_cached:
                        if cache_similarity:
                            st.success(
                                f"⚡ Cached Response (Similarity: {cache_similarity:.1%})")
                        else:
                            st.success("⚡ Cached Response")

                    st.caption(f"Method: {retrieval_method}")

                    # Expandable source details
                    with st.expander(f"📚 View {source_count} Sources", expanded=False):
                        for i, source in enumerate(sources, 1):
                            st.markdown(f"**Source {i}**")

                            # Source file and page info
                            source_file = source.get('source', 'Unknown')
                            page = source.get('page', 'Unknown')
                            accuracy_score = source.get('accuracy_score', 0.0)

                            st.write(f"📄 **File:** {source_file}")
                            st.write(f"📖 **Page:** {page}")
                            st.write(f"🎯 **Accuracy:** {accuracy_score:.2%}")

                            # Content preview - show full content with larger height
                            content = source.get(
                                'content', 'No content available')
                            st.text_area(
                                f"Content Preview {i}",
                                content,
                                height=300,  # Increased height for better content viewing
                                key=f"source_content_{assistant_message.id}_{i}",
                                disabled=True
                            )

                            # Additional metadata
                            metadata_info = source.get('metadata', {})
                            if metadata_info:
                                with st.expander(f"📋 Metadata {i}"):
                                    for key, value in metadata_info.items():
                                        # Don't repeat already shown info
                                        if key not in ['source', 'page']:
                                            st.write(
                                                f"**{key.title()}:** {value}")

                            st.divider()
                elif insufficient_info:
                    # LLM indicated insufficient information
                    st.markdown("### ℹ️ Insufficient Information")
                    st.info(
                        "The AI model indicated that the provided documents do not contain "
                        "enough information to answer this question reliably.")
                else:
                    # No sources at all
                    st.markdown("### ℹ️ No Sources Available")
                    st.write(
                        "No source information could be extracted for this response.")

            # Action buttons below the response
            st.markdown("---")
            col1, col2, col3 = st.columns([1, 1, 2])

            with col1:
                if st.button("🔄 Retry", key=f"retry_{assistant_message.id}", help="Regenerate response with current parameters"):
                    # Store the current query and assistant ID for retry
                    st.session_state.retry_query = user_query
                    st.session_state.retry_assistant_id = assistant_message.id
                    st.rerun()

            with col2:
                if st.button("📋 Copy", key=f"copy_{assistant_message.id}", help="Copy response to clipboard"):
                    # Use JavaScript to copy to clipboard
                    copy_text = assistant_message.content.replace(
                        '"', '\\"').replace('\n', '\\n')
                    st.markdown(f"""
                    <script>
                    navigator.clipboard.writeText("{copy_text}").then(function() {{
                        console.log('Text copied to clipboard');
                    }});
                    </script>
                    """, unsafe_allow_html=True)
                    st.success("📋 Response copied to clipboard!", icon="✅")

            if assistant_message.timestamp:
                st.caption(
                    f"🕒 {assistant_message.timestamp.strftime('%H:%M:%S')}")

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
                            st.write(
                                f"⚡ Response Time: {message.response_time:.2f}s")
                        if message.model_provider:
                            st.write(
                                f"🤖 Model: {message.model_provider.value}")
                        if message.source_documents:
                            st.write(
                                f"📚 Sources: {len(message.source_documents)} documents")

                if message.timestamp:
                    st.caption(f"🕒 {message.timestamp.strftime('%H:%M:%S')}")

    def _render_chat_input(self):
        """Render chat input section"""
        # Check if there's an edit request
        if 'edit_message_id' in st.session_state:
            st.markdown("### ✏️ Edit your question:")

            # Create a form for editing
            with st.form("edit_form"):
                edited_query = st.text_area(
                    "Edit your question:",
                    value=st.session_state.edit_query,
                    height=100,
                    placeholder="Type your edited question here..."
                )

                col1, col2 = st.columns(2)
                with col1:
                    submit_edit = st.form_submit_button(
                        "🔄 Update & Regenerate", type="primary")
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
                    edit_assistant_id = st.session_state.get(
                        'edit_assistant_id')
                    del st.session_state.edit_message_id
                    del st.session_state.edit_query
                    if 'edit_assistant_id' in st.session_state:
                        del st.session_state.edit_assistant_id

                    # Process the edited query (with replace mode for assistant message)
                    if edit_assistant_id:
                        st.session_state.replace_assistant_id = edit_assistant_id

                    self._process_user_query(
                        edited_query.strip(), is_edit=True)
                    return

                if cancel_edit:
                    # Clear edit state
                    if 'edit_message_id' in st.session_state:
                        del st.session_state.edit_message_id
                    if 'edit_query' in st.session_state:
                        del st.session_state.edit_query
                    if 'edit_assistant_id' in st.session_state:
                        del st.session_state.edit_assistant_id
                    st.rerun()
            return

        # Check if there's a retry query
        if 'retry_query' in st.session_state:
            retry_query = st.session_state.retry_query
            retry_assistant_id = st.session_state.get('retry_assistant_id')
            del st.session_state.retry_query  # Clear it after use
            if 'retry_assistant_id' in st.session_state:
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
            if not st.session_state.vector_store_initialized:
                st.error("❌ Please upload and process documents first!")
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
            unsafe_allow_html=True
        )

    def _initialize_llm(self, model_provider: str, api_key: str):
        """Initialize LLM service"""
        try:
            with st.spinner("🔧 Initializing LLM..."):
                # Initialize embeddings for vector store
                self.vector_store_manager.initialize_embeddings(
                    model_provider, api_key)

                # Initialize LLM
                self.llm_service.initialize_llm(model_provider, api_key)

                # Validate connection
                if self.llm_service.validate_api_connection():
                    st.session_state.llm_initialized = True
                    st.session_state.current_model_provider = model_provider
                    st.session_state.api_key = api_key  # Store API key for later use
                    st.success(
                        f"✅ {model_provider} LLM initialized successfully!")
                    self.logger.info(f"LLM initialized: {model_provider}")
                else:
                    st.error("❌ Failed to validate API connection")

        except Exception as e:
            st.error(f"❌ Failed to initialize LLM: {str(e)}")
            self.logger.error(f"LLM initialization failed: {str(e)}")

    def _process_documents(self, uploaded_files):
        """Process uploaded documents"""
        try:
            # Check if LLM is initialized first
            if not st.session_state.get('llm_initialized', False):
                st.error(
                    "❌ Please initialize LLM first before processing documents!")
                st.info(
                    "💡 Select a model provider, enter your API key, and click 'Initialize LLM'")
                return

            with st.spinner("📄 Processing documents..."):
                # Process documents
                documents = self.document_processor.process_uploaded_files(
                    uploaded_files)

                if not documents:
                    st.error("❌ No documents were successfully processed")
                    return

                # Create chunks
                all_chunks = []
                current_model_provider = st.session_state.get(
                    'current_model_provider', 'OpenAI')

                for doc in documents:
                    chunks = self.document_processor.create_chunks(
                        doc,
                        current_model_provider,
                        chunk_size=st.session_state.chunk_size,
                        chunk_overlap=st.session_state.chunk_overlap
                    )
                    all_chunks.extend(chunks)

                # Ensure embeddings are initialized for the current model provider
                # (This should already be done in _initialize_llm, but double-check)
                if not self.vector_store_manager.embeddings:
                    # Re-initialize embeddings if missing
                    try:
                        api_key = st.session_state.get('api_key', '')
                        if api_key:
                            self.vector_store_manager.initialize_embeddings(
                                current_model_provider, api_key)
                        else:
                            st.error(
                                "❌ API key not found. Please re-initialize LLM.")
                            return
                    except Exception as e:
                        st.error(
                            f"❌ Failed to initialize embeddings: {str(e)}")
                        return

                # Update or create vector store
                if st.session_state.vector_store_initialized:
                    vector_info = self.vector_store_manager.update_vector_store(
                        documents, all_chunks)
                    action = "updated"
                else:
                    vector_info = self.vector_store_manager.create_vector_store(
                        documents, all_chunks)
                    st.session_state.vector_store_initialized = True
                    action = "created"

                # Update session state
                st.session_state.documents.extend(documents)

                # Show success message
                st.success(f"""
                ✅ Successfully {action} knowledge base!
                - 📄 Documents: {len(documents)}
                - 🧩 Chunks: {len(all_chunks)}
                - 💾 Total Size: {format_file_size(vector_info.index_size_bytes)}
                """)

                self.logger.info(
                    f"Processed {len(documents)} documents with {len(all_chunks)} chunks")

        except Exception as e:
            st.error(f"❌ Document processing failed: {str(e)}")
            self.logger.error(
                f"Document processing failed: {str(e)}\n{traceback.format_exc()}")

    def _process_user_query(self, user_input: str, is_edit: bool = False, is_retry: bool = False):
        """Process user query and generate response"""
        try:
            # Check if we need to replace an existing assistant message
            replace_assistant_id = st.session_state.get('replace_assistant_id')
            if replace_assistant_id:
                del st.session_state.replace_assistant_id

            # For new queries (not edit/retry), add user message to conversation
            if not is_edit and not is_retry:
                user_message = ChatMessage(
                    id=str(uuid.uuid4()),
                    role=MessageRole.USER,
                    content=user_input
                )
                st.session_state.current_conversation.add_message(user_message)

                # Display user message immediately with edit button
                with st.chat_message("user"):
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        st.write(user_input)
                    with col2:
                        st.write("")  # Timestamp hizası için boşluk
                        if st.button("✏️ Edit", key=f"edit_active_{user_message.id}", help="Edit and regenerate"):
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
                            'current_model_provider', 'OpenAI')
                        api_key = st.session_state.get('api_key', '')

                        if not self.vector_store_manager.embeddings and api_key:
                            try:
                                self.vector_store_manager.initialize_embeddings(
                                    current_model_provider, api_key)
                            except Exception as e:
                                st.error(
                                    f"❌ Failed to initialize embeddings for search: {str(e)}")
                                return

                        # Use langchain-style search based on model provider
                        search_k = st.session_state.get('search_k', 5)

                        # Get search parameters
                        search_k = st.session_state.get('search_k', 5)
                        similarity_threshold = st.session_state.get(
                            'similarity_threshold', 0.75)  # Default to 0.75 for high quality

                        # Get model parameters
                        temperature = st.session_state.get(
                            'model_temperature', 0.7)

                        if current_model_provider == "OpenAI (API)":
                            search_result = self.vector_store_manager.search_documents_for_openai(
                                user_input, api_key, k=search_k, similarity_threshold=similarity_threshold,
                                temperature=temperature)
                        elif current_model_provider == "Local LLM (Qwen)":
                            # Use the same OpenAI search method but with Ollama configuration
                            search_result = self.vector_store_manager.search_documents_for_openai(
                                user_input, api_key, k=search_k, similarity_threshold=similarity_threshold,
                                temperature=temperature)
                        else:
                            search_result = {
                                "response": "Unsupported model provider"}

                        if not search_result or "Error" in search_result.get("response", ""):
                            response_text = search_result.get(
                                "response", "I couldn't find relevant information in the uploaded documents to answer your question.")
                            response_time = 0.0
                            assistant_message = ChatMessage(
                                id=str(uuid.uuid4()),
                                role=MessageRole.ASSISTANT,
                                content=response_text,
                                model_provider=ModelProvider(
                                    st.session_state.current_model_provider),
                                response_time=response_time
                            )
                        else:
                            # Create assistant message with the response and source information
                            sources = search_result.get("sources", [])
                            metadata = search_result.get("metadata", {})

                            assistant_message = ChatMessage(
                                id=str(uuid.uuid4()),
                                role=MessageRole.ASSISTANT,
                                content=search_result.get(
                                    "response", "No response generated"),
                                model_provider=ModelProvider(
                                    current_model_provider),
                                response_time=0.0,
                                tokens_used=0,
                                source_documents=[
                                    s.get("source", "Unknown") for s in sources],
                                metadata={
                                    'source_count': metadata.get("source_count", 0),
                                    'model': metadata.get("model", "Unknown"),
                                    'sources': sources,
                                    'retrieval_method': metadata.get("retrieval_method", "Unknown")
                                }
                            )

                            # Store response time for metrics
                            st.session_state.last_response_time = 0.0

                        # Add new assistant message to conversation
                        st.session_state.current_conversation.add_message(
                            assistant_message)

                        # Display response immediately
                        self._display_assistant_response(
                            assistant_message, user_input)
            else:
                # For replacement, process without chat context and rerun
                with st.spinner("🤔 Regenerating..."):
                    # Ensure embeddings are available for search
                    current_model_provider = st.session_state.get(
                        'current_model_provider', 'OpenAI')
                    api_key = st.session_state.get('api_key', '')

                    if not self.vector_store_manager.embeddings and api_key:
                        try:
                            self.vector_store_manager.initialize_embeddings(
                                current_model_provider, api_key)
                        except Exception as e:
                            st.error(
                                f"❌ Failed to initialize embeddings for search: {str(e)}")
                            return

                    # Get search parameters
                    search_k = st.session_state.get('search_k', 5)
                    similarity_threshold = st.session_state.get(
                        'similarity_threshold', 0.75)  # Default to 0.75 for high quality

                    # Get model parameters
                    temperature = st.session_state.get(
                        'model_temperature', 0.7)

                    if current_model_provider == "OpenAI (API)":
                        search_result = self.vector_store_manager.search_documents_for_openai(
                            user_input, api_key, k=search_k, similarity_threshold=similarity_threshold,
                            temperature=temperature)
                    elif current_model_provider == "Local LLM (Qwen)":
                        # Use the same OpenAI search method but with Ollama configuration
                        search_result = self.vector_store_manager.search_documents_for_openai(
                            user_input, api_key, k=search_k, similarity_threshold=similarity_threshold,
                            temperature=temperature)
                    else:
                        search_result = {
                            "response": "Unsupported model provider"}

                    if not search_result or "Error" in search_result.get("response", ""):
                        response_text = search_result.get(
                            "response", "I couldn't find relevant information in the uploaded documents to answer your question.")
                        response_time = 0.0
                        assistant_message = ChatMessage(
                            id=replace_assistant_id,
                            role=MessageRole.ASSISTANT,
                            content=response_text,
                            model_provider=ModelProvider(
                                st.session_state.current_model_provider),
                            response_time=response_time
                        )
                    else:
                        # Create assistant message with the response and source information
                        sources = search_result.get("sources", [])
                        metadata = search_result.get("metadata", {})

                        assistant_message = ChatMessage(
                            id=replace_assistant_id,
                            role=MessageRole.ASSISTANT,
                            content=search_result.get(
                                "response", "No response generated"),
                            model_provider=ModelProvider(
                                current_model_provider),
                            response_time=0.0,
                            tokens_used=0,
                            source_documents=[
                                s.get("source", "Unknown") for s in sources],
                            metadata={
                                'source_count': metadata.get("source_count", 0),
                                'model': metadata.get("model", "Unknown"),
                                'sources': sources,
                                'retrieval_method': metadata.get("retrieval_method", "Unknown")
                            }
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
                f"Query processing failed: {str(e)}\n{traceback.format_exc()}")

    def _display_assistant_response(self, assistant_message: ChatMessage, user_query: str):
        """Display assistant response with source information and action buttons"""
        # Display response in main area
        col1, col2 = st.columns([2, 1])

        with col1:
            st.write(assistant_message.content)

        # Show source information in right column
        with col2:
            if assistant_message.metadata and assistant_message.metadata.get('sources'):
                sources = assistant_message.metadata['sources']

                st.markdown("### 📊 Response Information")

                # Model and source info
                model_name = assistant_message.metadata.get('model', 'Unknown')
                source_count = assistant_message.metadata.get(
                    'source_count', 0)
                retrieval_method = assistant_message.metadata.get(
                    'retrieval_method', 'Unknown')

                st.metric("Model", model_name)
                st.metric("Sources Used", source_count)
                st.caption(f"Method: {retrieval_method}")

                # Expandable source details
                with st.expander(f"📚 View {source_count} Sources", expanded=False):
                    for i, source in enumerate(sources, 1):
                        st.markdown(f"**Source {i}**")

                        # Source file and page info
                        source_file = source.get('source', 'Unknown')
                        page = source.get('page', 'Unknown')
                        accuracy_score = source.get('accuracy_score', 0.0)

                        st.write(f"📄 **File:** {source_file}")
                        st.write(f"📖 **Page:** {page}")
                        st.write(f"🎯 **Accuracy:** {accuracy_score:.2%}")

                        # Content preview - show full content with larger height
                        content = source.get('content', 'No content available')
                        st.text_area(
                            f"Content Preview {i}",
                            content,
                            height=300,  # Increased height for better content viewing
                            key=f"source_content_new_{assistant_message.id}_{i}",
                            disabled=True
                        )

                        # Additional metadata
                        metadata_info = source.get('metadata', {})
                        if metadata_info:
                            with st.expander(f"📋 Metadata {i}"):
                                for key, value in metadata_info.items():
                                    # Don't repeat already shown info
                                    if key not in ['source', 'page']:
                                        st.write(f"**{key.title()}:** {value}")

                        st.divider()
            else:
                st.markdown("### ℹ️ No Sources Available")
                st.write(
                    "No source information could be extracted for this response.")

        # Action buttons below the response (only show if not in edit mode)
        if 'edit_message_id' not in st.session_state:
            st.markdown("---")
            col1, col2, col3 = st.columns([1, 1, 2])

            with col1:
                if st.button("🔄 Retry", key=f"retry_new_{assistant_message.id}", help="Regenerate response with current parameters"):
                    # Store the current query and assistant ID for retry
                    st.session_state.retry_query = user_query
                    st.session_state.retry_assistant_id = assistant_message.id
                    st.rerun()

            with col2:
                if st.button("📋 Copy", key=f"copy_new_{assistant_message.id}", help="Copy response to clipboard"):
                    # Use JavaScript to copy to clipboard
                    copy_text = assistant_message.content.replace(
                        '"', '\\"').replace('\n', '\\n')
                    st.markdown(f"""
                    <script>
                    navigator.clipboard.writeText("{copy_text}").then(function() {{
                        console.log('Text copied to clipboard');
                    }});
                    </script>
                    """, unsafe_allow_html=True)
                    st.success("✅ Response copied to clipboard!", icon="📋")

    def _clear_current_conversation(self):
        """Clear current conversation"""
        st.session_state.current_conversation = ConversationSession(
            id=str(uuid.uuid4()),
            title="New Conversation"
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
            with open(log_file, 'r', encoding='utf-8') as f:
                log_content = f.read()

            # Create download link
            b64 = base64.b64encode(log_content.encode('utf-8')).decode()
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
            with open(log_file, 'w', encoding='utf-8') as f:
                f.write("")
            st.success("✅ Logs cleared successfully!")

        except Exception as e:
            st.error(f"❌ Error clearing logs: {str(e)}")

    def _delete_chunks(self):
        """Delete all vector store chunks and clear system info"""
        try:
            # Clear vector store
            self.vector_store_manager.clear_vector_store()

            # Clear documents from session state
            st.session_state.documents = []

            # Clear conversation to reset system info
            st.session_state.current_conversation = ConversationSession(
                id=str(uuid.uuid4()),
                title="New Conversation"
            )

            st.success(
                "✅ All chunks deleted and system info cleared successfully!")

            # Force rerun to update UI
            st.rerun()

        except Exception as e:
            st.error(f"❌ Error deleting chunks: {str(e)}")

    def _clear_cache(self):
        """Clear question-answer cache"""
        try:
            # Get cache stats before clearing
            cache_stats = self.vector_store_manager.get_cache_stats()
            entries_count = cache_stats.get('total_entries', 0)

            # Clear cache
            self.vector_store_manager.clear_cache()

            st.success(
                f"✅ Cache cleared successfully! Removed {entries_count} cached entries.")

            # Force rerun to update UI
            st.rerun()

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
            data.append({
                'Timestamp': msg.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                'Role': msg.role.value,
                'Content': msg.content,
                'Model': msg.model_provider.value if msg.model_provider else '',
                'Response Time': msg.response_time if msg.response_time else '',
                'Tokens Used': msg.tokens_used if msg.tokens_used else ''
            })

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
