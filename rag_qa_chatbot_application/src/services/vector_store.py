"""
Vector store service using FAISS for RAG QA Chatbot Application
"""

import os
import pickle
import re
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
import numpy as np

from langchain.embeddings import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document as LangchainDocument

from ..config import config, CURRENT_DB_OPENAI_DIR, CURRENT_DB_QWEN_DIR
from ..models import Document, DocumentChunk, VectorStoreInfo, ModelProvider
from ..utils import app_logger, measure_execution_time, retry_on_exception, cached
from .cache_manager import CacheManager


class VectorStoreManager:
    """FAISS Vector Store Management Service"""

    def __init__(self):
        self.logger = app_logger
        self.index_path = config.vectorstore.index_path  # Just "faiss_index"
        self.vector_store: Optional[FAISS] = None
        self.embeddings = None
        self.current_model_provider = None
        self.document_metadata: Dict[str, Dict] = {}
        self.use_current_db = (
            False  # Flag for using current_db instead of regular vectorstore
        )
        self.db_mode = "current"  # "current", "new", "current+new"

        # Cached document count (to avoid recalculating on every question)
        # This should only be updated when documents are actually added
        # CRITICAL: Restore from Streamlit session state if available (to persist across reruns)
        self._cached_total_documents: Optional[int] = self._restore_cache_from_session()

        # Initialize cache manager
        self.cache_manager = CacheManager()

        # Set working directory to vectorstore directory
        path_separators = ["/", os.sep]
        has_path_separator = any(
            sep in config.vectorstore.index_path for sep in path_separators
        )
        self.vectorstore_dir = (
            Path(config.vectorstore.index_path).parent
            if has_path_separator
            else config.vectorstore.index_path.__class__(
                config.vectorstore.index_path
            ).parent
        )
        # For simple path like "faiss_index", use VECTORSTORE_DIR from config
        from ..config.settings import VECTORSTORE_DIR

        self.vectorstore_dir = VECTORSTORE_DIR

        # Ensure vectorstore directory exists
        self.vectorstore_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"Vector store index name: {self.index_path}")
        self.logger.info(f"Vector store directory: {self.vectorstore_dir}")

        # Log restored cache if available
        if self._cached_total_documents is not None:
            self.logger.info(
                f"Restored cached total documents count from session: {self._cached_total_documents}"
            )

    def _restore_cache_from_session(self) -> Optional[int]:
        """
        Restore cached document count from Streamlit session state.
        This ensures cache persists across Streamlit reruns.

        Returns:
            Cached document count if available, None otherwise
        """
        try:
            # Try to import streamlit (only if available, won't fail if not in Streamlit context)
            import streamlit as st

            # Get cache from session state
            cache_key = "vector_store_cached_total_documents"
            if cache_key in st.session_state:
                cached_value = st.session_state[cache_key]
                if (
                    cached_value is not None
                    and isinstance(cached_value, int)
                    and cached_value > 0
                ):
                    # Use app_logger directly since self.logger not yet initialized
                    app_logger.debug(
                        f"Restored cached total documents from session state: {cached_value}"
                    )
                    return cached_value
        except (ImportError, AttributeError, RuntimeError):
            # Not in Streamlit context or session state not available
            # This is OK, just return None
            pass

        return None

    def _save_cache_to_session(self, value: Optional[int]) -> None:
        """
        Save cached document count to Streamlit session state.
        This ensures cache persists across Streamlit reruns.

        Args:
            value: Document count to cache
        """
        try:
            # Try to import streamlit (only if available)
            import streamlit as st

            # Save cache to session state
            cache_key = "vector_store_cached_total_documents"
            if value is not None and value > 0:
                st.session_state[cache_key] = value
                self.logger.debug(
                    f"Saved cached total documents to session state: {value}"
                )
            elif cache_key in st.session_state:
                # Clear cache if value is None or 0
                del st.session_state[cache_key]
        except (ImportError, AttributeError, RuntimeError):
            # Not in Streamlit context or session state not available
            # This is OK, just skip saving
            pass

    def _convert_distance_to_similarity(self, distance_score: float) -> float:
        """
        Convert FAISS L2 distance to accuracy/relevance score (0-1 range)

        For normalized vectors (which OpenAI and Ollama embeddings use),
        L2 distance relates to cosine similarity as:
        cosine_similarity = 1 - (L2_distance^2 / 2)

        We use a more intuitive mapping for accuracy:
        - Distance 0.0 → Accuracy 100% (perfect match)
        - Distance 0.3 → Accuracy ~95% (excellent match)
        - Distance 0.5 → Accuracy ~88% (good match)
        - Distance 0.7 → Accuracy ~76% (fair match)
        - Distance 1.0 → Accuracy ~50% (poor match)
        - Distance >1.5 → Accuracy <25% (very poor match)

        Args:
            distance_score: FAISS L2 distance score

        Returns:
            Accuracy score between 0.0 and 1.0
        """
        # Warn for unusual scores
        if distance_score < 0:
            self.logger.warning(f"Negative distance score: {distance_score}")
            return 0.0

        if distance_score > 2.0:
            self.logger.debug(f"Very high distance score: {distance_score}")
            return 0.0

        # Convert L2 distance to cosine similarity
        # For normalized vectors: cos_sim = 1 - (dist^2 / 2)
        cosine_sim = 1.0 - (distance_score * distance_score / 2.0)

        # Apply boosting to make scores more interpretable
        # Small distances should result in higher scores
        # This makes the UI more intuitive for users
        if cosine_sim >= 0.7:
            # Already high similarity, keep it
            accuracy = cosine_sim
        elif cosine_sim >= 0.4:
            # Medium similarity - boost slightly
            # Map [0.4, 0.7] → [0.5, 0.7]
            accuracy = 0.5 + (cosine_sim - 0.4) * (0.2 / 0.3)
        elif cosine_sim >= 0.2:
            # Low similarity - boost moderately
            # Map [0.2, 0.4] → [0.3, 0.5]
            accuracy = 0.3 + (cosine_sim - 0.2) * (0.2 / 0.2)
        else:
            # Very low similarity - minimal boost
            # Map [0.0, 0.2] → [0.0, 0.3]
            accuracy = cosine_sim * 1.5

        # Clamp to [0, 1]
        result = max(0.0, min(1.0, accuracy))

        return result

    @measure_execution_time
    def initialize_embeddings(self, model_provider: str, api_key: str) -> None:
        """
        Initialize embeddings based on model provider

        Args:
            model_provider: Model provider name
            api_key: API key for the provider
        """
        try:
            if model_provider == "OpenAI (API)":
                self.embeddings = OpenAIEmbeddings(
                    model=config.model.openai_embedding_model, openai_api_key=api_key
                )
                # Log the embedding model being used for debugging
                self.logger.info(
                    f"Initialized OpenAI embeddings with model: {config.model.openai_embedding_model}"
                )
            elif model_provider == "Local LLM (Qwen)":
                # For local LLM, use Ollama embeddings
                self.embeddings = OllamaEmbeddings(
                    model=config.model.local_embedding_model,  # Use all-minilm
                    base_url=config.model.ollama_base_url.replace(
                        "/v1", ""
                    ),  # Remove /v1 for Ollama
                )
                self.logger.info(
                    f"Initialized Qwen embeddings with model: {config.model.local_embedding_model}"
                )
            else:
                raise ValueError(f"Unsupported model provider: {model_provider}")

            self.current_model_provider = model_provider
            self.logger.info(f"Initialized embeddings for {model_provider}")

            # Set embeddings for cache manager (for semantic similarity)
            self.cache_manager.set_embeddings(self.embeddings)

        except Exception as e:
            self.logger.error(f"Failed to initialize embeddings: {str(e)}")
            raise

    @measure_execution_time
    @retry_on_exception(max_retries=3)
    def create_vector_store(
        self, documents: List[Document], chunks: List[DocumentChunk]
    ) -> VectorStoreInfo:
        """
        Create new vector store from documents and chunks

        Args:
            documents: List of Document objects
            chunks: List of DocumentChunk objects

        Returns:
            VectorStoreInfo object with store information
        """
        if not self.embeddings:
            raise ValueError(
                "Embeddings not initialized. Call initialize_embeddings first."
            )

        try:
            # Prepare texts and metadata for FAISS
            texts = []
            metadatas = []

            for chunk in chunks:
                texts.append(chunk.content)

                # Find corresponding document
                doc = next((d for d in documents if d.id == chunk.document_id), None)
                doc_name = doc.name if doc else "Unknown"

                metadata = {
                    "chunk_id": chunk.id,
                    "document_id": chunk.document_id,
                    "document_name": doc_name,
                    "source": doc_name,  # Add source field for compatibility
                    # Use page from chunk metadata or chunk index
                    "page": chunk.metadata.get("page", chunk.chunk_index + 1),
                    "chunk_index": chunk.chunk_index,
                    "start_char": chunk.start_char,
                    "end_char": chunk.end_char,
                    **chunk.metadata,
                }
                metadatas.append(metadata)

            # Create FAISS vector store
            if texts:
                # Process in batches to avoid context length issues with Ollama embeddings
                batch_size = 32  # Process 32 chunks at a time

                if len(texts) <= batch_size:
                    # Small batch, process all at once
                    self.vector_store = FAISS.from_texts(
                        texts=texts,
                        embedding=self.embeddings,
                        metadatas=metadatas,
                        normalize_L2=True,
                    )
                else:
                    # Large batch, process incrementally
                    self.logger.info(
                        f"Processing {len(texts)} texts in batches of {batch_size}"
                    )

                    # Create initial vector store with first batch
                    first_batch_texts = texts[:batch_size]
                    first_batch_metas = metadatas[:batch_size]

                    self.vector_store = FAISS.from_texts(
                        texts=first_batch_texts,
                        embedding=self.embeddings,
                        metadatas=first_batch_metas,
                        normalize_L2=True,
                    )

                    # Add remaining batches
                    for i in range(batch_size, len(texts), batch_size):
                        batch_texts = texts[i : i + batch_size]
                        batch_metas = metadatas[i : i + batch_size]

                        self.logger.info(
                            f"Processing batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}"
                        )
                        self.vector_store.add_texts(
                            texts=batch_texts, metadatas=batch_metas
                        )

                # Save vector store
                self._save_vector_store()

                # Store document metadata
                self._save_document_metadata(documents)

                # For current_db mode, calculate and cache unique pubmed_id count
                total_docs = len(documents)
                if self.db_mode == "current" and self.vector_store:
                    unique_pubmed_count = self._count_unique_pubmed_ids()
                    if unique_pubmed_count > 0:
                        self._cached_total_documents = unique_pubmed_count
                        self._save_cache_to_session(unique_pubmed_count)
                        total_docs = unique_pubmed_count
                        self.logger.info(
                            f"Cached total documents count after creating store: {total_docs}"
                        )

                # Create store info
                store_info = VectorStoreInfo(
                    index_path=self.index_path,
                    total_documents=total_docs,
                    total_chunks=len(chunks),
                    embedding_model=self._get_embedding_model_name(),
                    index_size_bytes=self._get_index_size(),
                )

                self.logger.info(
                    f"Created vector store with {len(texts)} chunks from {len(documents)} documents"
                )
                return store_info

            else:
                raise ValueError("No text chunks to create vector store")

        except Exception as e:
            self.logger.error(f"Failed to create vector store: {str(e)}")
            raise

    @measure_execution_time
    def update_vector_store(
        self, new_documents: List[Document], new_chunks: List[DocumentChunk]
    ) -> VectorStoreInfo:
        """
        Update existing vector store with new documents

        Args:
            new_documents: List of new Document objects
            new_chunks: List of new DocumentChunk objects

        Returns:
            Updated VectorStoreInfo object
        """
        if not self.embeddings:
            raise ValueError("Embeddings not initialized")

        try:
            # Load existing vector store if not loaded
            if not self.vector_store:
                self._load_vector_store()

            # Prepare new texts and metadata
            new_texts = []
            new_metadatas = []

            for chunk in new_chunks:
                new_texts.append(chunk.content)

                doc = next(
                    (d for d in new_documents if d.id == chunk.document_id), None
                )
                doc_name = doc.name if doc else "Unknown"

                metadata = {
                    "chunk_id": chunk.id,
                    "document_id": chunk.document_id,
                    "document_name": doc_name,
                    "source": doc_name,  # Add source field for compatibility
                    # Use page from chunk metadata or chunk index
                    "page": chunk.metadata.get("page", chunk.chunk_index + 1),
                    "chunk_index": chunk.chunk_index,
                    "start_char": chunk.start_char,
                    "end_char": chunk.end_char,
                    **chunk.metadata,
                }
                new_metadatas.append(metadata)

            # Add new documents to existing vector store
            if new_texts:
                if self.vector_store:
                    # Add to existing store
                    self.vector_store.add_texts(
                        texts=new_texts, metadatas=new_metadatas
                    )
                else:
                    # Create new store if none exists
                    self.vector_store = FAISS.from_texts(
                        texts=new_texts,
                        embedding=self.embeddings,
                        metadatas=new_metadatas,
                    )

                # Save updated vector store
                self._save_vector_store()

                # Update document metadata
                self._update_document_metadata(new_documents)

                # Get updated info
                total_docs, total_chunks = self._get_store_stats()

                # For current_db mode, recalculate and cache unique pubmed_id count
                if self.db_mode == "current" and self.vector_store:
                    unique_pubmed_count = self._count_unique_pubmed_ids()
                    if unique_pubmed_count > 0:
                        self._cached_total_documents = unique_pubmed_count
                        self._save_cache_to_session(unique_pubmed_count)
                        total_docs = unique_pubmed_count
                        self.logger.info(
                            f"Updated cached total documents count after adding documents: {total_docs}"
                        )

                store_info = VectorStoreInfo(
                    index_path=self.index_path,
                    total_documents=total_docs,
                    total_chunks=total_chunks,
                    embedding_model=self._get_embedding_model_name(),
                    index_size_bytes=self._get_index_size(),
                )

                self.logger.info(
                    f"Updated vector store with {len(new_texts)} new chunks"
                )
                return store_info

            else:
                self.logger.warning("No new chunks to add to vector store")
                return self.get_store_info()

        except Exception as e:
            self.logger.error(f"Failed to update vector store: {str(e)}")
            raise

    def search_documents_for_openai(
        self,
        query: str,
        api_key: str,
        k: int = 5,
        temperature: float = None,
        max_tokens: int = None,
        new_vector_store: Optional[FAISS] = None,
    ) -> Dict[str, Any]:
        """
        Search documents using OpenAI approach with advanced retrieval techniques

        This method uses:
        1. Contextual Compression: Filters irrelevant content using LLM
        2. LLMChainExtractor: Extracts only query-relevant parts from documents
        3. similarity_search_with_score: Returns documents with their similarity scores

        Args:
            query: User query
            api_key: OpenAI API key
            k: Number of documents to retrieve
            temperature: LLM temperature
            max_tokens: Maximum tokens for LLM response
            new_vector_store: Optional new vector store for "new" or "current+new" modes

        Returns:
            Dictionary with response, sources, and metadata
        """
        # Check cache first
        cached_result = self.cache_manager.get(query, k=k)
        if cached_result:
            self.logger.info(f"Returning cached result for query: '{query[:50]}...'")
            return cached_result

        # Determine which vector store(s) to use based on db_mode
        current_db_vector_store = None
        new_db_vector_store = new_vector_store

        # CRITICAL: Only load current_db if NOT in "new" mode
        # In "new" mode, we should NEVER touch current_db
        if self.db_mode in ["current", "current+new"]:
            if not self.vector_store:
                # Load from current_db if in current mode
                # Preserve cached document count when loading
                current_db_path = self._get_current_db_path()
                if current_db_path:
                    self._load_vector_store(custom_path=current_db_path)
                else:
                    self._load_vector_store()
            current_db_vector_store = self.vector_store

            # CRITICAL: Ensure embeddings are set on the vector store after loading
            # This is essential for similarity search to work correctly
            if current_db_vector_store and self.embeddings:
                if (
                    not hasattr(current_db_vector_store, "embeddings")
                    or current_db_vector_store.embeddings is None
                ):
                    current_db_vector_store.embeddings = self.embeddings
                    self.logger.info("Set embeddings on vector store after loading")
                elif current_db_vector_store.embeddings != self.embeddings:
                    # Embeddings exist but are different - update them
                    self.logger.warning(
                        "Vector store embeddings differ from current embeddings, updating..."
                    )
                    current_db_vector_store.embeddings = self.embeddings
        else:
            # Explicitly clear current_db_vector_store for "new" mode
            # This ensures we never accidentally use current_db
            current_db_vector_store = None
            self.vector_store = None
            self.logger.info("Using new_db mode - current_db explicitly excluded")

        # Determine which vector store to use based on mode
        if self.db_mode == "new":
            # Use only new_db - NEVER use current_db
            if not new_db_vector_store:
                self.logger.warning("No new vector store available for search")
                return {
                    "response": "No documents found to search in new database.",
                    "sources": [],
                    "metadata": {"confidence": 0.0, "source_count": 0},
                }
            vector_store_to_use = new_db_vector_store
            self.logger.info("Using new_db for search (current_db excluded)")
        elif self.db_mode == "current+new":
            # Need to merge both stores
            # IMPORTANT: Merging should only happen in Docker, not on local filesystem
            # Check if we're in Docker by looking at environment or path
            import os

            is_docker = (
                os.path.exists("/.dockerenv")
                or os.environ.get("DOCKER_CONTAINER") == "true"
            )

            if not is_docker:
                self.logger.warning(
                    "current+new mode requires Docker environment. "
                    "Merging will be done in Docker only to protect current_db on local filesystem."
                )
                # If not in Docker, use only new_db to avoid modifying local current_db
                if new_db_vector_store:
                    vector_store_to_use = new_db_vector_store
                    self.logger.info(
                        "Using new_db only (not in Docker, protecting local current_db)"
                    )
                elif current_db_vector_store:
                    vector_store_to_use = current_db_vector_store
                    self.logger.info(
                        "Using current_db only (new_db not available, not in Docker)"
                    )
                else:
                    self.logger.warning("No vector stores available")
                    return {
                        "response": "No documents found to search.",
                        "sources": [],
                        "metadata": {"confidence": 0.0, "source_count": 0},
                    }
            elif not current_db_vector_store and not new_db_vector_store:
                self.logger.warning(
                    "No vector stores available for search (both current and new are empty)"
                )
                return {
                    "response": "No documents found to search.",
                    "sources": [],
                    "metadata": {"confidence": 0.0, "source_count": 0},
                }
            # Merge stores if both exist (only in Docker)
            elif current_db_vector_store and new_db_vector_store:
                # Merge the two stores (Docker environment)
                self.logger.info(
                    "Merging current_db and new_db for search (Docker environment)"
                )
                try:
                    # Merge FAISS stores by extracting all documents and creating a new merged store
                    # This approach is simpler but requires re-embedding (acceptable for merged searches)
                    try:
                        # Extract all documents from both stores
                        # FAISS docstore contains LangchainDocument objects with page_content and metadata
                        current_docs = []
                        if (
                            hasattr(current_db_vector_store, "docstore")
                            and current_db_vector_store.docstore
                        ):
                            if hasattr(current_db_vector_store.docstore, "_dict"):
                                current_docs = list(
                                    current_db_vector_store.docstore._dict.values()
                                )

                        new_docs = []
                        if (
                            hasattr(new_db_vector_store, "docstore")
                            and new_db_vector_store.docstore
                        ):
                            if hasattr(new_db_vector_store.docstore, "_dict"):
                                new_docs = list(
                                    new_db_vector_store.docstore._dict.values()
                                )

                        # Get embeddings and create new merged store
                        # Extract texts and metadatas from LangchainDocument objects
                        all_texts = []
                        all_metadatas = []
                        for doc in current_docs:
                            # LangchainDocument has page_content and metadata attributes
                            if hasattr(doc, "page_content"):
                                all_texts.append(doc.page_content)
                            else:
                                self.logger.warning(
                                    f"Document missing page_content: {type(doc)}"
                                )
                            if hasattr(doc, "metadata"):
                                all_metadatas.append(doc.metadata)
                            else:
                                all_metadatas.append({})

                        for doc in new_docs:
                            if hasattr(doc, "page_content"):
                                all_texts.append(doc.page_content)
                            else:
                                self.logger.warning(
                                    f"Document missing page_content: {type(doc)}"
                                )
                            if hasattr(doc, "metadata"):
                                all_metadatas.append(doc.metadata)
                            else:
                                all_metadatas.append({})

                        # Ensure texts and metadatas have same length
                        if len(all_texts) != len(all_metadatas):
                            self.logger.warning(
                                f"Mismatch: {len(all_texts)} texts vs {len(all_metadatas)} metadatas"
                            )
                            # Pad with empty dicts if needed
                            while len(all_metadatas) < len(all_texts):
                                all_metadatas.append({})

                        # Create merged store from all documents
                        if all_texts:
                            vector_store_to_use = FAISS.from_texts(
                                texts=all_texts,
                                embedding=self.embeddings,
                                metadatas=all_metadatas,
                                normalize_L2=True,
                            )
                            self.logger.info(
                                f"Successfully created merged store with {len(all_texts)} documents (current: {len(current_docs)}, new: {len(new_docs)})"
                            )
                        else:
                            raise ValueError("No documents to merge")
                    except Exception as merge_error:
                        self.logger.error(
                            f"Failed to create merged store from documents: {merge_error}"
                        )
                        # Fallback: use current_db only
                        vector_store_to_use = current_db_vector_store
                        self.logger.warning(
                            "Using current_db only due to merge failure"
                        )
                except Exception as e:
                    self.logger.error(f"Failed to merge stores: {e}")
                    # Last resort: use only current_db
                    vector_store_to_use = current_db_vector_store
                    self.logger.warning("Using current_db only due to merge failure")
            elif current_db_vector_store:
                vector_store_to_use = current_db_vector_store
                self.logger.info("Using current_db only (new_db not available)")
            else:
                vector_store_to_use = new_db_vector_store
                self.logger.info("Using new_db only (current_db not available)")
        else:
            # "current" mode - use only current_db
            if not current_db_vector_store:
                self.logger.warning("No vector store available for search")
                return {
                    "response": "No documents found to search.",
                    "sources": [],
                    "metadata": {"confidence": 0.0, "source_count": 0},
                }
            vector_store_to_use = current_db_vector_store
            self.logger.info("Using current_db for search")

        try:
            # Initialize OpenAI LLM
            from langchain_community.chat_models import ChatOpenAI
            from langchain.memory import ConversationBufferMemory
            from langchain.chains import ConversationalRetrievalChain
            from langchain.prompts import PromptTemplate
            from langchain.retrievers import ContextualCompressionRetriever
            from langchain.retrievers.document_compressors import LLMChainExtractor

            # Initialize LLM based on current provider
            if self.current_model_provider == "OpenAI (API)":
                llm = ChatOpenAI(
                    api_key=api_key,
                    model_name=config.model.openai_model,
                    temperature=(
                        temperature
                        if temperature is not None
                        else config.model.openai_temperature
                    ),
                    max_tokens=(
                        max_tokens
                        if max_tokens is not None
                        else config.model.openai_max_tokens
                    ),
                )
                model_display_name = f"OpenAI {config.model.openai_model}"
            elif self.current_model_provider == "Local LLM (Qwen)":
                llm = ChatOpenAI(
                    temperature=(
                        temperature
                        if temperature is not None
                        else config.model.local_temperature
                    ),
                    model_name=config.model.local_model,
                    openai_api_key="ollama",
                    openai_api_base=config.model.ollama_base_url,
                    max_tokens=(
                        max_tokens
                        if max_tokens is not None
                        else config.model.local_max_tokens
                    ),
                )
                model_display_name = f"Local Qwen {config.model.local_model}"
            else:
                # Fallback to OpenAI format
                llm = ChatOpenAI(
                    api_key=api_key,
                    model_name="gpt-4o",
                    temperature=temperature if temperature is not None else 0.7,
                    max_tokens=max_tokens if max_tokens is not None else 2000,
                )
                model_display_name = "gpt-4o"

            # Initialize memory for conversation
            memory = ConversationBufferMemory(
                memory_key="chat_history", return_messages=True, output_key="answer"
            )

            # Create custom QA prompt template with better instructions
            qa_prompt_template = """You are a highly intelligent and helpful AI assistant. Your task is to answer questions accurately based on the provided context.

CORE PRINCIPLES:
1. **Be Helpful**: Always try to provide useful information if ANY relevant content exists in the context
2. **Be Accurate**: Base your answer strictly on the provided context - never fabricate information
3. **Be Comprehensive**: Extract and synthesize ALL relevant information from the context
4. **Be Clear**: Provide structured, easy-to-understand answers

ANSWER GUIDELINES:
✓ If the context contains relevant information → Provide a detailed, complete answer
✓ If the context contains partial information → Answer what you can and note what's missing
✓ If the context mentions related concepts → Explain the connection and provide available details
✓ If multiple sources discuss the topic → Synthesize information from all relevant sources
✓ Always cite specific facts, numbers, and details from the context
✗ ONLY say "information not available" if the context is COMPLETELY unrelated to the question

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""

            qa_prompt = PromptTemplate(
                template=qa_prompt_template, input_variables=["context", "question"]
            )

            # Step 1: First verify basic similarity search works
            # This helps debug embedding issues
            try:
                test_results = vector_store_to_use.similarity_search_with_score(
                    query, k=min(5, k)
                )
                self.logger.info(
                    f"Basic similarity search test returned {len(test_results)} documents (query: '{query[:50]}...')"
                )
                if test_results:
                    self.logger.info(
                        f"First result similarity score: {test_results[0][1]:.4f}"
                    )
                else:
                    self.logger.warning(
                        "WARNING: Basic similarity search returned 0 results - this suggests an embedding mismatch!"
                    )
                    # Log embedding info for debugging
                    if (
                        hasattr(vector_store_to_use, "embeddings")
                        and vector_store_to_use.embeddings
                    ):
                        embedding_model = getattr(
                            vector_store_to_use.embeddings, "model", "unknown"
                        )
                        self.logger.warning(
                            f"Vector store embeddings model: {embedding_model}"
                        )
                    if hasattr(vector_store_to_use, "index") and hasattr(
                        vector_store_to_use.index, "ntotal"
                    ):
                        self.logger.warning(
                            f"Vector store index contains {vector_store_to_use.index.ntotal} vectors"
                        )
            except Exception as e:
                self.logger.error(f"Basic similarity search test failed: {str(e)}")

            # Step 1: Create base retriever with MMR for diversity and better recall
            # MMR (Maximal Marginal Relevance) balances relevance and diversity
            # More aggressive MMR settings for better coverage
            # Use the determined vector store (could be current_db, new_db, or merged)
            base_retriever = vector_store_to_use.as_retriever(
                search_type="mmr",  # Use MMR instead of pure similarity
                search_kwargs={
                    "k": k * 4,  # Get 4x more candidates for comprehensive coverage
                    "fetch_k": k
                    * 10,  # Fetch 10x candidates initially for MMR to choose from
                    "lambda_mult": 0.5,  # Lower = more diversity, higher = more relevance
                    # 0.5 gives good balance between finding relevant AND diverse content
                },
            )

            # TEMPORARY: Disable Contextual Compression for Ollama (too slow on CPU)
            # Use basic retriever for faster responses
            use_compression = self.current_model_provider != "Local LLM (Qwen)"

            if use_compression:
                # Step 2: Create LLM Chain Extractor for contextual compression
                compressor = LLMChainExtractor.from_llm(llm)

                # Step 3: Create Contextual Compression Retriever
                compression_retriever = ContextualCompressionRetriever(
                    base_compressor=compressor, base_retriever=base_retriever
                )

                self.logger.info(
                    f"Using Contextual Compression Retriever with LLM filtering for query: '{query[:50]}...'"
                )

                retriever_to_use = compression_retriever
            else:
                self.logger.info(
                    f"Using basic retriever (Compression disabled for Ollama/CPU) for query: '{query[:50]}...'"
                )
                retriever_to_use = base_retriever

            # Create conversational retrieval chain
            conversation_chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=retriever_to_use,
                memory=memory,
                return_source_documents=True,
                combine_docs_chain_kwargs={"prompt": qa_prompt},
            )

            # Get response using the contextual compression retriever
            response = conversation_chain(
                {
                    "question": query,
                    "chat_history": [],  # Reset chat history for each query
                }
            )

            # Get source documents from chain response
            # These are already filtered by LLM for relevance!
            chain_sources = response.get("source_documents", [])
            self.logger.info(
                f"Contextual Compression returned {len(chain_sources)} relevant documents"
            )

            # CRITICAL: Calculate similarity scores using ORIGINAL QUERY, not compressed content
            # Compressed documents have modified content, we need to match by metadata
            chain_sources_with_scores = {}

            if chain_sources:
                # Step 1: Get similarity scores from ORIGINAL QUERY
                self.logger.info(
                    f"Calculating similarity scores using original query: '{query[:50]}...'"
                )
                all_docs_with_scores = vector_store_to_use.similarity_search_with_score(
                    query, k=k * 3  # Get enough candidates to find all compressed docs
                )

                self.logger.info(
                    f"Original query search returned {len(all_docs_with_scores)} candidates"
                )

                # Step 2: Create mapping of chunk_id -> (distance, similarity) from original query results
                # Also create mapping of chunk_id -> original_content for highlighting
                doc_scores_map = {}
                original_content_map = {}  # chunk_id -> original page_content (before compression)
                for orig_doc, distance_score in all_docs_with_scores:
                    chunk_id = orig_doc.metadata.get("chunk_id")
                    if chunk_id:
                        similarity_score = self._convert_distance_to_similarity(
                            distance_score
                        )
                        doc_scores_map[chunk_id] = (distance_score, similarity_score)
                        # CRITICAL: Store original content for highlighting
                        # This is the UNCOMPRESSED content from vector store
                        original_content_map[chunk_id] = orig_doc.page_content
                        self.logger.debug(
                            f"  Original doc chunk_id={chunk_id}: distance={distance_score:.4f}, accuracy={similarity_score:.2%}"
                        )

                # Step 3: NO FILTERING BY QUERY-CONTENT SIMILARITY!
                # We'll filter later based on ANSWER-CONTENT similarity
                # This allows LLM to use any relevant information

                self.logger.info(
                    f"📚 Skipping query-content filtering. Using all {len(chain_sources)} LLM-retrieved sources."
                )
                self.logger.info(
                    f"⚡ Accuracy will be calculated based on ANSWER-content similarity (not query-content)."
                )

                # Store query-based scores for reference only (not for filtering)
                for compressed_doc in chain_sources:
                    chunk_id = compressed_doc.metadata.get("chunk_id")
                    if chunk_id and chunk_id in doc_scores_map:
                        distance_score, similarity_score = doc_scores_map[chunk_id]
                        chain_sources_with_scores[compressed_doc.page_content] = (
                            distance_score,
                            similarity_score,
                        )

                # Apply k limit if needed (but no similarity filtering!)
                if len(chain_sources) > k:
                    self.logger.info(
                        f"Limiting from {len(chain_sources)} to {k} sources (user's k setting)"
                    )
                    chain_sources = chain_sources[:k]

            self.logger.info(f"Final source count: {len(chain_sources)} (k={k})")

            if not chain_sources:
                return {
                    "response": "I couldn't find any sufficiently relevant information to answer your question.",
                    "sources": [],
                    "metadata": {
                        "source_count": 0,
                        "model": model_display_name,
                        "retrieval_method": "Contextual Compression + LLM Filtering",
                    },
                }

            # Extract source information from chain sources with accuracy scores
            sources = []
            total_accuracy = 0.0

            for i, doc in enumerate(chain_sources):
                # Try different metadata keys for source and page
                source = (
                    doc.metadata.get("source")
                    or doc.metadata.get("document_name")
                    or doc.metadata.get("filename")
                    or "Unknown"
                )

                page = (
                    doc.metadata.get("page")
                    or doc.metadata.get("chunk_index", 0) + 1
                    or "Unknown"
                )

                # Get accuracy score for this document
                accuracy_score = 0.60  # Default fallback - conservative estimate
                if doc.page_content in chain_sources_with_scores:
                    _, accuracy_score = chain_sources_with_scores[doc.page_content]
                    self.logger.debug(
                        f"Source {i+1} accuracy: {accuracy_score:.2%} from calculated score"
                    )
                else:
                    # If no score data available, use conservative default
                    self.logger.debug(
                        f"Source {i+1} accuracy: {accuracy_score:.2%} (default - no score data)"
                    )

                total_accuracy += accuracy_score

                # Extract metadata fields for structured JSON response
                doc_metadata = doc.metadata

                # Get document_id - use fallback strategy if missing
                document_id = doc_metadata.get("document_id", "")

                # If document_id is missing, try to generate it from available metadata
                # This is important for PubMed documents and legacy data
                if not document_id:
                    # Try pubmed_id first (for PubMed documents)
                    pubmed_id = doc_metadata.get("pubmed_id")
                    if pubmed_id:
                        document_id = str(pubmed_id)
                        self.logger.debug(
                            f"Using pubmed_id as document_id: {document_id}"
                        )
                    else:
                        # Try to extract ID from document_name (e.g., "PubMed_PMC7047764" -> "PMC7047764")
                        doc_name = (
                            doc_metadata.get("document_name")
                            or doc_metadata.get("source")
                            or ""
                        )
                        if doc_name:
                            # Extract ID patterns from document name
                            # Pattern 1: "PubMed_PMC7047764" -> "PMC7047764"
                            if "PMC" in doc_name:
                                pmc_match = re.search(r"PMC\d+", doc_name)
                                if pmc_match:
                                    document_id = pmc_match.group()
                                    self.logger.debug(
                                        f"Extracted document_id from PMC pattern: {document_id}"
                                    )
                            # Pattern 2: "PubMed_12345" -> "12345"
                            elif "PubMed_" in doc_name:
                                parts = doc_name.split("PubMed_")
                                if len(parts) > 1 and parts[1]:
                                    document_id = parts[1].split("_")[
                                        0
                                    ]  # Take first part after PubMed_
                                    self.logger.debug(
                                        f"Extracted document_id from PubMed pattern: {document_id}"
                                    )
                            # Pattern 3: Use document_name as-is if it looks like an ID
                            elif doc_name and not doc_name.lower().endswith(
                                (".pdf", ".docx", ".txt")
                            ):
                                document_id = doc_name
                                self.logger.debug(
                                    f"Using document_name as document_id: {document_id}"
                                )

                        # Last resort: generate a deterministic ID from chunk_id
                        # This ensures same document chunks get same document_id
                        if not document_id:
                            chunk_id = doc_metadata.get("chunk_id", "")
                            if chunk_id:
                                # Use first 8 characters of chunk_id as document_id (not ideal but better than empty)
                                document_id = (
                                    chunk_id[:8] if len(chunk_id) >= 8 else chunk_id
                                )
                                self.logger.warning(
                                    f"Generated document_id from chunk_id (not ideal): {document_id}. "
                                    f"Consider fixing metadata for chunk."
                                )

                document_name = (
                    doc_metadata.get("document_name")
                    or doc_metadata.get("source")
                    or doc_metadata.get("filename", "Unknown")
                )
                chunk_index = doc_metadata.get("chunk_index", 0)
                start_char = doc_metadata.get("start_char", 0)
                end_char = doc_metadata.get("end_char", 0)
                chunk_size = end_char - start_char if end_char > start_char else 0
                
                # Get the actual chunk_id from metadata (this is the real chunk UUID)
                actual_chunk_id = doc_metadata.get("chunk_id", "")
                
                # Also get page-specific position info
                start_char_in_page = doc_metadata.get("start_char_in_page", start_char)
                end_char_in_page = doc_metadata.get("end_char_in_page", end_char)

                # Determine if document is from new_db
                # Check if is_new_doc is explicitly set in metadata (from new_db_manager)
                # Otherwise use db_mode as fallback
                if "is_new_doc" in doc_metadata:
                    is_new_doc = doc_metadata.get("is_new_doc", False)
                elif self.db_mode == "new":
                    # In "new" mode, all documents are from new_db
                    is_new_doc = True
                elif self.db_mode == "current+new":
                    # In merged mode, documents from new_db typically don't have pubmed_id
                    # Documents from current_db usually have pubmed_id in metadata
                    is_new_doc = "pubmed_id" not in doc_metadata
                else:  # current mode
                    is_new_doc = False

                # Extract file type from document name or metadata
                file_type = doc_metadata.get("file_type", "")
                if not file_type:
                    # Try to extract from document name
                    doc_name_lower = document_name.lower()
                    if doc_name_lower.endswith(".pdf"):
                        file_type = "pdf"
                    elif doc_name_lower.endswith(".docx"):
                        file_type = "docx"
                    elif doc_name_lower.endswith(".txt"):
                        file_type = "txt"
                    else:
                        file_type = "unknown"

                # Get model provider
                model_provider = self.current_model_provider or "Unknown"

                # Create structured metadata JSON
                structured_metadata = {
                    # "Chunk_Id": doc_metadata.get("chunk_id", ""),  # Commented out as requested
                    "Document_Id": document_id,
                    "Document_Name": document_name,
                    "Chunk_Index": chunk_index,
                    "Start_Char": start_char,
                    "End_Char": end_char,
                    "Is_New_Doc": is_new_doc,
                    "File_Type": file_type,
                    "Chunk_Size": chunk_size,
                    "Model_Provider": model_provider,
                }

                # CRITICAL: Get original (uncompressed) content for highlighting
                # ContextualCompressionRetriever modifies doc.page_content
                # But we need the ORIGINAL content that matches the position metadata
                original_content = doc.page_content  # Default fallback
                if actual_chunk_id and original_content_map:
                    original_content = original_content_map.get(actual_chunk_id, doc.page_content)
                    if original_content != doc.page_content:
                        self.logger.debug(f"Using original content for chunk {actual_chunk_id} (compressed: {len(doc.page_content)} chars, original: {len(original_content)} chars)")

                source_info = {
                    "content": original_content,  # Use ORIGINAL content (matches highlight positions)
                    "original_content": original_content,  # Keep for backward compatibility
                    "metadata": doc.metadata,  # Keep original metadata for backward compatibility
                    "structured_metadata": structured_metadata,  # New structured metadata JSON
                    "page": page,
                    "source": source,
                    "chunk_id": actual_chunk_id if actual_chunk_id else str(i + 1),  # Use real chunk ID
                    "chunk_index_display": i + 1,  # Keep display index for backward compatibility
                    "start_char_in_page": start_char_in_page,  # Page-specific position
                    "end_char_in_page": end_char_in_page,  # Page-specific position
                    "accuracy_score": accuracy_score,  # Add accuracy score to each source
                }
                sources.append(source_info)

            # Get the answer from response FIRST
            answer_text = response.get("answer", "") or (
                response["chat_history"][-1].content
                if response.get("chat_history")
                else "No response generated"
            )

            # ==========================================
            # ANSWER-BASED ACCURACY CALCULATION
            # Calculate similarity between ANSWER and each source content
            # This shows which sources actually contributed to the answer
            # ==========================================
            self.logger.info(
                "📊 Calculating ANSWER-content similarity for accurate source attribution..."
            )

            try:
                import numpy as np

                # Get answer embedding once
                if hasattr(self.embeddings, "embed_query"):
                    answer_embedding = self.embeddings.embed_query(answer_text)
                else:
                    answer_embedding = self.embeddings.embed_documents([answer_text])[0]

                answer_vec = np.array(answer_embedding)
                answer_norm = np.linalg.norm(answer_vec)

                # Calculate answer-content similarities for each source
                for i, source in enumerate(sources):
                    content = source.get("content", "")

                    # Get content embedding
                    if hasattr(self.embeddings, "embed_query"):
                        content_embedding = self.embeddings.embed_query(content)
                    else:
                        content_embedding = self.embeddings.embed_documents([content])[
                            0
                        ]

                    content_vec = np.array(content_embedding)
                    content_norm = np.linalg.norm(content_vec)

                    # Cosine similarity between answer and content
                    if answer_norm > 0 and content_norm > 0:
                        similarity = np.dot(answer_vec, content_vec) / (
                            answer_norm * content_norm
                        )
                    else:
                        similarity = 0.0

                    # Update accuracy with answer-content similarity
                    # This shows which sources LLM actually used
                    source["accuracy_score"] = max(0.0, min(1.0, similarity))

                    self.logger.debug(
                        f"  Source {i+1}: answer-content similarity={similarity:.3f}"
                    )

                # Sort sources by ANSWER-based accuracy (highest first)
                # Now top sources are those that contributed most to the answer
                sources.sort(key=lambda x: x["accuracy_score"], reverse=True)

                # Re-calculate metrics with answer-based scores
                total_accuracy = sum(s["accuracy_score"] for s in sources)
                avg_accuracy = total_accuracy / len(sources) if sources else 0.0

                # Re-assign chunk IDs after sorting
                for idx, source in enumerate(sources):
                    source["chunk_id"] = idx + 1

                self.logger.info(
                    f"✅ Answer-content similarity: "
                    f"Top={sources[0]['accuracy_score']:.1%}, "
                    f"Avg={avg_accuracy:.1%}, "
                    f"Bottom={sources[-1]['accuracy_score']:.1%}"
                )

            except Exception as e:
                self.logger.warning(
                    f"⚠️  Answer-content similarity calculation failed: {e}. "
                    f"Using default scores."
                )
                # Keep original scores if answer-based scoring fails
                avg_accuracy = total_accuracy / len(sources) if sources else 0.0

            # Check if LLM indicates insufficient information (more strict detection)
            # Only trigger if LLM explicitly states lack of info in specific patterns
            insufficient_info_phrases = [
                "i don't have enough information in the provided documents",
                "i don't have sufficient information in the provided documents",
                "the provided documents do not contain enough information",
                "the context does not contain enough information",
                "there is not enough information in the provided documents",
                "i cannot find any information in the provided documents",
                "the documents provided do not contain information",
                "completely unrelated to the question",
            ]

            answer_lower = answer_text.lower()
            # Only mark as insufficient if we have a clear, explicit statement
            has_insufficient_info = any(
                phrase in answer_lower for phrase in insufficient_info_phrases
            )

            # Additional check: if answer is too short (<50 chars) and mentions "no info", it's likely insufficient
            if not has_insufficient_info and len(answer_text.strip()) < 50:
                short_insufficient_phrases = [
                    "no information",
                    "cannot answer",
                    "don't know",
                ]
                has_insufficient_info = any(
                    phrase in answer_lower for phrase in short_insufficient_phrases
                )

            # If LLM says there's no info, don't show sources
            if has_insufficient_info:
                self.logger.info(
                    f"LLM indicated insufficient information. Clearing sources from response."
                )
                result = {
                    "response": answer_text,
                    "sources": [],
                    "metadata": {
                        "source_count": 0,
                        "model": model_display_name,
                        "retrieval_method": "Contextual Compression + LLM Filtering",
                        "average_accuracy": 0.0,
                        "cached": False,
                        "insufficient_info": True,
                    },
                }
            else:
                result = {
                    "response": answer_text,
                    "sources": sources,
                    "metadata": {
                        "source_count": len(sources),
                        "model": model_display_name,
                        "retrieval_method": "Contextual Compression + LLM Filtering",
                        "average_accuracy": avg_accuracy,
                        "cached": False,
                        "insufficient_info": False,
                    },
                }

            # Store in cache for future queries with retrieval parameters
            # This ensures same question with different k gets separate cache entries
            self.cache_manager.put(query, result, k=k)

            return result

        except Exception as e:
            self.logger.error(f"Failed to search documents with OpenAI: {str(e)}")
            return {
                "response": f"Error during search: {str(e)}",
                "sources": [],
                "metadata": {"source_count": 0},
            }

    def _save_vector_store(self) -> None:
        """Save vector store to disk"""
        if self.vector_store:
            try:
                # Change to vectorstore directory and save with simple name
                original_cwd = os.getcwd()
                try:
                    os.chdir(self.vectorstore_dir)
                    self.logger.info(f"Changed to directory: {os.getcwd()}")

                    # Extract just the filename from index_path if it's a full path
                    path_separators = ["/", os.sep]
                    has_path_separator = any(
                        sep in self.index_path for sep in path_separators
                    )
                    index_name = (
                        Path(self.index_path).name
                        if has_path_separator
                        else self.index_path
                    )
                    self.logger.info(f"Saving vector store with name: {index_name}")

                    self.vector_store.save_local(index_name)

                    # Verify files were created
                    # FAISS creates a directory with the index name containing index.faiss and index.pkl
                    faiss_file = f"{index_name}/index.faiss"
                    pkl_file = f"{index_name}/index.pkl"
                    self.logger.info(
                        f"After save - FAISS file exists: {os.path.exists(faiss_file)}"
                    )
                    self.logger.info(
                        f"After save - PKL file exists: {os.path.exists(pkl_file)}"
                    )

                finally:
                    os.chdir(original_cwd)
                    self.logger.info(f"Restored working directory: {os.getcwd()}")

                self.logger.info(
                    f"Successfully saved vector store: {Path(self.index_path).name if ('/' in self.index_path or os.sep in self.index_path) else self.index_path}"
                )
            except Exception as e:
                self.logger.error(f"Failed to save vector store: {str(e)}")
                raise

    def set_db_mode(self, mode: str) -> None:
        """
        Set database mode: "current", "new", or "current+new"

        Args:
            mode: Database mode string
        """
        self.db_mode = mode
        self.logger.info(f"Database mode set to: {mode}")

    def _get_current_db_path(self) -> Optional[Path]:
        """Get current_db path based on model provider"""
        # CRITICAL: Never return current_db path if db_mode is "new"
        if self.db_mode == "new":
            self.logger.debug("Skipping current_db path - db_mode is 'new'")
            return None

        if self.current_model_provider == "OpenAI (API)":
            path = CURRENT_DB_OPENAI_DIR / "faiss_index"
            self.logger.info(f"Using OpenAI current_db path: {path.absolute()}")
            return path
        elif self.current_model_provider == "Local LLM (Qwen)":
            path = CURRENT_DB_QWEN_DIR / "faiss_index"
            self.logger.info(f"Using Qwen current_db path: {path.absolute()}")
            return path
        else:
            # If provider not set, check both paths and return the one that exists
            openai_path = CURRENT_DB_OPENAI_DIR / "faiss_index"
            qwen_path = CURRENT_DB_QWEN_DIR / "faiss_index"

            openai_exists = (openai_path / "index.faiss").exists() and (
                openai_path / "index.pkl"
            ).exists()
            qwen_exists = (qwen_path / "index.faiss").exists() and (
                qwen_path / "index.pkl"
            ).exists()

            if openai_exists:
                self.logger.info(
                    f"Provider not set, but OpenAI DB exists. Using: {openai_path.absolute()}"
                )
                return openai_path
            elif qwen_exists:
                self.logger.info(
                    f"Provider not set, but Qwen DB exists. Using: {qwen_path.absolute()}"
                )
                return qwen_path
            else:
                # Default to OpenAI path if neither exists
                self.logger.debug(
                    f"Provider not set and no DB found. Defaulting to OpenAI path: {openai_path.absolute()}"
                )
                return openai_path

    def _load_vector_store(self, custom_path: Optional[Path] = None) -> bool:
        """
        Load vector store from disk

        Args:
            custom_path: Optional custom path to load from (for current_db)
        """
        # CRITICAL: Never load current_db if db_mode is "new"
        if self.db_mode == "new":
            self.logger.debug("Skipping vector store load - db_mode is 'new'")
            return False

        try:
            # Determine which path to use
            if custom_path:
                load_dir = custom_path.parent
                index_name = custom_path.name
            elif self.db_mode == "current":
                current_db_path = self._get_current_db_path()
                if current_db_path:
                    load_dir = current_db_path.parent
                    index_name = current_db_path.name
                else:
                    load_dir = self.vectorstore_dir
                    index_name = self.index_path
            elif self.db_mode == "current+new":
                # For current+new, still load current_db (but merge only in Docker)
                current_db_path = self._get_current_db_path()
                if current_db_path:
                    load_dir = current_db_path.parent
                    index_name = current_db_path.name
                else:
                    load_dir = self.vectorstore_dir
                    index_name = self.index_path
            else:
                # Extract index name from path
                path_separators = ["/", os.sep]
                has_path_separator = any(
                    sep in self.index_path for sep in path_separators
                )
                index_name = (
                    Path(self.index_path).name
                    if has_path_separator
                    else self.index_path
                )
                load_dir = self.vectorstore_dir

            # Check files in the correct directory
            original_cwd = os.getcwd()
            try:
                os.chdir(load_dir)

                # FAISS creates a directory with the index name containing index.faiss and index.pkl
                faiss_file = f"{index_name}/index.faiss"
                pkl_file = f"{index_name}/index.pkl"

                self.logger.info(
                    f"Attempting to load vector store with name: {index_name}"
                )
                self.logger.info(f"Vector store directory: {load_dir}")
                self.logger.info(f"Looking for files: {faiss_file}, {pkl_file}")
                self.logger.info(f"FAISS file exists: {os.path.exists(faiss_file)}")
                self.logger.info(f"PKL file exists: {os.path.exists(pkl_file)}")
                self.logger.info(f"Embeddings available: {self.embeddings is not None}")

                if os.path.exists(faiss_file) and os.path.exists(pkl_file):
                    # Check if embeddings are initialized
                    if not self.embeddings:
                        self.logger.warning(
                            "Embeddings not initialized. Please initialize embeddings before loading vector store."
                        )
                        return False

                    # Load vector store with existing embeddings
                    self.vector_store = FAISS.load_local(
                        index_name,
                        self.embeddings,
                        allow_dangerous_deserialization=True,
                    )
                    # CRITICAL: Ensure embeddings are set on the vector store after loading
                    # This is important for similarity search to work correctly
                    if hasattr(self.vector_store, "embeddings"):
                        self.vector_store.embeddings = self.embeddings
                    self.logger.info(f"Successfully loaded vector store: {index_name}")

                    # Verify vector store is properly initialized
                    if self.vector_store:
                        # Check if vector store has embeddings attribute
                        if (
                            hasattr(self.vector_store, "embeddings")
                            and self.vector_store.embeddings
                        ):
                            embedding_model = getattr(
                                self.vector_store.embeddings, "model", "unknown"
                            )
                            self.logger.info(
                                f"Vector store embeddings model: {embedding_model}"
                            )
                        # Check vector count
                        if hasattr(self.vector_store, "index") and hasattr(
                            self.vector_store.index, "ntotal"
                        ):
                            self.logger.info(
                                f"Vector store contains {self.vector_store.index.ntotal} vectors"
                            )

                    # For current_db mode, calculate and cache unique pubmed_id count when first loaded
                    # Only calculate if cache is not already set (preserve existing cache)
                    if (
                        self.db_mode == "current"
                        and self._cached_total_documents is None
                    ):
                        unique_pubmed_count = self._count_unique_pubmed_ids()
                        if unique_pubmed_count > 0:
                            self._cached_total_documents = unique_pubmed_count
                            self._save_cache_to_session(unique_pubmed_count)
                            self.logger.info(
                                f"Initialized cached total documents count: {unique_pubmed_count}"
                            )
                    elif (
                        self.db_mode == "current"
                        and self._cached_total_documents is not None
                    ):
                        # Cache already exists, don't recalculate
                        self.logger.debug(
                            f"Preserving existing cached total documents count: {self._cached_total_documents}"
                        )

                    return True
                else:
                    self.logger.debug(f"Vector store files not found: {index_name}")

            finally:
                os.chdir(original_cwd)

        except Exception as e:
            self.logger.error(f"Failed to load vector store: {str(e)}")

        return False

    def _save_document_metadata(self, documents: List[Document]) -> None:
        """Save document metadata to disk"""
        metadata_file = self.vectorstore_dir / "document_metadata.pkl"

        # Convert documents to serializable format
        doc_metadata = {}
        for doc in documents:
            doc_metadata[doc.id] = {
                "name": doc.name,
                "file_type": doc.file_type.value,
                "file_size": doc.file_size,
                "file_hash": doc.file_hash,
                "upload_timestamp": doc.upload_timestamp.isoformat(),
                "chunk_count": doc.chunk_count,
                "metadata": doc.metadata,
            }

        self.document_metadata = doc_metadata

        with open(metadata_file, "wb") as f:
            pickle.dump(doc_metadata, f)

        self.logger.debug(f"Saved document metadata for {len(documents)} documents")

    def _update_document_metadata(self, new_documents: List[Document]) -> None:
        """Update document metadata with new documents"""
        metadata_file = self.vectorstore_dir / "document_metadata.pkl"

        # Load existing metadata
        if metadata_file.exists():
            try:
                with open(metadata_file, "rb") as f:
                    self.document_metadata = pickle.load(f)
            except Exception as e:
                self.logger.warning(f"Failed to load existing metadata: {str(e)}")
                self.document_metadata = {}

        # Add new documents
        for doc in new_documents:
            self.document_metadata[doc.id] = {
                "name": doc.name,
                "file_type": doc.file_type.value,
                "file_size": doc.file_size,
                "file_hash": doc.file_hash,
                "upload_timestamp": doc.upload_timestamp.isoformat(),
                "chunk_count": doc.chunk_count,
                "metadata": doc.metadata,
            }

        # Save updated metadata
        with open(metadata_file, "wb") as f:
            pickle.dump(self.document_metadata, f)

        self.logger.debug(
            f"Updated document metadata with {len(new_documents)} new documents"
        )

    def _count_unique_pubmed_ids(self) -> int:
        """
        Count unique pubmed_id values from current_db vector store metadata.
        This is used for current_db mode to get the accurate document count.

        Returns:
            Number of unique pubmed_id values, or 0 if not available
        """
        if not self.vector_store:
            self.logger.warning("Vector store not loaded, cannot count pubmed_ids")
            return 0

        try:
            unique_pubmed_ids = set()
            total_chunks_checked = 0

            if hasattr(self.vector_store, "docstore") and self.vector_store.docstore:
                if hasattr(self.vector_store.docstore, "_dict"):
                    docstore_dict = self.vector_store.docstore._dict
                    total_chunks_checked = len(docstore_dict)

                    for doc_metadata in docstore_dict.values():
                        if hasattr(doc_metadata, "metadata"):
                            metadata = doc_metadata.metadata
                            # Get pubmed_id from metadata
                            pubmed_id = metadata.get("pubmed_id")
                            if pubmed_id:
                                unique_pubmed_ids.add(str(pubmed_id))

            count = len(unique_pubmed_ids)

            # Validate result: if we have chunks but no pubmed_ids, something is wrong
            if total_chunks_checked > 0 and count == 0:
                self.logger.warning(
                    f"Found {total_chunks_checked} chunks but 0 unique pubmed_ids. "
                    f"This may indicate metadata issue. Check if pubmed_id field exists in metadata."
                )
            elif count > 0:
                self.logger.info(
                    f"Found {count} unique pubmed_id values from {total_chunks_checked} chunks in current_db"
                )
            else:
                self.logger.debug(
                    f"No pubmed_ids found (checked {total_chunks_checked} chunks)"
                )

            return count

        except Exception as e:
            self.logger.error(f"Error counting unique pubmed_ids: {e}")
            return 0

    def _get_store_stats(self) -> Tuple[int, int]:
        """Get total documents and chunks count"""
        if not self.document_metadata:
            metadata_file = Path(self.vectorstore_dir) / "document_metadata.pkl"
            if metadata_file.exists():
                try:
                    with open(metadata_file, "rb") as f:
                        self.document_metadata = pickle.load(f)
                except Exception:
                    self.document_metadata = {}

        total_docs = len(self.document_metadata)
        total_chunks = sum(
            doc.get("chunk_count", 0) for doc in self.document_metadata.values()
        )

        # If we have a loaded vector store, get actual chunk count
        if self.vector_store:
            try:
                if (
                    hasattr(self.vector_store, "docstore")
                    and self.vector_store.docstore
                ):
                    actual_chunks = len(self.vector_store.docstore._dict)
                elif hasattr(self.vector_store, "index") and hasattr(
                    self.vector_store.index, "ntotal"
                ):
                    actual_chunks = self.vector_store.index.ntotal
                else:
                    actual_chunks = 0

                self.logger.debug(
                    f"Metadata chunks: {total_chunks}, Actual chunks: {actual_chunks}"
                )
                # Use the actual count if available and greater than metadata count
                if actual_chunks > 0:
                    total_chunks = max(total_chunks, actual_chunks)
            except Exception as e:
                self.logger.debug(f"Could not get actual chunk count: {e}")

        return total_docs, total_chunks

    def _get_embedding_model_name(self) -> str:
        """Get current embedding model name"""
        if self.current_model_provider == "OpenAI (API)":
            return config.model.openai_embedding_model
        elif self.current_model_provider == "Local LLM (Qwen)":
            return config.model.local_embedding_model
        else:
            return "Unknown"

    def _get_index_size(self) -> int:
        """Get index file size in bytes"""
        try:
            # Determine which path to check based on db_mode
            if self.db_mode == "current":
                current_db_path = self._get_current_db_path()
                if current_db_path:
                    faiss_dir = current_db_path
                else:
                    # Fallback to regular vectorstore
                    path_separators = ["/", os.sep]
                    has_path_separator = any(
                        sep in self.index_path for sep in path_separators
                    )
                    index_name = (
                        Path(self.index_path).name
                        if has_path_separator
                        else self.index_path
                    )
                    faiss_dir = self.vectorstore_dir / index_name
            else:
                # Regular vectorstore path
                path_separators = ["/", os.sep]
                has_path_separator = any(
                    sep in self.index_path for sep in path_separators
                )
                index_name = (
                    Path(self.index_path).name
                    if has_path_separator
                    else self.index_path
                )
                faiss_dir = self.vectorstore_dir / index_name

            faiss_file = faiss_dir / "index.faiss"
            pkl_file = faiss_dir / "index.pkl"

            total_size = 0
            if faiss_file.exists():
                total_size += faiss_file.stat().st_size
            if pkl_file.exists():
                total_size += pkl_file.stat().st_size

            return total_size
        except Exception as e:
            self.logger.debug(f"Could not get index size: {e}")
        return 0

    def get_store_info(self) -> Optional[VectorStoreInfo]:
        """Get current vector store information"""
        # For current_db mode or current+new mode, use cached document count to avoid recalculation
        # The count should only change when documents are added, not when questions are asked
        if (
            self.db_mode in ["current", "current+new"]
            and self._cached_total_documents is not None
        ):
            # Use cached value - never recalculate if cache exists
            current_db_path = self._get_current_db_path()
            if current_db_path and (current_db_path / "index.faiss").exists():
                # Try to load vector store if not loaded yet (cache will be preserved)
                if not self.vector_store and self.embeddings:
                    loaded = self._load_vector_store(custom_path=current_db_path)
                    # _load_vector_store will preserve existing cache
                    self.logger.debug(
                        f"Vector store loaded, cache preserved: {self._cached_total_documents}"
                    )

                # Use cached value (always use cached value if it exists)
                if self._cached_total_documents is not None:
                    faiss_file = current_db_path / "index.faiss"
                    pkl_file = current_db_path / "index.pkl"
                    if faiss_file.exists() and pkl_file.exists():
                        # Get chunk count
                        total_chunks = 0
                        if self.vector_store:
                            try:
                                if (
                                    hasattr(self.vector_store, "docstore")
                                    and self.vector_store.docstore
                                ):
                                    if hasattr(self.vector_store.docstore, "_dict"):
                                        total_chunks = len(
                                            self.vector_store.docstore._dict
                                        )
                                elif hasattr(self.vector_store, "index") and hasattr(
                                    self.vector_store.index, "ntotal"
                                ):
                                    total_chunks = self.vector_store.index.ntotal
                            except Exception as e:
                                self.logger.debug(f"Could not get chunk count: {e}")

                        if total_chunks == 0:
                            # Estimate from file size
                            index_size = self._get_index_size()
                            if index_size > 0:
                                total_chunks = max(1, index_size // 4096)
                            else:
                                total_chunks = 2480  # Default fallback

                        return VectorStoreInfo(
                            index_path=str(current_db_path),
                            total_documents=self._cached_total_documents,
                            total_chunks=total_chunks,
                            embedding_model=self._get_embedding_model_name(),
                            index_size_bytes=self._get_index_size(),
                        )

        # Try to load vector store if embeddings are initialized
        # IMPORTANT: Only load if db_mode is "current" or "current+new" - never load for "new" mode
        # IMPORTANT: Only calculate cache if it doesn't exist - never overwrite existing cache!
        if (
            not self.vector_store
            and self.embeddings
            and self.db_mode in ["current", "current+new"]
        ):
            self.logger.debug("Vector store not loaded, attempting to load...")
            # Try loading from current_db if in current or current+new mode
            if self.db_mode in ["current", "current+new"]:
                current_db_path = self._get_current_db_path()
                if current_db_path:
                    loaded = self._load_vector_store(custom_path=current_db_path)
                    if loaded:
                        # Only calculate and cache unique pubmed_id count if cache doesn't exist
                        # This prevents overwriting cache when questions are asked
                        if self._cached_total_documents is None:
                            unique_count = self._count_unique_pubmed_ids()
                            if unique_count > 0:
                                self._cached_total_documents = unique_count
                                self._save_cache_to_session(unique_count)
                                self.logger.info(
                                    f"Initialized cached total documents count: {self._cached_total_documents}"
                                )
                            else:
                                self.logger.warning(
                                    f"Could not count unique pubmed_ids (returned {unique_count}), keeping cache as None"
                                )
                        else:
                            self.logger.debug(
                                f"Preserving existing cached total documents count: {self._cached_total_documents}"
                            )
                    self.logger.info(
                        f"Attempted to load current_db: {current_db_path}, loaded={loaded}"
                    )
                else:
                    loaded = self._load_vector_store()
            if not loaded:
                self.logger.debug(
                    "Failed to load vector store (this is OK if embeddings not initialized or db_mode is 'new')"
                )

        # Determine which path to check based on db_mode
        if self.db_mode in ["current", "current+new"]:
            current_db_path = self._get_current_db_path()
            if current_db_path:
                faiss_dir = current_db_path
                # For current_db, index_path should reflect the current_db location
                index_path_str = str(current_db_path)
                self.logger.debug(f"Checking current_db path: {faiss_dir}")
            else:
                # Fallback to regular vectorstore
                path_separators = ["/", os.sep]
                has_path_separator = any(
                    sep in self.index_path for sep in path_separators
                )
                index_name = (
                    Path(self.index_path).name
                    if has_path_separator
                    else self.index_path
                )
                faiss_dir = self.vectorstore_dir / index_name
                index_path_str = self.index_path
        else:
            # Regular vectorstore path
            path_separators = ["/", os.sep]
            has_path_separator = any(sep in self.index_path for sep in path_separators)
            index_name = (
                Path(self.index_path).name if has_path_separator else self.index_path
            )
            faiss_dir = self.vectorstore_dir / index_name
            index_path_str = self.index_path

        faiss_file = faiss_dir / "index.faiss"
        pkl_file = faiss_dir / "index.pkl"

        self.logger.debug(
            f"Checking FAISS files: {faiss_file} (exists={faiss_file.exists()}), {pkl_file} (exists={pkl_file.exists()})"
        )

        if faiss_file.exists() and pkl_file.exists():
            # Try to get stats from loaded vector store first
            total_docs = 0
            total_chunks = 0

            # If vector store is loaded, get stats directly from it
            if self.vector_store:
                try:
                    if (
                        hasattr(self.vector_store, "docstore")
                        and self.vector_store.docstore
                    ):
                        if hasattr(self.vector_store.docstore, "_dict"):
                            total_chunks = len(self.vector_store.docstore._dict)
                            # Count unique documents from metadata
                            # For current_db mode or current+new mode, count unique pubmed_id values
                            # For other modes, use source/document_name fields
                            if self.db_mode in ["current", "current+new"]:
                                # Only recalculate if cache is not set
                                # This prevents overwriting the cache when questions are asked
                                if self._cached_total_documents is None:
                                    # Count unique pubmed_id values (this is the accurate count for current_db)
                                    unique_pubmed_ids = set()
                                    for (
                                        doc_metadata
                                    ) in self.vector_store.docstore._dict.values():
                                        if hasattr(doc_metadata, "metadata"):
                                            metadata = doc_metadata.metadata
                                            pubmed_id = metadata.get("pubmed_id")
                                            if pubmed_id:
                                                unique_pubmed_ids.add(str(pubmed_id))
                                    total_docs = (
                                        len(unique_pubmed_ids)
                                        if unique_pubmed_ids
                                        else 0
                                    )
                                    # Cache this value so it doesn't change when questions are asked
                                    if total_docs > 0:
                                        self._cached_total_documents = total_docs
                                        self._save_cache_to_session(total_docs)
                                        self.logger.info(
                                            f"Cached total documents count from pubmed_id: {total_docs}"
                                        )
                                else:
                                    # Use cached value instead of recalculating
                                    total_docs = self._cached_total_documents
                                    self.logger.debug(
                                        f"Using cached total documents count: {total_docs}"
                                    )
                                self.logger.debug(
                                    f"Found {total_docs} unique documents from {total_chunks} chunks using pubmed_id (cached: {self._cached_total_documents is not None})"
                                )
                            else:
                                # For non-current_db modes, use source/document_name fields
                                unique_doc_sources = set()
                                for (
                                    doc_metadata
                                ) in self.vector_store.docstore._dict.values():
                                    if hasattr(doc_metadata, "metadata"):
                                        metadata = doc_metadata.metadata
                                        # Priority: source > document_name > document_id > filename
                                        # This matches what's shown in response information
                                        doc_source = (
                                            metadata.get("source")
                                            or metadata.get("document_name")
                                            or metadata.get("filename")
                                            or metadata.get("document_id")
                                        )
                                        if doc_source:
                                            unique_doc_sources.add(doc_source)
                                total_docs = (
                                    len(unique_doc_sources) if unique_doc_sources else 0
                                )
                                self.logger.debug(
                                    f"Found {total_docs} unique documents from {total_chunks} chunks using source/document_name fields"
                                )
                    if total_chunks == 0 and hasattr(self.vector_store, "index"):
                        if hasattr(self.vector_store.index, "ntotal"):
                            total_chunks = self.vector_store.index.ntotal
                except Exception as e:
                    self.logger.debug(f"Could not get stats from vector store: {e}")

            # If still 0, try to get from metadata (but skip for current_db as it may not have metadata file)
            if total_chunks == 0 and self.db_mode != "current":
                total_docs, total_chunks = self._get_store_stats()
            elif total_chunks == 0:
                # For current_db, try to estimate from file size or use a default
                # We can't read FAISS without embeddings, so use file size as indicator
                index_size = self._get_index_size()
                if index_size > 0:
                    # Estimate chunks based on file size (rough estimate: ~4KB per chunk)
                    # This is just for display, not accurate
                    total_chunks = max(1, index_size // 4096)
                    total_docs = 1  # Can't determine doc count without loading
                    self.logger.debug(
                        f"Estimated chunks from file size: {total_chunks}"
                    )

            # If we still have 0, but vector store is loaded, try one more time
            if total_chunks == 0 and self.vector_store:
                try:
                    if hasattr(self.vector_store.index, "ntotal"):
                        total_chunks = self.vector_store.index.ntotal
                        self.logger.debug(
                            f"Got chunk count from FAISS index: {total_chunks}"
                        )
                    elif hasattr(self.vector_store, "docstore") and hasattr(
                        self.vector_store.docstore, "_dict"
                    ):
                        total_chunks = len(self.vector_store.docstore._dict)
                        self.logger.debug(
                            f"Got chunk count from docstore: {total_chunks}"
                        )
                except Exception as e:
                    self.logger.warning(f"Could not get chunk count directly: {e}")

            # If we have chunks or docs, or if files exist (even without accurate counts), return info
            if (
                total_chunks > 0
                or total_docs > 0
                or (faiss_file.exists() and pkl_file.exists())
            ):
                # For current_db mode, prefer cached value if available
                if (
                    self.db_mode == "current"
                    and self._cached_total_documents is not None
                    and self._cached_total_documents > 0
                ):
                    final_total_docs = self._cached_total_documents
                else:
                    final_total_docs = total_docs if total_docs > 0 else 1

                return VectorStoreInfo(
                    index_path=index_path_str,
                    total_documents=final_total_docs,
                    total_chunks=(
                        total_chunks if total_chunks > 0 else 1
                    ),  # At least 1 if files exist
                    embedding_model=self._get_embedding_model_name(),
                    index_size_bytes=self._get_index_size(),
                )
        else:
            self.logger.debug(
                f"FAISS files not found at {faiss_dir}: {faiss_file.exists()=}, {pkl_file.exists()=}"
            )

        return None

    def clear_vector_store(self) -> None:
        """Clear vector store and metadata"""
        try:
            # Remove index files in directory format
            path_separators = ["/", os.sep]
            has_path_separator = any(sep in self.index_path for sep in path_separators)
            index_name = (
                Path(self.index_path).name if has_path_separator else self.index_path
            )

            # FAISS creates a directory with index files
            faiss_dir = self.vectorstore_dir / index_name
            metadata_file = self.vectorstore_dir / "document_metadata.pkl"

            # Remove FAISS directory if it exists
            if faiss_dir.exists():
                import shutil

                shutil.rmtree(faiss_dir)

            # Remove metadata file
            if metadata_file.exists():
                metadata_file.unlink()

            # Clear in-memory objects
            self.vector_store = None
            self.document_metadata = {}
            self._cached_total_documents = None  # Clear cached count
            self._save_cache_to_session(None)  # Clear from session state too

            self.logger.info("Cleared vector store and metadata")

        except Exception as e:
            self.logger.error(f"Failed to clear vector store: {str(e)}")
            raise

    def clear_cache(self) -> None:
        """Clear question-answer cache"""
        try:
            self.cache_manager.clear()
            self.logger.info("Cache cleared successfully")
        except Exception as e:
            self.logger.error(f"Failed to clear cache: {str(e)}")
            raise

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        try:
            return self.cache_manager.get_stats()
        except Exception as e:
            self.logger.error(f"Failed to get cache stats: {str(e)}")
            return {
                "enabled": False,
                "total_entries": 0,
                "hits": 0,
                "misses": 0,
                "hit_rate": 0.0,
            }


__all__ = ["VectorStoreManager"]
