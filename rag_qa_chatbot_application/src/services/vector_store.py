"""
Vector store service using FAISS for RAG QA Chatbot Application
"""

import os
import pickle
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
import numpy as np

from langchain.embeddings import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document as LangchainDocument

from ..config import config
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

    def _filter_by_similarity_threshold(
        self, docs_with_scores: List[Tuple], similarity_threshold: float
    ) -> List[Tuple]:
        """
        Filter documents by similarity threshold

        Args:
            docs_with_scores: List of (document, distance_score) tuples
            similarity_threshold: Minimum similarity score (0.0 to 1.0)

        Returns:
            Filtered list of (document, distance_score) tuples
        """
        filtered_docs = []

        for doc, distance_score in docs_with_scores:
            similarity_score = self._convert_distance_to_similarity(distance_score)

            # Only include documents that meet the similarity threshold
            if similarity_score >= similarity_threshold:
                filtered_docs.append((doc, distance_score))

        return filtered_docs

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
            elif model_provider == "Local LLM (Qwen)":
                # For local LLM, use Ollama embeddings
                self.embeddings = OllamaEmbeddings(
                    model=config.model.local_embedding_model,  # Use all-minilm
                    base_url=config.model.ollama_base_url.replace(
                        "/v1", ""
                    ),  # Remove /v1 for Ollama
                    # Increase context window for embeddings (default 512 is too small)
                    num_ctx=2048,
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

                # Create store info
                store_info = VectorStoreInfo(
                    index_path=self.index_path,
                    total_documents=len(documents),
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
        similarity_threshold: float = 0.70,
        temperature: float = None,
        max_tokens: int = None,
    ) -> Dict[str, Any]:
        """
        Search documents using OpenAI approach with advanced retrieval techniques

        This method uses:
        1. Contextual Compression: Filters irrelevant content using LLM
        2. LLMChainExtractor: Extracts only query-relevant parts from documents
        3. Multi-stage filtering: Embedding similarity + LLM relevance check

        Args:
            query: User query
            api_key: OpenAI API key
            k: Number of documents to retrieve
            similarity_threshold: Minimum similarity score (0.0 to 1.0)
            temperature: LLM temperature
            max_tokens: Maximum tokens for LLM response

        Returns:
            Dictionary with response, sources, and metadata
        """
        # Check cache first
        cached_result = self.cache_manager.get(
            query, k=k, similarity_threshold=similarity_threshold
        )
        if cached_result:
            self.logger.info(f"Returning cached result for query: '{query[:50]}...'")
            return cached_result

        if not self.vector_store:
            self._load_vector_store()

        if not self.vector_store:
            self.logger.warning("No vector store available for search")
            return {
                "response": "No documents found to search.",
                "sources": [],
                "metadata": {"confidence": 0.0, "source_count": 0},
            }

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

            # Step 1: Create base retriever with MMR for diversity and better recall
            # MMR (Maximal Marginal Relevance) balances relevance and diversity
            # More aggressive MMR settings for better coverage
            base_retriever = self.vector_store.as_retriever(
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
                all_docs_with_scores = self.vector_store.similarity_search_with_score(
                    query, k=k * 3  # Get enough candidates to find all compressed docs
                )

                self.logger.info(
                    f"Original query search returned {len(all_docs_with_scores)} candidates"
                )

                # Step 2: Create mapping of chunk_id -> (distance, similarity) from original query results
                doc_scores_map = {}
                for orig_doc, distance_score in all_docs_with_scores:
                    chunk_id = orig_doc.metadata.get("chunk_id")
                    if chunk_id:
                        similarity_score = self._convert_distance_to_similarity(
                            distance_score
                        )
                        doc_scores_map[chunk_id] = (distance_score, similarity_score)
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

            self.logger.info(
                f"Final source count: {len(chain_sources)} (threshold={similarity_threshold:.2%}, k={k})"
            )

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

                source_info = {
                    "content": doc.page_content,  # Show full content instead of truncating
                    "metadata": doc.metadata,
                    "page": page,
                    "source": source,
                    "chunk_id": i + 1,
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
            # This ensures same question with different k/threshold gets separate cache entries
            self.cache_manager.put(
                query, result, k=k, similarity_threshold=similarity_threshold
            )

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

    def _load_vector_store(self) -> bool:
        """Load vector store from disk"""
        try:
            # Extract index name from path
            path_separators = ["/", os.sep]
            has_path_separator = any(sep in self.index_path for sep in path_separators)
            index_name = (
                Path(self.index_path).name if has_path_separator else self.index_path
            )

            # Check files in the correct directory
            original_cwd = os.getcwd()
            try:
                os.chdir(self.vectorstore_dir)

                # FAISS creates a directory with the index name containing index.faiss and index.pkl
                index_dir = index_name
                faiss_file = f"{index_name}/index.faiss"
                pkl_file = f"{index_name}/index.pkl"

                self.logger.info(
                    f"Attempting to load vector store with name: {index_name}"
                )
                self.logger.info(f"Vector store directory: {self.vectorstore_dir}")
                self.logger.info(f"Looking for files: {faiss_file}, {pkl_file}")
                self.logger.info(f"FAISS file exists: {os.path.exists(faiss_file)}")
                self.logger.info(f"PKL file exists: {os.path.exists(pkl_file)}")
                self.logger.info(f"Embeddings available: {self.embeddings is not None}")

                # Debug logs to verify state
                self.logger.debug(f"Embeddings object: {self.embeddings}")
                self.logger.debug(f"Vector store directory: {self.vectorstore_dir}")
                self.logger.debug(f"FAISS file exists: {os.path.exists(faiss_file)}")
                self.logger.debug(f"PKL file exists: {os.path.exists(pkl_file)}")

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
                    self.logger.info(f"Successfully loaded vector store: {index_name}")
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
            # Check for FAISS files in directory format
            path_separators = ["/", os.sep]
            has_path_separator = any(sep in self.index_path for sep in path_separators)
            index_name = (
                Path(self.index_path).name if has_path_separator else self.index_path
            )

            # FAISS creates a directory with index.faiss and index.pkl files
            faiss_dir = self.vectorstore_dir / index_name
            faiss_file = faiss_dir / "index.faiss"
            pkl_file = faiss_dir / "index.pkl"

            total_size = 0
            if faiss_file.exists():
                total_size += faiss_file.stat().st_size
            if pkl_file.exists():
                total_size += pkl_file.stat().st_size

            return total_size
        except Exception:
            pass
        return 0

    def get_store_info(self) -> Optional[VectorStoreInfo]:
        """Get current vector store information"""
        # Only try to load vector store if embeddings are initialized
        if not self.vector_store and self.embeddings:
            self.logger.debug("Vector store not loaded, attempting to load...")
            loaded = self._load_vector_store()
            if not loaded:
                self.logger.debug(
                    "Failed to load vector store (this is OK if embeddings not initialized)"
                )

        # Check for FAISS files in directory format
        path_separators = ["/", os.sep]
        has_path_separator = any(sep in self.index_path for sep in path_separators)
        index_name = (
            Path(self.index_path).name if has_path_separator else self.index_path
        )

        # FAISS creates a directory with index.faiss and index.pkl files
        faiss_dir = self.vectorstore_dir / index_name
        faiss_file = faiss_dir / "index.faiss"
        pkl_file = faiss_dir / "index.pkl"

        if faiss_file.exists() and pkl_file.exists():
            total_docs, total_chunks = self._get_store_stats()

            # If we still have 0 chunks but files exist, try to get count directly from vector store
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

            return VectorStoreInfo(
                index_path=self.index_path,
                total_documents=total_docs,
                total_chunks=total_chunks,
                embedding_model=self._get_embedding_model_name(),
                index_size_bytes=self._get_index_size(),
            )
        else:
            self.logger.debug(
                f"FAISS files not found: {faiss_file.exists()=}, {pkl_file.exists()=}"
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
