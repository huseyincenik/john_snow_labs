"""
New Database Manager for handling temporary document databases
This manages databases for newly uploaded documents (not persisted)
"""

import os
from pathlib import Path
from typing import List, Optional
from langchain_community.vectorstores import FAISS
from langchain.schema import Document as LangchainDocument

from ..config import config
from ..models import Document, DocumentChunk
from ..utils import app_logger


class NewDBManager:
    """Manages temporary vector databases for new documents"""

    def __init__(self):
        self.logger = app_logger
        self.base_dir = Path(__file__).parent.parent.parent
        self.new_db_dir = self.base_dir / "data" / "new_db"
        self.new_db_dir.mkdir(parents=True, exist_ok=True)

        # In-memory vector stores (not persisted)
        self.new_vector_stores: dict[str, FAISS] = {}

    def create_new_vector_store(
        self,
        documents: List[Document],
        chunks: List[DocumentChunk],
        embeddings,
        provider_name: str,
    ) -> FAISS:
        """
        Create a new temporary vector store for uploaded documents

        Args:
            documents: List of Document objects
            chunks: List of DocumentChunk objects
            embeddings: Embeddings model
            provider_name: Model provider name (for identification)

        Returns:
            FAISS vector store
        """
        try:
            # Prepare texts and metadata
            texts = []
            metadatas = []

            for chunk in chunks:
                texts.append(chunk.content)

                # Find corresponding document
                doc = next((d for d in documents if d.id == chunk.document_id), None)
                doc_name = doc.name if doc else "Unknown"

                # Validate chunk has document_id
                if not chunk.document_id:
                    self.logger.warning(
                        f"Chunk {chunk.id} is missing document_id. "
                        f"Available document IDs: {[d.id for d in documents]}"
                    )

                # Start with chunk.metadata, then override with our critical fields
                # This ensures critical fields (document_id, chunk_id) are never overwritten
                metadata = {
                    **chunk.metadata,  # First spread chunk metadata
                    # Then override with critical fields (these will take precedence)
                    "chunk_id": chunk.id,
                    "document_id": chunk.document_id,
                    "document_name": doc_name,
                    "source": doc_name,
                    "page": chunk.metadata.get("page", chunk.chunk_index + 1),
                    "chunk_index": chunk.chunk_index,
                    "start_char": chunk.start_char,
                    "end_char": chunk.end_char,
                    "is_new_doc": True,  # Mark as new document
                }

                # Final validation: ensure document_id is set in metadata
                if not metadata.get("document_id"):
                    self.logger.error(
                        f"document_id is missing in metadata for chunk {chunk.id}. "
                        f"chunk.document_id={chunk.document_id}, "
                        f"chunk.metadata keys={list(chunk.metadata.keys())}"
                    )

                metadatas.append(metadata)

            if not texts:
                raise ValueError("No text chunks to create vector store")

            # Create FAISS vector store
            batch_size = 32

            if len(texts) <= batch_size:
                vector_store = FAISS.from_texts(
                    texts=texts,
                    embedding=embeddings,
                    metadatas=metadatas,
                    normalize_L2=True,
                )
            else:
                # Process in batches
                self.logger.info(
                    f"Processing {len(texts)} texts in batches of {batch_size}"
                )

                first_batch_texts = texts[:batch_size]
                first_batch_metas = metadatas[:batch_size]

                vector_store = FAISS.from_texts(
                    texts=first_batch_texts,
                    embedding=embeddings,
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
                    vector_store.add_texts(texts=batch_texts, metadatas=batch_metas)

            # Store in memory (not persisted)
            store_key = f"{provider_name}_new"
            self.new_vector_stores[store_key] = vector_store

            self.logger.info(
                f"Created new vector store with {len(texts)} chunks for {provider_name}"
            )
            return vector_store

        except Exception as e:
            self.logger.error(f"Failed to create new vector store: {str(e)}")
            raise

    def get_new_vector_store(self, provider_name: str) -> Optional[FAISS]:
        """
        Get existing new vector store for a provider

        Args:
            provider_name: Model provider name

        Returns:
            FAISS vector store or None
        """
        store_key = f"{provider_name}_new"
        return self.new_vector_stores.get(store_key)

    def add_to_new_vector_store(
        self,
        vector_store: FAISS,
        documents: List[Document],
        chunks: List[DocumentChunk],
    ) -> None:
        """
        Add new documents to existing new vector store

        Args:
            vector_store: Existing FAISS vector store
            documents: List of new Document objects
            chunks: List of new DocumentChunk objects
        """
        try:
            new_texts = []
            new_metadatas = []

            for chunk in chunks:
                new_texts.append(chunk.content)

                doc = next((d for d in documents if d.id == chunk.document_id), None)
                doc_name = doc.name if doc else "Unknown"

                # Start with chunk.metadata, then override with our critical fields
                # This ensures critical fields (document_id, chunk_id) are never overwritten
                metadata = {
                    **chunk.metadata,  # First spread chunk metadata
                    # Then override with critical fields (these will take precedence)
                    "chunk_id": chunk.id,
                    "document_id": chunk.document_id,
                    "document_name": doc_name,
                    "source": doc_name,
                    "page": chunk.metadata.get("page", chunk.chunk_index + 1),
                    "chunk_index": chunk.chunk_index,
                    "start_char": chunk.start_char,
                    "end_char": chunk.end_char,
                    "is_new_doc": True,
                }
                new_metadatas.append(metadata)

            if new_texts:
                vector_store.add_texts(texts=new_texts, metadatas=new_metadatas)
                self.logger.info(f"Added {len(new_texts)} new chunks to vector store")

        except Exception as e:
            self.logger.error(f"Failed to add to new vector store: {str(e)}")
            raise

    def clear_new_vector_store(self, provider_name: Optional[str] = None) -> None:
        """
        Clear new vector stores

        Args:
            provider_name: If provided, clear only for this provider.
                          If None, clear all new stores.
        """
        if provider_name:
            store_key = f"{provider_name}_new"
            if store_key in self.new_vector_stores:
                del self.new_vector_stores[store_key]
                self.logger.info(f"Cleared new vector store for {provider_name}")
        else:
            self.new_vector_stores.clear()
            self.logger.info("Cleared all new vector stores")

    def search_new_vector_store(
        self, query: str, vector_store: FAISS, k: int = 5
    ) -> List[tuple]:
        """
        Search in new vector store

        Args:
            query: Search query
            vector_store: FAISS vector store to search
            k: Number of results

        Returns:
            List of (document, score) tuples
        """
        try:
            results = vector_store.similarity_search_with_score(query, k=k)
            return results
        except Exception as e:
            self.logger.error(f"Failed to search new vector store: {str(e)}")
            return []


__all__ = ["NewDBManager"]
