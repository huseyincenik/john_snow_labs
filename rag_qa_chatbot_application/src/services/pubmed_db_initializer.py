"""
PubMed Database Initializer
Creates FAISS databases for PubMed dataset with both OpenAI and Qwen embeddings
"""

import os
from pathlib import Path
from typing import Optional
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings

from ..config import config, CURRENT_DB_OPENAI_DIR, CURRENT_DB_QWEN_DIR
from ..utils import app_logger
from .data_initializer import DataInitializer


class PubMedDBInitializer:
    """Initializes PubMed FAISS databases for both OpenAI and Qwen embeddings"""

    def __init__(self):
        self.logger = app_logger
        self.data_initializer = DataInitializer()
        self.openai_db_path = CURRENT_DB_OPENAI_DIR / "faiss_index"
        self.qwen_db_path = CURRENT_DB_QWEN_DIR / "faiss_index"

        # Log paths for debugging
        self.logger.info(f"Current DB OpenAI path: {self.openai_db_path.absolute()}")
        self.logger.info(f"Current DB Qwen path: {self.qwen_db_path.absolute()}")

        # Ensure directories exist (only creates if they don't exist)
        CURRENT_DB_OPENAI_DIR.mkdir(parents=True, exist_ok=True)
        CURRENT_DB_QWEN_DIR.mkdir(parents=True, exist_ok=True)
        self.logger.info(
            f"Current DB directories ensured: OpenAI={CURRENT_DB_OPENAI_DIR.exists()}, Qwen={CURRENT_DB_QWEN_DIR.exists()}"
        )

    def initialize_databases(
        self, openai_api_key: Optional[str] = None
    ) -> dict[str, bool]:
        """
        Initialize both OpenAI and Qwen databases

        Args:
            openai_api_key: OpenAI API key (if available)

        Returns:
            Dictionary with status for each database: {"openai": True/False, "qwen": True/False}
        """
        results = {"openai": False, "qwen": False}

        # Check if databases already exist
        openai_exists = self._check_db_exists(self.openai_db_path)
        qwen_exists = self._check_db_exists(self.qwen_db_path)

        self.logger.info(
            f"Database existence check - OpenAI: {openai_exists} at {self.openai_db_path.absolute()}, "
            f"Qwen: {qwen_exists} at {self.qwen_db_path.absolute()}"
        )

        if openai_exists and qwen_exists:
            self.logger.info(
                "Both PubMed databases already exist, skipping initialization"
            )
            return {"openai": True, "qwen": True}

        # Get PubMed documents
        self.logger.info("Initializing PubMed dataset...")
        df, documents = self.data_initializer.initialize_pubmed_dataset()

        if not documents:
            self.logger.error("Failed to get PubMed documents")
            return results

        self.logger.info(f"Got {len(documents)} PubMed documents")

        # Split documents into chunks using JohnSnowLabs-style splitter
        # Similar to embedding_retrieval.JohnSnowLabsLangChainCharSplitter
        chunk_size = 500
        chunk_overlap = 50
        splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )

        all_texts = []
        all_metadatas = []

        for doc in documents:
            chunks = splitter.split_text(doc.page_content)
            for chunk in chunks:
                if chunk.strip():
                    all_texts.append(chunk)
                    all_metadatas.append(doc.metadata.copy())

        self.logger.info(f"Created {len(all_texts)} chunks from PubMed documents")

        # Initialize OpenAI database if not exists
        if not openai_exists and openai_api_key:
            self.logger.info("Creating OpenAI PubMed database...")
            try:
                embeddings = OpenAIEmbeddings(
                    model=config.model.openai_embedding_model,
                    openai_api_key=openai_api_key,
                )
                results["openai"] = self._create_database(
                    all_texts, all_metadatas, embeddings, self.openai_db_path
                )
            except Exception as e:
                self.logger.error(f"Failed to create OpenAI database: {str(e)}")

        # Initialize Qwen database if not exists
        if not qwen_exists:
            self.logger.info("Creating Qwen PubMed database...")
            try:
                embeddings = OllamaEmbeddings(
                    model=config.model.local_embedding_model,
                    base_url=config.model.ollama_base_url.replace("/v1", ""),
                    num_ctx=2048,
                )
                results["qwen"] = self._create_database(
                    all_texts, all_metadatas, embeddings, self.qwen_db_path
                )
            except Exception as e:
                self.logger.error(f"Failed to create Qwen database: {str(e)}")
                self.logger.warning(
                    "Qwen database creation failed. This is OK if Ollama is not running yet."
                )

        return results

    def _check_db_exists(self, db_path: Path) -> bool:
        """Check if FAISS database already exists"""
        faiss_file = db_path / "index.faiss"
        pkl_file = db_path / "index.pkl"
        return faiss_file.exists() and pkl_file.exists()

    def _create_database(
        self, texts: list[str], metadatas: list[dict], embeddings, db_path: Path
    ) -> bool:
        """
        Create FAISS database at specified path

        Args:
            texts: List of text chunks
            metadatas: List of metadata dictionaries
            embeddings: Embeddings model
            db_path: Path to save the database

        Returns:
            True if successful, False otherwise
        """
        try:
            # Ensure parent directory exists (only creates if doesn't exist)
            db_path.parent.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Creating database at: {db_path.absolute()}")

            # Process in batches
            batch_size = 32

            if len(texts) <= batch_size:
                vector_store = FAISS.from_texts(
                    texts=texts,
                    embedding=embeddings,
                    metadatas=metadatas,
                    normalize_L2=True,
                )
            else:
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

            # Save to disk
            original_cwd = os.getcwd()
            try:
                os.chdir(db_path.parent)
                vector_store.save_local(db_path.name)
                self.logger.info(f"Saved database to {db_path}")
            finally:
                os.chdir(original_cwd)

            return True

        except Exception as e:
            self.logger.error(f"Failed to create database at {db_path}: {str(e)}")
            return False


__all__ = ["PubMedDBInitializer"]
