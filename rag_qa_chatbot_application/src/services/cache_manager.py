"""
Cache manager service for RAG QA Chatbot Application
Implements semantic caching with similarity matching for question-answer pairs
Uses pure semantic (embedding-based) similarity for accurate matching
"""

import time
import hashlib
import pickle
import copy
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime, timedelta
from collections import OrderedDict
import numpy as np

from ..config import config
from ..config.settings import DATA_DIR
from ..utils import app_logger


class CacheEntry:
    """Cache entry containing question, answer, and metadata with semantic embedding"""

    def __init__(
        self,
        question: str,
        answer: Dict[str, Any],
        embedding: Optional[np.ndarray] = None,
    ):
        self.question = question
        self.answer = answer
        self.embedding = embedding  # Semantic embedding for similarity matching
        self.timestamp = datetime.now()
        self.hit_count = 0
        self.last_accessed = datetime.now()

    def is_expired(self, ttl: int) -> bool:
        """Check if cache entry is expired based on TTL"""
        return (datetime.now() - self.timestamp).total_seconds() > ttl

    def update_access(self):
        """Update last accessed time and hit count"""
        self.last_accessed = datetime.now()
        self.hit_count += 1


class CacheManager:
    """
    Semantic cache manager for storing and retrieving question-answer pairs
    Uses pure embedding-based semantic similarity for accurate matching
    """

    def __init__(self, embeddings=None):
        self.logger = app_logger
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.embeddings = embeddings

        # Cache configuration
        self.enable_cache = config.cache.enable_cache
        self.cache_ttl = config.cache.cache_ttl
        self.max_cache_size = config.cache.max_cache_size

        # Cache file path
        self.cache_dir = DATA_DIR / "cache"
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_file = self.cache_dir / "qa_cache.pkl"

        # Semantic similarity threshold for cache matching
        # VERY HIGH threshold to avoid false matches between related but different topics
        # e.g., "Climeworks" vs "Carbon Capture" should NOT match (they're different companies)
        # Set to 0.96 (96%) so only nearly identical questions match
        self.similarity_threshold = 0.90

        # Statistics
        self.stats = {"hits": 0, "misses": 0, "total_queries": 0, "cache_saves": 0}

        # Load existing cache if available
        self._load_cache()

        self.logger.info(
            f"Cache Manager initialized (enabled={self.enable_cache}, "
            f"ttl={self.cache_ttl}s, max_size={self.max_cache_size}, "
            f"similarity_threshold={self.similarity_threshold:.1%})"
        )

    def set_embeddings(self, embeddings):
        """Set embeddings model for semantic similarity"""
        self.embeddings = embeddings
        self.logger.info("Embeddings model set for cache manager")

    def _generate_cache_key(
        self, question: str, k: int = None, similarity_threshold: float = None
    ) -> str:
        """
        Generate cache key from question and retrieval parameters

        Including k and similarity_threshold in the key ensures that:
        - Same question with different retrieval settings gets separate cache entries
        - More precise cache matching and fewer false positives

        Args:
            question: Query question
            k: Number of documents to retrieve (optional)
            similarity_threshold: Similarity threshold for retrieval (optional)

        Returns:
            MD5 hash of the normalized question and parameters
        """
        # Normalize question
        normalized_q = question.lower().strip()

        # If parameters provided, include them in cache key for more precise matching
        if k is not None or similarity_threshold is not None:
            key_string = f"{normalized_q}|k={k}|threshold={similarity_threshold}"
        else:
            key_string = normalized_q

        return hashlib.md5(key_string.encode()).hexdigest()

    def _compute_similarity(
        self, embedding1: np.ndarray, embedding2: np.ndarray
    ) -> float:
        """
        Compute cosine similarity between two semantic embeddings

        This uses pure semantic similarity based on embeddings,
        which captures the actual meaning of questions.

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector

        Returns:
            Similarity score between 0 and 1 (1 = identical meaning, 0 = completely different)
        """
        try:
            if embedding1 is None or embedding2 is None:
                return 0.0

            # Normalize embeddings to unit vectors
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)

            if norm1 == 0 or norm2 == 0:
                return 0.0

            # Compute cosine similarity: dot product of normalized vectors
            # For text embeddings, result is typically in range [0, 1]
            # (embeddings are in positive space)
            similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)

            # Ensure result is strictly between 0 and 1
            # Clip to handle any numerical errors
            return float(max(0.0, min(1.0, similarity)))

        except Exception as e:
            self.logger.error(f"Error computing semantic similarity: {str(e)}")
            return 0.0

    def _get_question_embedding(self, question: str) -> Optional[np.ndarray]:
        """Get embedding for a question"""
        if not self.embeddings:
            return None

        try:
            # Generate embedding using the embeddings model
            # Check if it's OpenAI or Ollama embeddings
            if hasattr(self.embeddings, "embed_query"):
                embedding = self.embeddings.embed_query(question)
            elif hasattr(self.embeddings, "embed_documents"):
                # Fallback to embed_documents with a list
                embedding = self.embeddings.embed_documents([question])[0]
            else:
                self.logger.error(
                    "Embeddings model does not have embed_query or embed_documents method"
                )
                return None

            return np.array(embedding)
        except Exception as e:
            self.logger.error(f"Error generating embedding for question: {str(e)}")
            return None

    def _find_similar_question(
        self,
        question: str,
        question_embedding: Optional[np.ndarray],
        k: int = None,
        similarity_threshold: float = None,
    ) -> Optional[Tuple[str, CacheEntry, float]]:
        """
        Find most similar question in cache using pure semantic similarity

        Args:
            question: Query question
            question_embedding: Semantic embedding of the query question
            k: Number of documents parameter (for matching)
            similarity_threshold: Similarity threshold parameter (for matching)

        Returns:
            Tuple of (cache_key, cache_entry, similarity_score) or None
        """
        # First try exact match with parameters
        exact_cache_key = self._generate_cache_key(question, k, similarity_threshold)
        if exact_cache_key in self.cache:
            entry = self.cache[exact_cache_key]
            if not entry.is_expired(self.cache_ttl):
                self.logger.debug(
                    f"Exact cache match found with parameters k={k}, threshold={similarity_threshold}"
                )
                return (exact_cache_key, entry, 1.0)

        # If no embedding available, try exact match without parameters
        if question_embedding is None:
            cache_key = self._generate_cache_key(question)
            if cache_key in self.cache:
                entry = self.cache[cache_key]
                if not entry.is_expired(self.cache_ttl):
                    return (cache_key, entry, 1.0)
            return None

        best_match = None
        best_similarity = 0.0
        candidates = []

        # Search through cache for semantically similar questions
        for cache_key, entry in self.cache.items():
            # Skip expired entries
            if entry.is_expired(self.cache_ttl):
                continue

            # Skip entries without embeddings
            if entry.embedding is None:
                continue

            # Compute semantic similarity between question embeddings
            similarity = self._compute_similarity(question_embedding, entry.embedding)

            # Track candidates for debugging
            if similarity >= self.similarity_threshold * 0.8:  # Log near-matches too
                candidates.append((entry.question[:60], similarity))

            # Update best match if this is better and meets the STRICT threshold
            if similarity > best_similarity and similarity >= self.similarity_threshold:
                best_similarity = similarity
                best_match = (cache_key, entry, similarity)

        # Enhanced logging for debugging cache behavior
        if candidates:
            self.logger.debug(f"Cache similarity candidates for '{question[:60]}...':")
            for cand_q, cand_sim in sorted(
                candidates, key=lambda x: x[1], reverse=True
            )[:3]:
                marker = (
                    "✓ MATCH"
                    if cand_sim >= self.similarity_threshold
                    else "✗ Below threshold"
                )
                self.logger.debug(f"  - {cand_sim:.3f} {marker}: '{cand_q}...'")

        # Log best match
        if best_match:
            _, entry, sim = best_match
            self.logger.info(
                f"✓ Cache HIT: similarity={sim:.3f} (threshold={self.similarity_threshold:.3f}) "
                f"for question '{entry.question[:60]}...'"
            )
        else:
            self.logger.debug(
                f"✗ Cache MISS: No match above threshold {self.similarity_threshold:.3f} "
                f"for '{question[:60]}...'"
            )

        return best_match

    def get(
        self, question: str, k: int = 5, similarity_threshold: float = 0.9
    ) -> Optional[Dict[str, Any]]:
        """
        Get cached answer for a question or similar question using semantic similarity

        Args:
            question: User question
            k: Number of documents parameter (for cache key uniqueness)
            similarity_threshold: Similarity threshold parameter (for cache key uniqueness)

        Returns:
            Cached answer dictionary or None if not found
        """
        if not self.enable_cache:
            return None

        self.stats["total_queries"] += 1

        try:
            # Generate semantic embedding for the question
            question_embedding = self._get_question_embedding(question)

            # Find semantically similar question in cache
            # Pass retrieval parameters for more precise matching
            match = self._find_similar_question(
                question, question_embedding, k, similarity_threshold
            )

            if match:
                cache_key, entry, similarity = match

                # Update access statistics
                entry.update_access()

                # Move to end (most recently used)
                self.cache.move_to_end(cache_key)

                self.stats["hits"] += 1

                self.logger.info(
                    f"Cache HIT (semantic similarity={similarity:.2%}, hits={entry.hit_count}): '{question[:50]}...'"
                )

                # Return cached answer with metadata about cache hit
                cached_answer = copy.deepcopy(entry.answer)

                # Ensure metadata exists and is a dictionary
                if "metadata" not in cached_answer:
                    cached_answer["metadata"] = {}
                elif not isinstance(cached_answer["metadata"], dict):
                    cached_answer["metadata"] = {}

                # Add cache metadata without overwriting existing data
                cached_answer["metadata"]["cached"] = True
                cached_answer["metadata"]["cache_similarity"] = similarity
                cached_answer["metadata"]["cache_hit_count"] = entry.hit_count
                cached_answer["metadata"]["original_question"] = entry.question

                return cached_answer

            # Cache miss
            self.stats["misses"] += 1
            self.logger.debug(f"Cache MISS: '{question[:50]}...'")
            return None

        except Exception as e:
            self.logger.error(f"Error retrieving from cache: {str(e)}")
            return None

    def put(
        self,
        question: str,
        answer: Dict[str, Any],
        k: int = None,
        similarity_threshold: float = None,
    ) -> None:
        """
        Store question-answer pair in cache with semantic embedding

        Args:
            question: User question
            answer: Answer dictionary from RAG system
            k: Number of documents parameter (for cache key uniqueness)
            similarity_threshold: Similarity threshold parameter (for cache key uniqueness)
        """
        if not self.enable_cache:
            return

        try:
            # Generate cache key including retrieval parameters for precise matching
            cache_key = self._generate_cache_key(question, k, similarity_threshold)

            # Generate semantic embedding for similarity matching
            question_embedding = self._get_question_embedding(question)

            # Create cache entry with embedding
            entry = CacheEntry(
                question=question, answer=answer, embedding=question_embedding
            )

            # Check if cache is full (LRU eviction)
            if len(self.cache) >= self.max_cache_size:
                # Remove least recently used (first item in OrderedDict)
                removed_key = next(iter(self.cache))
                removed_entry = self.cache.pop(removed_key)

                self.logger.debug(
                    f"Cache full, removed LRU entry: '{removed_entry.question[:50]}...'"
                )

            # Add to cache
            self.cache[cache_key] = entry
            self.stats["cache_saves"] += 1

            self.logger.debug(
                f"Cached question-answer pair with semantic embedding: '{question[:50]}...'"
            )

            # Periodically save cache to disk (every 10 entries)
            if self.stats["cache_saves"] % 10 == 0:
                self._save_cache()

        except Exception as e:
            self.logger.error(f"Error storing in cache: {str(e)}")

    def clear(self) -> None:
        """Clear all cache entries"""
        self.cache.clear()
        self.stats = {"hits": 0, "misses": 0, "total_queries": 0, "cache_saves": 0}
        self._save_cache()
        self.logger.info("Cache cleared")

    def remove_expired(self) -> int:
        """
        Remove expired entries from cache

        Returns:
            Number of entries removed
        """
        expired_keys = [
            key for key, entry in self.cache.items() if entry.is_expired(self.cache_ttl)
        ]

        for key in expired_keys:
            del self.cache[key]

        if expired_keys:
            self.logger.info(f"Removed {len(expired_keys)} expired cache entries")

        return len(expired_keys)

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        # Remove expired entries before calculating stats
        self.remove_expired()

        hit_rate = (
            (self.stats["hits"] / self.stats["total_queries"] * 100)
            if self.stats["total_queries"] > 0
            else 0.0
        )

        return {
            "enabled": self.enable_cache,
            "total_entries": len(self.cache),
            "max_size": self.max_cache_size,
            "hits": self.stats["hits"],
            "misses": self.stats["misses"],
            "total_queries": self.stats["total_queries"],
            "hit_rate": hit_rate,
            "cache_saves": self.stats["cache_saves"],
            "ttl_seconds": self.cache_ttl,
        }

    def _save_cache(self) -> None:
        """Save cache to disk with semantic embeddings"""
        try:
            # Remove expired entries before saving
            self.remove_expired()

            cache_data = {
                "cache": dict(self.cache),
                "stats": self.stats,
                "timestamp": datetime.now(),
            }

            with open(self.cache_file, "wb") as f:
                pickle.dump(cache_data, f)

            self.logger.debug(
                f"Cache saved to disk ({len(self.cache)} entries with semantic embeddings)"
            )

        except Exception as e:
            self.logger.error(f"Error saving cache: {str(e)}")

    def _load_cache(self) -> None:
        """Load cache from disk with semantic embeddings"""
        if not self.cache_file.exists():
            self.logger.debug("No existing cache file found")
            return

        try:
            with open(self.cache_file, "rb") as f:
                cache_data = pickle.load(f)

            self.cache = OrderedDict(cache_data.get("cache", {}))
            self.stats = cache_data.get("stats", self.stats)

            # Remove expired entries
            expired_count = self.remove_expired()

            self.logger.info(
                f"Cache loaded from disk ({len(self.cache)} active entries, "
                f"{expired_count} expired entries removed)"
            )

        except Exception as e:
            self.logger.error(f"Error loading cache: {str(e)}")
            self.cache = OrderedDict()

    def __del__(self):
        """Save cache when object is destroyed"""
        try:
            if self.cache:
                self._save_cache()
        except:
            pass


__all__ = ["CacheManager", "CacheEntry"]
