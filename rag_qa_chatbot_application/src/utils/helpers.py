"""
Utility functions for RAG QA Chatbot Application
"""
import hashlib
import time
from typing import Any, Dict, List, Optional
from functools import wraps
from ..utils.logger import app_logger


class SimpleCache:
    """Simple in-memory cache implementation"""

    def __init__(self, max_size: int = 100, ttl: int = 3600):
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.max_size = max_size
        self.ttl = ttl

    def _generate_key(self, *args, **kwargs) -> str:
        """Generate cache key from arguments"""
        key_data = str(args) + str(sorted(kwargs.items()))
        return hashlib.md5(key_data.encode()).hexdigest()

    def _is_expired(self, timestamp: float) -> bool:
        """Check if cache entry is expired"""
        return time.time() - timestamp > self.ttl

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        if key in self.cache:
            entry = self.cache[key]
            if not self._is_expired(entry['timestamp']):
                app_logger.debug(f"Cache hit for key: {key[:10]}...")
                return entry['value']
            else:
                del self.cache[key]
                app_logger.debug(f"Cache expired for key: {key[:10]}...")
        return None

    def set(self, key: str, value: Any) -> None:
        """Set value in cache"""
        if len(self.cache) >= self.max_size:
            # Remove oldest entry
            oldest_key = min(self.cache.keys(),
                             key=lambda k: self.cache[k]['timestamp'])
            del self.cache[oldest_key]

        self.cache[key] = {
            'value': value,
            'timestamp': time.time()
        }
        app_logger.debug(f"Cached value for key: {key[:10]}...")

    def clear(self) -> None:
        """Clear all cache"""
        self.cache.clear()
        app_logger.info("Cache cleared")


# Global cache instance
cache = SimpleCache()


def cached(ttl: int = 3600):
    """Decorator for caching function results"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            cache_key = cache._generate_key(func.__name__, *args, **kwargs)

            # Try to get from cache
            result = cache.get(cache_key)
            if result is not None:
                return result

            # Execute function and cache result
            result = func(*args, **kwargs)
            cache.set(cache_key, result)
            return result
        return wrapper
    return decorator


def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format"""
    if size_bytes == 0:
        return "0B"

    size_names = ["B", "KB", "MB", "GB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1

    return f"{size_bytes:.1f}{size_names[i]}"


def truncate_text(text: str, max_length: int = 100) -> str:
    """Truncate text to specified length"""
    if len(text) <= max_length:
        return text
    return text[:max_length-3] + "..."


def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe storage"""
    import re
    # Remove or replace invalid characters
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # Remove leading/trailing spaces and dots
    filename = filename.strip(' .')
    return filename


def validate_file_type(filename: str, allowed_extensions: tuple) -> bool:
    """Validate file type based on extension"""
    from pathlib import Path
    file_ext = Path(filename).suffix.lower()
    return file_ext in allowed_extensions


def get_file_hash(file_content: bytes) -> str:
    """Generate hash for file content"""
    return hashlib.sha256(file_content).hexdigest()


def retry_on_exception(max_retries: int = 3, delay: float = 1.0):
    """Decorator to retry function on exception"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        app_logger.warning(
                            f"Attempt {attempt + 1} failed for {func.__name__}: {str(e)}. "
                            f"Retrying in {delay} seconds..."
                        )
                        time.sleep(delay)
                    else:
                        app_logger.error(
                            f"All {max_retries} attempts failed for {func.__name__}: {str(e)}"
                        )

            raise last_exception
        return wrapper
    return decorator


def measure_execution_time(func):
    """Decorator to measure function execution time"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time

        app_logger.info(
            f"{func.__name__} executed in {execution_time:.2f} seconds")
        return result
    return wrapper


__all__ = [
    "SimpleCache",
    "cache",
    "cached",
    "format_file_size",
    "truncate_text",
    "sanitize_filename",
    "validate_file_type",
    "get_file_hash",
    "retry_on_exception",
    "measure_execution_time"
]
