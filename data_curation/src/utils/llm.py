"""LLM integration via OpenRouter (OpenAI-compatible)."""

from typing import Dict, Any, Optional, Tuple, Type

from openai import (
    APITimeoutError,
    APIConnectionError,
    AsyncOpenAI,
    RateLimitError,
)
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_exponential

from config.settings import settings


class LLMProvider:
    """Base class for LLM providers."""

    def __init__(self, provider_name: str, model_name: str):
        self.provider_name = provider_name
        self.model_name = model_name

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        response_format: Optional[Dict[str, Any]] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Generate response from LLM."""
        raise NotImplementedError


def _normalize_base_url(raw: Optional[str]) -> Optional[str]:
    """Return sanitized base URL or None if missing."""
    if raw is None:
        return None
    value = raw.strip().strip("'\"")
    if not value:
        return None
    return value


class OpenAIProvider(LLMProvider):
    """OpenAI (or OpenAI-compatible) chat completion provider."""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4-turbo-preview",
        base_url: Optional[str] = None,
        provider_name: str = "openai",
    ):
        client_kwargs: Dict[str, Any] = {"api_key": api_key}
        normalized_url = _normalize_base_url(base_url)
        if provider_name in {"openai", "qwen"}:
            normalized_url = normalized_url or settings.openrouter_api_base
            client_kwargs.setdefault(
                "default_headers",
                {
                    "HTTP-Referer": settings.openrouter_referer,
                    "X-Title": settings.openrouter_app_title,
                },
            )
        elif not normalized_url:
            normalized_url = "https://api.openai.com/v1"
        if normalized_url:
            client_kwargs["base_url"] = normalized_url
        self.client = AsyncOpenAI(**client_kwargs)
        self.model = model
        self.base_url = normalized_url
        super().__init__(provider_name=provider_name, model_name=model)

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        response_format: Optional[Dict[str, Any]] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Generate response using OpenAI API."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        timeout = settings.llm_request_timeout

        # Remove 'openrouter/' prefix from model name if base_url is OpenRouter
        # OpenRouter expects model names without the prefix when using their API
        model_name = self.model
        if self.base_url and "openrouter" in self.base_url.lower():
            if model_name.startswith("openrouter/"):
                model_name = model_name[len("openrouter/"):]

        kwargs = {
            "model": model_name,
            "messages": messages,
            "temperature": temperature,
            "timeout": timeout,
        }

        if response_format:
            kwargs["response_format"] = response_format

        if max_tokens:
            kwargs["max_tokens"] = max_tokens

        # Add extra_body for Qwen models (e.g., reasoning_effort)
        if extra_body:
            kwargs["extra_body"] = extra_body

        response = await self._create_completion_with_retry(**kwargs)
        content = response.choices[0].message.content
        
        # Handle cases where content might be None (e.g., reasoning mode, tool calls)
        if content is None:
            # Check for reasoning_content (Qwen 3 reasoning mode)
            if hasattr(response.choices[0].message, "reasoning_content") and response.choices[0].message.reasoning_content:
                # If only reasoning_content exists, return empty string (should not happen with reasoning disabled)
                return ""
            # Check for tool_calls
            if hasattr(response.choices[0].message, "tool_calls") and response.choices[0].message.tool_calls:
                # Tool calls not expected for tagger, return empty
                return ""
            # If content is None and no alternatives, raise error
            raise ValueError(f"LLM response content is None. Finish reason: {response.choices[0].finish_reason}")
        
        return content

    async def _create_completion_with_retry(self, **kwargs):
        """Execute chat completion with retry/backoff on transient failures."""
        retryable: Tuple[Type[BaseException], ...] = (
            APIConnectionError,
            APITimeoutError,
            RateLimitError,
        )
        attempts = max(1, settings.llm_retry_attempts)
        
        # Log the request for debugging
        import logging
        logger = logging.getLogger(__name__)
        logger.debug(f"LLM API call: model={kwargs.get('model')}, messages_count={len(kwargs.get('messages', []))}")
        
        try:
            if attempts == 1:
                response = await self.client.chat.completions.create(**kwargs)
            else:
                retrying = AsyncRetrying(
                    reraise=True,
                    stop=stop_after_attempt(attempts),
                    wait=wait_exponential(multiplier=1, min=1, max=10),
                    retry=retry_if_exception_type(retryable),
                )
                async for attempt in retrying:
                    with attempt:
                        response = await self.client.chat.completions.create(**kwargs)
                        break
            
            # Log response details for debugging
            if response and response.choices:
                logger.debug(f"LLM API response: finish_reason={response.choices[0].finish_reason}, has_content={response.choices[0].message.content is not None}")
            else:
                logger.error(f"LLM API response is empty or has no choices: {response}")
            
            return response
        except Exception as e:
            logger.error(f"LLM API call failed: {type(e).__name__}: {str(e)}")
            raise


def _resolve_openrouter_model(provider: str, override: Optional[str]) -> str:
    if override:
        return override
    normalized = provider.lower()
    if normalized == "openai":
        return settings.openrouter_model_openai
    if normalized == "qwen":
        return settings.openrouter_model_qwen
    raise ValueError(f"Unsupported LLM provider: {provider}")


def get_llm_provider(
    provider_name: Optional[str] = None,
    model_override: Optional[str] = None,
) -> LLMProvider:
    """Return an LLM provider instance based on configuration or overrides."""
    provider = (provider_name or settings.default_llm_provider).lower()
    model = _resolve_openrouter_model(provider, model_override)
    return OpenAIProvider(
        api_key=settings.openrouter_api_key,
        model=model,
        base_url=settings.openrouter_api_base,
        provider_name=provider,
    )
