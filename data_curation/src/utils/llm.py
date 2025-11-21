"""LLM integration for OpenAI and local models."""

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
        client_kwargs = {"api_key": api_key}
        normalized_url = _normalize_base_url(base_url)
        if not normalized_url and provider_name == "openai":
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
    ) -> str:
        """Generate response using OpenAI API."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Provider'a göre timeout seç
        timeout = settings.llm_request_timeout
        if self.provider_name == "qwen":
            timeout = settings.qwen_request_timeout

        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "timeout": timeout,
        }

        if response_format:
            kwargs["response_format"] = response_format

        if max_tokens:
            kwargs["max_tokens"] = max_tokens

        response = await self._create_completion_with_retry(**kwargs)
        return response.choices[0].message.content

    async def _create_completion_with_retry(self, **kwargs):
        """Execute chat completion with retry/backoff on transient failures."""
        retryable: Tuple[Type[BaseException], ...] = (
            APIConnectionError,
            APITimeoutError,
            RateLimitError,
        )
        attempts = max(1, settings.llm_retry_attempts)
        if attempts == 1:
            return await self.client.chat.completions.create(**kwargs)

        retrying = AsyncRetrying(
            reraise=True,
            stop=stop_after_attempt(attempts),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            retry=retry_if_exception_type(retryable),
        )
        async for attempt in retrying:
            with attempt:
                return await self.client.chat.completions.create(**kwargs)


def get_llm_provider(
    provider_name: Optional[str] = None,
    model_override: Optional[str] = None,
) -> LLMProvider:
    """Return an LLM provider instance based on configuration or overrides."""
    provider = (provider_name or settings.default_llm_provider).lower()
    if provider == "openai":
        return OpenAIProvider(
            api_key=settings.openai_api_key,
            model=model_override or settings.openai_model,
            base_url=None,
            provider_name="openai",
        )
    if provider == "qwen":
        return OpenAIProvider(
            api_key=settings.qwen_api_key,
            model=model_override or settings.qwen_model,
            base_url=settings.qwen_api_base,
            provider_name="qwen",
        )
    # Treat "local" provider as OpenAI-compatible server (Gemma, etc.)
    if provider == "local" or settings.use_local_llm:
        return OpenAIProvider(
            api_key=settings.local_api_key,
            model=model_override or settings.local_model_name,
            base_url=settings.local_api_base,
            provider_name="local",
        )
    raise ValueError(f"Unsupported LLM provider: {provider}")
