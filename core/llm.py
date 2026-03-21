"""
core/llm.py — LLM factory supporting Groq, OpenAI, Ollama, and Anthropic.

Set LLM_PROVIDER in .env to switch between providers.
Supports:
    groq     — Groq cloud (default, fastest)
    openai   — OpenAI API
    anthropic — Anthropic Claude
    ollama   — Fully local via Ollama

.env examples:
    LLM_PROVIDER=groq
    GROQ_API_KEY=gsk_...

    LLM_PROVIDER=openai
    OPENAI_API_KEY=sk-...

    LLM_PROVIDER=anthropic
    ANTHROPIC_API_KEY=sk-ant-...

    LLM_PROVIDER=ollama
    OLLAMA_BASE_URL=http://localhost:11434
    OLLAMA_MODEL=llama3.1:70b

    LLM_PROVIDER=gemini
    GOOGLE_API_KEY=AIza...
"""
import os
import asyncio
import threading
from functools import lru_cache
from dotenv import load_dotenv

load_dotenv()

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").lower()
_OLLAMA_LOCK = threading.Lock()


class _SerializedLLM:
    """Serialize local-model calls so Ollama does not exhaust request slots."""

    def __init__(self, inner, sync_lock=None):
        self._inner = inner
        self._sync_lock = sync_lock

    def bind_tools(self, *args, **kwargs):
        bound = self._inner.bind_tools(*args, **kwargs)
        return _SerializedLLM(bound, self._sync_lock)

    def invoke(self, *args, **kwargs):
        if self._sync_lock is None:
            return self._inner.invoke(*args, **kwargs)
        with self._sync_lock:
            return self._inner.invoke(*args, **kwargs)

    async def ainvoke(self, *args, **kwargs):
        if self._sync_lock is None:
            return await self._inner.ainvoke(*args, **kwargs)
        await asyncio.to_thread(self._sync_lock.acquire)
        try:
            return await self._inner.ainvoke(*args, **kwargs)
        finally:
            self._sync_lock.release()

    def __getattr__(self, name):
        return getattr(self._inner, name)


@lru_cache(maxsize=16)
def get_llm(role: str = "default", temperature: float = 0.0):
    """
    Returns a cached LLM instance for the given role.
    Roles:
        "fast"    — small/cheap model for routing decisions
        "coder"   — powerful model for data analysis & code generation
        "default" — general purpose
    """
    providers = {
        "groq": _build_groq,
        "openai": _build_openai,
        "anthropic": _build_anthropic,
        "ollama": _build_ollama,
        "gemini": _build_gemini,
    }
    builder = providers.get(LLM_PROVIDER, _build_groq)
    return builder(role, temperature)


def _build_groq(role: str, temperature: float):
    from langchain_groq import ChatGroq
    model_map = {
        "fast":    os.getenv("GROQ_FAST_MODEL",  "llama-3.3-70b-versatile"),
        "coder":   os.getenv("GROQ_CODER_MODEL", "llama-3.3-70b-versatile"),
        "default": os.getenv("GROQ_MODEL",       "llama-3.3-70b-versatile"),
    }
    return ChatGroq(
        model=model_map.get(role, model_map["default"]),
        temperature=temperature,
        api_key=os.getenv("GROQ_API_KEY"),
    )


def _build_openai(role: str, temperature: float):
    try:
        from langchain_openai import ChatOpenAI
    except ImportError:
        raise ImportError("Run: pip install langchain-openai")
    model_map = {
        "fast":    os.getenv("OPENAI_FAST_MODEL",  "gpt-4o-mini"),
        "coder":   os.getenv("OPENAI_CODER_MODEL", "gpt-4o"),
        "default": os.getenv("OPENAI_MODEL",       "gpt-4o"),
    }
    return ChatOpenAI(
        model=model_map.get(role, model_map["default"]),
        temperature=temperature,
        api_key=os.getenv("OPENAI_API_KEY"),
    )


def _build_anthropic(role: str, temperature: float):
    try:
        from langchain_anthropic import ChatAnthropic
    except ImportError:
        raise ImportError("Run: pip install langchain-anthropic")
    model_map = {
        "fast":    os.getenv("ANTHROPIC_FAST_MODEL",  "claude-haiku-4-5-20251001"),
        "coder":   os.getenv("ANTHROPIC_CODER_MODEL", "claude-sonnet-4-6"),
        "default": os.getenv("ANTHROPIC_MODEL",       "claude-sonnet-4-6"),
    }
    return ChatAnthropic(
        model=model_map.get(role, model_map["default"]),
        temperature=temperature,
        api_key=os.getenv("ANTHROPIC_API_KEY"),
    )


def _build_ollama(role: str, temperature: float):
    try:
        from langchain_ollama import ChatOllama
    except ImportError:
        raise ImportError("Run: pip install langchain-ollama")
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    model_map = {
        "fast":    os.getenv("OLLAMA_FAST_MODEL",  os.getenv("OLLAMA_MODEL", "llama3.1:8b")),
        "coder":   os.getenv("OLLAMA_CODER_MODEL", os.getenv("OLLAMA_MODEL", "llama3.1:70b")),
        "default": os.getenv("OLLAMA_MODEL",       "llama3.1:70b"),
    }
    model = ChatOllama(
        model=model_map.get(role, model_map["default"]),
        base_url=base_url,
        temperature=temperature,
    )
    return _SerializedLLM(model, _OLLAMA_LOCK)


def _build_gemini(role: str, temperature: float):
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
    except ImportError:
        raise ImportError("Run: pip install langchain-google-genai")
    model_map = {
        "fast":    os.getenv("GEMINI_FAST_MODEL",  "gemini-1.5-flash"),
        "coder":   os.getenv("GEMINI_CODER_MODEL", "gemini-1.5-pro"),
        "default": os.getenv("GEMINI_MODEL",       "gemini-1.5-pro"),
    }
    return ChatGoogleGenerativeAI(
        model=model_map.get(role, model_map["default"]),
        temperature=temperature,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )
