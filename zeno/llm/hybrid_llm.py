"""
ZENO Hybrid LLM - Kimi K2.5 (NVIDIA NIM) + LocalLLM Fallback

Phase 6: Hybrid LLM Routing

Purpose:
Route planning and code-generation prompts to Kimi K2.5 for higher quality,
fall back to LocalLLM (Ollama) if Kimi is unavailable or errors.

Rules:
- DO NOT remove or modify LocalLLM
- DO NOT modify ChatAgent behavior (chat role always uses LocalLLM)
- PlannerAgent uses HybridLLM(role="planner")
- DeveloperAgent uses HybridLLM(role="code")
- All other roles fall through to LocalLLM directly

Design:
- HybridLLM exposes the same .generate() interface as LocalLLM
  so it is a drop-in replacement with no agent changes needed.
- Kimi is attempted first for "planner" and "code" roles.
- Any exception from Kimi triggers silent fallback to LocalLLM.
- NVIDIA NIM endpoint is used (change KIMI_BASE_URL to suit your setup).
"""
import requests


import logging
import os
from pathlib import Path
from typing import Optional

from zeno.llm import LocalLLM, QWEN_3B_INSTRUCT

logger = logging.getLogger(__name__)

# ============================================================================
# Load .env file for Kimi credentials
# ============================================================================

def _load_env_file():
    """Load environment variables from .env file if it exists."""
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        try:
            from dotenv import load_dotenv
            load_dotenv(env_path)
            logger.debug(f"Loaded environment from {env_path}")
        except ImportError:
            logger.warning("python-dotenv not installed, trying manual parsing...")
            # Manual parsing if dotenv not available
            with open(env_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        os.environ[key.strip()] = value.strip()

_load_env_file()

# ============================================================================
# Kimi K2.5 configuration
# Set KIMI_API_KEY in your environment (or .env file).
# KIMI_BASE_URL: NVIDIA NIM endpoint (default below) or your own deployment.
# ============================================================================

KIMI_API_KEY  = os.environ.get("KIMI_API_KEY", "")
KIMI_BASE_URL = os.environ.get(
    "KIMI_BASE_URL",
    "https://integrate.api.nvidia.com/v1",   # NVIDIA NIM default
)
KIMI_MODEL = os.environ.get("KIMI_MODEL", "moonshotai/kimi-k2.5")

# Roles that use Kimi — everything else goes straight to LocalLLM
KIMI_ROLES = {"planner", "code"}


class KimiResponse:
    """Thin wrapper so HybridLLM returns the same shape as LocalLLM responses."""
    def __init__(self, text: str):
        self.text = text


class HybridLLM:
    """
    Drop-in replacement for LocalLLM that routes to Kimi K2.5 first.

    The .generate() signature is intentionally compatible with LocalLLM.generate()
    so PlannerAgent and DeveloperAgent need no changes beyond swapping their
    self.llm reference at construction time.

    Usage (in start.py):
        from zeno.llm.hybrid_llm import HybridLLM
        hybrid = HybridLLM(local_llm)
        planner = PlannerAgent(hybrid)
        developer_agent = DeveloperAgent(hybrid)
    """

    def __init__(self, local_llm: LocalLLM):
        """
        Args:
            local_llm: Existing LocalLLM instance (Ollama) used as fallback.
        """
        self._local = local_llm
        self._kimi_available = bool(KIMI_API_KEY)

        if self._kimi_available:
            logger.info(
                "HybridLLM initialized — Kimi K2.5 primary, LocalLLM fallback"
            )
        else:
            logger.warning(
                "HybridLLM: KIMI_API_KEY not set — will use LocalLLM for all roles. "
                "Set KIMI_API_KEY env var to enable Kimi."
            )

    def generate(
        self,
        prompt: str,
        model: str = QWEN_3B_INSTRUCT,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        kimi_timeout: int = 90,
        local_timeout: int = 120,
        role: str = "chat",          # NEW: callers pass role="planner" / role="code"
        **kwargs,
    ):
        """
        Generate a response.

        Routing:
        - role in {"planner", "code"} AND KIMI_API_KEY is set → try Kimi first
        - All other roles, or Kimi failure → LocalLLM

        Args:
            prompt:       Full prompt string.
            model:        Ollama model name (used only when falling back to LocalLLM).
            temperature:  Sampling temperature.
            max_tokens:   Token budget for the response.
            kimi_timeout:  Seconds before giving up on Kimi (default 90).
            local_timeout: Seconds before giving up on LocalLLM (default 120).
            role:         "planner", "code", or "chat".

        Returns:
            Object with a .text attribute containing the generated string.
        """
        # Strip caller-supplied 'timeout' from kwargs so it doesn't clash
        # with the explicit timeout= we pass to LocalLLM below
        kwargs.pop("timeout", None)

        if self._kimi_available and role in KIMI_ROLES:
            try:
                return self._kimi_generate(
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=kimi_timeout,
                    
                )
            except Exception as e:
                logger.warning(
                    "HybridLLM: Kimi failed for role=%s (%s) — falling back to LocalLLM",
                    role, e,
                )

        # Fallback (or non-Kimi role)
        return self._local.generate(
            prompt=prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=local_timeout,
            **kwargs,
        )

    def _kimi_generate(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
        timeout: int,
    ) -> KimiResponse:
        """
        Call Kimi K2.5 via NVIDIA NIM (OpenAI-compatible endpoint).

        Raises any exception on failure so the caller can fall back.
        """
        # Import here so a missing openai package doesn't break startup
        try:
            from openai import OpenAI
        except ImportError as e:
            raise RuntimeError(
                "openai package not installed. "
                "Run: pip install openai"
            ) from e

        logger.debug("HybridLLM: sending request to Kimi K2.5...")

        # =====================================================================
        # NVIDIA API Catalog requires raw HTTP instead of OpenAI SDK
        # =====================================================================
        if "integrate.api.nvidia.com" in KIMI_BASE_URL:

            invoke_url = f"{KIMI_BASE_URL}/chat/completions"

            headers = {
                "Authorization": f"Bearer {KIMI_API_KEY}",
                "Accept": "application/json",
                "Content-Type": "application/json",
            }

            payload = {
                "model": KIMI_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": max_tokens,
                "chat_template_kwargs": {"thinking": True},
                "stream": False,
            }

            resp = requests.post(invoke_url, headers=headers, json=payload, timeout=timeout)
            resp.raise_for_status()

            data = resp.json()
            text = data["choices"][0]["message"]["content"] or ""

        else:
            # Keep original OpenAI SDK logic for non-NVIDIA endpoints
            client = OpenAI(
                base_url=KIMI_BASE_URL,
                api_key=KIMI_API_KEY,
            )

            response = client.chat.completions.create(
                model=KIMI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
            )

            text = response.choices[0].message.content or ""

        logger.debug("HybridLLM: Kimi responded (%d chars)", len(text))
        return KimiResponse(text=text)