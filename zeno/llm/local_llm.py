"""
ZENO Local LLM Runtime - Ollama Client Abstraction

Responsibilities:
- Provide clean, reliable interface to Ollama-hosted LLMs
- Handle HTTP communication with local Ollama instance
- Parse and structure LLM responses
- Explicit error handling for all failure modes
- Offline-only operation (no cloud APIs)

Supported Models:
- Qwen 2.5 3B Instruct (q4_0): Low-RAM conversational and planning (CURRENT RECOMMENDED - ~2-3GB RAM)
- Llama 3.2 3B: Low-RAM alternative (deprecated due to compatibility issues)
- Mistral 7B: High-quality conversation, reasoning, planning (requires ~8GB RAM)
- Qwen 2.5 Coder: Code generation & debugging (Phase 4+)

Future Extensions:
- Streaming support (not implemented in Phase 2)
- Multi-turn conversation handling (agent responsibility)
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import json

# Use urllib for HTTP to avoid external dependencies
import urllib.request
import urllib.error
from urllib.parse import urljoin

logger = logging.getLogger(__name__)


# Model Constants (Recommended for different use cases)
MISTRAL_7B = "mistral"  # 7B - High RAM requirement (~8GB) - legacy
QWEN_CODER = "qwen2.5-coder"  # For code generation (Phase 4+)
LLAMA3_3B = "llama3.2:3b"  # 3B - Low RAM - deprecated, llama3.2 has compatibility issues
QWEN_3B_INSTRUCT = "qwen2.5:3b-instruct-q4_0"  # 3B quantized - CURRENT RECOMMENDED MODEL
LLAMA_1B_INSTRUCT = "llama3.2:1b"               # 1B - Ultra-fast chat model


class OllamaError(Exception):
    """Base exception for all Ollama-related errors"""
    pass


class OllamaConnectionError(OllamaError):
    """Raised when cannot connect to Ollama server"""
    pass


class OllamaModelNotFoundError(OllamaError):
    """Raised when requested model is not available"""
    pass


class OllamaTimeoutError(OllamaError):
    """Raised when request exceeds timeout"""
    pass


class OllamaInvalidResponseError(OllamaError):
    """Raised when Ollama returns malformed response"""
    pass


@dataclass
class LLMResponse:
    """
    Structured response from LLM generation.
    
    Attributes:
        text: Generated text content
        model: Model name that generated the response
        duration_ms: Generation duration in milliseconds
        prompt_tokens: Number of tokens in prompt (if available)
        completion_tokens: Number of tokens in completion (if available)
        raw_response: Full Ollama response dict for debugging
    """
    text: str
    model: str
    duration_ms: float
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    raw_response: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate response"""
        if not self.text:
            logger.warning("LLMResponse created with empty text")
        if self.duration_ms < 0:
            raise ValueError("duration_ms cannot be negative")


class LocalLLM:
    """
    Client abstraction for local Ollama LLM runtime.
    
    Provides offline-only LLM inference via Ollama's HTTP API.
    Designed for deterministic, reliable operation with explicit error handling.
    
    Example:
        >>> client = LocalLLM()
        >>> response = client.generate(
        ...     prompt="Explain quantum computing briefly",
        ...     model=MISTRAL_7B,
        ...     temperature=0.7
        ... )
        >>> print(response.text)
    """
    
    DEFAULT_BASE_URL = "http://localhost:11434"
    DEFAULT_TIMEOUT = 120  # seconds
    
    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        timeout: int = DEFAULT_TIMEOUT
    ):
        """
        Initialize Ollama client.
        
        Args:
            base_url: Ollama server URL (default: http://localhost:11434)
            timeout: Default request timeout in seconds (default: 120)
            
        Note:
            Does not validate connection on initialization (lazy failure).
            Use health_check() to verify Ollama is running.
        """
        self.base_url = base_url.rstrip('/')
        self.default_timeout = timeout
        
        logger.info(f"LocalLLM initialized (base_url={self.base_url}, timeout={timeout}s)")
    
    def generate(
        self,
        prompt: str,
        model: str,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[list[str]] = None,
        timeout: Optional[int] = None
    ) -> LLMResponse:
        """
        Generate text using specified model.
        
        Args:
            prompt: Input prompt text (caller must flatten any structure)
            model: Model name (use MISTRAL_7B or QWEN_CODER constants)
            temperature: Sampling temperature (0.0-1.0, higher = more random)
            top_p: Nucleus sampling threshold (0.0-1.0)
            max_tokens: Maximum tokens to generate
            stop: List of stop sequences
            timeout: Request timeout in seconds (overrides default)
            
        Returns:
            LLMResponse with generated text and metadata
            
        Raises:
            OllamaConnectionError: Cannot reach Ollama server
            OllamaModelNotFoundError: Model not found/pulled
            OllamaTimeoutError: Request exceeded timeout
            OllamaInvalidResponseError: Malformed response from Ollama
            OllamaError: Other Ollama-related errors
            
        Example:
            >>> response = client.generate(
            ...     prompt="Write a Python function to reverse a string",
            ...     model=QWEN_CODER,
            ...     temperature=0.3,
            ...     max_tokens=500
            ... )
        """
        if not prompt:
            raise ValueError("Prompt cannot be empty")
        if not model:
            raise ValueError("Model cannot be empty")
        
        # Build request payload
        request_timeout = timeout if timeout is not None else self.default_timeout
        payload = self._build_request_payload(
            prompt=prompt,
            model=model,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stop=stop
        )
        
        logger.info(f"Generating with model={model}, prompt_len={len(prompt)}, timeout={request_timeout}s")
        start_time = time.time()
        
        try:
            # Make HTTP request to Ollama
            raw_response = self._make_request(
                endpoint="/api/generate",
                payload=payload,
                timeout=request_timeout
            )
            
            # Parse response
            llm_response = self._parse_response(
                raw_response=raw_response,
                model=model,
                start_time=start_time
            )
            
            logger.info(
                f"Generation completed: model={model}, "
                f"duration={llm_response.duration_ms:.0f}ms, "
                f"response_len={len(llm_response.text)}"
            )
            
            return llm_response
            
        except OllamaError:
            # Re-raise Ollama errors as-is
            raise
        except Exception as e:
            logger.error(f"Unexpected error during generation: {e}", exc_info=True)
            raise OllamaError(f"Unexpected error: {e}") from e
    
    def _build_request_payload(
        self,
        prompt: str,
        model: str,
        temperature: Optional[float],
        top_p: Optional[float],
        max_tokens: Optional[int],
        stop: Optional[list[str]]
    ) -> Dict[str, Any]:
        """Build Ollama API request payload"""
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False  # Streaming not implemented in Phase 2
        }
        
        # Build options dict for generation parameters
        options = {}
        
        if temperature is not None:
            if not 0.0 <= temperature <= 2.0:
                raise ValueError(f"temperature must be between 0.0 and 2.0, got {temperature}")
            options["temperature"] = temperature
        
        if top_p is not None:
            if not 0.0 <= top_p <= 1.0:
                raise ValueError(f"top_p must be between 0.0 and 1.0, got {top_p}")
            options["top_p"] = top_p
        
        if max_tokens is not None:
            if max_tokens <= 0:
                raise ValueError(f"max_tokens must be positive, got {max_tokens}")
            options["num_predict"] = max_tokens  # Ollama uses num_predict
        
        if options:
            payload["options"] = options
        
        if stop:
            payload["stop"] = stop
        
        return payload
    
    def _make_request(
        self,
        endpoint: str,
        payload: Dict[str, Any],
        timeout: int
    ) -> Dict[str, Any]:
        """
        Make HTTP request to Ollama API.
        
        Args:
            endpoint: API endpoint (e.g., /api/generate)
            payload: Request payload dict
            timeout: Request timeout in seconds
            
        Returns:
            Parsed JSON response
            
        Raises:
            OllamaConnectionError: Connection failed
            OllamaTimeoutError: Request timed out
            OllamaModelNotFoundError: Model not found
            OllamaInvalidResponseError: Invalid response
        """
        url = urljoin(self.base_url, endpoint)
        
        # Prepare request
        data = json.dumps(payload).encode('utf-8')
        headers = {
            'Content-Type': 'application/json'
        }
        
        request = urllib.request.Request(
            url,
            data=data,
            headers=headers,
            method='POST'
        )
        
        try:
            # Make request with timeout
            with urllib.request.urlopen(request, timeout=timeout) as response:
                response_data = response.read().decode('utf-8')
                return json.loads(response_data)
                
        except urllib.error.HTTPError as e:
            # Handle HTTP errors
            error_body = ""
            try:
                error_body = e.read().decode('utf-8')
            except:
                pass
            
            if e.code == 404:
                # Model not found
                model = payload.get('model', 'unknown')
                raise OllamaModelNotFoundError(
                    f"Model '{model}' not found.\n"
                    f"Run: ollama pull {model}"
                ) from e
            else:
                raise OllamaError(
                    f"HTTP {e.code} error from Ollama: {error_body or e.reason}"
                ) from e
                
        except urllib.error.URLError as e:
            # Connection errors
            if "timed out" in str(e).lower() or isinstance(e.reason, TimeoutError):
                raise OllamaTimeoutError(
                    f"Request timed out after {timeout}s. "
                    f"Model may be large or server overloaded."
                ) from e
            else:
                raise OllamaConnectionError(
                    f"Cannot connect to Ollama at {self.base_url}. "
                    f"Ensure Ollama is running.\n"
                    f"Error: {e.reason}"
                ) from e
                
        except json.JSONDecodeError as e:
            raise OllamaInvalidResponseError(
                f"Invalid JSON response from Ollama: {e}"
            ) from e
        
        except Exception as e:
            # Catch-all for unexpected errors
            logger.error(f"Unexpected error in _make_request: {e}", exc_info=True)
            raise OllamaError(f"Unexpected request error: {e}") from e
    
    def _parse_response(
        self,
        raw_response: Dict[str, Any],
        model: str,
        start_time: float
    ) -> LLMResponse:
        """
        Parse Ollama response into LLMResponse.
        
        Args:
            raw_response: Raw JSON response from Ollama
            model: Model name used
            start_time: Request start timestamp
            
        Returns:
            Structured LLMResponse
            
        Raises:
            OllamaInvalidResponseError: Missing required fields
        """
        try:
            # Extract generated text
            text = raw_response.get("response", "")
            if not text and not raw_response.get("done", False):
                raise OllamaInvalidResponseError(
                    "Response missing 'response' field or incomplete"
                )
            
            # Calculate duration
            duration_ms = (time.time() - start_time) * 1000
            
            # Extract token counts if available
            prompt_tokens = None
            completion_tokens = None
            
            if "prompt_eval_count" in raw_response:
                prompt_tokens = raw_response["prompt_eval_count"]
            
            if "eval_count" in raw_response:
                completion_tokens = raw_response["eval_count"]
            
            return LLMResponse(
                text=text,
                model=model,
                duration_ms=duration_ms,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                raw_response=raw_response
            )
            
        except KeyError as e:
            raise OllamaInvalidResponseError(
                f"Response missing required field: {e}"
            ) from e
        except Exception as e:
            raise OllamaInvalidResponseError(
                f"Error parsing response: {e}"
            ) from e
    
    def health_check(self, timeout: Optional[int] = None) -> bool:
        """
        Check if Ollama server is reachable and healthy.
        
        Args:
            timeout: Request timeout in seconds (uses default if None)
            
        Returns:
            True if Ollama is reachable and responding
            
        Raises:
            OllamaConnectionError: Cannot reach Ollama
            
        Example:
            >>> client = LocalLLM()
            >>> if client.health_check():
            ...     print("Ollama is running")
        """
        check_timeout = timeout if timeout is not None else 5  # Shorter timeout for health check
        url = urljoin(self.base_url, "/api/tags")  # List models endpoint
        
        logger.info(f"Performing health check: {url}")
        
        try:
            request = urllib.request.Request(url, method='GET')
            with urllib.request.urlopen(request, timeout=check_timeout) as response:
                if response.status == 200:
                    logger.info("Health check passed - Ollama is running")
                    return True
                else:
                    raise OllamaConnectionError(
                        f"Ollama responded with status {response.status}"
                    )
                    
        except urllib.error.URLError as e:
            raise OllamaConnectionError(
                f"Ollama health check failed at {self.base_url}. "
                f"Ensure Ollama is running.\n"
                f"Error: {e.reason}"
            ) from e
        except Exception as e:
            raise OllamaConnectionError(
                f"Health check failed: {e}"
            ) from e
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get client configuration information.
        
        Returns:
            Dict with client settings
        """
        return {
            'base_url': self.base_url,
            'default_timeout': self.default_timeout,
            'supported_models': [QWEN_3B_INSTRUCT, LLAMA3_3B, MISTRAL_7B, QWEN_CODER],
            'recommended_low_ram': QWEN_3B_INSTRUCT,
            'streaming_supported': False  # Not in Phase 2
        }