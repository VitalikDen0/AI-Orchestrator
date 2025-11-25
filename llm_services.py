"""LLM services: llama.cpp wrapper and model management utilities."""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Lazy imports for llama-cpp-python
try:
    from llama_cpp import Llama  # type: ignore
    LLAMA_CPP_AVAILABLE = True
    logger.debug("llama-cpp-python is available")
except ImportError:
    Llama = None  # type: ignore
    LLAMA_CPP_AVAILABLE = False
    logger.warning("llama-cpp-python not available, LlamaCppWrapper will not work")


class LlamaCppWrapper:
    """Wrapper for llama-cpp-python that mimics LM Studio API."""

    def __init__(
        self,
        model_path: str,
        params: Dict[str, Any],
        logger_instance: Optional[logging.Logger] = None,
    ) -> None:
        """Initialize llama.cpp model wrapper.

        Args:
            model_path: Path to .gguf model file.
            params: Model parameters (n_ctx, n_gpu_layers, etc.).
            logger_instance: Logger for output messages.
        """
        self.logger = logger_instance or logger
        self.model_path = model_path
        self.params = params
        self.llm: Any = None
        self.model_id: Optional[str] = None
        self._is_loading = False
        self._load_error: Optional[str] = None

        self.logger.debug(
            "LlamaCppWrapper initialized: model_path=%s, params=%s",
            model_path,
            params,
        )

    def load_model(self) -> bool:
        """Load model into memory."""
        if self._is_loading:
            self.logger.warning("Model is already loading, skipping...")
            return False

        if self.llm is not None:
            self.logger.info("Model already loaded")
            return True

        try:
            self._is_loading = True
            self._load_error = None

            if not LLAMA_CPP_AVAILABLE:
                raise ImportError("llama-cpp-python is not installed")

            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model not found: {self.model_path}")

            model_basename = os.path.basename(self.model_path)
            self.logger.info("Loading model via llama.cpp: %s", model_basename)
            self.logger.info(
                "Parameters: n_ctx=%s, n_gpu_layers=%s",
                self.params.get("n_ctx"),
                self.params.get("n_gpu_layers"),
            )

            if Llama is None:
                raise ImportError("Llama class is unavailable")

            # Prepare parameters strictly from provided params dict
            llama_params = dict(self.params)
            llama_params["model_path"] = self.model_path
            llama_params.setdefault("verbose", False)

            load_start = time.time()
            self.llm = Llama(**llama_params)  # type: ignore
            load_duration = time.time() - load_start

            self.model_id = model_basename
            self.logger.info(
                "Model loaded successfully: %s (took %.2f seconds)",
                self.model_id,
                load_duration,
            )

            # Log context size
            context_size = self.llm.n_ctx()  # type: ignore
            self.logger.info("Context size: %d tokens", context_size)

            return True

        except Exception as e:
            self._load_error = str(e)
            self.logger.error("Failed to load model: %s", e, exc_info=True)
            self.llm = None
            return False
        finally:
            self._is_loading = False

    def unload_model(self) -> None:
        """Unload model from memory and free GPU resources."""
        if self.llm is not None:
            self.logger.info("Unloading model from memory...")
            del self.llm
            self.llm = None
            self.model_id = None

            # Force GPU memory cleanup if torch is available
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    self.logger.info("GPU memory cleared")
            except ImportError:
                self.logger.debug("Torch not available, skipping GPU cleanup")
            except Exception as exc:
                self.logger.warning("Failed to clear GPU memory: %s", exc)

    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.llm is not None

    def get_model_info(self) -> Dict[str, Any]:
        """Return model information (mimics LM Studio /v1/models)."""
        if not self.is_loaded():
            self.logger.debug("Model not loaded, returning empty info")
            return {"data": []}

        return {
            "data": [
                {
                    "id": self.model_id,
                    "object": "model",
                    "owned_by": "llama-cpp-python",
                    "permission": [],
                }
            ]
        }

    def get_context_info(self) -> Dict[str, int]:
        """Return model context information."""
        if not self.is_loaded():
            self.logger.warning("Model not loaded, returning zero context")
            return {"max_context": 0, "safe_context": 0}

        max_ctx = self.llm.n_ctx()  # type: ignore
        safe_ctx = int(max_ctx * 0.8)
        self.logger.debug("Context info: max=%d, safe=%d", max_ctx, safe_ctx)
        return {"max_context": max_ctx, "safe_context": safe_ctx}

    def create_chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = -1,
        stream: bool = False,
        top_p: float = 0.95,
        top_k: int = 40,
        repeat_penalty: float = 1.1,
    ) -> Dict[str, Any]:
        """Create chat completion (mimics LM Studio /v1/chat/completions).

        Args:
            messages: List of messages in OpenAI format.
            temperature: Generation temperature (0.0-2.0).
            max_tokens: Maximum token count (None or -1 = unlimited).
            stream: Stream generation (not yet supported).
            top_p: Top-p sampling parameter.
            top_k: Top-k sampling parameter.
            repeat_penalty: Repeat penalty.

        Returns:
            Response in OpenAI API format.

        Raises:
            RuntimeError: If model is not loaded.
        """
        if not self.is_loaded():
            raise RuntimeError("Model not loaded. Call load_model() first.")

        try:
            start_method = time.time()
            self.logger.info(
                "Starting generation: temp=%.2f, max_tokens=%s, messages=%d",
                temperature,
                max_tokens,
                len(messages),
            )

            # Log prompt size for debugging
            total_prompt_size = sum(len(str(m.get("content", ""))) for m in messages)
            self.logger.info("Total prompt size: %d characters", total_prompt_size)

            # Call llama.cpp for generation
            start_llm_call = time.time()
            self.logger.info("BEGIN llm.create_chat_completion()...")

            # llama.cpp expects None for unlimited tokens
            max_tokens_arg = (
                max_tokens if isinstance(max_tokens, int) and max_tokens > 0 else None
            )

            response = self.llm.create_chat_completion(  # type: ignore
                messages=messages,  # type: ignore
                temperature=temperature,
                max_tokens=max_tokens_arg,
                top_p=top_p,
                top_k=top_k,
                repeat_penalty=repeat_penalty,
                stream=stream,
            )

            end_llm_call = time.time()
            llm_duration = end_llm_call - start_llm_call

            # Extract token usage for speed calculation
            usage = response.get("usage", {})  # type: ignore
            completion_tokens = usage.get("completion_tokens", 0)
            prompt_tokens = usage.get("prompt_tokens", 0)

            tokens_per_sec = completion_tokens / llm_duration if llm_duration > 0 else 0

            self.logger.info(
                "END llm.create_chat_completion(): %.2f seconds", llm_duration
            )
            self.logger.info(
                "Tokens: prompt=%d, completion=%d, total=%d",
                prompt_tokens,
                completion_tokens,
                prompt_tokens + completion_tokens,
            )
            self.logger.info("Generation speed: %.1f tokens/sec", tokens_per_sec)

            total_duration = time.time() - start_method
            self.logger.info("Total method duration: %.2f seconds", total_duration)

            # Warn about low generation speed
            if tokens_per_sec < 50:
                self.logger.warning(
                    "LOW SPEED! Expected >100 tokens/sec, got %.1f", tokens_per_sec
                )
                self.logger.warning(
                    "Possible causes: GPU not used, large prompt, or slow model"
                )
            else:
                self.logger.info("Speed is normal - GPU working properly!")

            return response  # type: ignore

        except Exception as e:
            self.logger.error("Generation failed: %s", e, exc_info=True)
            raise

    def reconnect(self) -> bool:
        """Reconnect to model (reload)."""
        self.logger.info("Reconnecting to model (reload)...")
        self.unload_model()
        return self.load_model()


def is_model_running_lm_studio(lm_studio_url: str, model_name: str, logger_instance: Optional[logging.Logger] = None) -> bool:
    """Check if model is running in LM Studio via API.
    
    Args:
        lm_studio_url: LM Studio API URL (e.g., "http://localhost:1234").
        model_name: Model name or path fragment to search for.
        logger_instance: Logger for output messages.
        
    Returns:
        True if model is loaded and running.
    """
    log = logger_instance or logger
    try:
        import requests
        
        response = requests.get(f"{lm_studio_url}/v1/models", timeout=10)
        if response.status_code == 200:
            data = response.json()
            for model in data.get("data", []):
                if model_name in model.get("id", "") and model.get("isLoaded", False):
                    log.debug("Model %s is running", model_name)
                    return True
        log.debug("Model %s is not running", model_name)
        return False
    except Exception as e:
        log.error("Failed to check model status for %s: %s", model_name, e, exc_info=True)
        return False


def get_model_context_info_lm_studio(
    lm_studio_url: str,
    model_search_terms: List[str],
    default_context: int = 262144,
    logger_instance: Optional[logging.Logger] = None,
) -> Dict[str, int]:
    """Get model context information from LM Studio API.
    
    Args:
        lm_studio_url: LM Studio API URL.
        model_search_terms: List of terms to search for in model IDs.
        default_context: Default context size if not found.
        logger_instance: Logger for output messages.
        
    Returns:
        Dictionary with max_context and safe_context keys.
    """
    log = logger_instance or logger
    try:
        import requests
        
        response = requests.get(f"{lm_studio_url}/v1/models", timeout=10)
        if response.status_code == 200:
            data = response.json()
            
            # Search for target model by keywords
            target_model = None
            for model in data.get("data", []):
                model_id = model.get("id", "").lower()
                for term in model_search_terms:
                    if term.lower() in model_id:
                        target_model = model
                        log.info("Found target model: %s", model.get("id"))
                        break
                if target_model:
                    break
            
            if target_model:
                # Try to extract context from model metadata if available
                # LM Studio may provide this in model object
                log.debug("Using target model metadata for context info")
                max_ctx = default_context  # Fallback
                safe_ctx = int(max_ctx * 0.8)
                return {"max_context": max_ctx, "safe_context": safe_ctx}
        
        log.warning("Could not get model context info from API, using defaults")
        return {
            "max_context": default_context,
            "safe_context": int(default_context * 0.8),
        }
    
    except Exception as e:
        log.warning("Error getting model context info: %s", e, exc_info=True)
        return {
            "max_context": default_context,
            "safe_context": int(default_context * 0.8),
        }


def load_model_lm_studio(
    lm_studio_url: str,
    model_path: str,
    logger_instance: Optional[logging.Logger] = None,
) -> bool:
    """Load model in LM Studio via API.
    
    Args:
        lm_studio_url: LM Studio API URL.
        model_path: Path to model file.
        logger_instance: Logger for output messages.
        
    Returns:
        True if model loaded successfully.
    """
    log = logger_instance or logger
    try:
        import requests
        
        payload = {"model": model_path, "load": True}
        log.info("Loading model via LM Studio API: %s", os.path.basename(model_path))
        
        response = requests.post(
            f"{lm_studio_url}/v1/models/load",
            json=payload,
            timeout=60
        )
        
        if response.status_code == 200:
            log.info("Model loaded successfully via API")
            return True
        else:
            log.warning(
                "Failed to load model via API: status=%d, response=%s",
                response.status_code,
                response.text[:200],
            )
            return False
    
    except Exception as e:
        log.error("Error loading model via API: %s", e, exc_info=True)
        return False


def unload_model_lm_studio(
    lm_studio_url: str,
    model_id: str,
    logger_instance: Optional[logging.Logger] = None,
) -> bool:
    """Unload model from LM Studio via API.
    
    Args:
        lm_studio_url: LM Studio API URL.
        model_id: Model ID to unload.
        logger_instance: Logger for output messages.
        
    Returns:
        True if model unloaded successfully.
    """
    log = logger_instance or logger
    try:
        import requests
        
        log.info("Unloading model via LM Studio API: %s", model_id)
        response = requests.post(
            f"{lm_studio_url}/v1/models/unload",
            json={"model": model_id},
            timeout=30
        )
        
        if response.status_code == 200:
            log.info("Model unloaded successfully")
            return True
        else:
            log.warning(
                "Failed to unload model: status=%d, response=%s",
                response.status_code,
                response.text[:200],
            )
            return False
    
    except Exception as e:
        log.error("Error unloading model: %s", e, exc_info=True)
        return False


def ask_qwen_for_prompt(
    lm_studio_url: str,
    model_id: str,
    question: str,
    logger_instance: Optional[logging.Logger] = None,
) -> Optional[str]:
    """
    Запрос к Qwen (или другой модели) для генерации промтов изображений.
    
    Args:
        lm_studio_url: URL API LM Studio
        model_id: ID модели для использования
        question: Запрос пользователя
        logger_instance: Опциональный логгер
        
    Returns:
        Сгенерированный промт или None при ошибке
    """
    log = logger_instance or logger
    import requests
    
    payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": "Ты — ассистент для генерации идеальных промтов для Stable Diffusion. Твоя задача — создать идеальный промт для генерации изображения на основе запроса пользователя. ВАЖНО: prompt и negative_prompt должны быть ТОЛЬКО на английском языке, иначе будет ошибка! ВСЕГДА включай negative_prompt - это обязательное поле! Формируй промт и настройки строго в формате JSON: {\"prompt\":..., \"negative_prompt\":..., \"params\":{...}}. Пример negative_prompt: '(worst quality, low quality, normal quality:1.4), (deformed, distorted, disfigured:1.3), poorly drawn, bad anatomy'. Не добавляй ничего лишнего!"},
            {"role": "user", "content": f"Вопрос: {question}\n\nВАЖНО: prompt и negative_prompt должны быть ТОЛЬКО на английском языке! ОБЯЗАТЕЛЬНО включи negative_prompt в JSON!"}
        ],
        "temperature": 0.2,
        "max_tokens": 1024,
        "stream": False
    }
    
    try:
        resp = requests.post(
            f"{lm_studio_url}/v1/chat/completions", 
            json=payload, 
            headers={"Content-Type": "application/json"},
            timeout=60
        )
        if resp.status_code == 200:
            result = resp.json()
            content = result["choices"][0]["message"]["content"].strip()
            return content
        else:
            log.error(f"Ошибка Qwen: {resp.status_code} - {resp.text}")
            return None
    except Exception as e:
        log.error(f"Ошибка запроса к Qwen: {e}")
        return None


__all__ = [
    "LlamaCppWrapper",
    "LLAMA_CPP_AVAILABLE",
    "is_model_running_lm_studio",
    "get_model_context_info_lm_studio",
    "load_model_lm_studio",
    "unload_model_lm_studio",
    "ask_qwen_for_prompt",
]
