"""Централизованные настройки приложения.

Модуль объединяет все значения, которые имеет смысл менять без правки
основного кода: пути к моделям, параметры контекста, конфигурацию
логов, директории с ресурсами и т.д. 1.py импортирует эти константы и
работает с локальными копиями, чтобы не модифицировать сам конфиг.
"""

from __future__ import annotations

import os
from typing import Any, Dict
from pathlib import Path

# ---------------------------------------------------------------------------
# GPU / аппаратная конфигурация
# ---------------------------------------------------------------------------

GPU_CONFIG: Dict[str, Any] = {
    # CUDA устройство по умолчанию. При необходимости можно указать "0,1" или "-1".
    "cuda_visible_devices": "0",
    # Уровень подробности логов llama.cpp. 40 = ERROR, 30 = WARNING, 20 = INFO.
    "llama_log_level": "40",
    # Сообщение, которое выводим на старте для ясности, какая видеокарта ожидается.
    "force_gpu_message": "RTX 5060 Ti (compute capability 12.0)",
}

# ---------------------------------------------------------------------------
# Основная языковая модель (llama.cpp)
# ---------------------------------------------------------------------------

# Если False — используем LM Studio HTTP API, а не локальную сборку llama.cpp
USE_LLAMA_CPP: bool = True

# Путь к GGUF-файлу модели в локальной папке проекта (models/).
BASE_DIR = Path(__file__).resolve().parent
LLAMA_CPP_MODEL_PATH: str = str(
    BASE_DIR / "models" / "nvidia_Orchestrator-8B-Q4_K_M.gguf"
)

# Базовый тип квантизации KV-кэша, пока не проверили доступные форматы у llama_cpp.
LLAMA_KV_Q8_DEFAULT: int = 8

# Стартовые параметры инициализации llama.cpp.
# Код оркестратора делает deepcopy, поэтому runtime-изменения не затрагивают конфиг напрямую.
LLAMA_CPP_PARAMS: Dict[str, Any] = {
    "n_ctx": 32768,
    "n_gpu_layers": -1,
    "n_threads": 4,
    "n_batch": 2048,
    "use_mlock": False,
    "use_mmap": True,
    "offload_kqv": True,
    "flash_attn": True,
    "type_k": LLAMA_KV_Q8_DEFAULT,
    "type_v": LLAMA_KV_Q8_DEFAULT,
    "verbose": False,
    "seed": -1,
    "mul_mat_q": True,
    "logits_all": False,
    "embedding": False,
    "last_n_tokens_size": 64,
}

# Параметры генерации текста: температура, top-k/p и т.д.
LLAMA_CPP_GENERATION_PARAMS: Dict[str, Any] = {
    "temperature": 0.7,
    "max_tokens": None,
    "top_p": 0.9,
    "top_k": 40,
    "repeat_penalty": 1.1,
    "stream": False,
}

# ---------------------------------------------------------------------------
# Vision модель (например, Moondream2 через LM Studio или llama.cpp)
# ---------------------------------------------------------------------------

# Идентификатор модели vision в LM Studio (можно переопределить переменной окружения).
VISION_MODEL_ID: str = os.getenv("VISION_MODEL_ID", "moondream2-llamafile")

VISION_MODEL_LOAD_ARGS: Dict[str, Any] = {
    "n_ctx": 2048,
    "n_gpu_layers": 24,
    "n_threads": 8,
    "n_batch": 512,
    "offload_kqv": True,
    "flash_attn": True,
    "type_k": "q8_0",
    "type_v": "q8_0",
}

VISION_GENERATION_PARAMS: Dict[str, Any] = {
    "temperature": 0.3,
    "max_tokens": 2048,
    "stream": False,
}

# Имя Markdown-файла с инструкциями для анализа изображений.
VISION_PROMPT_FILENAME: str = "vision_analysis_prompt.md"

# Fallback-текст, используемый, если файл отсутствует или пуст.
VISION_FALLBACK_PROMPT: str = (
    "Ты анализируешь изображение. Опиши ключевые объекты, их расположение, цвета и текст. "
    "Будь краток, но информативен. Не делай выводов, которых не видно. Если что-то непонятно, "
    "честно сообщи об этом."
)

# ---------------------------------------------------------------------------
# Настройки векторного хранилища (ChromaDB)
# ---------------------------------------------------------------------------

# Путь к директории, куда сохраняется ChromaDB PersistentClient.
CHROMA_DB_PATH: str = "./chroma_db"

# Коллекция для боевых данных (используется классом ChromaDBManager).
CHROMADB_DEFAULT_COLLECTION_NAME: str = "conversation_memory"
CHROMADB_DEFAULT_COLLECTION_METADATA: Dict[str, Any] = {
    "description": "Векторное хранилище диалогов и предпочтений пользователя",
    "hnsw:space": "cosine",
}

# Флаг включения сохранения диалогов в ChromaDB
CHROMADB_ENABLE_MEMORY: bool = False
# Сколько дней хранить записи по умолчанию при очистке
CHROMADB_CLEANUP_DEFAULT_DAYS: int = 30

# Коллекция, которую использует вспомогательная функция load_chromadb.
CHROMADB_BACKGROUND_COLLECTION_NAME: str = "ai_memories"

# Модель sentence-transformers для построения эмбеддингов по умолчанию.
CHROMADB_EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

# Стоит ли пытаться ускорять хранилище на GPU по умолчанию.
CHROMADB_USE_GPU_BY_DEFAULT: bool = True

# ---------------------------------------------------------------------------
# Тайминги, контекст и прочие числовые параметры
# ---------------------------------------------------------------------------

DEFAULT_MAX_CONTEXT_LENGTH: int = 262_144
DEFAULT_SAFE_CONTEXT_LENGTH: int = 32_768
DEFAULT_MAX_RETRIES: int = 3
AUTO_DISABLE_DELAY_SECONDS: int = 300  # 5 минут
DEFAULT_SIMILARITY_THRESHOLD: float = 0.7
HISTORY_MAX_LENGTH: int = 200

# ---------------------------------------------------------------------------
# Пути и директории проекта
# ---------------------------------------------------------------------------

PROMPTS_DIR_NAME: str = "prompt"
PLUGINS_DIR_NAME: str = "plugins"
OUTPUT_DIR_NAME: str = "output"

# ---------------------------------------------------------------------------
# Подключения к внешним сервисам
# ---------------------------------------------------------------------------

DEFAULT_LM_STUDIO_URL: str = "http://localhost:1234"

# ---------------------------------------------------------------------------
# Логирование
# ---------------------------------------------------------------------------

LOG_FILE_NAME: str = "ai_orchestrator.log"
FILE_LOG_FORMAT: str = "%(asctime)s - %(levelname)s - %(message)s"
CONSOLE_LOG_LEVEL: str = "INFO"  # Уровень для StreamHandler

# ---------------------------------------------------------------------------
# Флаги доступности зависимостей (значения по умолчанию)
# ---------------------------------------------------------------------------

OCR_AVAILABLE_DEFAULT: bool = True
CHROMADB_AVAILABLE_DEFAULT: bool = True
TORCH_AVAILABLE_DEFAULT: bool = True

__all__ = [
    "GPU_CONFIG",
    "USE_LLAMA_CPP",
    "LLAMA_CPP_MODEL_PATH",
    "LLAMA_KV_Q8_DEFAULT",
    "LLAMA_CPP_PARAMS",
    "LLAMA_CPP_GENERATION_PARAMS",
    "VISION_MODEL_ID",
    "VISION_MODEL_LOAD_ARGS",
    "VISION_GENERATION_PARAMS",
    "VISION_PROMPT_FILENAME",
    "VISION_FALLBACK_PROMPT",
    "CHROMA_DB_PATH",
    "CHROMADB_DEFAULT_COLLECTION_NAME",
    "CHROMADB_DEFAULT_COLLECTION_METADATA",
    "CHROMADB_BACKGROUND_COLLECTION_NAME",
    "CHROMADB_EMBEDDING_MODEL",
    "CHROMADB_USE_GPU_BY_DEFAULT",
    "CHROMADB_ENABLE_MEMORY",
    "CHROMADB_CLEANUP_DEFAULT_DAYS",
    "DEFAULT_MAX_CONTEXT_LENGTH",
    "DEFAULT_SAFE_CONTEXT_LENGTH",
    "DEFAULT_MAX_RETRIES",
    "AUTO_DISABLE_DELAY_SECONDS",
    "DEFAULT_SIMILARITY_THRESHOLD",
    "HISTORY_MAX_LENGTH",
    "PROMPTS_DIR_NAME",
    "PLUGINS_DIR_NAME",
    "OUTPUT_DIR_NAME",
    "DEFAULT_LM_STUDIO_URL",
    "LOG_FILE_NAME",
    "FILE_LOG_FORMAT",
    "CONSOLE_LOG_LEVEL",
    "OCR_AVAILABLE_DEFAULT",
    "CHROMADB_AVAILABLE_DEFAULT",
    "TORCH_AVAILABLE_DEFAULT",
]
