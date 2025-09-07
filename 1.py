#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI PowerShell Orchestrator with Google Search Integration
Интегрирует LM Studio, PowerShell команды и поиск Google

ОБНОВЛЕНО: Теперь использует прямую интеграцию со Stable Diffusion для генерации изображений
ОБНОВЛЕНО: Добавлено векторное хранилище ChromaDB для преодоления ограничений контекста

ТРЕБУЕМЫЕ БИБЛИОТЕКИ:
pip install pyautogui mss pillow requests diffusers transformers torch torchvision accelerate safetensors chromadb sentence-transformers
"""

import requests
import subprocess
import json
import time
import sys as _sys
import os
import base64
import io
import re
import threading
import tempfile
import shutil
import math
import pyautogui
import mss
import queue
import logging
import argparse
from typing import Dict, Any, List, Union, Optional, TYPE_CHECKING
import urllib.parse
from PIL import Image
from io import BytesIO
from collections import defaultdict
import asyncio
import telegram
from telegram import Update, Bot
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from dotenv import load_dotenv

# Помощь статическим анализаторам: явные объявления для опциональных внешних символов
from typing import Any as _Any
chromadb: _Any = None
Settings: _Any = None
SentenceTransformer: _Any = None
torch: _Any = None
_imageio: _Any = None
_pygame: _Any = None

# Импорты для ChromaDB и векторного поиска
try:
    import chromadb
    from chromadb.config import Settings
    from sentence_transformers import SentenceTransformer
    import numpy as np
    import torch
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    print("⚠️ ChromaDB не установлен. Установите: pip install chromadb sentence-transformers")

# Проверки доступности опциональных модулей
try:
    import torch as _torch
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

try:
    import diffusers as _diffusers
    DIFFUSERS_AVAILABLE = True
except Exception:
    DIFFUSERS_AVAILABLE = False

try:
    import imageio as _imageio
    IMAGEIO_AVAILABLE = True
except Exception:
    IMAGEIO_AVAILABLE = False

try:
    import pygame as _pygame
    PYGAME_AVAILABLE = True
except Exception:
    PYGAME_AVAILABLE = False

# Загружаем переменные окружения из .env файла
# override=True - перезаписываем существующие переменные
load_dotenv(override=True)

# Determine if running in web mode (show verbose console logs)
IS_WEB = any(arg == '--web' for arg in _sys.argv)

# Настройка логирования: всегда пишем подробный файл, но в консоль показываем INFO только в --web
log_file = "ai_orchestrator.log"
root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)

# File handler: keep full INFO logs for later inspection
file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
file_handler.setLevel(logging.INFO)
file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)
root_logger.addHandler(file_handler)

# Console handler: verbose only for --web, otherwise warnings+ only
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(file_formatter)
root_logger.addHandler(console_handler)

logger = logging.getLogger(__name__)

# Disable chromadb telemetry / noisy chromadb logs entirely
try:
    chroma_logger = logging.getLogger('chromadb')
    chroma_logger.disabled = True
except Exception:
    pass

# Filter for telemetry messages: only applied to console handler so file logs keep full info
class TelemetryFilter(logging.Filter):
    def __init__(self, patterns=None):
        super().__init__()
        self.patterns = patterns or ["Failed to send telemetry event", "telemetry", "capture() takes"]

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            msg = record.getMessage()
        except Exception:
            msg = ''
        for p in self.patterns:
            if p in msg:
                return False
        return True

# Apply telemetry filter to console only
try:
    console_handler.addFilter(TelemetryFilter())
except Exception:
    pass


class _StderrFilterWriter:
    """A file-like wrapper for stderr that filters lines containing given patterns."""
    def __init__(self, orig, patterns):
        self.orig = orig
        self.patterns = patterns

    def write(self, data):
        if not data:
            return
        for p in self.patterns:
            if p in data:
                return
        try:
            self.orig.write(data)
        except Exception:
            pass

    def flush(self):
        try:
            self.orig.flush()
        except Exception:
            pass
if TYPE_CHECKING:
    try:
        import chromadb  # type: ignore
        from chromadb.config import Settings  # type: ignore
    except Exception:
        pass
    # Stubs for optional heavy libraries so Pylance doesn't warn
    try:
        import imageio as _imageio  # type: ignore
    except Exception:
        pass
    try:
        import pygame as _pygame  # type: ignore
    except Exception:
        pass
    try:
        import diffusers as _diffusers  # type: ignore
    except Exception:
        pass
    try:
        import torch as _torch  # type: ignore
    except Exception:
        pass


from contextlib import contextmanager

@contextmanager
def suppress_stderr_patterns(patterns):
    orig = _sys.stderr
    try:
        _sys.stderr = _StderrFilterWriter(orig, patterns)
        yield
    finally:
        _sys.stderr = orig

### НОВОЕ: Класс для работы с векторным хранилищем ChromaDB ###
class ChromaDBManager:
    """
    Менеджер векторного хранилища ChromaDB для преодоления ограничений контекста
    и обучения на предпочтениях пользователя
    """
    
    def __init__(self, db_path: str = "chroma_db", embedding_model: str = "all-MiniLM-L6-v2", use_gpu: bool = True):
        """
        Инициализация ChromaDB менеджера
        
        Args:
            db_path: Путь к базе данных ChromaDB
            embedding_model: Модель для создания эмбеддингов (784 размерности)
            use_gpu: Использовать GPU для создания эмбеддингов (если доступен)
        """
        self.db_path = db_path
        self.embedding_model = embedding_model
        self.use_gpu = use_gpu
        self.client = None
        self.collection = None
        self.embedding_model_obj = None
        self.initialized = False
        
        # Создаем папку для базы данных если её нет
        os.makedirs(db_path, exist_ok=True)
        
        # Инициализируем ChromaDB
        self._initialize_chromadb()
    
    def _initialize_chromadb(self):
        """Инициализация ChromaDB клиента и коллекции"""
        try:
            if not CHROMADB_AVAILABLE:
                logger.warning("⚠️ ChromaDB недоступен, векторное хранилище отключено")
                return
            
            # Инициализируем клиент ChromaDB (подавляем телеметрию в stderr)
            with suppress_stderr_patterns(["Failed to send telemetry event", "capture() takes", "telemetry"]):
                self.client = chromadb.PersistentClient(
                    path=self.db_path,
                    settings=Settings(
                        anonymized_telemetry=False,
                        allow_reset=True
                    )
                )
            
            # Создаем или получаем коллекцию
            self.collection = self.client.get_or_create_collection(
                name="conversation_memory",
                metadata={"description": "Векторное хранилище диалогов и предпочтений пользователя"}
            )
            
            # Загружаем модель для эмбеддингов
            logger.info(f"📦 Загружаю модель эмбеддингов: {self.embedding_model}")
            
            # Проверяем доступность GPU
            device = "cuda" if self.use_gpu and torch.cuda.is_available() else "cpu"
            logger.info(f"🔧 Используется устройство: {device}")
            
            # Получаем информацию о GPU
            gpu_info = self.get_gpu_info()
            
            self.embedding_model_obj = SentenceTransformer(self.embedding_model, device=device)
            
            # Проверяем размерность эмбеддингов
            test_embedding = self.embedding_model_obj.encode("test")
            embedding_dim = len(test_embedding)
            logger.info(f"✅ Модель эмбеддингов загружена, размерность: {embedding_dim}")
            
            # Проверяем количество записей в базе
            if self.collection is None:
                logger.warning("⚠️ Коллекция ChromaDB не инициализирована при попытке получить count")
                count = 0
            else:
                count = self.collection.count()
            logger.info(f"📊 База данных содержит {count} записей")
            
            self.initialized = True
            logger.info("✅ ChromaDB успешно инициализирован")
            
        except Exception as e:
            logger.error(f"❌ Ошибка инициализации ChromaDB: {e}")
            self.initialized = False
    
    def get_gpu_info(self) -> Dict[str, Any]:
        """
        Получает информацию о GPU для ChromaDB
        
        Returns:
            Словарь с информацией о GPU
        """
        gpu_info = {
            "gpu_available": False,
            "gpu_name": None,
            "gpu_memory": None,
            "device_used": "cpu"
        }
        
        try:
            if torch.cuda.is_available():
                gpu_info["gpu_available"] = True
                gpu_info["gpu_name"] = torch.cuda.get_device_name(0)
                gpu_info["gpu_memory"] = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
                gpu_info["device_used"] = "cuda" if self.use_gpu else "cpu"
                
                logger.info(f"🎮 GPU доступен: {gpu_info['gpu_name']}")
                logger.info(f"💾 GPU память: {gpu_info['gpu_memory']:.1f} GB")
            else:
                logger.info("💻 GPU недоступен, используется CPU")
                
        except Exception as e:
            logger.warning(f"⚠️ Ошибка получения информации о GPU: {e}")
            
        return gpu_info
    
    def add_conversation_memory(self, user_message: str, ai_response: str,
                               context: str = "", metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Добавляет диалог в векторное хранилище
        
        Args:
            user_message: Сообщение пользователя
            ai_response: Ответ ИИ
            context: Дополнительный контекст
            metadata: Дополнительные метаданные
            
        Returns:
            True если успешно добавлено, False при ошибке
        """
        if not self.initialized:
            return False
        
        try:
            # Создаем уникальный ID для записи
            timestamp = int(time.time())
            record_id = f"conv_{timestamp}_{hash(user_message) % 10000}"
            
            # Объединяем текст для создания эмбеддинга
            combined_text = f"User: {user_message}\nAI: {ai_response}"
            if context:
                combined_text += f"\nContext: {context}"
            
            # Создаем эмбеддинг (если модель доступна)
            if not self.initialized or self.embedding_model_obj is None:
                logger.warning("⚠️ Эмбеддинговая модель не инициализирована, пропускаю добавление в ChromaDB")
                return False
            embedding = self.embedding_model_obj.encode(combined_text).tolist()
            
            # Подготавливаем метаданные
            record_metadata = {
                "timestamp": timestamp,
                "user_message": user_message,
                "ai_response": ai_response,
                "context": context,
                "type": "conversation"
            }
            
            if metadata:
                record_metadata.update(metadata)
            
            # Добавляем в коллекцию
                if self.collection is None:
                    logger.warning("⚠️ Коллекция ChromaDB не инициализирована при попытке add")
                    return False
                self.collection.add(
                embeddings=[embedding],
                documents=[combined_text],
                metadatas=[record_metadata],
                ids=[record_id]
            )
            
            logger.info(f"💾 Добавлена запись в ChromaDB: {record_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Ошибка добавления в ChromaDB: {e}")
            return False
    
    def add_user_preference(self, preference_text: str, category: str = "general",
                           metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Добавляет предпочтение пользователя в векторное хранилище
        
        Args:
            preference_text: Текст предпочтения
            category: Категория предпочтения
            metadata: Дополнительные метаданные
            
        Returns:
            True если успешно добавлено, False при ошибке
        """
        if not self.initialized:
            return False
        
        try:
            # Создаем уникальный ID
            timestamp = int(time.time())
            record_id = f"pref_{timestamp}_{hash(preference_text) % 10000}"
            
            # Создаем эмбеддинг (если модель доступна)
            if not self.initialized or self.embedding_model_obj is None:
                logger.warning("⚠️ Эмбеддинговая модель не инициализирована, пропускаю добавление предпочтения")
                return False
            embedding = self.embedding_model_obj.encode(preference_text).tolist()
            
            # Подготавливаем метаданные
            record_metadata = {
                "timestamp": timestamp,
                "preference_text": preference_text,
                "category": category,
                "type": "preference"
            }
            
            if metadata:
                record_metadata.update(metadata)
            
            # Добавляем в коллекцию
                if self.collection is None:
                    logger.warning("⚠️ Коллекция ChromaDB не инициализирована при попытке add preference")
                    return False
                self.collection.add(
                embeddings=[embedding],
                documents=[preference_text],
                metadatas=[record_metadata],
                ids=[record_id]
            )
            
            logger.info(f"💾 Добавлено предпочтение в ChromaDB: {record_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Ошибка добавления предпочтения в ChromaDB: {e}")
            return False
    
    def search_similar_conversations(self, query: str, n_results: int = 5,
                                   similarity_threshold: float = None) -> List[Dict[str, Any]]:
        """
        Ищет похожие диалоги в векторном хранилище
        
        Args:
            query: Поисковый запрос
            n_results: Количество результатов
            similarity_threshold: Порог схожести (автоматический если None)
            
        Returns:
            Список найденных диалогов с метаданными
        """
        if not self.initialized:
            return []
        
        try:
            # Создаем эмбеддинг для запроса
            if not self.initialized or self.embedding_model_obj is None:
                logger.warning("⚠️ Эмбеддинговая модель не инициализирована, поиск невозможен")
                return []
            
            logger.info(f"🔍 Ищем похожие диалоги для запроса: '{query}'")
            query_embedding = self.embedding_model_obj.encode(query).tolist()
            
            # Ищем похожие записи (если коллекция доступна)
            if self.collection is None:
                logger.warning("⚠️ Коллекция ChromaDB не доступна, поиск невозможен")
                return []
            
            # Увеличиваем количество результатов для лучшего поиска
            search_results = max(n_results * 3, 15)
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=search_results,
                where={"type": "conversation"}  # type: ignore[arg-type]
            )
            
            # Анализируем результаты для определения адаптивного порога
            filtered_results = []
            found_count = 0
            
            # Защищаемся от отсутствия ключей или пустых результатов
            if isinstance(results, dict) and results:
                distances = results.get('distances')
                ids = results.get('ids')
                documents = results.get('documents')
                metadatas = results.get('metadatas')

                if distances and isinstance(distances, list) and distances and distances[0]:
                    logger.info(f"📊 Обработка {len(distances[0])} результатов поиска")
                    
                    # Вычисляем адаптивный порог если не задан
                    if similarity_threshold is None:
                        similarities = [1 - d for d in distances[0]]
                        if similarities:
                            max_sim = max(similarities)
                            avg_sim = sum(similarities) / len(similarities)
                            
                            # Адаптивный порог: берем результаты выше среднего, но не выше 0.5
                            if max_sim > 0.1:
                                adaptive_threshold = min(avg_sim + 0.1, 0.3, max_sim - 0.05)
                            else:
                                adaptive_threshold = -0.2  # Очень низкий порог для слабых совпадений
                            
                            logger.info(f"🎯 Адаптивный порог схожести: {adaptive_threshold:.3f} (макс: {max_sim:.3f}, средн: {avg_sim:.3f})")
                        else:
                            adaptive_threshold = 0.1
                    else:
                        adaptive_threshold = similarity_threshold
                    
                    for i, distance in enumerate(distances[0]):
                        # ChromaDB возвращает расстояния, конвертируем в схожесть
                        similarity = 1 - distance
                        
                        # Логируем для отладки
                        if i < 3:  # Показываем первые 3 результата
                            logger.info(f"   Результат {i+1}: схожесть={similarity:.3f}, расстояние={distance:.3f}")
                        
                        if similarity >= adaptive_threshold:
                            # Проверяем, что остальные структуры содержат нужные элементы
                            doc = None
                            meta = None
                            idv = None
                            try:
                                idv = ids[0][i] if ids and ids[0] and len(ids[0]) > i else None
                                doc = documents[0][i] if documents and documents[0] and len(documents[0]) > i else None
                                meta = metadatas[0][i] if metadatas and metadatas[0] and len(metadatas[0]) > i else None
                            except Exception as e:
                                logger.warning(f"⚠️ Ошибка извлечения данных для результата {i}: {e}")
                                continue

                            result = {
                                'id': idv,
                                'document': doc,
                                'metadata': meta,
                                'similarity': similarity,
                                'distance': distance
                            }
                            filtered_results.append(result)
                            found_count += 1
                            
                            # Ограничиваем количество результатов
                            if found_count >= n_results:
                                break
                    
                    # Если ничего не найдено даже с адаптивным порогом, берем лучшие результаты
                    if not filtered_results and distances[0]:
                        logger.info(f"⚠️ Ничего не найдено с порогом {adaptive_threshold:.3f}, берем {min(3, len(distances[0]))} лучших результата")
                        best_results = min(3, len(distances[0]))
                        for i in range(best_results):
                            distance = distances[0][i]
                            similarity = 1 - distance
                            
                            try:
                                idv = ids[0][i] if ids and ids[0] and len(ids[0]) > i else None
                                doc = documents[0][i] if documents and documents[0] and len(documents[0]) > i else None
                                meta = metadatas[0][i] if metadatas and metadatas[0] and len(metadatas[0]) > i else None
                                
                                result = {
                                    'id': idv,
                                    'document': doc,
                                    'metadata': meta,
                                    'similarity': similarity,
                                    'distance': distance
                                }
                                filtered_results.append(result)
                            except Exception as e:
                                logger.warning(f"⚠️ Ошибка извлечения лучшего результата {i}: {e}")
                else:
                    logger.warning("⚠️ Пустые результаты поиска в ChromaDB")
            else:
                logger.warning("⚠️ Некорректный формат результатов поиска")
            
            logger.info(f"✅ Найдено {len(filtered_results)} похожих диалогов")
            return filtered_results
            
        except Exception as e:
            logger.error(f"❌ Ошибка поиска в ChromaDB: {e}")
            return []
    
    def search_user_preferences(self, query: str, category: Optional[str] = None,
                               n_results: int = 3) -> List[Dict[str, Any]]:
        """
        Ищет предпочтения пользователя
        
        Args:
            query: Поисковый запрос
            category: Категория предпочтений (опционально)
            n_results: Количество результатов
            
        Returns:
            Список найденных предпочтений
        """
        if not self.initialized:
            return []
        
        try:
            # Создаем эмбеддинг для запроса
            if not self.initialized or self.embedding_model_obj is None:
                logger.warning("⚠️ Эмбеддинговая модель не инициализирована, поиск предпочтений невозможен")
                return []
            query_embedding = self.embedding_model_obj.encode(query).tolist()
            
            # Формируем условия поиска
            where_condition = {"type": "preference"}
            if category:
                where_condition["category"] = category
            
            # Ищем похожие записи
            if self.collection is None:
                logger.warning("⚠️ Коллекция ChromaDB не доступна, поиск предпочтений невозможен")
                return []
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where_condition  # type: ignore[arg-type]
            )
            
            # Формируем результат
            preferences = []
            if isinstance(results, dict) and results:
                docs = results.get('documents')
                distances = results.get('distances')
                ids = results.get('ids')
                metadatas = results.get('metadatas')

                if docs and isinstance(docs, list) and docs and docs[0]:
                    for i, document in enumerate(docs[0]):
                        try:
                            dist = distances[0][i] if distances and distances[0] and len(distances[0]) > i else None
                            sim = 1 - dist if dist is not None else 0.0
                            pref = {
                                'id': ids[0][i] if ids and ids[0] and len(ids[0]) > i else None,
                                'preference_text': document,
                                'metadata': metadatas[0][i] if metadatas and metadatas[0] and len(metadatas[0]) > i else {},
                                'similarity': sim
                            }
                            preferences.append(pref)
                        except Exception:
                            continue
            
            logger.info(f"🔍 Найдено {len(preferences)} предпочтений пользователя")
            return preferences
            
        except Exception as e:
            logger.error(f"❌ Ошибка поиска предпочтений в ChromaDB: {e}")
            return []
    
    def get_conversation_context(self, query: str, max_context_length: int = 2000) -> str:
        """
        Получает релевантный контекст из предыдущих диалогов
        
        Args:
            query: Текущий запрос пользователя
            max_context_length: Максимальная длина контекста
            
        Returns:
            Строка с релевантным контекстом
        """
        if not self.initialized:
            return ""
        
        try:
            # Ищем похожие диалоги с адаптивным порогом
            similar_conversations = self.search_similar_conversations(
                query, n_results=5, similarity_threshold=None  # Автоматический порог
            )
            
            if not similar_conversations:
                logger.info("📚 Релевантный контекст не найден")
                return ""
            
            logger.info(f"📚 Найдено {len(similar_conversations)} релевантных диалогов для контекста")
            
            # Формируем контекст
            context_parts = []
            current_length = 0
            
            for i, conv in enumerate(similar_conversations):
                # Извлекаем пользовательское сообщение из метаданных если доступно
                user_msg = ""
                ai_resp = ""
                
                if conv.get('metadata') and isinstance(conv['metadata'], dict):
                    user_msg = conv['metadata'].get('user_message', '')
                    ai_resp = conv['metadata'].get('ai_response', '')
                
                # Если метаданные недоступны, пытаемся извлечь из документа
                if not user_msg and conv.get('document'):
                    doc = conv['document']
                    if 'User:' in doc and 'AI:' in doc:
                        parts = doc.split('AI:', 1)
                        if len(parts) >= 2:
                            user_part = parts[0].replace('User:', '').strip()
                            user_msg = user_part
                
                if user_msg:
                    conv_text = f"Похожий запрос #{i+1} (схожесть: {conv['similarity']:.3f}):\n"
                    conv_text += f"Пользователь: {user_msg[:200]}{'...' if len(user_msg) > 200 else ''}\n"
                    
                    if ai_resp and len(ai_resp) < 300:  # Включаем короткие ответы
                        clean_ai_resp = ai_resp.replace('<think>', '').replace('</think>', '')
                        if len(clean_ai_resp) < 200:
                            conv_text += f"Ответ: {clean_ai_resp[:150]}{'...' if len(clean_ai_resp) > 150 else ''}\n"
                    
                    conv_text += "\n"
                    
                    if current_length + len(conv_text) <= max_context_length:
                        context_parts.append(conv_text)
                        current_length += len(conv_text)
                    else:
                        break
            
            context = "".join(context_parts)
            if context:
                logger.info(f"✅ Сформирован контекст длиной {len(context)} символов из {len(context_parts)} диалогов")
            return context
            
        except Exception as e:
            logger.error(f"❌ Ошибка получения контекста из ChromaDB: {e}")
            return ""
    
    def get_user_preferences_summary(self, query: Optional[str] = None) -> str:
        """
        Получает краткое резюме предпочтений пользователя
        
        Args:
            query: Контекстный запрос (опционально)
            
        Returns:
            Строка с резюме предпочтений
        """
        if not self.initialized:
            return ""
        
        try:
            # Ищем релевантные предпочтения
            if query:
                preferences = self.search_user_preferences(query, n_results=5)
            else:
                # Получаем все предпочтения
                results = self.collection.get(where={"type": "preference"})  # type: ignore[arg-type]
                preferences = []
                if isinstance(results, dict) and results:
                    docs = results.get('documents')
                    metadatas = results.get('metadatas')
                    if docs:
                        for i, doc in enumerate(docs):
                            pref_meta = metadatas[i] if metadatas and len(metadatas) > i else {}
                            preference = {
                                'preference_text': doc,
                                'metadata': pref_meta,
                                'similarity': 1.0  # Для общих предпочтений
                            }
                            preferences.append(preference)
            
            if not preferences:
                return ""
            
            # Формируем резюме
            summary_parts = ["Предпочтения пользователя:"]
            
            for pref in preferences[:3]:  # Ограничиваем 3 предпочтениями
                category = pref['metadata'].get('category', 'general')
                summary_parts.append(f"- {category}: {pref['preference_text'][:100]}...")
            
            summary = "\n".join(summary_parts)
            logger.info(f"📋 Сформировано резюме предпочтений длиной {len(summary)} символов")
            return summary
            
        except Exception as e:
            logger.error(f"❌ Ошибка получения резюме предпочтений: {e}")
            return ""
    
    def cleanup_old_records(self, days_to_keep: int = 30) -> int:
        """
        Удаляет старые записи из базы данных
        
        Args:
            days_to_keep: Количество дней для хранения записей
            
        Returns:
            Количество удаленных записей
        """
        if not self.initialized:
            return 0
        
        try:
            cutoff_timestamp = int(time.time()) - (days_to_keep * 24 * 60 * 60)
            
            # Получаем все записи
            if self.collection is None:
                logger.warning("⚠️ Коллекция ChromaDB не доступна, очистка записей пропущена")
                return 0
            results = self.collection.get()

            # Защита от отсутствия ключей/пустых результатов
            if not isinstance(results, dict) or not results:
                return 0

            ids = results.get('ids')
            metadatas = results.get('metadatas')
            if not ids:
                return 0

            # Находим записи для удаления
            ids_to_delete = []
            if metadatas:
                for i, metadata in enumerate(metadatas):
                    timestamp = metadata.get('timestamp', 0) if isinstance(metadata, dict) else 0
                    if timestamp < cutoff_timestamp:
                        # Защищаем доступ к ids
                        if ids and len(ids) > i:
                            ids_to_delete.append(ids[i])
            
            # Удаляем старые записи
            if ids_to_delete:
                if self.collection is None:
                    logger.warning("⚠️ Коллекция ChromaDB не доступна, удаление невозможно")
                    return 0
                self.collection.delete(ids=ids_to_delete)
                logger.info(f"🧹 Удалено {len(ids_to_delete)} старых записей из ChromaDB")
                return len(ids_to_delete)
            
            return 0
            
        except Exception as e:
            logger.error(f"❌ Ошибка очистки ChromaDB: {e}")
            return 0
    
    def get_database_stats(self) -> Dict[str, Any]:
        """
        Получает статистику базы данных
        
        Returns:
            Словарь со статистикой
        """
        if not self.initialized:
            return {"error": "ChromaDB не инициализирован"}
        
        try:
            if self.collection is None:
                logger.warning("⚠️ Коллекция ChromaDB не доступна, статистика недоступна")
                return {"error": "ChromaDB не инициализирован"}

            total_count = self.collection.count()
            # Подсчитываем по типам
            conversations = self.collection.get(where={"type": "conversation"})  # type: ignore[arg-type]
            preferences = self.collection.get(where={"type": "preference"})  # type: ignore[arg-type]

            conv_ids = conversations.get('ids') if isinstance(conversations, dict) else None
            pref_ids = preferences.get('ids') if isinstance(preferences, dict) else None

            stats = {
                "total_records": total_count,
                "conversations": len(conv_ids) if conv_ids else 0,
                "preferences": len(pref_ids) if pref_ids else 0,
                "database_path": self.db_path,
                "embedding_model": self.embedding_model
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"❌ Ошибка получения статистики ChromaDB: {e}")
            return {"error": str(e)}

### НОВОЕ: Функция для сжатия изображений ###
def image_to_base64_balanced(image_path: str, max_size=(500, 500), palette_colors=12) -> str:
    """
    Конвертирует изображение в PNG base64 без ч/б и quantize, только ресайз (если нужно).
    """
    try:
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            img.thumbnail(max_size, Image.Resampling.LANCZOS)
            buf = BytesIO()
            img.save(buf, format="PNG", optimize=True)
            return base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception as e:
        logger.error(f"Ошибка кодирования (balanced) {image_path}: {e}")
        return ""

class AIOrchestrator:
    def extract_video_frames(self, video_path: str, fps: int = 1) -> list:
        """
        Извлекает по одному кадру на каждую секунду видео (fps=1).
        Возвращает список кортежей (таймкод, base64 PNG).
        """
        frames = []
        temp_dir = tempfile.mkdtemp()
        try:
            # Получаем длительность видео через ffprobe
            cmd = [
                'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1', video_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            duration = float(result.stdout.strip()) if result.returncode == 0 else 0
            if duration == 0:
                return []
            # Извлекаем кадры с помощью ffmpeg
            # -vf fps=1: по одному кадру в секунду
            frame_pattern = os.path.join(temp_dir, 'frame_%05d.png')
            cmd = [
                'ffmpeg', '-i', video_path, '-vf', f'fps={fps}', '-q:v', '2', frame_pattern, '-hide_banner', '-loglevel', 'error'
            ]
            subprocess.run(cmd, check=True)
            # Собираем кадры и таймкоды
            total_frames = int(math.ceil(duration))
            for i in range(1, total_frames + 1):
                frame_path = os.path.join(temp_dir, f'frame_{i:05d}.png')
                if not os.path.exists(frame_path):
                    continue
                # Таймкод в формате [HH:MM:SS]
                sec = i - 1
                h = sec // 3600
                m = (sec % 3600) // 60
                s = sec % 60
                timecode = f"[{h:02}:{m:02}:{s:02}]"
                # base64 через существующую функцию
                b64 = image_to_base64_balanced(frame_path)
                frames.append((timecode, b64))
            return frames
        except Exception as e:
            self.logger.error(f"Ошибка извлечения кадров: {e}")
            return []
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
    def download_youtube_video(self, url: str, out_dir: Optional[str] = None) -> Optional[str]:
        """
        Скачивает видео с YouTube по ссылке (использует yt-dlp)
        Возвращает путь к mp4-файлу или пустую строку
        """
        if out_dir is None:
            out_dir = os.path.join(os.path.dirname(__file__), "Video")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "yt_video.%(ext)s")
        
        # Проверяем наличие cookies
        cookies_path = self.get_youtube_cookies_path()
        use_cookies = False
        
        if cookies_path and self.check_cookies_validity(cookies_path):
            use_cookies = True
            self.logger.info("🍪 Использую cookies для аутентификации YouTube")
        else:
            self.logger.info("ℹ️ Cookies не найдены или невалидны, использую базовые параметры")
            if not cookies_path:
                self.suggest_cookies_update()
        
        # Базовые параметры для yt-dlp
        base_cmd = [
            "yt-dlp",
            "--force-ipv4",
            "--user-agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "--extractor-args", "youtube:player_client=android",  # Используем Android клиент
            "--no-check-certificate",  # Игнорируем SSL ошибки
            "--prefer-insecure",  # Предпочитаем HTTP
            "--geo-bypass",  # Обход геоблокировки
            "--geo-bypass-country", "US",  # Страна для обхода
            "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4/best[ext=mp4]/best",
            "-o", out_path
        ]
        
        # Добавляем cookies если доступны
        if cookies_path:
            base_cmd.extend(["--cookies", str(cookies_path)])  # type: ignore[arg-type]
        
        # Добавляем URL в конец
        cmd = base_cmd + [url]
        
        try:
            self.logger.info(f"Скачиваю видео с YouTube: {url}")
            # Логируем команду в одну строку для избежания обрезания
            cmd_str = " ".join(cmd)
            self.logger.info(f"Команда: {cmd_str}")
            
            # Запускаем с таймаутом
            result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=300)
            
            if result.stdout:
                self.logger.info(f"yt-dlp stdout: {result.stdout}")
            if result.stderr:
                self.logger.warning(f"yt-dlp stderr: {result.stderr}")
            
            # Найти скачанный файл
            for fname in os.listdir(out_dir):
                if fname.startswith("yt_video") and fname.endswith('.mp4'):
                    self.logger.info(f"✅ Видео успешно скачано: {fname}")
                    return os.path.join(out_dir, fname)
            
            self.logger.warning("⚠️ Файл не найден после скачивания")
            return ""
            
        except subprocess.TimeoutExpired:
            self.logger.error("❌ Таймаут скачивания видео (5 минут)")
            return ""
        except subprocess.CalledProcessError as e:
            self.logger.error(f"❌ Ошибка yt-dlp: {e}")
            if e.stderr:
                self.logger.error(f"stderr: {e.stderr}")
            return ""
        except Exception as e:
            self.logger.error(f"❌ Неожиданная ошибка скачивания видео: {e}")
            
            # Пробуем альтернативный метод с другими параметрами
            self.logger.info("🔄 Пробую альтернативный метод скачивания...")
            try:
                alt_cmd = [
                    "yt-dlp",
                    "--force-ipv4",
                    "--user-agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                    "--extractor-args", "youtube:player_client=web",
                    "--no-check-certificate",
                    "--geo-bypass",
                    "--geo-bypass-country", "US",
                    "-f", "best[ext=mp4]/best",
                    "-o", out_path
                ]
                
                # Добавляем cookies если доступны
                if cookies_path:
                    alt_cmd.extend(["--cookies", str(cookies_path)])  # type: ignore[arg-type]
                
                alt_cmd.append(url)
                
                # Логируем команду в одну строку
                alt_cmd_str = " ".join(alt_cmd)
                self.logger.info(f"Альтернативная команда: {alt_cmd_str}")
                result = subprocess.run(alt_cmd, check=True, capture_output=True, text=True, timeout=300)
                
                # Найти скачанный файл
                for fname in os.listdir(out_dir):
                    if fname.startswith("yt_video") and fname.endswith('.mp4'):
                        self.logger.info(f"✅ Видео успешно скачано альтернативным методом: {fname}")
                        return os.path.join(out_dir, fname)
                        
            except Exception as alt_e:
                self.logger.error(f"❌ Альтернативный метод также не сработал: {alt_e}")
                
                # Пробуем третий метод с максимально простыми параметрами
                self.logger.info("🔄 Пробую третий метод (минимальные параметры)...")
                try:
                    simple_cmd = [
                        "yt-dlp",
                        "--force-ipv4",
                        "--user-agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                        "--no-check-certificate",
                        "-f", "best",
                        "-o", out_path
                    ]
                    # Добавляем cookies если доступны
                    if use_cookies and cookies_path:
                        simple_cmd.extend(["--cookies", str(cookies_path)])  # type: ignore[arg-type]

                    simple_cmd.append(url)
                    
                    # Логируем команду в одну строку
                    simple_cmd_str = " ".join(simple_cmd)
                    self.logger.info(f"Третий метод: {simple_cmd_str}")
                    result = subprocess.run(simple_cmd, check=True, capture_output=True, text=True, timeout=300)
                    
                    # Найти скачанный файл
                    for fname in os.listdir(out_dir):
                        if fname.startswith("yt_video") and fname.endswith('.mp4'):
                            self.logger.info(f"✅ Видео успешно скачано третьим методом: {fname}")
                            return os.path.join(out_dir, fname)
                            
                except Exception as simple_e:
                    self.logger.error(f"❌ Третий метод также не сработал: {simple_e}")
            
            return ""
    
    def check_vpn_status(self) -> bool:
        """
        Проверяет статус VPN соединения
        """
        try:
            import requests
            # Пробуем получить IP адрес
            response = requests.get("https://ifconfig.me", timeout=10)
            if response.status_code == 200:
                ip = response.text.strip()
                self.logger.info(f"🌐 Текущий IP адрес: {ip}")
                
                # Проверяем, не из РФ ли IP
                ru_ips = ["185.", "31.", "46.", "37.", "95.", "178.", "79.", "5.", "176.", "195."]
                if any(ip.startswith(prefix) for prefix in ru_ips):
                    self.logger.warning("⚠️ IP адрес похож на российский. VPN может не работать корректно.")
                    return False
                else:
                    self.logger.info("✅ IP адрес не из РФ. VPN работает.")
                    return True
            else:
                self.logger.warning(f"⚠️ Не удалось проверить IP: {response.status_code}")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Ошибка проверки VPN: {e}")
            return False

    def get_youtube_info(self, url: str) -> dict:
        """
        Получает информацию о YouTube видео без скачивания
        """
        try:
            import json
            # Проверяем наличие cookies
            cookies_path = self.get_youtube_cookies_path()
            use_cookies = False
            
            if cookies_path and self.check_cookies_validity(cookies_path):
                use_cookies = True
                self.logger.info("🍪 Использую cookies для получения информации о видео")
            
            # Базовые параметры для yt-dlp
            base_cmd = [
                "yt-dlp",
                "--force-ipv4",
                "--user-agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "--extractor-args", "youtube:player_client=android",
                "--no-check-certificate",
                "--geo-bypass",
                "--dump-json"
            ]
            
            # Добавляем cookies если доступны
            if use_cookies:
                base_cmd.extend(["--cookies", str(cookies_path)])  # type: ignore[arg-type]
            
            # Добавляем URL в конец
            cmd = base_cmd + [url]
            
            self.logger.info("📋 Получаю информацию о YouTube видео...")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0 and result.stdout:
                try:
                    import json
                    info = json.loads(result.stdout)
                    title = info.get('title', 'Неизвестное видео')
                    duration = info.get('duration', 0)
                    uploader = info.get('uploader', 'Неизвестный автор')
                    
                    self.logger.info(f"✅ Информация получена: {title} ({duration}с) от {uploader}")
                    return {
                        'title': title,
                        'duration': duration,
                        'uploader': uploader,
                        'success': True
                    }
                except json.JSONDecodeError:
                    self.logger.error("❌ Ошибка парсинга JSON информации о видео")
                    return {'success': False, 'error': 'JSON parse error'}
            else:
                self.logger.error(f"❌ Не удалось получить информацию: {result.stderr}")
                
                # Пробуем альтернативный метод без Android клиента
                self.logger.info("🔄 Пробую альтернативный метод получения информации...")
                try:
                    alt_cmd = [
                        "yt-dlp",
                        "--force-ipv4",
                        "--user-agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                        "--extractor-args", "youtube:player_client=web",
                        "--no-check-certificate",
                        "--geo-bypass",
                        "--dump-json"
                    ]
                    
                    # Добавляем cookies если доступны
                    if use_cookies:
                        alt_cmd.extend(["--cookies", str(cookies_path)])  # type: ignore[arg-type]
                    
                    alt_cmd.append(url)
                    
                    self.logger.info("🔄 Альтернативная команда для получения информации...")
                    alt_result = subprocess.run(alt_cmd, capture_output=True, text=True, timeout=60)
                    
                    if alt_result.returncode == 0 and alt_result.stdout:
                        try:
                            import json
                            info = json.loads(alt_result.stdout)
                            title = info.get('title', 'Неизвестное видео')
                            duration = info.get('duration', 0)
                            uploader = info.get('uploader', 'Неизвестный автор')
                            
                            self.logger.info(f"✅ Информация получена альтернативным методом: {title} ({duration}с) от {uploader}")
                            return {
                                'title': title,
                                'duration': duration,
                                'uploader': uploader,
                                'success': True
                            }
                        except json.JSONDecodeError:
                            self.logger.error("❌ Ошибка парсинга JSON альтернативным методом")
                            return {'success': False, 'error': 'JSON parse error (alt method)'}
                    else:
                        self.logger.error(f"❌ Альтернативный метод также не сработал: {alt_result.stderr}")
                        return {'success': False, 'error': result.stderr}
                        
                except Exception as alt_e:
                    self.logger.error(f"❌ Ошибка альтернативного метода: {alt_e}")
                    return {'success': False, 'error': result.stderr}
                
        except Exception as e:
            self.logger.error(f"❌ Ошибка получения информации о видео: {e}")
            return {'success': False, 'error': str(e)}

    def check_youtube_accessibility(self, url: str) -> bool:
        """
        Проверяет доступность YouTube ссылки различными методами
        """
        try:
            # Проверяем наличие cookies
            cookies_path = self.get_youtube_cookies_path()
            use_cookies = False
            
            if cookies_path and self.check_cookies_validity(cookies_path):
                use_cookies = True
                self.logger.info("🍪 Использую cookies для проверки доступности")
            
            # Базовые параметры для yt-dlp
            base_cmd = [
                "yt-dlp",
                "--force-ipv4",
                "--user-agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "--extractor-args", "youtube:player_client=android",
                "--no-check-certificate",
                "--geo-bypass",
                "--list-formats"
            ]
            
            # Добавляем cookies если доступны
            if use_cookies:
                base_cmd.extend(["--cookies", str(cookies_path)])  # type: ignore[arg-type]
            
            # Добавляем URL в конец
            test_cmd = base_cmd + [url]
            
            self.logger.info("🔍 Проверяю доступность YouTube ссылку...")
            result = subprocess.run(test_cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                self.logger.info("✅ YouTube ссылка доступна")
                return True
            else:
                self.logger.warning(f"⚠️ YouTube ссылка недоступна: {result.stderr}")
                
                # Пробуем альтернативный метод с web клиентом
                self.logger.info("🔄 Пробую альтернативный метод проверки...")
                try:
                    alt_test_cmd = [
                        "yt-dlp",
                        "--force-ipv4",
                        "--user-agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                        "--extractor-args", "youtube:player_client=web",
                        "--no-check-certificate",
                        "--geo-bypass",
                        "--list-formats"
                    ]
                    
                    # Добавляем cookies если доступны
                    if use_cookies:
                        alt_test_cmd.extend(["--cookies", str(cookies_path)])  # type: ignore[arg-type]
                    
                    alt_test_cmd.append(url)
                    
                    alt_result = subprocess.run(alt_test_cmd, capture_output=True, text=True, timeout=60)
                    
                    if alt_result.returncode == 0:
                        self.logger.info("✅ YouTube ссылка доступна через альтернативный метод")
                        return True
                    else:
                        self.logger.warning(f"⚠️ YouTube ссылка недоступна и через альтернативный метод: {alt_result.stderr}")
                        return False
                        
                except Exception as alt_e:
                    self.logger.error(f"❌ Ошибка альтернативной проверки: {alt_e}")
                    return False
                
        except Exception as e:
            self.logger.error(f"❌ Ошибка проверки доступности YouTube: {e}")
            return False

    def _auto_load_brain_model(self):
        """Автоматически загружает модель мозга при инициализации"""
        try:
            # Проверяем, запущена ли модель через прямой запрос к API
            try:
                resp = requests.get(f"{self.lm_studio_url}/v1/models", timeout=10)
                if resp.status_code == 200:
                    data = resp.json()
                    model_loaded = False
                    for m in data.get("data", []):
                        if self.brain_model in m.get("id", "") and m.get("isLoaded", False):
                            model_loaded = True
                            # Сохраняем короткий ID модели для API вызовов
                            self.brain_model_id = m.get("id")
                            self.logger.info(f"✅ Модель мозга уже загружена: {os.path.basename(self.brain_model)} (ID: {self.brain_model_id})")
                            return
                else:
                    self.logger.warning(f"⚠️ Не удалось проверить статус моделей: {resp.status_code}")
            except Exception as e:
                self.logger.warning(f"⚠️ Ошибка проверки статуса моделей: {e}")
            
            # Если модель не загружена, пытаемся загрузить
            self.logger.info(f"🧠 Автоматически загружаю модель мозга: {os.path.basename(self.brain_model)}")
            
            # Пытаемся загрузить модель через API
            payload = {
                "model": self.brain_model,
                "load": True
            }
            
            try:
                resp = requests.post(f"{self.lm_studio_url}/v1/models/load", json=payload, timeout=30)
                if resp.status_code == 200:
                    self.logger.info("✅ Модель мозга успешно загружена через API")
                    # Получаем короткий ID модели после загрузки
                    self._update_brain_model_id()
                else:
                    self.logger.warning(f"⚠️ Не удалось загрузить модель через API: {resp.status_code}")
                    # Пробуем запустить через LM Studio
                    self.launch_model(self.brain_model)
                    self.logger.info("🔄 Запускаю модель через LM Studio...")
                    # Пытаемся получить ID модели после запуска
                    self._update_brain_model_id()
            except Exception as e:
                self.logger.warning(f"⚠️ Ошибка API загрузки модели: {e}")
                # Пробуем запустить через LM Studio
                self.launch_model(self.brain_model)
                self.logger.info("🔄 Запускаю модель через LM Studio...")
                # Пытаемся получить ID модели после запуска
                self._update_brain_model_id()
                
        except Exception as e:
            self.logger.error(f"❌ Ошибка автозагрузки модели мозга: {e}")
    
    def _update_brain_model_id(self):
        """Обновляет короткий ID модели мозга из API"""
        try:
            resp = requests.get(f"{self.lm_studio_url}/v1/models", timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                for m in data.get("data", []):
                    if self.brain_model in m.get("id", "") and m.get("isLoaded", False):
                        self.brain_model_id = m.get("id")
                        self.logger.info(f"✅ Обновлен ID модели мозга: {self.brain_model_id}")
                        return
                self.logger.warning("⚠️ Не удалось найти загруженную модель для получения ID")
            else:
                self.logger.warning(f"⚠️ Не удалось получить список моделей для обновления ID: {resp.status_code}")
        except Exception as e:
            self.logger.warning(f"⚠️ Ошибка обновления ID модели мозга: {e}")
    
    def _check_ffmpeg(self):
        """Проверяет наличие ffmpeg в системе для конвертации аудио"""
        try:
            result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                self.logger.info("✅ ffmpeg найден в системе")
            else:
                self.logger.warning("⚠️ ffmpeg найден, но не может быть запущен")
        except (subprocess.CalledProcessError, FileNotFoundError):
            self.logger.warning("⚠️ ffmpeg не найден в системе. Установите ffmpeg для конвертации аудио.")
            self.logger.info("💡 Скачайте с https://ffmpeg.org/download.html")
        except Exception as e:
            self.logger.warning(f"⚠️ Ошибка проверки ffmpeg: {e}")
    
    def is_model_running(self, model_name: str) -> bool:
        """
        Проверяет, запущена ли модель в LM Studio через /v1/models
        """
        try:
            resp = requests.get(f"{self.lm_studio_url}/v1/models")
            if resp.status_code == 200:
                data = resp.json()
                for m in data.get("data", []):
                    if model_name in m.get("id", "") and m.get("isLoaded", False):
                        return True
            return False
        except Exception as e:
            self.logger.error(f"Ошибка проверки модели {model_name}: {e}")
            return False

    def get_model_context_info(self) -> Dict[str, int]:
        """
        Получает информацию о максимальном контексте модели из LM Studio API
        Возвращает словарь с max_context и safe_context
        """
        try:
            resp = requests.get(f"{self.lm_studio_url}/v1/models", timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                
                # Ищем нашу модель по ключевым словам
                target_model = None
                search_terms = ["huihui-qwen3-4b-thinking", "qwen3-4b", "thinking"]
                
                for m in data.get("data", []):
                    model_id = m.get("id", "").lower()
                    for term in search_terms:
                        if term.lower() in model_id:
                            target_model = m
                            self.logger.info(f"🎯 Найдена модель: {m.get('id')}")
                            break
                    if target_model:
                        break
                
                if target_model:
                    # Пытаемся получить информацию через запрос к модели
                    context_info = self._get_context_info_via_chat(target_model.get("id"))
                    if context_info:
                        return context_info
                    
                    # Если не удалось получить через чат, сохраняем ID модели для использования
                    if not hasattr(self, 'brain_model_id') or not self.brain_model_id:
                        self.brain_model_id = target_model.get("id")
                        self.logger.info(f"✅ Сохранен ID модели мозга: {self.brain_model_id}")
                
            # Если не удалось получить информацию, используем значения по умолчанию
            self.logger.warning("⚠️ Не удалось получить информацию о контексте модели, используем значения по умолчанию")
            return {
                "max_context": 262144,
                "safe_context": 32768
            }
        except Exception as e:
            self.logger.warning(f"⚠️ Ошибка получения информации о контексте модели: {e}")
            return {
                "max_context": 262144,
                "safe_context": 32768
            }

    def _get_context_info_via_chat(self, model_id: str) -> Optional[Dict[str, int]]:
        """
        Пытается получить информацию о контексте через запрос к модели
        """
        try:
            # Простой запрос для получения информации о модели
            payload = {
                "model": model_id,
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 1,
                "temperature": 0
            }
            
            resp = requests.post(f"{self.lm_studio_url}/v1/chat/completions", json=payload, timeout=10)
            
            if resp.status_code == 200:
                data = resp.json()
                
                # Проверяем поле stats
                stats = data.get("stats", {})
                if stats:
                    # Ищем информацию о контексте в stats
                    context_length = None
                    if "context_length" in stats:
                        context_length = stats["context_length"]
                    elif "max_context" in stats:
                        context_length = stats["max_context"]
                    elif "max_tokens" in stats:
                        context_length = stats["max_tokens"]
                    
                    if context_length:
                        safe_context = max(context_length // 8, 32768)
                        self.logger.info(f"✅ Найден context_length в stats: {context_length}")
                        return {
                            "max_context": context_length,
                            "safe_context": safe_context
                        }
                
                # Проверяем другие поля на предмет информации о контексте
                for key, value in data.items():
                    if isinstance(value, dict) and ("context" in key.lower() or "token" in key.lower()):
                        self.logger.debug(f"🔍 Проверяем поле {key}: {value}")
                
                self.logger.debug("❌ Информация о контексте не найдена в ответе модели")
                return None
            else:
                self.logger.warning(f"❌ Ошибка запроса к модели: {resp.status_code}")
                return None
                
        except Exception as e:
            self.logger.warning(f"❌ Ошибка получения информации через чат: {e}")
            return None

    def _initialize_dynamic_context(self):
        """
        Инициализирует динамические параметры контекста на основе информации о модели
        """
        try:
            context_info = self.get_model_context_info()
            self.max_context_length = context_info["max_context"]
            self.safe_context_length = context_info["safe_context"]
            self.logger.info(f"📊 Контекст инициализирован: максимум {self.max_context_length:,}, безопасный {self.safe_context_length:,}")
        except Exception as e:
            self.logger.warning(f"⚠️ Ошибка инициализации динамического контекста: {e}")
            # Оставляем значения по умолчанию
            self.max_context_length = 262144
            self.safe_context_length = 32768

    def _trim_context_if_needed(self):
        """
        Обрезает контекст если он превышает безопасные лимиты
        Использует self.current_context_length (total_tokens) для проверки
        """
        if self.current_context_length > self.max_context_length:
            # Критический лимит - агрессивная обрезка
            self.conversation_history = self.conversation_history[-2:]  # Оставляем только 2 последних сообщения
            self.logger.warning(f"Критическое превышение контекста ({self.current_context_length:,} > {self.max_context_length:,}) - агрессивная обрезка истории")
        elif self.current_context_length > self.safe_context_length:
            # Превышение безопасного лимита - аккуратная обрезка
            self.conversation_history = self.conversation_history[-5:]  # Оставляем только 5 последних сообщений
            self.logger.warning(f"Превышение безопасного контекста ({self.current_context_length:,} > {self.safe_context_length:,}) - аккуратная обрезка истории")
        elif self.current_context_length > self.safe_context_length * 0.8:
            # Приближение к безопасному лимиту - профилактическая обрезка
            self.conversation_history = self.conversation_history[-10:]  # Оставляем только 10 последних сообщений
            self.logger.info(f"Приближение к безопасному лимиту ({self.current_context_length:,} > {self.safe_context_length * 0.8:,}) - профилактическая обрезка истории")

    def launch_model(self, model_path: str):
        """
        Запускает модель через LM Studio (локально, subprocess)
        """
        try:
            # threading уже импортирован в начале файла
            lmstudio_exe = os.getenv("LMSTUDIO_EXE", r"C:\Program Files\LM Studio\LM Studio.exe")
            self.logger.info(f"Запускаю модель: {model_path}")
            threading.Thread(target=lambda: os.system(f'"{lmstudio_exe}" --model "{model_path}"'), daemon=True).start()
        except Exception as e:
            self.logger.error(f"Ошибка запуска модели: {e}")

    def ask_qwen(self, question: str) -> Optional[str]:
        """Запрос к Qwen для генерации промтов изображений
        Использует основной мозг для генерации промтов
        """
        # Используем основной мозг для генерации промтов
        image_model = self.brain_model_id if hasattr(self, 'brain_model_id') and self.brain_model_id else self.brain_model
        payload = {
            "model": image_model,
            "messages": [
                {"role": "system", "content": "Ты — ассистент для генерации идеальных промтов для Stable Diffusion. Твоя задача — создать идеальный промт для генерации изображения на основе запроса пользователя. ВАЖНО: prompt и negative_prompt должны быть ТОЛЬКО на английском языке, иначе будет ошибка! Формируй промт и настройки строго в формате JSON: {\"prompt\":..., \"negative_prompt\":..., \"params\":{...}}. Не добавляй ничего лишнего!"},
                {"role": "user", "content": f"Вопрос: {question}\n\nВАЖНО: prompt и negative_prompt должны быть ТОЛЬКО на английском языке! Если они не на английском — это ошибка!"}
            ],
            "temperature": 0.2,
            "max_tokens": 1024,
            "stream": False
        }
        try:
            resp = requests.post(f"{self.lm_studio_url}/v1/chat/completions", json=payload, headers={"Content-Type": "application/json"})
            if resp.status_code == 200:
                result = resp.json()
                return result["choices"][0]["message"]["content"].strip()
            else:
                self.logger.error(f"Ошибка Qwen: {resp.status_code} - {resp.text}")
                return None
        except Exception as e:
            self.logger.error(f"Ошибка запроса к Qwen: {e}")
            return None

    def get_youtube_cookies_path(self) -> Optional[str]:
        """
        Получает путь к файлу cookies для YouTube
        Возвращает путь к файлу или None если файл не найден
        """
        cookies_file = "youtube_cookies.txt"
        
        # Сначала ищем в текущей рабочей директории
        cookies_path = os.path.join(os.getcwd(), cookies_file)
        if os.path.exists(cookies_path) and os.path.getsize(cookies_path) > 0:
            self.logger.info(f"🍪 Найден файл cookies в рабочей директории: {cookies_file}")
            return cookies_path
        
        # Затем ищем в директории скрипта
        cookies_path = os.path.join(os.path.dirname(__file__), cookies_file)
        if os.path.exists(cookies_path) and os.path.getsize(cookies_path) > 0:
            self.logger.info(f"🍪 Найден файл cookies в директории скрипта: {cookies_file}")
            return cookies_path
        
        # Если файл не найден нигде
        self.logger.info(f"ℹ️ Файл cookies не найден: {cookies_file}")
        return None

    def check_cookies_validity(self, cookies_path: str) -> bool:
        """
        Проверяет валидность файла cookies
        """
        try:
            with open(cookies_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Проверяем базовую структуру
            if not content.strip():
                return False
                
            # Проверяем наличие YouTube доменов
            youtube_domains = ['youtube.com', '.youtube.com', 'google.com', '.google.com']
            has_youtube = any(domain in content for domain in youtube_domains)
            
            if not has_youtube:
                self.logger.warning("⚠️ В файле cookies не найдены домены YouTube")
                return False
                
            # Проверяем формат (должен содержать табуляции)
            if '\t' not in content:
                self.logger.warning("⚠️ Неверный формат файла cookies (отсутствуют табуляции)")
                return False
                
            self.logger.info("✅ Файл cookies валиден")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка проверки cookies: {e}")
            return False

    def suggest_cookies_update(self):
        """
        Предлагает пользователю обновить cookies
        """
        self.logger.info("💡 Для улучшения работы с YouTube рекомендуется:")
        self.logger.info("   1. Запустить: python extract_chrome_cookies.py")
        self.logger.info("   2. Закрыть Chrome перед извлечением")
        self.logger.info("   3. Войти в YouTube через VPN")
        self.logger.info("   4. Cookies обновляются каждые 2-3 месяца")

    def generate_image_stable_diffusion(self, prompt: str, negative_prompt: str, params: dict) -> Optional[str]:
        """Генерация изображения через прямую интеграцию со Stable Diffusion"""
        start_time = time.time()
        
        # Автоматически включаем генерацию изображений при необходимости
        if not getattr(self, 'use_image_generation', False):
            self.logger.info("🔧 Автоматически включаю генерацию изображений")
            self.use_image_generation = True
            # Запускаем таймер автоматического выключения
            self.auto_disable_tools("image_generation")
        
        # Параметры по умолчанию
        default_params = {
            "seed": -1,
            "steps": 30,
            "width": 1024,
            "height": 1024,
            "cfg": 7.0,
            "sampler_name": "dpmpp_2m",
            "scheduler": "karras"
        }
        
        # Обновляем параметры пользовательскими значениями
        gen_params = default_params.copy()
        gen_params.update(params)
        
        # Исправляем seed если он -1
        if gen_params["seed"] == -1:
            import random
            gen_params["seed"] = random.randint(0, 2**32 - 1)
            self.logger.info(f"🎲 Сгенерирован случайный seed: {gen_params['seed']}")
        
        self.logger.info(f"🔧 Параметры генерации: {gen_params}")
        
        try:
            # Устанавливаем необходимые зависимости
            self._install_diffusers_dependencies()
            
            # Импортируем необходимые библиотеки (рекомендованные подмодули для совместимости с Pylance)
            from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipeline  # type: ignore
            from diffusers.schedulers.scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler  # type: ignore
            import torch
            
            # Путь к модели
            model_path = os.getenv("STABLE_DIFFUSION_MODEL_PATH", "J:\\ComfyUI\\models\\checkpoints\\novaAnime_v20.safetensors")
            
            # Проверяем существование модели
            if not os.path.exists(model_path):
                self.logger.error(f"❌ Модель не найдена: {model_path}")
                return None
            
            self.logger.info(f"📦 Загружаю модель: {model_path}")
            
            # Загружаем pipeline
            pipe = StableDiffusionPipeline.from_single_file(
                model_path,
                torch_dtype=torch.float16,
                use_safetensors=True
            )
            
            # Перемещаем на GPU если доступен
            if torch.cuda.is_available():
                pipe = pipe.to("cuda")
                self.logger.info("🚀 Модель перемещена на GPU")
            else:
                self.logger.warning("⚠️ GPU недоступен, использую CPU")
            
            # Настраиваем scheduler
            if gen_params["sampler_name"] == "dpmpp_2m":
                pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
                self.logger.info("⚙️ Использую DPMSolverMultistepScheduler")
            
            # Генерируем изображение
            self.logger.info(f"🎨 Генерирую изображение: {prompt[:50]}...")

            result = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=gen_params["steps"],
                guidance_scale=gen_params["cfg"],
                width=gen_params["width"],
                height=gen_params["height"],
                generator=torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(gen_params["seed"])
            )

            # Получаем изображение: результат pipe может быть объектом с атрибутом images или кортежем (image, extras)
            image = None
            try:
                imgs = getattr(result, 'images', None)
                if imgs:
                    image = imgs[0]
                elif isinstance(result, (tuple, list)) and len(result) > 0:
                    image = result[0]
            except Exception:
                image = None

            if image is None:
                raise RuntimeError('Не удалось получить изображение из результата pipeline')

            # Нормализация: если image — numpy array или torch tensor, конвертируем в PIL.Image
            try:
                import numpy as _np
                try:
                    import torch as _torch
                except Exception:
                    _torch = None

                if hasattr(image, 'save'):
                    img_to_save = image
                elif _np and isinstance(image, _np.ndarray):
                    img_to_save = Image.fromarray(image.astype('uint8'))
                elif _torch and _torch.is_tensor(image):
                    arr = image.detach().cpu().numpy()
                    img_to_save = Image.fromarray(arr.astype('uint8'))
                else:
                    # Попытка сконвертировать общим способом
                    img_to_save = Image.fromarray(_np.array(image).astype('uint8'))
            except Exception:
                # В крайнем случае пытаемся работать напрямую — пусть вызов .save выбросит понятную ошибку
                img_to_save = image
            # Подсказка для статического анализатора: гарантируем, что img_to_save рассматривается как PIL.Image
            try:
                from typing import cast
                img_to_save = cast(Image.Image, img_to_save)
            except Exception:
                pass
            
            # Сохраняем изображение
            output_dir = os.path.join(os.path.dirname(__file__), "Images", "generated")
            os.makedirs(output_dir, exist_ok=True)
            
            filename = f"ConsoleTest_{gen_params['seed']}.png"
            output_path = os.path.join(output_dir, filename)
            
            # Сохраняем PIL.Image
            try:
                # Если уже PIL.Image
                if isinstance(img_to_save, Image.Image):
                    img_to_save.save(output_path)
                    self.logger.info(f"💾 Изображение сохранено: {output_path}")
                else:
                    # Пытаемся сохранить оригинальный объект как PIL
                    if isinstance(image, Image.Image):
                        image.save(output_path)
                        self.logger.info(f"💾 Изображение сохранено (fallback): {output_path}")
                    else:
                        import numpy as _np
                        Image.fromarray(_np.array(image).astype('uint8')).save(output_path)
                        self.logger.info(f"💾 Изображение сохранено (converted fallback): {output_path}")
            except Exception:
                self.logger.error(f"❌ Не удалось сохранить изображение ни одним из способов")
            
            # Автоматически открываем изображение
            try:
                subprocess.run(["start", output_path], shell=True, check=True)
                self.logger.info("🖼️ Изображение автоматически открыто")
            except Exception as e:
                self.logger.warning(f"⚠️ Не удалось открыть изображение: {e}")
            
            # Конвертируем в base64
            buf = BytesIO()
            try:
                if isinstance(img_to_save, Image.Image):
                    img_to_save.save(buf, format="PNG")
                elif isinstance(image, Image.Image):
                    image.save(buf, format="PNG")
                else:
                    import numpy as _np
                    Image.fromarray(_np.array(image).astype('uint8')).save(buf, format="PNG")
            except Exception:
                self.logger.error("❌ Не удалось сконвертировать изображение в буфер PNG")
            img_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
            
            return img_b64
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка генерации изображения: {e}")
            import traceback
            traceback.print_exc()
            return None
        finally:
            # Записываем метрику производительности
            response_time = time.time() - start_time
            self.add_performance_metric("image_generation", response_time)
            self.logger.info(f"🎨 Изображение сгенерировано за {response_time:.2f} сек")

    def generate_video_stable_diffusion(self, prompt: str, negative_prompt: str, params: dict) -> Optional[str]:
        """Генерация видео через прямую интеграцию со Stable Diffusion"""
        start_time = time.time()
        
        # Автоматически включаем генерацию изображений при необходимости
        if not getattr(self, 'use_image_generation', False):
            self.logger.info("🔧 Автоматически включаю генерацию изображений")
            self.use_image_generation = True
            # Запускаем таймер автоматического выключения
            self.auto_disable_tools("image_generation")
        
        # Параметры по умолчанию для видео
        default_params = {
            "seed": -1,
            "steps": 20,
            "width": 512,
            "height": 512,
            "cfg": 7.0,
            "num_frames": 24,
            "fps": 8,
            "key_frames": 4
        }
        
        # Обновляем параметры пользовательскими значениями
        gen_params = default_params.copy()
        gen_params.update(params)
        
        # Исправляем seed если он -1
        if gen_params["seed"] == -1:
            import random
            gen_params["seed"] = random.randint(0, 2**32 - 1)
            self.logger.info(f"🎲 Сгенерирован случайный seed: {gen_params['seed']}")
        
        self.logger.info(f"🔧 Параметры генерации видео: {gen_params}")
        
        try:
            # Устанавливаем необходимые зависимости
            self._install_diffusers_dependencies()
            
            # Импортируем необходимые библиотеки (рекомендованные подмодули для совместимости с Pylance)
            from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipeline  # type: ignore
            from diffusers.schedulers.scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler  # type: ignore
            import torch
            from PIL import Image
            import numpy as np
            import imageio  # type: ignore
            
            # Путь к модели
            model_path = os.getenv("STABLE_DIFFUSION_MODEL_PATH", "J:\\ComfyUI\\models\\checkpoints\\novaAnime_v20.safetensors")
            
            # Проверяем существование модели
            if not os.path.exists(model_path):
                self.logger.error(f"❌ Модель не найдена: {model_path}")
                return None
            
            self.logger.info(f"📦 Загружаю модель: {model_path}")
            
            # Загружаем pipeline
            pipe = StableDiffusionPipeline.from_single_file(
                model_path,
                torch_dtype=torch.float16,
                use_safetensors=True
            )
            
            # Перемещаем на GPU если доступен
            if torch.cuda.is_available():
                pipe = pipe.to("cuda")
                self.logger.info("🚀 Модель перемещена на GPU")
            else:
                self.logger.warning("⚠️ GPU недоступен, использую CPU")
            
            # Настраиваем scheduler
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
            self.logger.info("⚙️ Использую DPMSolverMultistepScheduler")
            
            # Параметры генерации
            generation_config = {
                "width": gen_params["width"],
                "height": gen_params["height"],
                "num_inference_steps": gen_params["steps"],
                "guidance_scale": gen_params["cfg"],
                "num_images_per_prompt": 1
            }
            
            self.logger.info(f"🎬 Генерирую {gen_params['num_frames']} кадров для видео...")
            
            frames = []
            key_frames = gen_params["key_frames"]
            
            # Создаем вариации промпта для ключевых кадров
            key_prompts = [
                prompt,
                self._add_dynamic_elements(prompt, 1, key_frames),
                self._add_dynamic_elements(prompt, 2, key_frames),
                self._add_dynamic_elements(prompt, 3, key_frames)
            ]
            
            # Генерируем ключевые кадры
            for i in range(key_frames):
                seed = gen_params["seed"] + i * 50  # Разные seed'ы
                generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(seed)
                
                with torch.no_grad():
                    result = pipe(
                        prompt=key_prompts[i],
                        negative_prompt=negative_prompt,
                        generator=generator,
                        **generation_config
                    )
                
                # Безопасно извлекаем изображение из результата pipeline
                frame_img = None
                try:
                    imgs = getattr(result, 'images', None)
                    if imgs:
                        frame_img = imgs[0]
                    elif isinstance(result, (tuple, list)) and len(result) > 0:
                        frame_img = result[0]
                except Exception:
                    frame_img = None

                if frame_img is None:
                    raise RuntimeError('Не удалось получить кадр из результата pipeline')

                frames.append(frame_img)
                self.logger.info(f"  ✅ Ключевой кадр {i+1} готов")
            
            # Создаем интерполированные кадры между ключевыми кадрами
            frames_per_segment = gen_params["num_frames"] // (key_frames - 1)
            
            for segment in range(key_frames - 1):
                img1 = np.array(frames[segment])
                img2 = np.array(frames[segment + 1])
                
                for i in range(frames_per_segment):
                    # Вычисляем коэффициент интерполяции
                    t = i / frames_per_segment
                    
                    # Используем более плавную интерполяцию (ease-in-out)
                    t_smooth = 3 * t * t - 2 * t * t * t
                    
                    # Интерполяция между двумя изображениями
                    interpolated_array = img1 * (1 - t_smooth) + img2 * t_smooth
                    
                    # Конвертируем обратно в PIL Image
                    interpolated_image = Image.fromarray(interpolated_array.astype(np.uint8))
                    frames.append(interpolated_image)
                    
                    frame_num = segment * frames_per_segment + i + 1
                    self.logger.info(f"  ✅ Кадр {frame_num}/{gen_params['num_frames']} готов (сегмент {segment+1}, интерполяция: {t_smooth:.2f})")
            
            # Добавляем последний ключевой кадр если нужно
            if len(frames) < gen_params["num_frames"]:
                frames.append(frames[-1])
                self.logger.info(f"  ✅ Добавлен финальный кадр")
            
            frames = frames[:gen_params["num_frames"]]  # Убеждаемся, что возвращаем нужное количество кадров
            
            # Создаем папку для выходных файлов
            output_dir = os.path.join(os.path.dirname(__file__), "Videos", "generated")
            os.makedirs(output_dir, exist_ok=True)
            
            # Сохраняем кадры
            self.logger.info("💾 Сохраняю кадры...")
            for i, frame in enumerate(frames):
                frame_path = os.path.join(output_dir, f"video_frame_{i:03d}.png")
                try:
                    if hasattr(frame, 'save'):
                        try:
                            from typing import cast
                            frame = cast(Image.Image, frame)
                        except Exception:
                            pass
                        frame.save(frame_path)
                    else:
                        # Попытка привести numpy array / tensor к PIL Image
                        import numpy as _np
                        try:
                            import torch as _torch
                        except Exception:
                            _torch = None

                        if _torch and _torch.is_tensor(frame):
                            arr = frame.detach().cpu().numpy()
                            Image.fromarray(arr.astype('uint8')).save(frame_path)
                        elif isinstance(frame, _np.ndarray):
                            Image.fromarray(frame.astype('uint8')).save(frame_path)
                        else:
                            Image.fromarray(_np.array(frame).astype('uint8')).save(frame_path)

                    self.logger.info(f"  💾 Кадр {i+1} сохранен: {frame_path}")
                except Exception as e:
                    self.logger.warning(f"⚠️ Не удалось сохранить кадр {i+1}: {e}")
            
            # Создаем видео
            video_path = os.path.join(output_dir, f"ConsoleVideo_{gen_params['seed']}.mp4")
            self.logger.info(f"🎬 Создаю видео: {video_path}")
            
            # Конвертируем PIL изображения в numpy массивы
            video_frames = []
            for frame in frames:
                frame_array = np.array(frame)
                video_frames.append(frame_array)
            
            # Создаем видео с высоким качеством
            imageio.mimsave(video_path, video_frames, fps=gen_params["fps"], quality=8)
            
            self.logger.info(f"✅ Видео создано: {video_path}")
            
            # Автоматически открываем видео
            try:
                subprocess.run(["start", video_path], shell=True, check=True)
                self.logger.info("🎬 Видео автоматически открыто")
            except Exception as e:
                self.logger.warning(f"⚠️ Не удалось открыть видео: {e}")
            
            return video_path
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка генерации видео: {e}")
            import traceback
            traceback.print_exc()
            return None
        finally:
            # Записываем метрику производительности
            response_time = time.time() - start_time
            self.add_performance_metric("video_generation", response_time)
            self.logger.info(f"🎬 Видео сгенерировано за {response_time:.2f} сек")

    def _add_dynamic_elements(self, prompt, frame_index, total_frames):
        """Добавляет динамические элементы к промпту в зависимости от номера кадра"""
        
        # Базовые динамические элементы для разных типов промптов
        dynamic_elements = {
            "pose": [
                "slight head turn", "head turning", "looking to the side", "looking up", "looking down",
                "slight body movement", "body turning", "arm movement", "hand gesture", "finger movement",
                "eye movement", "blinking", "mouth movement", "smile change", "expression change"
            ],
            "lighting": [
                "slight lighting change", "light shift", "shadow movement", "highlight change",
                "ambient light variation", "light intensity change", "color temperature shift"
            ],
            "camera": [
                "slight camera movement", "camera angle change", "zoom effect", "perspective shift",
                "depth change", "focus adjustment", "blur variation"
            ],
            "motion": [
                "motion blur", "movement lines", "wind effect", "hair movement", "clothing movement",
                "particle effects", "energy flow", "magical effects", "sparkle effects"
            ]
        }
        
        # Определяем тип промпта
        prompt_lower = prompt.lower()
        
        # Выбираем подходящие динамические элементы
        if any(word in prompt_lower for word in ["anime", "girl", "boy", "character", "person"]):
            # Для персонажей добавляем движения и выражения
            elements = dynamic_elements["pose"] + dynamic_elements["motion"]
        elif any(word in prompt_lower for word in ["landscape", "nature", "scenery", "background"]):
            # Для пейзажей добавляем изменения освещения и камеры
            elements = dynamic_elements["lighting"] + dynamic_elements["camera"]
        else:
            # Для остальных используем все элементы
            elements = dynamic_elements["pose"] + dynamic_elements["lighting"] + dynamic_elements["camera"] + dynamic_elements["motion"]
        
        # Выбираем элемент в зависимости от номера кадра
        if elements:
            # Равномерно распределяем элементы по кадрам
            element_index = int((frame_index / total_frames) * len(elements))
            selected_element = elements[element_index % len(elements)]
            
            # Добавляем элемент к промпту
            enhanced_prompt = f"{prompt}, {selected_element}"
            
            # Добавляем интенсивность изменения в зависимости от прогресса
            progress = frame_index / total_frames
            if progress > 0.5:
                enhanced_prompt += ", subtle animation"
            
            return enhanced_prompt
        
        return prompt
    
    def _install_diffusers_dependencies(self):
        """Устанавливает необходимые зависимости для diffusers"""
        try:
            import diffusers
            import torch
            self.logger.info("✅ diffusers и torch уже установлены")
            return
        except ImportError:
            self.logger.info("📦 Устанавливаю зависимости для diffusers...")
            
            try:
                subprocess.run([_sys.executable, "-m", "pip", "install", "diffusers", "transformers", "torch", "torchvision", "accelerate", "safetensors"], 
                             check=True, capture_output=True)
                self.logger.info("✅ Зависимости установлены успешно")
            except subprocess.CalledProcessError as e:
                self.logger.error(f"❌ Ошибка установки зависимостей: {e}")
                raise

    def show_image_base64_temp(self, b64img: str):
        """Показать изображение из base64 на 5 секунд"""
        try:
            # В веб-режиме отключаем всплывающее окно показа
            if not getattr(self, 'show_images_locally', True):
                return
            img = Image.open(BytesIO(base64.b64decode(b64img)))
            img.show()
            time.sleep(5)
            img.close()
        except Exception as e:
            self.logger.error(f"Ошибка показа изображения: {e}")
    def find_new_audio(self) -> Optional[str]:
        """Находит новый аудиофайл для обработки"""
        audio_extensions = ['.mp3', '.wav', '.m4a', '.flac', '.ogg', '.aac']
        
        # Ищем в папке Audio
        audio_dir = os.path.join(self.base_dir, 'Audio')
        if os.path.exists(audio_dir):
            for file in os.listdir(audio_dir):
                if any(file.lower().endswith(ext) for ext in audio_extensions):
                    # Проверяем, что файл не помечен как обработанный
                    if '.used' not in file and not file.endswith('.used'):
                        file_path = os.path.join(audio_dir, file)
                        # Проверяем, что файл действительно существует и не пустой
                        if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                            return file_path
        
        return ""

    def mark_audio_used(self, audio_path: str):
        """Удаляет аудиофайл после обработки"""
        try:
            if os.path.exists(audio_path):
                # Удаляем файл полностью
                os.remove(audio_path)
                self.logger.info(f"✅ Аудиофайл удален после обработки: {os.path.basename(audio_path)}")
        except Exception as e:
            self.logger.error(f"❌ Ошибка при удалении аудиофайла: {e}")

    def transcribe_audio_whisper(self, audio_path: str, lang: str = "ru", use_separator: bool = True) -> Optional[str]:
        """
        Распознаёт аудио через whisper-cli. Если use_separator=True, предварительно выделяет вокал через audio-separator.
        Возвращает текст транскрипта (выводит только один раз при получении).
        """
        start_time = time.time()
        
        # Автоматически включаем audio модель при необходимости
        if not getattr(self, 'use_audio', False):
            self.logger.info("🔧 Автоматически включаю audio модель")
            self.use_audio = True
            # Запускаем таймер автоматического выключения
            self.auto_disable_tools("audio")
        
        # Проверяем и загружаем whisper модель если нужно
        if not self.check_whisper_setup():
            return "[Whisper error] Проблемы с настройкой Whisper. Проверьте наличие whisper-cli.exe и модели."
        
        try:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            exe_path = os.path.join(base_dir, "Release", "whisper-cli.exe")
            model_path = os.path.join(base_dir, "models", "whisper-large-v3-q8_0.gguf")
            
            # Проверяем существование whisper-cli.exe
            if not os.path.exists(exe_path):
                return "[Whisper error] Не найден whisper-cli.exe в папке Release"
            
            # Проверяем существование модели
            if not os.path.exists(model_path):
                return "[Whisper error] Не найдена модель whisper в папке models"
            
            audio_for_whisper = audio_path
            
            # Используем audio separator если включен
            if use_separator:
                try:
                    from audio_separator.separator import Separator
                    self.logger.info("🎵 Использую audio-separator для выделения вокала...")
                    out_dir = os.path.join(base_dir, "separated")
                    os.makedirs(out_dir, exist_ok=True)
                    separator = Separator(output_dir=out_dir)
                    separator.load_model(model_filename='htdemucs_ft.yaml')
                    output_files = separator.separate(audio_path)
                    vocals_path = None
                    for file_path in output_files:
                        if '(Vocals)' in os.path.basename(file_path):
                            vocals_path = file_path  # audio-separator возвращает полный путь
                            self.logger.info(f"[SUCCESS] Вокал найден: {vocals_path}")
                            break
                    if not vocals_path:
                        self.logger.warning("⚠️ Не удалось найти файл с голосом после разделения дорожек, использую оригинал")
                    else:
                        audio_for_whisper = vocals_path
                except ImportError:
                    self.logger.warning("⚠️ Не установлена библиотека audio-separator. Пытаюсь установить автоматически...")
                    try:
                        import subprocess
                        subprocess.run([_sys.executable, "-m", "pip", "install", "audio-separator"], 
                                     capture_output=True, check=True)
                        self.logger.info("✅ audio-separator успешно установлен")
                        # Повторно пытаемся импортировать
                        from audio_separator.separator import Separator
                        self.logger.info("🎵 Использую audio-separator для выделения вокала...")
                        out_dir = os.path.join(base_dir, "separated")
                        os.makedirs(out_dir, exist_ok=True)
                        separator = Separator(output_dir=out_dir)
                        separator.load_model(model_filename='htdemucs_ft.yaml')
                        output_files = separator.separate(audio_path)
                        vocals_path = None
                        for file_path in output_files:
                            if '(Vocals)' in os.path.basename(file_path):
                                vocals_path = file_path  # audio-separator возвращает полный путь
                                self.logger.info(f"[SUCCESS] Вокал найден: {vocals_path}")
                                break
                        if not vocals_path:
                            self.logger.warning("⚠️ Не удалось найти файл с голосом после разделения дорожек, использую оригинал")
                        else:
                            audio_for_whisper = vocals_path
                    except Exception as install_error:
                        self.logger.warning(f"⚠️ Не удалось установить audio-separator: {install_error}")
                        self.logger.info("ℹ️ Продолжаю без разделения дорожек")
                except Exception as e:
                    self.logger.warning(f"⚠️ Ошибка audio-separator: {e}, использую оригинал")
            
            # Конвертируем аудио в WAV формат для Whisper (если это не уже WAV)
            if not audio_for_whisper.lower().endswith('.wav'):
                wav_path = self.convert_audio_to_wav(audio_for_whisper)
                if wav_path:
                    audio_for_whisper = wav_path
                    self.logger.info(f"✅ Аудио конвертировано в WAV: {os.path.basename(wav_path)}")
                else:
                    self.logger.warning("⚠️ Не удалось конвертировать в WAV, использую оригинал")
            else:
                self.logger.info("✅ Аудио уже в WAV формате")
            
            # Переименовать используемый файл в .used.расширение
            base_used, ext_used = os.path.splitext(audio_for_whisper)
            used_path = base_used + ".used" + ext_used
            try:
                if os.path.exists(audio_for_whisper):
                    os.rename(audio_for_whisper, used_path)
                    self.logger.info(f"✅ Аудиофайл переименован в: {os.path.basename(used_path)}")
                else:
                    self.logger.warning(f"⚠️ Аудиофайл не найден для переименования: {audio_for_whisper}")
                    used_path = audio_for_whisper
            except Exception as e:
                self.logger.error(f"Ошибка переименования аудио после whisper: {e}")
                # Если не удалось переименовать, используем оригинальный файл
                used_path = audio_for_whisper
            
            # Теперь используем used_path для whisper
            cmd = [exe_path, "--model", model_path]
            if lang:
                cmd += ["--language", lang]
            cmd.append(used_path)
            self.logger.info(f"[INFO] Запуск Whisper: {' '.join(cmd)}")
            import subprocess
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, encoding="utf-8", errors="replace")
            transcript = result.stdout.strip() if result.stdout else ""
            if transcript:
                self.logger.info("\n=== ТРАНСКРИПТ АУДИО ===\n" + transcript)
                return transcript
            
            # Очистка временных файлов если был separator
            if use_separator and 'separated' in audio_for_whisper:
                try:
                    separated_dir = os.path.dirname(audio_for_whisper)
                    if os.path.exists(separated_dir):
                        shutil.rmtree(separated_dir)
                        self.logger.info("🧹 Временные файлы audio-separator очищены")
                except Exception as e:
                    self.logger.warning(f"⚠️ Не удалось очистить временные файлы: {e}")
            
            err = result.stderr.strip() if result.stderr else ""
            return f"[Whisper error] Не удалось получить транскрипт. STDERR: {err}"
        except Exception as e:
            error_msg = f"Исключение whisper-cli: {str(e)}"
            self.logger.error(error_msg)
            return f"[Whisper error] {error_msg}"
        finally:
            # Записываем метрику производительности
            response_time = time.time() - start_time
            self.add_performance_metric("whisper_transcription", response_time)
            self.logger.info(f"🎤 Whisper обработал за {response_time:.2f} сек")

    def convert_audio_to_wav(self, audio_path: str) -> Optional[str]:
        """
        Конвертирует аудиофайл в WAV формат для Whisper.
        Возвращает путь к WAV файлу или None при ошибке.
        """
        try:
            if not audio_path or not os.path.exists(audio_path):
                return None
            
            # Если уже WAV, не конвертируем
            if audio_path.lower().endswith('.wav'):
                return audio_path
            
            # Проверяем наличие ffmpeg
            try:
                subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
            except (subprocess.CalledProcessError, FileNotFoundError):
                self.logger.warning("⚠️ ffmpeg не найден в системе. Установите ffmpeg для конвертации аудио.")
                return None
            
            # Создаем временную папку для конвертации
            temp_dir = os.path.join(os.path.dirname(audio_path), "temp_convert")
            os.makedirs(temp_dir, exist_ok=True)
            
            # Имя выходного WAV файла
            base_name = os.path.splitext(os.path.basename(audio_path))[0]
            wav_path = os.path.join(temp_dir, f"{base_name}.wav")
            
            # Команда для конвертации через ffmpeg
            cmd = [
                'ffmpeg', '-i', audio_path,
                '-acodec', 'pcm_s16le',  # 16-bit PCM
                '-ar', '16000',          # 16kHz sample rate (оптимально для Whisper)
                '-ac', '1',              # моно
                '-y',                    # перезаписать существующий файл
                wav_path
            ]
            
            self.logger.info(f"🔄 Конвертирую аудио в WAV: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0 and os.path.exists(wav_path):
                self.logger.info(f"✅ Конвертация успешна: {os.path.basename(wav_path)}")
                return wav_path
            else:
                self.logger.error(f"❌ Ошибка конвертации: {result.stderr}")
                return None
                
        except Exception as e:
            self.logger.error(f"❌ Ошибка конвертации аудио в WAV: {e}")
            return None

    def check_whisper_setup(self) -> bool:
        """
        Проверяет настройку Whisper: наличие whisper-cli.exe и модели.
        Возвращает True если всё готово, False если есть проблемы.
        """
        try:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            exe_path = os.path.join(base_dir, "Release", "whisper-cli.exe")
            model_path = os.path.join(base_dir, "models", "whisper-large-v3-q8_0.gguf")
            
            # Проверяем whisper-cli.exe
            if not os.path.exists(exe_path):
                self.logger.error(f"❌ Не найден whisper-cli.exe в папке Release: {exe_path}")
                self.logger.info("💡 Скачайте whisper.cpp с https://github.com/ggerganov/whisper.cpp")
                return False
            
            # Проверяем модель
            if not os.path.exists(model_path):
                self.logger.warning(f"⚠️ Не найдена модель whisper в папке models: {model_path}")
                self.logger.info("🔄 Пытаюсь автоматически скачать модель...")
                if self.download_whisper_model():
                    self.logger.info("✅ Модель whisper успешно загружена")
                else:
                    self.logger.error("❌ Не удалось загрузить модель whisper")
                    self.logger.info("💡 Скачайте модель whisper-large-v3-q8_0.gguf вручную")
                    return False
            
            # Проверяем права на выполнение
            try:
                result = subprocess.run([exe_path, "--help"], capture_output=True, text=True, timeout=10)
                if result.returncode != 0:
                    self.logger.warning("⚠️ whisper-cli.exe не может быть запущен")
                    return False
            except Exception as e:
                self.logger.warning(f"⚠️ Ошибка запуска whisper-cli.exe: {e}")
                return False
            
            self.logger.info("✅ Whisper настройка проверена успешно")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка проверки настройки Whisper: {e}")
            return False

    def download_whisper_model(self) -> bool:
        """
        Автоматически скачивает модель whisper-large-v3-q8_0.gguf.
        Возвращает True если успешно, False если ошибка.
        """
        try:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            models_dir = os.path.join(base_dir, "models")
            os.makedirs(models_dir, exist_ok=True)
            
            model_name = "whisper-large-v3-q8_0.gguf"
            model_path = os.path.join(models_dir, model_name)
            
            # URL для скачивания модели (используем Hugging Face)
            model_url = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-q8_0.bin"
            
            self.logger.info(f"📥 Скачиваю модель whisper: {model_name}")
            self.logger.info(f"🔗 URL: {model_url}")
            
            # Скачиваем модель
            response = requests.get(model_url, stream=True, timeout=300)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(model_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            percent = (downloaded / total_size) * 100
                            self.logger.info(f"📊 Прогресс: {percent:.1f}% ({downloaded}/{total_size} байт)")
            
            self.logger.info(f"✅ Модель скачана: {model_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка скачивания модели whisper: {e}")
            return False

    def download_youtube_audio(self, url: str, out_dir: Optional[str] = None) -> str:
        """
        Скачивает аудиодорожку с YouTube по ссылке (использует yt-dlp)
        Возвращает путь к аудиофайлу или пустую строку
        """
        # subprocess уже импортирован в начале файла
        if out_dir is None:
            out_dir = os.path.join(os.path.dirname(__file__), "Audio")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "yt_audio.%(ext)s")
        # Проверяем наличие cookies
        cookies_path = self.get_youtube_cookies_path()
        use_cookies = False
        
        if cookies_path and self.check_cookies_validity(cookies_path):
            use_cookies = True
            self.logger.info("🍪 Использую cookies для аутентификации YouTube")
        else:
            self.logger.info("ℹ️ Cookies не найдены или невалидны, использую базовые параметры")
        
        # Базовые параметры для yt-dlp
        base_cmd = [
            "yt-dlp",
            "--force-ipv4",
            "--user-agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "--extractor-args", "youtube:player_client=android",  # Используем Android клиент
            "--no-check-certificate",  # Игнорируем SSL ошибки
            "--prefer-insecure",  # Предпочитаем HTTP
            "--geo-bypass",  # Обход геоблокировки
            "--geo-bypass-country", "US",  # Страна для обхода
            "-f", "bestaudio[ext=m4a]/bestaudio/best",
            "--extract-audio", "--audio-format", "wav",  # Сразу в WAV для Whisper
            "-o", out_path
        ]

        # Добавляем cookies если доступны
        if use_cookies:
            base_cmd.extend(["--cookies", str(cookies_path)])  # type: ignore[arg-type]
        
        # Добавляем URL в конец
        cmd = base_cmd + [url]
        
        try:
            self.logger.info(f"Скачиваю аудио с YouTube: {url}")
            # Логируем команду в одну строку для избежания обрезания
            cmd_str = " ".join(cmd)
            self.logger.info(f"Команда: {cmd_str}")
            
            # Запускаем с таймаутом
            result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=300)
            
            if result.stdout:
                self.logger.info(f"yt-dlp stdout: {result.stdout}")
            if result.stderr:
                self.logger.warning(f"yt-dlp stderr: {result.stderr}")
            
            # Найти скачанный файл
            for fname in os.listdir(out_dir):
                if fname.startswith("yt_audio") and fname.endswith(('.wav', '.m4a', '.mp3', '.ogg', '.flac')):
                    self.logger.info(f"✅ Аудио успешно скачано: {fname}")
                    return os.path.join(out_dir, fname)
            
            self.logger.warning("⚠️ Аудиофайл не найден после скачивания")
            return ""
            
        except subprocess.TimeoutExpired:
            self.logger.error("❌ Таймаут скачивания аудио (5 минут)")
            return ""
        except subprocess.CalledProcessError as e:
            self.logger.error(f"❌ Ошибка yt-dlp: {e}")
            if e.stderr:
                self.logger.error(f"stderr: {e.stderr}")
            return ""
        except Exception as e:
            self.logger.error(f"❌ Неожиданная ошибка скачивания аудио: {e}")
            
            # Пробуем альтернативный метод с другими параметрами
            self.logger.info("🔄 Пробую альтернативный метод скачивания...")
            try:
                alt_cmd = [
                    "yt-dlp",
                    "--force-ipv4",
                    "--user-agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                    "--extractor-args", "youtube:player_client=web",
                    "--no-check-certificate",
                    "--geo-bypass",
                    "--geo-bypass-country", "US",
                    "-f", "bestaudio",
                    "--extract-audio", "--audio-format", "wav",  # Сразу в WAV для Whisper
                    "-o", out_path
                ]

                # Добавляем cookies если доступны
                if use_cookies:
                    alt_cmd.extend(["--cookies", str(cookies_path)])  # type: ignore[arg-type]

                alt_cmd.append(url)
                
                # Логируем команду в одну строку
                alt_cmd_str = " ".join(alt_cmd)
                self.logger.info(f"Альтернативная команда: {alt_cmd_str}")
                result = subprocess.run(alt_cmd, check=True, capture_output=True, text=True, timeout=300)
                
                # Найти скачанный файл
                for fname in os.listdir(out_dir):
                    if fname.startswith("yt_audio") and fname.endswith(('.wav', '.m4a', '.mp3', '.ogg', '.flac')):
                        self.logger.info(f"✅ Аудио успешно скачано альтернативным методом: {fname}")
                        return os.path.join(out_dir, fname)
                        
            except Exception as alt_e:
                self.logger.error(f"❌ Альтернативный метод также не сработал: {alt_e}")
                
                # Пробуем третий метод с максимально простыми параметрами
                self.logger.info("🔄 Пробую третий метод (минимальные параметры)...")
                try:
                    simple_cmd = [
                        "yt-dlp",
                        "--force-ipv4",
                        "--user-agent", "Mozilla/0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                        "--no-check-certificate",
                        "-f", "bestaudio",
                        "--extract-audio", "--audio-format", "mp3",
                        "-o", out_path
                    ]
                    
                    # Добавляем cookies если доступны
                    if use_cookies:
                        simple_cmd.extend(["--cookies", str(cookies_path)])  # type: ignore[arg-type]
                    
                    simple_cmd.append(url)
                    
                    self.logger.info(f"Третий метод: {' '.join(simple_cmd)}")
                    result = subprocess.run(simple_cmd, check=True, capture_output=True, text=True, timeout=300)
                    
                    # Найти скачанный файл
                    for fname in os.listdir(out_dir):
                        if fname.startswith("yt_audio") and fname.endswith(('.m4a', '.mp3', '.wav', '.ogg', '.flac')):
                            self.logger.info(f"✅ Аудио успешно скачано третьим методом: {fname}")
                            return os.path.join(out_dir, fname)
                            
                except Exception as simple_e:
                    self.logger.error(f"❌ Третий метод также не сработал: {simple_e}")
            
            return ""
    def find_new_image(self) -> str:
        """
        Находит первое новое изображение (png или jpg) в папке Photos, игнорируя файлы с .used перед расширением
        """
        photos_dir = os.path.join(os.path.dirname(__file__), "Photos")
        if not os.path.exists(photos_dir):
            return ""
        for fname in os.listdir(photos_dir):
            lower = fname.lower()
            if lower.endswith(('.png', '.jpg', '.jpeg')) and '.used' not in os.path.splitext(lower)[0]:
                return os.path.join(photos_dir, fname)
        return ""

    def mark_image_used(self, image_path: str):
        """
        Переименовывает изображение, чтобы нейросеть его больше не использовала
        """
        if not image_path:
            return
        base, ext = os.path.splitext(image_path)
        new_path = base + ".used" + ext
        try:
            os.rename(image_path, new_path)
        except Exception as e:
            self.logger.error(f"Ошибка переименования изображения: {e}")
    def extract_think_content(self, text: str) -> Optional[str]:
        """
        Извлекает содержимое из блока <think> или альтернативных маркеров размышлений.
        Поддерживает форматы:
        - <think>...</think> 
        - <|begin_of_thought|>...<|end_of_thought|>
        - BEGIN_OF_THOUGHT...END_OF_THOUGHT

        Args:
            text: Входной текст для поиска блока размышлений

        Returns:
            Optional[str]: Содержимое блока размышлений или None если блок не найден
        """
        # Паттерны для поиска (case-insensitive)
        patterns = [
            r'<think>(.*?)</think>',
            r'<\|begin_of_thought\|>(.*?)<\|end_of_thought\|>',
            r'BEGIN_OF_THOUGHT(.*?)END_OF_THOUGHT'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                content = match.group(1).strip()
                if content:  # Проверяем что контент не пустой
                    return content
        
        return None

    def extract_first_json(self, text: str, allow_json_in_think: bool = False) -> str:
        """
        Извлекает первый корректный JSON-блок из текста.
        
        Args:
            text: Входной текст для поиска JSON
            allow_json_in_think: Искать ли JSON внутри блока think
            
        Returns:
            str: Найденный JSON или исходный текст если JSON не найден
        """
        # Сначала ищем think-блок и удаляем его из текста если он есть
        think_content = None
        clean_text = text
        
        patterns = [
            r'<think>.*?</think>',
            r'<\|begin_of_thought\|>.*?<\|end_of_thought\|>',
            r'BEGIN_OF_THOUGHT.*?END_OF_THOUGHT'
        ]
        
        for pattern in patterns:
            think_match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if think_match:
                think_content = self.extract_think_content(think_match.group(0))
                clean_text = re.sub(pattern, '', text, flags=re.DOTALL | re.IGNORECASE).strip()
                break
        
        # Сначала пробуем найти чистый JSON (без обрамлений)
        json_in_text = self._extract_json_from_text(clean_text)
        if json_in_text:
            return json_in_text
        
        # Если не нашли, пробуем искать JSON с обрамлением и удалять его
        json_with_wrapper = re.search(r'```(?:json)?\s*(.*?)\s*```', clean_text, re.DOTALL)
        if json_with_wrapper:
            potential_json = json_with_wrapper.group(1).strip()
            json_in_wrapper = self._extract_json_from_text(potential_json)
            if json_in_wrapper:
                return json_in_wrapper
            
        # Если разрешено и есть think-контент - ищем JSON там
        if allow_json_in_think and think_content:
            json_in_think = self._extract_json_from_text(think_content)
            if json_in_think:
                return json_in_think
        
        return text  # если не найдено, вернуть исходное
    
    def _extract_json_from_text(self, text: str) -> str:
        """
        Вспомогательная функция для извлечения JSON из текста
        """
        # re уже импортирован в начале файла
        stack = []
        start = None
        
        for i, c in enumerate(text):
            if c == '{':
                if not stack:
                    start = i
                stack.append('{')
            elif c == '}':
                if stack:
                    stack.pop()
                    if not stack and start is not None:
                        return text[start:i+1]
        
        return ""

    def _smart_json_parse(self, s: str):
        """Умный парсер JSON с несколькими попытками автокоррекции.

        Возвращает tuple (data_or_none, fixes_list).
        """
        logger.info(f"🔍 Парсинг JSON: {s[:200]}...")
        try:
            return json.loads(s), []
        except Exception as e:
            fixes = [f"Первый парсинг не удался: {e}"]

        # Попытка закрыть скобки
        open_braces = s.count('{')
        close_braces = s.count('}')
        if open_braces > close_braces:
            s2 = s + '}' * (open_braces - close_braces)
            fixes.append(f"Добавлено {open_braces - close_braces} }} для баланса скобок")
            try:
                return json.loads(s2), fixes
            except Exception:
                pass
        elif close_braces > open_braces:
            s2 = re.sub(r'}+$', '', s)
            fixes.append(f"Удалены лишние закрывающие скобки")
            try:
                return json.loads(s2), fixes
            except Exception:
                pass

        # Заменяем одинарные кавычки на двойные если это безопасно
        if "'" in s and '"' not in s:
            s2 = s.replace("'", '"')
            try:
                return json.loads(s2), fixes+['Заменены одинарные кавычки на двойные']
            except Exception:
                pass

        # Удаляем лишние запятые перед закрывающей скобкой
        s3 = re.sub(r',\s*([}\]])', r'\1', s)
        try:
            return json.loads(s3), fixes+['Удалены лишние запятые']
        except Exception:
            pass

        # Оборачиваем ключи в кавычки (грубая попытка)
        s4 = re.sub(r'([,{]\s*)([a-zA-Z0-9_]+)\s*:', r'\1"\2":', s3)
        try:
            return json.loads(s4), fixes+['Добавлены кавычки к ключам']
        except Exception:
            pass

        # Исправляем незакрытые строки
        s5 = re.sub(r'([^\"])\s*$', r'\1"', s4)
        try:
            return json.loads(s5), fixes+['Исправлены незакрытые строки']
        except Exception:
            pass

        # Финальная попытка с очисткой от непечатаемых символов
        s6 = re.sub(r'[^\x20-\x7E]', '', s5)
        try:
            return json.loads(s6), fixes+['Очищены непечатаемые символы']
        except Exception as e2:
            fixes.append(f"Не удалось распарсить даже после исправлений: {e2}")

        return None, fixes
    def __init__(self, lm_studio_url: str = "http://localhost:1234", 
                 google_api_key: str = "", google_cse_id: str = ""):
        """
        Инициализация оркестратора
        
        Args:
            lm_studio_url: URL сервера LM Studio
            google_api_key: API ключ Google Custom Search
            google_cse_id: ID поисковой системы Google CSE
        """
        self.lm_studio_url = lm_studio_url.rstrip("/")
        self.google_api_key = google_api_key
        self.google_cse_id = google_cse_id
        # unify logger usage for instance methods
        self.logger = logger
        self.conversation_history: List[Dict[str, Any]] = []
        self.brain_model = "J:/models-LM Studio/mradermacher/Huihui-Qwen3-4B-Thinking-2507-abliterated-GGUF/Huihui-Qwen3-4B-Thinking-2507-abliterated.Q4_K_S.gguf"
        self.brain_model_id = None  # Короткий ID модели для API вызовов
        self.use_separator = True  # По умолчанию True, чтобы убрать предупреждение Pylance
        self.use_image_generation = False  # По умолчанию отключена генерация изображений
        # Тумблеры функционала (визуал и аудио)
        self.use_vision = False
        self.use_audio = False
        # Управление локальным показом изображений (для веб-режима можно отключить)
        self.show_images_locally = True
        # Хранилище последнего сгенерированного изображения (base64) и ответа
        self.last_generated_image_b64 = None
        self.last_final_response = ""
        
        # Динамическое управление контекстом
        self.max_context_length = 262144  # Максимальный контекст (временно)
        self.safe_context_length = 32768   # Безопасный контекст (временно)
        self.current_context_length = 0    # Текущий размер контекста
        
        # Метрики производительности
        self.performance_metrics = []  # Список метрик производительности
        
        # Счетчик попыток для предотвращения зацикливания
        self.retry_count = 0
        self.max_retries = 3
        
        # Постоянная голосовая запись
        self.continuous_recording = False
        self.audio_queue = queue.Queue()
        self.recording_thread = None
        
        # Таймеры для автоматического выключения инструментов
        self.tool_timers = {}
        self.auto_disable_delay = 300  # Выключать инструменты через 5 минут после использования
        
        # Автоматически запускаем модель мозга при инициализации
        self._auto_load_brain_model()
        
        # Инициализируем динамические параметры контекста после загрузки модели
        self._initialize_dynamic_context()
        
        # Инициализируем базовую директорию
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Инициализируем ChromaDB для векторного хранилища
        self.chromadb_manager = ChromaDBManager(
            db_path=os.path.join(self.base_dir, "chroma_db"),
            use_gpu=True  # Включаем поддержку GPU
        )
        
        # Проверяем наличие ffmpeg для конвертации аудио
        self._check_ffmpeg()
        
        # Telegram Bot настройки
        self.telegram_bot_token = ""
        self.telegram_allowed_user_id = ""
        

        # Универсальный системный промпт для оркестратора
        self.system_prompt = """
ВЫ - ИНТЕЛЛЕКТУАЛЬНЫЙ АССИСТЕНТ С ДОСТУПОМ К ВЕКТОРНОЙ ПАМЯТИ (ChromaDB):

У вас есть доступ к векторной базе данных ChromaDB, которая хранит:
1. Все предыдущие диалоги с пользователем
2. Предпочтения пользователя, извлеченные из разговоров
3. Контекстную информацию для улучшения ответов

ВАЖНЫЕ ПРИНЦИПЫ РАБОТЫ С ПАМЯТЬЮ:
- ВСЕ диалоги автоматически сохраняются в память после каждого ответа
- Вы НЕ используете память напрямую в новых чатах - только если это релевантно
- Если пользователь спрашивает о чем-то, что обсуждалось ранее, вы можете вспомнить это из памяти
- Предпочтения пользователя помогают персонализировать ответы
- Память работает автоматически - вам не нужно явно обращаться к ней

ИНФОРМАЦИЯ О ГЕНЕРАЦИИ ИЗОБРАЖЕНИЙ:

Категории и теги для генерации изображений (каждый тег подписан, обязательные отмечены [!]):

[Универсальные] — базовые теги, почти всегда нужны для высокого качества:
- masterpiece [!] — всегда использовать для лучшего качества
- best quality [!] — всегда использовать для лучшего качества
- extremely detailed [!] — всегда использовать для детализации
- high quality [!] — всегда использовать для качества
- 4k / 8k / 16k resolution — высокое разрешение (опционально)
- dynamic pose — динамичная поза (опционально)
- random pose — случайная поза (опционально)
- various pose — разные позы (опционально)
- random composition — случайная композиция (опционально)
- random clothes — случайная одежда (опционально)
- no specific character — без конкретного персонажа (опционально)
- solo — один персонаж (опционально)
- multiple characters / group — группа персонажей (опционально)
- close-up — крупный план (опционально)
- full body — полный рост (опционально)
- upper body — по пояс (опционально)
- cropped to knees / cropped tight / half body — обрезка кадра (опционально)
- view from below / bird's eye view / side view / front view / back view — ракурс (опционально)
- floating / levitating — парящий (опционально)
- random background / abstract background / surreal background — фон (опционально)
- soft lighting / dramatic lighting / natural lighting — освещение (опционально)
- cinematic lighting — кинематографичное освещение (опционально)
- beautifully lit — красиво освещено (опционально)
- natural colors / vibrant colors / muted colors — цвета (опционально)
- atmospheric — атмосферно (опционально)
- detailed background — детализированный фон (опционально)
- intricately detailed — сложная детализация (опционально)
- ornate — украшения (опционально)
- simple background — минималистичный фон (опционально)
- medium breasts / small breasts / large breasts — размер груди (опционально)
- wide hips / slim hips / athletic build / petite — тип фигуры (опционально)
- cute face / beautiful eyes / expressive eyes / smile / neutral expression / serious expression — выражение лица (опционально)

[NSFW] — для откровенных сцен, использовать только если требуется:
- nude — обнажённая натура
- lewd — пошлость
- explicit — откровенность
- uncensored — без цензуры
- cleavage — декольте
- nipples visible — видны соски
- medium breasts / large breasts / small breasts — размер груди
- wide hips — широкие бёдра
- ass visible — видна попа
- sexy pose — сексуальная поза
- dynamic pose / random pose — динамика
- legs cropped to knees — акцент на ногах
- solo — один персонаж
- 1girl / 1boy / 1person — один персонаж без имени
- multiple girls / multiple boys — группа
- erotic / sensual / seductive pose — эротика
- bed scene / erotic setting / dim lighting — постельная сцена
- soft skin / smooth skin — мягкая кожа
- skin exposed — открытая кожа
- no clothes / minimal clothes / random clothes — одежда
- random background — случайный фон
- random hair color / natural hair color — цвет волос
- messy hair / flowing hair — растрёпанные волосы
- natural lighting / moody lighting / warm lighting — освещение

[NSFW - negative prompt] — всегда добавлять для фильтрации багов:
- worst quality [!]
- low quality [!]
- blurry [!]
- jpeg artifacts [!]
- watermark [!]
- signature [!]
- disfigured [!]
- malformed limbs [!]
- bad anatomy [!]
- poorly drawn face [!]
- extra limbs [!]
- missing limbs [!]
- out of frame [!]
- mutilated [!]
- mutated hands [!]
- extra fingers [!]
- text [!]
- error [!]
- cropped [!]
- duplicate [!]
- lowres [!]
- bad proportions [!]
- squint [!]
- grainy [!]
- ugly [!]

[SFW] — для безопасных сцен, без NSFW:
- sfw [!]
- clothed — одет(а)
- random clothes — случайная одежда
- casual clothes / elegant clothes / formal clothes — стиль одежды
- dynamic pose / random pose — динамика
- walking / sitting / standing / running / jumping — поза/движение
- smiling / happy expression / neutral expression — выражение лица
- cute face / beautiful eyes / expressive eyes — лицо
- solo / group — количество персонажей
- wide shot / medium shot / close-up — план
- background: natural / city / forest / abstract / random background — фон
- bright lighting / natural lighting / studio lighting — освещение
- scenic view — пейзаж
- colorful / vibrant colors / pastel colors — цвета
- hair color random / natural hair colors / random hairstyle — волосы
- standing on grass / street / indoors / outdoors — окружение
- hands visible / face visible — видимость частей тела
- wearing hat / scarf / jacket / dress — аксессуары
- full body / half body / cropped — кадрирование

[SFW - negative prompt] — всегда добавлять для фильтрации артефактов и NSFW:
- nude [!]
- nsfw [!]
- lewd [!]
- explicit [!]
- uncleared skin [!]
- cleavage [!]
- nipples [!]
- bad anatomy [!]
- malformed [!]
- low quality [!]
- jpeg artifacts [!]
- watermark [!]
- signature [!]
- text [!]
- blurry [!]
- distorted [!]
- out of frame [!]
- duplicate [!]
- extra limbs [!]
- missing limbs [!]
- mutated [!]
- squint [!]
- grainy [!]
- ugly [!]

[Дополнительные теги] — для случайности и вариативности:
- random hair color — случайный цвет волос
- random eye color — случайный цвет глаз
- random skin tone — случайный тон кожи
- random background — случайный фон
- random lighting — случайное освещение
- dynamic lighting — динамичное освещение
- soft shadows — мягкие тени
- motion blur — эффект движения
- motion lines — линии движения
- floating — парящий
- wind blowing hair / wind effect — ветер
- glowing elements / magical atmosphere — магия
- surreal / abstract shapes — сюрреализм
- random accessories — случайные аксессуары
- random pose transitions — смена поз
- random facial expression — выражение лица
- random angle — угол
- random camera position — позиция камеры
- asymmetrical design — асимметрия
- broken pattern — нарушенный паттерн
- glitch effect — глитч-эффект
- pastel colors / neon colors / monochrome — цветовые схемы

---

Тебя зовут Нейро. Ты — интеллектуальный программный оркестратор, который может выполнять команды PowerShell, управлять мышью и клавиатурой, создавать и читать файлы, искать информацию в интернете, анализировать изображения и видео, а также генерировать изображения.

ТЫ ОСОБЕННО ХОРОШ В:
- Анализе и понимании сложных задач
- Разбиении задач на логические шаги
- Использовании инструментов для достижения цели
- Адаптации к изменениям и исправлению ошибок
- Объяснении своих действий и решений

СТРОГО СОБЛЮДАЙ СЛЕДУЮЩИЕ ПРАВИЛА:

2. ВСЕГДА отвечай в формате JSON с одним из следующих действий:
   - "powershell" — для выполнения команд PowerShell
   - "search" — для поиска в интернете
   - "generate_image" — для генерации изображения (только если включена генерация изображений)
   - "speak" — для озвучки важного текста (только самое важное, что нужно сразу услышать)
   - "response" — для финального ответа пользователю
   - "move_mouse" — переместить мышь (x, y)
   - "left_click" — клик левой кнопкой мыши (x, y)
   - "right_click" — клик правой кнопкой мыши (x, y)
   - "scroll_up" — прокрутка вверх (pixels)
   - "scroll_down" — прокрутка вниз (pixels)
   - "mouse_down" — зажать левую кнопку мыши (x, y)
   - "mouse_up" — отпустить левую кнопку мыши (x, y)
   - "drag_and_drop" — перетащить мышью (x1, y1, x2, y2)
   - "type_text" — ввести текст (text)
   - "take_screenshot" — сделать скриншот экрана для анализа

3. ПРАВИЛО ОЗВУЧКИ: Используй действие "speak" только для самого важного текста, который нужно сразу услышать. 
   Остальной текст (объяснения, детали, дополнительная информация) помещай в обычный ответ "response".
   Например, если пользователь спрашивает "сколько 2+2", озвучь только "Будет 4", а объяснения и детали 
   помести в обычный текстовый ответ.

4. Формат JSON для управления мышью:
{
  "action": "move_mouse",
  "x": 123,
  "y": 456,
  "description": "Переместить мышь на кнопку 'ОК'"
}
{
  "action": "left_click",
  "x": 123,
  "y": 456,
  "description": "Кликнуть по кнопке 'ОК'"
}
{
  "action": "right_click",
  "x": 123,
  "y": 456,
  "description": "ПКМ по объекту"
}
{
  "action": "scroll_up",
  "pixels": 100,
  "description": "Прокрутить вверх"
}
{
  "action": "scroll_down",
  "pixels": 100,
  "description": "Прокрутить вниз"
}
{
  "action": "mouse_down",
  "x": 100,
  "y": 200,
  "description": "Зажать ЛКМ для выделения"
}
{
  "action": "mouse_up",
  "x": 200,
  "y": 200,
  "description": "Отпустить ЛКМ после выделения"
}
{
  "action": "drag_and_drop",
  "x1": 100,
  "y1": 200,
  "x2": 300,
  "y2": 400,
  "description": "Перетащить объект"
}
{
  "action": "type_text",
  "text": "пример текста",
  "description": "Ввести текст"
}
{
  "action": "take_screenshot",
  "description": "Сделать скриншот для анализа"
}

ПРИМЕР ИСПОЛЬЗОВАНИЯ ОЗВУЧКИ:
{
  "action": "speak",
  "text": "Важный текст для озвучки",
  "voice": "male",
  "language": "ru",
  "description": "Озвучить важную информацию"
}

ПРИМЕР ГЕНЕРАЦИИ ИЗОБРАЖЕНИЙ:
{
  "action": "generate_image",
  "text": "masterpiece, best quality, extremely detailed, anime girl, full body, detailed face, bright colors, standing pose",
  "negative_prompt": "(worst quality, low quality, normal quality:1.4), (deformed, distorted, disfigured:1.3), poorly drawn, bad anatomy",
  "description": "Генерирую изображение аниме девочки"
}

5. ОБРАБОТКА ЗАПРОСОВ:
   - На простые приветствия ("привет", "hello", "как дела") отвечай дружелюбно действием "response"
   - Для команд управления ПК (клик, движение мыши, команды) используй соответствующие действия
   - Если запрос неясен, переспроси пользователя действием "response"
   - ДЛЯ ГЕНЕРАЦИИ ИЗОБРАЖЕНИЙ: используй действие "generate_image" с полем "text" содержащим промпт на английском языке

6. ФОРМАТ JSON ДЛЯ ГЕНЕРАЦИИ ИЗОБРАЖЕНИЙ:
   ОБЯЗАТЕЛЬНО используй точно такой формат:
   {
     "action": "generate_image",
     "text": "промпт на английском языке с тегами",
     "negative_prompt": "негативный промпт (опционально)",
     "description": "краткое описание что генерируешь"
   }
   
   Поле "text" должно содержать основной промпт для Stable Diffusion на английском языке.
   Поле "negative_prompt" содержит негативный промпт (что НЕ должно быть на изображении).
   НИКОГДА не используй теги <think> или другие форматы - только чистый JSON!

7. РАБОТА СО СКРИНШОТАМИ:
   - При получении команды, связанной с управлением ПК (клик, движение мыши), сначала сделай скриншот для анализа текущего состояния экрана.
   - Vision-модель опишет содержимое экрана, включая расположение объектов.
   - На основе описания принимай решения о координатах для действий.
   - После выполнения действия можно сделать новый скриншот для проверки результата.

8. ОБРАТНАЯ СВЯЗЬ И АДАПТАЦИЯ:
   - Если результат действия — изображение (скриншот после действия), укажи что это изображение после выполнения команды.
   - Анализируй изменения на экране после действий и сообщай об успехе/неудаче.
   - При ошибках предлагай альтернативные решения.
   - Учись на своих действиях и улучшай стратегию.

9. СТРАТЕГИЧЕСКОЕ МЫШЛЕНИЕ:
   - Всегда планируй несколько шагов вперед
   - Учитывай возможные ошибки и альтернативы
   - Если задача сложная — разбивай на подзадачи
   - Проверяй результаты каждого шага перед следующим

10. НИКОГДА не пиши обычный текст вне JSON!

ПРИМЕРЫ ОТВЕТОВ:

Простое приветствие:
{
  "action": "response",
  "content": "Привет! Я Нейро, ваш AI-помощник. Чем могу помочь?"
}

Пример с Powershell:
{
  "action": "powershell",
  "command": "New-Item -Path 'C:\\\\Users\\\\vital\\\\Desktop\\\\НоваяПапка' -ItemType Directory -Force",
  "description": "Создаю папку 'НоваяПапка' на рабочем столе"
}

Пример с озвучкой:
{
  "action": "speak",
  "text": "Текст, который нужно озвучить",
  "voice": "male",
  "language": "ru",
  "description": "Текст, который не будет озвучен, а как дополнение к озвучке, пояснение"
}
{
  "action": "response",
  "content": "Полноый текст ответа, не будет озвучен, нужен как менее важный текст"
}

11. ГЕНЕРАЦИЯ ИЗОБРАЖЕНИЙ: Если пользователь использует слова "сгенерируй", "нарисуй", "создай изображение", "покажи как выглядит", "визуализируй", "изобрази" или подобные по смыслу, И генерация изображений включена, используй действие "generate_image" с подробным описанием. ВАЖНО: После успешной генерации изображения система автоматически завершит диалог - НЕ пытайся генерировать повторно!
12. Если задача требует несколько шагов (например, поиск + создание файла), всегда строй цепочку действий: сначала "search", затем обработай результат и только потом "powershell" для создания/записи файла, и только после этого — "response".
13. После каждого шага жди результат и только потом предлагай следующий JSON-действие.
14. Если пользователь просит сохранить или обработать результат поиска, обязательно сгенерируй команду для создания/записи файла через PowerShell.
15. Для файлов с русским текстом всегда используй кодировку utf-8 (encoding='utf-8' или 65001) и явно указывай это в PowerShell-команде (например, параметр -Encoding UTF8).
16. В JSON-ответах ВСЕ обратные слэши (\\) должны быть экранированы (\\\\), особенно в путях файлов и строках PowerShell.
17. Поисковые запросы делай максимально краткими и точными.
18. Если результат команды или поиска очень большой, проси пользователя уточнить или обрезай вывод до 2000 символов.
19. Если задача полностью решена, обязательно заверши цепочку действием "response".
20. Не повторяй одни и те же действия без необходимости.
21. Если не уверен, уточни у пользователя.
22. Директория Desktop: C:\\Users\\vital\\Desktop

НОВЫЕ ПРАВИЛА ДЛЯ РАБОТЫ С ИЗОБРАЖЕНИЯМИ И ВИДЕО:
23. Если тебе предоставлено изображение, детально опиши его содержимое в начале ответа.
24. При анализе изображений уделяй внимание тексту, цифрам, диаграммам и другим данным.
25. Если на изображении есть текст, перепиши его точно и полностью.
26. При наличии изображения и пользовательского запроса, сначала анализируй изображение, затем выполняй запрос.
27. Если в запросе присутствует секция [Покадровое описание видео]: ... — это хронологическая последовательность описаний кадров видео с таймкодами. Используй эти описания для анализа происходящего в видео, связывай объекты и события по времени.
28. Если есть секция [Текст из аудио]: ... с таймкодами, это синхронизированный текст аудиодорожки. Используй таймкоды для сопоставления текста и визуального ряда.
29. При анализе видео учитывай, что каждый таймкод соответствует определённому моменту времени. Можно делать выводы о развитии событий, появлении/исчезновении объектов, действиях и т.д.
30. Если есть и аудио, и покадровое описание — старайся анализировать их совместно, чтобы дать максимально точный и информативный ответ.
31. Если несколько подряд идущих кадров имеют одинаковое описание — объединяй их в диапазон таймкодов [start-end]: описание. Если одинаковые, но не подряд — собирай список таймкодов [t1, t2, t3]: описание.

ПОМНИ: Ты не просто исполнитель команд, а интеллектуальный помощник, который думает, планирует и адаптируется!
"""

    def auto_disable_tools(self, tool_name: Optional[str] = None):
        """Автоматически выключает инструмент через заданное время после использования"""
        import threading
        import time
        
        def disable_tool(tool_name):
            time.sleep(self.auto_disable_delay)
            if tool_name == 'image_generation':
                if hasattr(self, 'use_image_generation'):
                    self.use_image_generation = False
                    logger.info(f"🔧 Автоматически выключил {tool_name}")
            elif tool_name == 'vision':
                if hasattr(self, 'use_vision'):
                    self.use_vision = False
                    logger.info(f"🔧 Автоматически выключил {tool_name}")
            elif tool_name == 'audio':
                if hasattr(self, 'use_audio'):
                    self.use_audio = False
                    logger.info(f"🔧 Автоматически выключил {tool_name}")
        
        # Если указан конкретный инструмент, запускаем таймер только для него
        if tool_name:
            if tool_name not in self.tool_timers or not self.tool_timers[tool_name].is_alive():
                timer = threading.Thread(target=disable_tool, args=(tool_name,), daemon=True)
                self.tool_timers[tool_name] = timer
                timer.start()
                logger.info(f"⏰ Запустил таймер автоматического выключения для {tool_name}")
        else:
            # Запускаем таймеры для всех активных инструментов
            for tool_name in ['image_generation', 'vision', 'audio']:
                if tool_name not in self.tool_timers or not self.tool_timers[tool_name].is_alive():
                    timer = threading.Thread(target=disable_tool, args=(tool_name,), daemon=True)
                    self.tool_timers[tool_name] = timer
                    timer.start()
                    logger.info(f"⏰ Запустил таймер автоматического выключения для {tool_name}")
                
    def _log(self, message: str, level: str = "INFO"):
        """Логирование с временной меткой в файл и консоль"""
        timestamp = time.strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {level}: {message}"
        
        # Логируем в файл
        if level == "ERROR":
            logger.error(message)
        elif level == "WARNING":
            logger.warning(message)
        else:
            logger.info(message)
        
        # Выводим в консоль
        print(formatted_message)
    
    def get_context_info(self) -> str:
        """Возвращает информацию о текущем использовании контекста"""
        return f"Контекст: {self.current_context_length:,} токенов / {self.safe_context_length:,} (безопасный) / {self.max_context_length:,} (максимум)"

    def add_performance_metric(self, action: str, response_time: float, context_length: int = 0):
        """Добавляет метрику производительности"""
        metric = {
            "timestamp": time.time(),
            "action": action,
            "response_time": response_time,
            "context_length": context_length
        }
        self.performance_metrics.append(metric)
        
        # Ограничиваем количество метрик
        if len(self.performance_metrics) > 100:
            self.performance_metrics.pop(0)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Возвращает статистику производительности"""
        if not self.performance_metrics:
            return {"total_actions": 0, "avg_response_time": 0, "recent_metrics": []}
        
        total_actions = len(self.performance_metrics)
        avg_response_time = sum(m["response_time"] for m in self.performance_metrics) / total_actions
        recent_metrics = self.performance_metrics[-10:]  # Последние 10 метрик
        
        return {
            "total_actions": total_actions,
            "avg_response_time": round(avg_response_time, 3),
            "recent_metrics": recent_metrics
        }

    def take_screenshot(self) -> str:
        """
        Делает скриншот основного монитора и возвращает base64
        """
        try:
            # mss уже импортирован в начале файла
            with mss.mss() as sct:
                # Скриншот основного монитора
                monitor = sct.monitors[1]  # 0 - все мониторы, 1 - основной
                screenshot = sct.grab(monitor)
                # Конвертируем в PIL Image
                img = Image.frombytes("RGB", screenshot.size, screenshot.rgb)
                # Сжимаем для экономии места
                img.thumbnail((1280, 720), Image.Resampling.LANCZOS)
                # Конвертируем в base64
                buf = BytesIO()
                img.save(buf, format="PNG", optimize=True)
                return base64.b64encode(buf.getvalue()).decode("ascii")
        except ImportError:
            logger.warning("mss не установлен, используем pyautogui")
            try:
                # pyautogui уже импортирован в начале файла
                screenshot = pyautogui.screenshot()
                screenshot.thumbnail((1280, 720), Image.Resampling.LANCZOS)
                buf = BytesIO()
                screenshot.save(buf, format="PNG", optimize=True)
                return base64.b64encode(buf.getvalue()).decode("ascii")
            except ImportError:
                logger.error("pyautogui не установлен")
                return ""
        except Exception as e:
            logger.error(f"Ошибка создания скриншота: {e}")
            return ""

    def move_mouse(self, x: int, y: int) -> Dict[str, Any]:
        """Перемещение мыши в координаты (x, y)"""
        try:
            # pyautogui уже импортирован в начале файла
            pyautogui.moveTo(x, y, duration=0.2)
            return {"success": True, "message": f"Мышь перемещена в ({x}, {y})"}
        except ImportError:
            return {"success": False, "error": "pyautogui не установлен"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def left_click(self, x: int, y: int) -> Dict[str, Any]:
        """Клик левой кнопкой мыши по координатам (x, y)"""
        try:
            # pyautogui уже импортирован в начале файла
            pyautogui.click(x, y)
            return {"success": True, "message": f"ЛКМ клик в ({x}, {y})"}
        except ImportError:
            return {"success": False, "error": "pyautogui не установлен"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def right_click(self, x: int, y: int) -> Dict[str, Any]:
        """Клик правой кнопкой мыши по координатам (x, y)"""
        try:
            # pyautogui уже импортирован в начале файла
            pyautogui.rightClick(x, y)
            return {"success": True, "message": f"ПКМ клик в ({x}, {y})"}
        except ImportError:
            return {"success": False, "error": "pyautogui не установлен"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def scroll(self, pixels: int) -> Dict[str, Any]:
        """Прокрутка колесиком мыши. Положительные значения - вверх, отрицательные - вниз"""
        try:
            # pyautogui уже импортирован в начале файла
            pyautogui.scroll(pixels)
            direction = "вверх" if pixels > 0 else "вниз"
            return {"success": True, "message": f"Прокрутка {direction} на {abs(pixels)} пикселей"}
        except ImportError:
            return {"success": False, "error": "pyautogui не установлен"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def mouse_down(self, x: int, y: int) -> Dict[str, Any]:
        """Зажать левую кнопку мыши в координатах (x, y)"""
        try:
            # pyautogui уже импортирован в начале файла
            pyautogui.moveTo(x, y)
            pyautogui.mouseDown(button='left')
            return {"success": True, "message": f"ЛКМ зажата в ({x}, {y})"}
        except ImportError:
            return {"success": False, "error": "pyautogui не установлен"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def mouse_up(self, x: int, y: int) -> Dict[str, Any]:
        """Отпустить левую кнопку мыши в координатах (x, y)"""
        try:
            # pyautogui уже импортирован в начале файле
            pyautogui.moveTo(x, y)
            pyautogui.mouseUp(button='left')
            return {"success": True, "message": f"ЛКМ отпущена в ({x}, {y})"}
        except ImportError:
            return {"success": False, "error": "pyautogui не установлен"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def drag_and_drop(self, x1: int, y1: int, x2: int, y2: int) -> Dict[str, Any]:
        """Перетащить мышью из (x1, y1) в (x2, y2)"""
        try:
            import pyautogui
            pyautogui.dragTo(x2, y2, duration=0.5, button='left')
            return {"success": True, "message": f"Перетаскивание из ({x1}, {y1}) в ({x2}, {y2})"}
        except ImportError:
            return {"success": False, "error": "pyautogui не установлен"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def type_text(self, text: str) -> Dict[str, Any]:
        """Ввести текст"""
        try:
            import pyautogui
            pyautogui.typewrite(text, interval=0.05)
            return {"success": True, "message": f"Введён текст: {text}"}
        except ImportError:
            return {"success": False, "error": "pyautogui не установлен"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def start_continuous_recording(self):
        """Запуск постоянной голосовой записи"""
        if self.continuous_recording:
            return
        
        self.continuous_recording = True
        self.recording_thread = threading.Thread(target=self._continuous_recording_worker, daemon=True)
        self.recording_thread.start()
        logger.info("Постоянная голосовая запись запущена")

    def stop_continuous_recording(self):
        """Остановка постоянной голосовой записи"""
        self.continuous_recording = False
        if self.recording_thread:
            self.recording_thread.join(timeout=2)
        logger.info("Постоянная голосовая запись остановлена")

    def _continuous_recording_worker(self):
        """Воркер для постоянной голосовой записи (заглушка - нужна реализация через веб-интерфейс)"""
        # Эта функция будет вызываться из веб-интерфейса через API
        while self.continuous_recording:
            try:
                # Проверяем очередь на наличие аудиочанков
                if not self.audio_queue.empty():
                    audio_data = self.audio_queue.get_nowait()
                    # Обрабатываем аудио
                    self._process_audio_chunk(audio_data)
                time.sleep(0.1)
            except Exception as e:
                logger.error(f"Ошибка в continuous recording worker: {e}")

    def _process_audio_chunk(self, audio_data: bytes):
        """Обработка чанка аудио из постоянной записи"""
        try:
            # Сохраняем чанк во временный файл
            temp_dir = os.path.join(os.path.dirname(__file__), "temp_audio")
            os.makedirs(temp_dir, exist_ok=True)
            temp_file = os.path.join(temp_dir, f"chunk_{int(time.time())}.wav")
            
            with open(temp_file, 'wb') as f:
                f.write(audio_data)
            
            # Распознаём аудио
            transcript = self.transcribe_audio_whisper(temp_file, use_separator=False)
            
            if transcript and not transcript.startswith("[Whisper error]"):
                # Проверяем, содержит ли текст команду или имя "Алиса"
                if self._is_valid_command(transcript):
                    logger.info(f"Получена команда из голоса: {transcript}")
                    # Делаем скриншот для контекста
                    screenshot_b64 = self.take_screenshot()
                    vision_desc = ""
                    if screenshot_b64:
                        vision_desc = self.call_vision_model(screenshot_b64)
                    
                    # Формируем запрос для мозга
                    brain_input = f"[Скриншот экрана]: {vision_desc}\n\nГолосовая команда: {transcript}"
                    
                    # Отправляем в мозг
                    ai_response = self.call_brain_model(brain_input)
                    self.process_ai_response(ai_response)
                else:
                    # Игнорируем бессмысленные фразы
                    pass
            
            # Удаляем временный файл
            try:
                os.remove(temp_file)
            except Exception:
                pass
                
        except Exception as e:
            logger.error(f"Ошибка обработки аудиочанка: {e}")

    def _is_valid_command(self, text: str) -> bool:
        """Всегда возвращает True — фильтрация отключена, все команды проходят к нейросети"""
        return True

    def call_vision_model(self, image_base64: str) -> str:
        """
        Отправка изображения только в vision-модель ("глаза")
        Возвращает описание изображения (текст).
        """
        start_time = time.time()
        
        # Автоматически включаем vision модель при необходимости
        if not getattr(self, 'use_vision', False):
            logger.info("🔧 Автоматически включаю vision модель")
            self.use_vision = True
            # Запускаем таймер автоматического выключения
            self.auto_disable_tools("vision")
        
        try:
            payload = {
                "model": "moondream2-llamafile",
                "messages": [
                    {"role": "user", "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}
                    ]}
                ],
                "temperature": 0.0,
                "max_tokens": 2048,
                "stream": False
            }
            logger.info("Отправляю изображение в vision-модель (глаза)")
            response = requests.post(
                f"{self.lm_studio_url}/v1/chat/completions",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"].strip()
            else:
                error_msg = f"Ошибка vision-модели: {response.status_code} - {response.text}"
                logger.error(error_msg)
                return f"[Vision error] {error_msg}"
        except Exception as e:
            error_msg = f"Исключение vision: {str(e)}"
            logger.error(error_msg)
            return f"[Vision error] {error_msg}"
        finally:
            # Записываем метрику производительности
            response_time = time.time() - start_time
            self.add_performance_metric("vision_processing", response_time)
            logger.info(f"👁️ Vision обработал за {response_time:.2f} сек")

    def call_brain_model(self, user_message: str, vision_desc: str = "") -> str:
        """
        Отправка текстового запроса (и опционально описания изображения) в "мозг" (текстовая модель)
        """
        start_time = time.time()
        try:
            # Улучшаем промпт с помощью памяти ChromaDB
            enhanced_system_prompt = self.enhance_prompt_with_memory(user_message, self.system_prompt)
            
            messages: List[Dict[str, Any]] = [
                {"role": "system", "content": enhanced_system_prompt}
            ]
            
            # Добавляем историю разговора с управлением контекстом
            messages.extend(self.conversation_history)
            
            # Добавляем описание изображения, если есть
            if vision_desc:
                messages.append({"role": "user", "content": vision_desc})
            # Добавляем основной запрос пользователя
            messages.append({"role": "user", "content": user_message})
            
            # Динамическое управление контекстом - подсчитываем длину
            total_length = sum(len(str(msg.get("content", ""))) for msg in messages)
            self.current_context_length = total_length
            
            # Обрезаем контекст если необходимо (используем предварительную оценку)
            if total_length > self.safe_context_length:
                self.conversation_history = self.conversation_history[-5:]  # Оставляем только 5 последних сообщений
                logger.warning(f"Превышение безопасного контекста ({total_length:,} символов > {self.safe_context_length:,}) - аккуратная обрезка истории")
            
            # Пересобираем сообщения после обрезки
            messages = [{"role": "system", "content": enhanced_system_prompt}]
            messages.extend(self.conversation_history)
            if vision_desc:
                messages.append({"role": "user", "content": vision_desc})
            messages.append({"role": "user", "content": user_message})
            
            payload = {
                "model": self.brain_model_id if hasattr(self, 'brain_model_id') and self.brain_model_id else self.brain_model,
                "messages": messages,
                "temperature": 0.1,
                "max_tokens": 32767,
                "stream": False
            }
            logger.info(f"Отправляю запрос в мозг: {user_message[:100]}...")
            response = requests.post(
                f"{self.lm_studio_url}/v1/chat/completions",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            if response.status_code == 200:
                result = response.json()
                ai_response = result["choices"][0]["message"]["content"].strip()
                
                # Извлекаем информацию о токенах из ответа модели
                usage_info = result.get("usage", {})
                prompt_tokens = usage_info.get("prompt_tokens", 0)
                completion_tokens = usage_info.get("completion_tokens", 0)
                total_tokens = usage_info.get("total_tokens", 0)
                
                # Обновляем текущий размер контекста на основе total_tokens
                if total_tokens > 0:
                    self.current_context_length = total_tokens
                    logger.info(f"📊 Реальные токены: prompt={prompt_tokens}, completion={completion_tokens}, total={total_tokens}")
                    
                    # Обрезаем контекст на основе реальных токенов
                    self._trim_context_if_needed()
                
                # Добавляем в историю разговора (если ответ не пустой)
                if ai_response and ai_response != "{}":
                    self.conversation_history.append({"role": "user", "content": user_message})
                    self.conversation_history.append({"role": "assistant", "content": ai_response})
                    
                    # Автоматически сохраняем диалог в ChromaDB
                    self.auto_save_conversation(user_message, ai_response, vision_desc)
                    
                    # Извлекаем предпочтения пользователя из диалога
                    self.extract_preferences_from_response(user_message, ai_response)
                
                return ai_response
            else:
                error_msg = f"Ошибка brain-модели: {response.status_code} - {response.text}"
                logger.error(error_msg)
                return f"[Brain error] {error_msg}"
        except Exception as e:
            error_msg = f"Исключение brain: {str(e)}"
            logger.error(error_msg)
            return f"[Brain error] {error_msg}"
        finally:
            # Записываем метрику производительности
            response_time = time.time() - start_time
            self.add_performance_metric("brain_response", response_time, self.current_context_length)
            logger.info(f"🧠 Мозг ответил за {response_time:.2f} сек")

    def execute_powershell(self, command: str) -> Dict[str, Any]:
        """
        Выполнение PowerShell команды
        
        Args:
            command: PowerShell команда
        
        Returns:
            Словарь с результатом выполнения
        """
        try:
            orig_command = command
            # Автоисправление: заменяем '&&' на PowerShell-совместимый синтаксис
            if '&&' in command:
                parts = [p.strip() for p in command.split('&&')]
                # Если первая часть cd, делаем push-location, затем вторую команду
                if parts[0].lower().startswith('cd '):
                    dir_path = parts[0][3:].strip().strip('"\'')
                    command = f"Push-Location '{dir_path}'; {parts[1]} ; Pop-Location"
                else:
                    # Просто объединяем через ';'
                    command = ' ; '.join(parts)
                logger.info(f"PowerShell: автоисправлен '&&' -> ';' или Push-Location: {command}")
            logger.info(f"Выполняю PowerShell: {command}")
            # Выполняем команду PowerShell с декодированием cp1251 и защитой
            result = subprocess.run(
                ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", command],
                capture_output=True,
                text=True,
                encoding='cp1251',
                errors='replace',
                timeout=60
            )
            success = result.returncode == 0
            # Защита от None
            output = (result.stdout if success else result.stderr) or ""
            logger.info(f"PowerShell результат (код: {result.returncode}): {output[:200]}...")
            return {
                "success": success,
                "returncode": result.returncode,
                "output": output,
                "error": (result.stderr or "") if not success else ""
            }
        except subprocess.TimeoutExpired:
            error_msg = "Команда превысила лимит времени выполнения (60 сек)"
            logger.error(error_msg)
            return {"success": False, "returncode": -1, "output": "", "error": error_msg}
        except Exception as e:
            error_msg = f"Ошибка выполнения PowerShell: {str(e)}"
            logger.error(error_msg)
            return {"success": False, "returncode": -1, "output": "", "error": error_msg}

    def google_search(self, query: str, num_results: int = 10) -> List[Dict[str, str]]:
        """
        Поиск в Google Custom Search API
        
        Args:
            query: Поисковый запрос
            num_results: Количество результатов для парсинга (по умолчанию 10)
            
        Returns:
            Список результатов поиска
        """
        try:
            if not self.google_api_key or not self.google_cse_id:
                return [{"error": "Google API ключ или CSE ID не настроены"}]
            
            logger.info(f"Выполняю поиск Google: {query}")
            
            # Кодируем запрос для URL
            encoded_query = urllib.parse.quote(query)
            
            # Формируем URL для Google Custom Search API (максимум 10 результатов)
            url = f"https://www.googleapis.com/customsearch/v1?key={self.google_api_key}&cx={self.google_cse_id}&q={encoded_query}&num=10"
            
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                if "items" not in data:
                    return [{"error": "Результаты не найдены"}]
                
                # Берем первые num_results результатов для парсинга (максимум 10)
                actual_results = min(num_results, 10)
                search_results = []
                for i, item in enumerate(data["items"][:actual_results]):
                    result = {
                        "title": item.get("title", ""),
                        "url": item.get("link", ""),
                        "snippet": item.get("snippet", "")
                    }
                    
                    # Пытаемся получить содержимое страницы
                    try:
                        page_response = requests.get(result["url"], timeout=5, headers={
                            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                        })
                        if page_response.status_code == 200:
                            # Берем первые 2000 символов текста для полного анализа
                            content = page_response.text[:2000]
                            result["content"] = content
                        else:
                            result["content"] = "Не удалось получить содержимое страницы"
                    except:
                        result["content"] = "Ошибка при получении содержимого страницы"
                    
                    search_results.append(result)
                    logger.info(f"Получен результат {i+1}: {result['title']}")
                
                logger.info(f"Поиск завершен: найдено {len(search_results)} результатов")
                return search_results
            else:
                error_msg = f"Ошибка Google Search API: {response.status_code}"
                logger.error(error_msg)
                return [{"error": error_msg}]
                
        except Exception as e:
            error_msg = f"Ошибка поиска Google: {str(e)}"
            logger.error(error_msg)
            return [{"error": error_msg}]

    def process_ai_response(self, ai_response: str) -> bool:
        """Light wrapper to keep `process_ai_response` simple for static analysers.

        Реализует прокси к подробной реализации `_process_ai_response_impl`.
        Это уменьшает сложность видимой функции для Pylance.
        """
        return self._process_ai_response_impl(ai_response)

    # --- Разделённые обработчики действий (уменьшают сложность основного метода) ---
    def _is_english_simple(self, s: str) -> bool:
        if not s:
            return False
        allowed_chars = 0
        total_chars = len(s)
        for c in s:
            if (c.isascii() and (c.isalpha() or c.isdigit()) or
                c in '.,;:-_=+!@#$%^&*()[]{}<>?/\\|`~\'\" ' or
                c.isspace()):
                allowed_chars += 1
        return allowed_chars / total_chars > 0.85

    def _handle_powershell(self, action_data: Dict[str, Any]) -> Union[bool, str]:
        command = action_data.get("command", "")
        description = action_data.get("description", "")
        logger.info(f"\n🔧 ВЫПОЛНЕНИЕ КОМАНДЫ: {description}")
        logger.info(f"📝 Команда: {command}")
        result = self.execute_powershell(command)
        if result["success"]:
            feedback = f"Команда '{command}' выполнена успешно. Результат: {result['output']}"
        else:
            feedback = f"Команда '{command}' завершилась с ошибкой: {result.get('error','') }"
        follow_up = self.call_brain_model(feedback)
        return follow_up

    def _handle_search(self, action_data: Dict[str, Any]) -> Union[bool, str]:
        query = action_data.get("query", "")
        description = action_data.get("description", "")
        logger.info(f"\n🔍 ПОИСК В ИНТЕРНЕТЕ: {description}")
        logger.info(f"🔎 Запрос: {query}")
        search_results = self.google_search(query)
        results_text = "Результаты поиска:\n"
        for i, result in enumerate(search_results, 1):
            if "error" in result:
                results_text += f"{i}. Ошибка: {result['error']}\n"
            else:
                results_text += f"{i}. {result['title']}\n"
                results_text += f"   URL: {result['url']}\n"
                results_text += f"   Описание: {result['snippet']}\n"
                results_text += f"   Содержимое: {result.get('content', '')}\n\n"
        logger.info("✅ Поиск завершен")
        follow_up = self.call_brain_model(f"Результаты поиска по запросу '{query}': {results_text}")
        return follow_up

    def _handle_take_screenshot(self, action_data: Dict[str, Any]) -> Union[bool, str]:
        logger.info(f"\n📸 СОЗДАНИЕ СКРИНШОТА")
        if not getattr(self, 'use_vision', False):
            logger.info("🔧 Автоматически включаю vision модель")
            self.use_vision = True
            self.auto_disable_tools("vision")
        screenshot_b64 = self.take_screenshot()
        if screenshot_b64:
            vision_desc = self.call_vision_model(screenshot_b64)
            feedback = f"Скриншот экрана получен. Описание от vision-модели: {vision_desc}"
        else:
            feedback = "Не удалось создать скриншот"
        follow_up = self.call_brain_model(feedback)
        return follow_up

    def _handle_move_mouse(self, action_data: Dict[str, Any]) -> Union[bool, str]:
        x = action_data.get("x", 0)
        y = action_data.get("y", 0)
        description = action_data.get("description", "")
        logger.info(f"\n🖱️ ПЕРЕМЕЩЕНИЕ МЫШИ: {description}")
        result = self.move_mouse(x, y)
        feedback = f"Мышь перемещена в координаты ({x}, {y})" if result.get("success") else f"Ошибка перемещения мыши: {result.get('error','') }"
        follow_up = self.call_brain_model(feedback)
        return follow_up

    def _handle_left_click(self, action_data: Dict[str, Any]) -> Union[bool, str]:
        x = action_data.get("x", 0)
        y = action_data.get("y", 0)
        result = self.left_click(x, y)
        feedback = f"Клик ЛКМ выполнен в координатах ({x}, {y})" if result.get("success") else f"Ошибка клика: {result.get('error','') }"
        follow_up = self.call_brain_model(feedback)
        return follow_up

    def _handle_right_click(self, action_data: Dict[str, Any]) -> Union[bool, str]:
        x = action_data.get("x", 0)
        y = action_data.get("y", 0)
        result = self.right_click(x, y)
        feedback = f"Клик ПКМ выполнен в координатах ({x}, {y})" if result.get("success") else f"Ошибка клика ПКМ: {result.get('error','') }"
        follow_up = self.call_brain_model(feedback)
        return follow_up

    def _handle_scroll(self, action: str, action_data: Dict[str, Any]) -> Union[bool, str]:
        pixels = action_data.get("pixels", 100)
        if action == "scroll_down":
            pixels = -pixels
        result = self.scroll(pixels)
        feedback = f"Прокрутка выполнена: {result.get('message','') }" if result.get("success") else f"Ошибка прокрутки: {result.get('error','') }"
        follow_up = self.call_brain_model(feedback)
        return follow_up

    def _handle_mouse_down(self, action_data: Dict[str, Any]) -> Union[bool, str]:
        x = action_data.get("x", 0)
        y = action_data.get("y", 0)
        result = self.mouse_down(x, y)
        feedback = f"ЛКМ зажата в координатах ({x}, {y})" if result.get("success") else f"Ошибка зажатия ЛКМ: {result.get('error','') }"
        follow_up = self.call_brain_model(feedback)
        return follow_up

    def _handle_mouse_up(self, action_data: Dict[str, Any]) -> Union[bool, str]:
        x = action_data.get("x", 0)
        y = action_data.get("y", 0)
        result = self.mouse_up(x, y)
        feedback = f"ЛКМ отпущена в координатах ({x}, {y})" if result.get("success") else f"Ошибка отпускания ЛКМ: {result.get('error','') }"
        follow_up = self.call_brain_model(feedback)
        return follow_up

    def _handle_drag_and_drop(self, action_data: Dict[str, Any]) -> Union[bool, str]:
        x1 = action_data.get("x1", 0)
        y1 = action_data.get("y1", 0)
        x2 = action_data.get("x2", 0)
        y2 = action_data.get("y2", 0)
        result = self.drag_and_drop(x1, y1, x2, y2)
        feedback = f"Перетаскивание выполнено от ({x1}, {y1}) к ({x2}, {y2})" if result.get("success") else f"Ошибка перетаскивания: {result.get('error','') }"
        follow_up = self.call_brain_model(feedback)
        return follow_up

    def _handle_type_text(self, action_data: Dict[str, Any]) -> Union[bool, str]:
        text = action_data.get("text", "")
        result = self.type_text(text)
        feedback = f"Текст введён: {text}" if result.get("success") else f"Ошибка ввода текста: {result.get('error','') }"
        follow_up = self.call_brain_model(feedback)
        return follow_up

    def _handle_generate_image(self, action_data: Dict[str, Any]) -> Union[bool, str]:
        if not getattr(self, 'use_image_generation', False):
            logger.info("🔧 Автоматически включаю генерацию изображений")
            self.use_image_generation = True
            self.auto_disable_tools("image_generation")
        if not getattr(self, 'use_image_generation', False):
            logger.error("❌ Генерация изображений отключена")
            follow_up = self.call_brain_model("Генерация изображений отключена. Предложи альтернативный способ помочь пользователю.")
            return follow_up

        description = action_data.get("description", "")
        text = action_data.get("text", "")
        style = action_data.get("style", "")
        negative_prompt = action_data.get("negative_prompt", "")
        prompt = text.strip() if text else (description or "")
        if style and prompt:
            prompt += f", {style}"

        params = {}
        if not isinstance(params, dict):
            params = {}

        if not prompt:
            follow_up = self.call_brain_model("Нейросеть вернула пустой промт для генерации изображения. Попроси её создать описание.")
            return follow_up

        if not self._is_english_simple(prompt):
            follow_up = self.call_brain_model(f"Нейросеть вернула промт не на английском языке: {prompt}. Попроси её создать промт на английском.")
            return follow_up

        neg = negative_prompt.strip() if negative_prompt else ""
        if not neg or not self._is_english_simple(neg):
            neg = "(worst quality, low quality, normal quality:1.4)"

        # default params and validation (kept simple here)
        default_params = {"seed": -1, "steps": 30, "width": 1024, "height": 1024, "cfg": 4.0}
        gen_params = default_params.copy()

        img_b64 = self.generate_image_stable_diffusion(prompt, neg, gen_params)
        if img_b64:
            self.last_generated_image_b64 = img_b64
            self.show_image_base64_temp(img_b64)
            final_msg = f"✅ Изображение успешно сгенерировано по вашему описанию: {description}\n🎨 Использованный промт: {prompt}"
            self.last_final_response = final_msg
            logger.info(final_msg)
            return False
        else:
            follow_up = self.call_brain_model(f"Не удалось сгенерировать изображение по описанию: {description}.")
            return follow_up

    def _handle_generate_video(self, action_data: Dict[str, Any]) -> Union[bool, str]:
        if not getattr(self, 'use_image_generation', False):
            logger.info("🔧 Автоматически включаю генерацию изображений")
            self.use_image_generation = True
            self.auto_disable_tools("image_generation")
        if not getattr(self, 'use_image_generation', False):
            follow_up = self.call_brain_model("Генерация изображений отключена. Предложи альтернативный способ помочь пользователю.")
            return follow_up

        description = action_data.get("description", "")
        text = action_data.get("text", "")
        style = action_data.get("style", "")
        negative_prompt = action_data.get("negative_prompt", "")
        prompt = text.strip() if text else (description or "")
        if style and prompt:
            prompt += f", {style}"

        if not prompt:
            follow_up = self.call_brain_model("Нейросеть вернула пустой промт для генерации видео. Попроси её создать описание.")
            return follow_up

        # negative prompt handling
        neg = negative_prompt.strip() if negative_prompt else ""
        fallback_negative = "(worst quality, low quality, normal quality:1.4), (deformed, distorted, disfigured:1.3), poorly drawn, bad anatomy, text, watermark"
        if not neg or not self._is_english_simple(neg):
            neg = fallback_negative

        # default video params and validation
        default_params = {"seed": -1, "steps": 20, "width": 512, "height": 512, "cfg": 7.0, "num_frames": 24, "fps": 8}
        params = action_data.get("params", {}) or {}
        gen_params = default_params.copy()
        if isinstance(params, dict):
            for key, value in params.items():
                if key in default_params:
                    try:
                        if key in ["seed", "steps", "width", "height", "num_frames", "fps"]:
                            gen_params[key] = int(value)
                        elif key == "cfg":
                            gen_params[key] = float(value)
                        else:
                            gen_params[key] = value
                    except (ValueError, TypeError):
                        logger.warning(f"⚠️ Неверный параметр {key}={value}, используется значение по умолчанию")

        # basic bounds
        if gen_params["steps"] < 1 or gen_params["steps"] > 100:
            gen_params["steps"] = 20
        if gen_params["width"] < 64 or gen_params["width"] > 2048:
            gen_params["width"] = 512
        if gen_params["height"] < 64 or gen_params["height"] > 2048:
            gen_params["height"] = 512

        video_path = self.generate_video_stable_diffusion(prompt, neg, gen_params)
        if video_path:
            final_msg = f"✅ Видео успешно сгенерировано по вашему описанию: {description}\n📁 Путь к видео: {video_path}"
            self.last_final_response = final_msg
            logger.info(final_msg)
            return False
        else:
            follow_up = self.call_brain_model(f"Не удалось сгенерировать видео по описанию: {description}.")
            return follow_up

    def _handle_speak(self, action_data: Dict[str, Any]) -> Union[bool, str]:
        text_to_speak = action_data.get("text", "")
        voice = action_data.get("voice", "male")
        language = action_data.get("language", "ru")
        if not text_to_speak:
            follow_up = self.call_brain_model("Текст для озвучки пустой. Укажите текст в поле 'text'.")
            return follow_up
        audio_path = self.text_to_speech(text_to_speak, voice, language)
        if audio_path:
            follow_up = self.call_brain_model(f"Текст успешно озвучен: {text_to_speak}. Аудиофайл сохранен: {os.path.basename(audio_path)}")
        else:
            follow_up = self.call_brain_model(f"Не удалось озвучить текст: {text_to_speak}.")
        return follow_up

    def _handle_response(self, action_data: Dict[str, Any]) -> Union[bool, str]:
        content = action_data.get("content", "")
        self.last_final_response = content
        logger.info(f"\n🤖 ФИНАЛЬНЫЙ ОТВЕТ:")
        logger.info(content)
        return False

    def _process_ai_response_impl(self, ai_response: str) -> bool:
        """
        Подробная реализация обработки ответа AI (перенесена из original `process_ai_response`).

        Args:
            ai_response: JSON ответ от AI

        Returns:
            True если нужно продолжать диалог, False если завершить
        """
        # Итеративный обработчик цепочек follow_up: избегаем рекурсии.
        next_input: Optional[str] = ai_response
        attempts = 0
        while next_input is not None and attempts <= self.max_retries:
            attempts += 1
            try:
                json_str = self.extract_first_json(next_input, allow_json_in_think=True)
                if not json_str or json_str == next_input:
                    logger.info("💬 Модель вернула текстовый ответ без JSON")
                    if len(next_input.strip()) > 5 and not next_input.strip().startswith('{'):
                        logger.info("💬 Использую текстовый ответ как финальный")
                        self.last_final_response = next_input.strip()
                        return False
                    think_content = self.extract_think_content(next_input)
                    if think_content:
                        json_in_think = self._extract_json_from_text(think_content)
                        if json_in_think:
                            logger.info("🔍 Найден JSON внутри <think> блока, использую его")
                            json_str = json_in_think
                        else:
                            feedback = "Модель вернула неполный ответ. Пожалуйста, сформулируй полный ответ или действие."
                            next_input = self.call_brain_model(feedback)
                            continue
                    else:
                        feedback = "Модель вернула неполный ответ. Пожалуйста, сформулируй полный ответ или действие."
                        next_input = self.call_brain_model(feedback)
                        continue

                action_data, fixes = self._smart_json_parse(json_str)
                if fixes:
                    logger.warning(f"⚠️ Исправления JSON: {'; '.join(fixes)}")
                if not action_data:
                    logger.error(f"❌ Не удалось распарсить JSON даже после исправлений:\n{json_str}")
                    self.retry_count += 1
                    if self.retry_count > self.max_retries:
                        logger.warning(f"🔄 Достигнут лимит попыток ({self.max_retries}), завершаю диалог")
                        self.retry_count = 0
                        self.last_final_response = "Извините, возникла проблема с обработкой запроса. Попробуйте переформулировать вопрос."
                        return False
                    think_content = self.extract_think_content(next_input)
                    if think_content:
                        logger.info("💭 Обнаружен блок размышлений - модель размышляет, но не генерирует действие")
                        logger.info(f"💭 Содержимое: {think_content[:200]}...")
                        if len(think_content) > 20 and any(word in think_content.lower() for word in ['привет', 'hello', 'здравствуй']):
                            logger.info("💭 Обнаружено приветствие в размышлениях, завершаю диалог")
                            self.retry_count = 0
                            self.last_final_response = "Привет! Я Нейро, ваш AI-помощник. Чем могу помочь?"
                            return False
                        if self.retry_count >= 2 and 'правил' in think_content.lower():
                            logger.info("💭 Модель зацикливается на правилах, принудительно завершаю")
                            self.retry_count = 0
                            self.last_final_response = "Привет! Как дела? Чем могу помочь?"
                            return False
                        next_input = self.call_brain_model("Пожалуйста, дай простой дружелюбный ответ или конкретное действие. Не анализируй правила.")
                        continue
                    else:
                        next_input = self.call_brain_model("Пожалуйста, дай простой ответ или конкретное действие.")
                        continue

                if 'action' not in action_data:
                    logger.warning("⚠️ В JSON отсутствует ключ 'action', добавляю action: 'unknown'")
                    action_data['action'] = 'unknown'

                if action_data == {} or (len(action_data) == 1 and 'action' in action_data and action_data['action'] == 'unknown'):
                    logger.info("💭 Модель вернула пустой JSON - возможно, она размышляет")
                    think_content = self.extract_think_content(next_input)
                    if think_content and len(think_content) > 20:
                        logger.info("💭 Использую содержимое размышлений как ответ")
                        self.last_final_response = think_content
                        return False
                    next_input = self.call_brain_model("Модель вернула пустой JSON. Пожалуйста, сформулируй конкретный ответ или действие.")
                    continue

                action = action_data.get("action")
                self.retry_count = 0

                # Вызов соответствующего хендлера и обработка результата (следующий ввод или завершение)
                handler_result = None
                if action == "powershell":
                    handler_result = self._handle_powershell(action_data)
                elif action == "search":
                    handler_result = self._handle_search(action_data)
                elif action == "take_screenshot":
                    handler_result = self._handle_take_screenshot(action_data)
                elif action == "move_mouse":
                    handler_result = self._handle_move_mouse(action_data)
                elif action == "left_click":
                    handler_result = self._handle_left_click(action_data)
                elif action == "right_click":
                    handler_result = self._handle_right_click(action_data)
                elif action in ["scroll_up", "scroll_down"]:
                    handler_result = self._handle_scroll(action, action_data)
                elif action == "mouse_down":
                    handler_result = self._handle_mouse_down(action_data)
                elif action == "mouse_up":
                    handler_result = self._handle_mouse_up(action_data)
                elif action == "drag_and_drop":
                    handler_result = self._handle_drag_and_drop(action_data)
                elif action == "type_text":
                    handler_result = self._handle_type_text(action_data)
                elif action == "generate_image":
                    handler_result = self._handle_generate_image(action_data)
                elif action == "generate_video":
                    handler_result = self._handle_generate_video(action_data)
                elif action == "speak":
                    handler_result = self._handle_speak(action_data)
                elif action == "response":
                    handler_result = self._handle_response(action_data)
                else:
                    logger.warning(f"❓ Неизвестное действие: {action}")
                    return False

                # Обработка результата хендлера: False => завершить, str => новый ввод для итерации
                if handler_result is False:
                    return False
                elif isinstance(handler_result, str):
                    next_input = handler_result
                    continue
                else:
                    # Если хендлер вернул None или неожиданный тип — завершаем
                    logger.error("❌ Хендлер вернул некорректный результат, завершаю")
                    return False

            except json.JSONDecodeError as e:
                logger.error(f"❌ Ошибка парсинга JSON ответа AI: {e}")
                logger.info(f"📝 Ответ AI: {next_input}")
                return False
            except Exception as e:
                logger.error(f"❌ Ошибка обработки ответа AI: {str(e)}")
                return False

        logger.warning("🔄 Превышен цикл попыток обработки follow_up или достигнут лимит. Завершаю.")
        return False

    def run_interactive(self):
        """Запуск интерактивного режима (глаза, аудио, мозг)"""
        # Показываем сообщения только в консольном режиме
        if getattr(self, 'show_images_locally', True):
            logger.info("🚀 AI PowerShell Оркестратор запущен!")
            logger.info("💡 Если в папке Photos есть изображение или в Audio есть аудиофайл, сначала будет анализ глазами/ушами, затем вы сможете задать вопрос для мозга.")
            logger.info(f"🧠 Модель: {os.path.basename(self.brain_model)}")
            logger.info(f"📊 {self.get_context_info()}")
            logger.info("💻 Доступные команды: 'stats' (метрики), 'reset' (сброс), 'logs' (логи), 'export' (экспорт), 'memory' (память), 'gpu' (видеокарта), 'search' (поиск), 'preferences' (предпочтения), 'cleanup' (очистка), 'exit' (выход)")
            logger.info("="*60)

        vision_desc = ""
        audio_text = ""
        while True:
            try:
                # 1. Проверяем наличие нового изображения
                image_path = self.find_new_image()
                image_base64 = ""
                if image_path:
                    # Показываем сообщения только в консольном режиме
                    if getattr(self, 'show_images_locally', True):
                        logger.info(f"📸 Найдено изображение: {os.path.basename(image_path)}")
                    image_base64 = image_to_base64_balanced(image_path)
                    if image_base64:
                        if getattr(self, 'show_images_locally', True):
                            logger.info(f"✅ Изображение обработано (размер: {len(image_base64)} символов)")
                        # Сохраняем копию base64-изображения в корень проекта
                        try:
                            # base64 и io уже импортированы в начале файла
                            img_bytes = base64.b64decode(image_base64)
                            with open(os.path.join(os.path.dirname(__file__), "last_sent_image.png"), "wb") as f:
                                f.write(img_bytes)
                            if getattr(self, 'show_images_locally', True):
                                logger.info("🖼️ Сжатое изображение сохранено как last_sent_image.png")
                        except Exception as e:
                            if getattr(self, 'show_images_locally', True):
                                logger.warning(f"⚠️ Не удалось сохранить last_sent_image.png: {e}")
                        # Отправляем изображение в vision-модель
                        vision_desc = self.call_vision_model(image_base64)
                        if getattr(self, 'show_images_locally', True):
                            logger.info("\n👁️ Описание изображения (глаза):\n" + vision_desc)
                        self.mark_image_used(image_path)
                    else:
                        if getattr(self, 'show_images_locally', True):
                            logger.error("❌ Ошибка обработки изображения")
                else:
                    vision_desc = ""

                audio_path = self.find_new_audio()
                if audio_path:
                    # Показываем сообщения только в консольном режиме
                    if getattr(self, 'show_images_locally', True):
                        logger.info(f"🔊 Найден аудиофайл: {os.path.basename(audio_path)}")
                        # Запросить язык у пользователя
                        lang = input("�� Введите язык аудиофайла (например, ru, en, etc..) или Enter для ru: ").strip() or "ru"
                    else:
                        # В веб-режиме используем русский по умолчанию
                        lang = "ru"
                    audio_text = self.transcribe_audio_whisper(audio_path, lang=lang, use_separator=getattr(self, 'use_separator', True))
                    # Транскрипт уже выведен внутри transcribe_audio_whisper, не дублируем
                else:
                    audio_text = ""

                # 3. Запрашиваем у пользователя текстовый вопрос
                try:
                    if getattr(self, 'show_images_locally', True):
                        user_input = input("\n👤 Ваш вопрос (или Enter для пропуска, либо вставьте ссылку на YouTube): ").strip()
                    else:
                        # В веб-режиме не запрашиваем ввод
                        user_input = ""
                except EOFError:
                    # Если ввод из файла/pipe, используем пустую строку
                    user_input = ""
                    if getattr(self, 'show_images_locally', True):
                        logger.info("📝 Ввод из файла/pipe, продолжаю...")
                if user_input.lower() in ['exit', 'quit', 'выход']:
                    if getattr(self, 'show_images_locally', True):
                        logger.info("👋 До свидания!")
                    break
                if user_input.lower() in ['stats', 'метрики', 'статистика']:
                    # Показываем метрики производительности только в консольном режиме
                    if getattr(self, 'show_images_locally', True):
                        stats = self.get_performance_stats()
                        logger.info("\n📊 МЕТРИКИ ПРОИЗВОДИТЕЛЬНОСТИ:")
                        logger.info(f"   Всего действий: {stats['total_actions']}")
                        logger.info(f"   Среднее время ответа: {stats['avg_response_time']} сек")
                        if stats['recent_metrics']:
                            logger.info("   Последние действия:")
                            for metric in stats['recent_metrics'][-5:]:  # Показываем последние 5
                                timestamp = time.strftime("%H:%M:%S", time.localtime(metric['timestamp']))
                                logger.info(f"     [{timestamp}] {metric['action']}: {metric['response_time']:.2f} сек")
                        logger.info(f"   {self.get_context_info()}")
                    continue
                if user_input.lower() in ['reset', 'сброс', 'очистить']:
                    # Сбрасываем метрики и историю
                    self.performance_metrics.clear()
                    self.conversation_history.clear()
                    self.current_context_length = 0
                    if getattr(self, 'show_images_locally', True):
                        logger.info("🔄 Метрики и история сброшены")
                    continue
                if user_input.lower() in ['logs', 'логи']:
                    # Показываем последние записи из лог-файла только в консольном режиме
                    if getattr(self, 'show_images_locally', True):
                        try:
                            with open("ai_orchestrator.log", "r", encoding="utf-8") as f:
                                lines = f.readlines()
                                logger.info("\n📝 ПОСЛЕДНИЕ ЗАПИСИ В ЛОГЕ:")
                                for line in lines[-10:]:  # Показываем последние 10 строк
                                    logger.info(f"   {line.strip()}")
                        except Exception as e:
                            logger.error(f"Ошибка чтения лог-файла: {e}")
                    continue
                if user_input.lower() in ['export', 'экспорт']:
                    # Экспортируем метрики в JSON файл только в консольном режиме
                    if getattr(self, 'show_images_locally', True):
                        try:
                            stats = self.get_performance_stats()
                            export_data = {
                                "export_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                                "performance_stats": stats,
                                "context_info": {
                                    "current": self.current_context_length,
                                    "safe": self.safe_context_length,
                                    "max": self.max_context_length
                                }
                            }
                            filename = f"metrics_export_{int(time.time())}.json"
                            with open(filename, "w", encoding="utf-8") as f:
                                json.dump(export_data, f, ensure_ascii=False, indent=2)
                            logger.info(f"📊 Метрики экспортированы в {filename}")
                        except Exception as e:
                            logger.error(f"Ошибка экспорта метрик: {e}")
                    continue
                if user_input.lower() in ['memory', 'память', 'mem']:
                    # Показываем статистику памяти ChromaDB
                    if getattr(self, 'show_images_locally', True):
                        try:
                            stats = self.get_memory_stats()
                            logger.info("\n🧠 СТАТИСТИКА ПАМЯТИ CHROMADB:")
                            if "error" not in stats:
                                logger.info(f"   Всего записей: {stats['total_records']}")
                                logger.info(f"   Диалогов: {stats['conversations']}")
                                logger.info(f"   Предпочтений: {stats['preferences']}")
                                logger.info(f"   Путь к БД: {stats['database_path']}")
                                logger.info(f"   Модель эмбеддингов: {stats['embedding_model']}")
                            else:
                                logger.error(f"   Ошибка получения статистики: {stats['error']}")
                        except Exception as e:
                            logger.error(f"Ошибка получения статистики памяти: {e}")
                    continue
                if user_input.lower() in ['gpu', 'видеокарта', 'gpuinfo']:
                    # Показываем информацию о GPU для ChromaDB
                    if getattr(self, 'show_images_locally', True):
                        try:
                            gpu_info = self.get_gpu_info()
                            logger.info("\n🎮 ИНФОРМАЦИЯ О GPU ДЛЯ CHROMADB:")
                            if "error" not in gpu_info:
                                logger.info(f"   GPU доступен: {'Да' if gpu_info['gpu_available'] else 'Нет'}")
                                if gpu_info['gpu_available']:
                                    logger.info(f"   Название GPU: {gpu_info['gpu_name']}")
                                    logger.info(f"   Память GPU: {gpu_info['gpu_memory']:.1f} GB")
                                logger.info(f"   Используемое устройство: {gpu_info['device_used']}")
                            else:
                                logger.error(f"   Ошибка получения информации о GPU: {gpu_info['error']}")
                        except Exception as e:
                            logger.error(f"Ошибка получения информации о GPU: {e}")
                    continue
                if user_input.lower() in ['cleanup', 'очистка', 'clean']:
                    # Очищаем старые записи из памяти
                    if getattr(self, 'show_images_locally', True):
                        try:
                            days = input("🗑️ Введите количество дней для хранения записей (по умолчанию 30): ").strip()
                            days_to_keep = int(days) if days.isdigit() else 30
                            deleted_count = self.cleanup_old_memory(days_to_keep)
                            logger.info(f"🧹 Удалено {deleted_count} старых записей из памяти")
                        except Exception as e:
                            logger.error(f"Ошибка очистки памяти: {e}")
                    continue
                if user_input.lower() in ['search', 'поиск', 'find']:
                    # Поиск похожих диалогов в памяти
                    if getattr(self, 'show_images_locally', True):
                        try:
                            query = input("🔍 Введите поисковый запрос: ").strip()
                            if query:
                                results = self.search_similar_conversations(query, n_results=3)
                                if results:
                                    logger.info(f"\n🔍 НАЙДЕНО {len(results)} ПОХОЖИХ ДИАЛОГОВ:")
                                    for i, result in enumerate(results, 1):
                                        logger.info(f"   {i}. Схожесть: {result['similarity']:.2f}")
                                        logger.info(f"      ID: {result['id']}")
                                        logger.info(f"      Текст: {result['document'][:100]}...")
                                        logger.info(f"      Время: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(result['metadata']['timestamp']))}")
                                        logger.info("")
                                else:
                                    logger.info("🔍 Похожих диалогов не найдено")
                        except Exception as e:
                            logger.error(f"Ошибка поиска: {e}")
                    continue
                if user_input.lower() in ['preferences', 'предпочтения', 'prefs']:
                    # Показываем предпочтения пользователя
                    if getattr(self, 'show_images_locally', True):
                        try:
                            query = input("👤 Введите контекст для поиска предпочтений (или Enter для всех): ").strip()
                            preferences = self.get_user_preferences(query if query else None)
                            if preferences:
                                logger.info(f"\n👤 ПРЕДПОЧТЕНИЯ ПОЛЬЗОВАТЕЛЯ:")
                                logger.info(preferences)
                            else:
                                logger.info("👤 Предпочтения не найдены")
                        except Exception as e:
                            logger.error(f"Ошибка получения предпочтений: {e}")
                    continue
                if not user_input:
                    continue

                # 4. Перехват: если есть YouTube-ссылка, скачиваем видео и аудио, обрабатываем аудио, затем формируем brain_input: вопрос пользователя (без ссылки, вместо неё название ролика), затем текст из аудио, и только после этого отправляем brain_input в мозг
                # re уже импортирован в начале файла
                yt_url_match = re.search(r'https?://(?:www\.)?(?:youtube\.com|youtu\.be)/\S+', user_input)
                yt_processed = False  # Флаг для отслеживания обработки YouTube
                if yt_url_match:
                    yt_url = yt_url_match.group(0)
                    logger.info("🔗 Обнаружена ссылка на YouTube, скачиваю видео...")
                    
                    # Проверяем VPN статус перед скачиванием
                    if not self.check_vpn_status():
                        logger.warning("⚠️ VPN может не работать корректно. YouTube может быть недоступен.")
                        user_input_no_url = re.sub(r'https?://\S+', '[YouTube недоступен - проверьте VPN]', user_input).strip()
                        brain_input = f"{user_input_no_url}\n\n[ОШИБКА]: Не удалось скачать YouTube видео. Проверьте VPN соединение."
                        ai_response = self.call_brain_model(brain_input)
                        continue
                    
                    # Проверяем доступность YouTube ссылки
                    if not self.check_youtube_accessibility(yt_url):
                        logger.error("❌ YouTube ссылка недоступна. Проверьте VPN или ссылку.")
                        user_input_no_url = re.sub(r'https?://\S+', '[YouTube недоступен]', user_input).strip()
                        brain_input = f"{user_input_no_url}\n\n[ОШИБКА]: YouTube ссылка недоступна. Проверьте VPN или ссылку."
                        ai_response = self.call_brain_model(brain_input)
                        continue
                    
                    # Получаем информацию о видео
                    video_info = self.get_youtube_info(yt_url)
                    if video_info.get('success'):
                        video_title = video_info['title']
                        logger.info(f"📹 Название видео: {video_title}")
                    else:
                        video_title = "Неизвестное видео"
                        logger.warning(f"⚠️ Не удалось получить название: {video_info.get('error', 'Неизвестная ошибка')}")
                    yt_video = self.download_youtube_video(yt_url)
                    if yt_video:
                        logger.info(f"✅ Видео скачано: {yt_video}")
                        # video_title уже получен выше из get_youtube_info
                    else:
                        logger.error("❌ Не удалось скачать видео с YouTube")
                        video_title = "Видео не скачано"
                    logger.info("🔗 Скачиваю аудиодорожку для транскрипции...")
                    yt_audio = self.download_youtube_audio(yt_url)
                    if yt_audio:
                        logger.info(f"✅ Аудио скачано: {yt_audio}")
                        # Определяем язык по названию видео, а не по имени файла
                        if "english" in video_title.lower() or "eng" in video_title.lower():
                            lang = "en"
                        elif "рус" in video_title.lower() or "russian" in video_title.lower():
                            lang = "ru"
                        else:
                            # Автоматически определяем язык по названию видео
                            # Для Rick Astley и подобных - английский
                            if any(word in video_title.lower() for word in ["rick", "astley", "never", "gonna", "give", "you", "up"]):
                                lang = "en"
                                logger.info("🌐 Автоматически определен язык: английский (по названию видео)")
                            else:
                                # По умолчанию используем русский
                                lang = "ru"
                                logger.info("🌐 Используется язык по умолчанию: русский")
                        audio_text = self.transcribe_audio_whisper(yt_audio, lang=lang, use_separator=getattr(self, 'use_separator', True))
                        # --- VISION ПОКАДРОВО ---
                        vision_frames_desc = ""
                        if getattr(self, 'use_vision', False) and yt_video:
                            logger.info("🖼️ Извлекаю кадры из видео для vision...")
                            frames = self.extract_video_frames(yt_video, fps=1)
                            frame_results = []  # [(timecode, desc)]
                            for idx, (timecode, b64) in enumerate(frames):
                                if not b64:
                                    continue
                                vision_prompt = "Describe absolutely everything you see in the image, including all small details, their positions, and any visible text. Be as detailed as possible."
                                desc = self.call_vision_model(b64 + "\n" + vision_prompt)
                                frame_results.append((timecode, desc))
                                logger.info(f"[VISION][{timecode}] {desc}")
                            # Группировка одинаковых описаний
                            # collections.defaultdict уже импортирован в начале файла
                            grouped = []  # [(list_of_timecodes, desc)]
                            prev_desc = None
                            prev_times = []
                            for i, (tc, desc) in enumerate(frame_results):
                                if prev_desc is None:
                                    prev_desc = desc
                                    prev_times = [tc]
                                elif desc == prev_desc:
                                    prev_times.append(tc)
                                else:
                                    grouped.append((prev_times, prev_desc))
                                    prev_desc = desc
                                    prev_times = [tc]
                            if prev_times:
                                grouped.append((prev_times, prev_desc))
                            # Формируем текст с диапазонами и списками
                            def format_timecodes(times):
                                def tc_to_sec(tc):
                                    h, m, s = map(int, tc.strip('[]').split(':'))
                                    return h*3600 + m*60 + s
                                if len(times) == 1:
                                    return times[0]
                                secs = [tc_to_sec(t) for t in times]
                                sorted_pairs = sorted(zip(secs, times))
                                secs_sorted, times_sorted = zip(*sorted_pairs)
                                # Проверяем, есть ли диапазон подряд
                                ranges = []
                                start = end = secs_sorted[0]
                                start_tc = times_sorted[0]
                                for i in range(1, len(secs_sorted)):
                                    if secs_sorted[i] == end + 1:
                                        end = secs_sorted[i]
                                    else:
                                        if start != end:
                                            ranges.append(f"[{start_tc}-{times_sorted[i-1]}]")
                                        else:
                                            ranges.append(f"{start_tc}")
                                        start = end = secs_sorted[i]
                                        start_tc = times_sorted[i]
                                if start != end:
                                    ranges.append(f"[{start_tc}-{times_sorted[-1]}]")
                                else:
                                    ranges.append(f"{start_tc}")
                                return ', '.join(ranges)
                            for times, desc in grouped:
                                vision_frames_desc += f"{format_timecodes(times)}: {desc}\n"
                        # Формируем brain_input: вопрос пользователя (без ссылки, вместо неё название ролика), ответы vision по кадрам, затем текст из аудио (с таймкодами)
                        user_input_no_url = re.sub(r'https?://\S+', f'[Видео]: {video_title}' if video_title else '', user_input).strip()
                        brain_input = ""
                        brain_input += user_input_no_url
                        if vision_frames_desc:
                            brain_input += f"\n[Покадровое описание видео]:\n{vision_frames_desc}"
                        if audio_text:
                            brain_input += f"\n[Текст из аудио]:\n{audio_text}"
                        # Отправляем в мозг и обрабатываем ответ
                        ai_response = self.call_brain_model(brain_input)
                        # Показываем информацию о контексте
                        logger.info(f"📊 {self.get_context_info()}")
                        continue_dialog = self.process_ai_response(ai_response)
                        if not continue_dialog:
                            logger.info("\n" + "="*60)
                        yt_processed = True  # Отмечаем, что YouTube обработан
                        continue  # пропускаем стандартный brain_input ниже
                    else:
                        logger.error("❌ Не удалось скачать аудио с YouTube")

                # 5. Собираем итоговый запрос для мозга (только если не было YouTube обработки)
                if not yt_processed:  # Используем флаг вместо yt_url_match
                    brain_input = ""
                    if vision_desc:
                        brain_input += f"[Описание изображения]:\n{vision_desc}\n"
                    if audio_text:
                        brain_input += f"[Текст из аудио]:\n{audio_text}\n"
                    brain_input += user_input

                    # 6. Отправляем запрос в мозг
                    ai_response = self.call_brain_model(brain_input)
                    # Показываем информацию о контексте
                    logger.info(f"📊 {self.get_context_info()}")
                    # Обрабатываем ответ AI
                    continue_dialog = self.process_ai_response(ai_response)
                    if not continue_dialog:
                        logger.info("\n" + "="*60)

            except Exception as e:
                if isinstance(e, KeyboardInterrupt):
                    logger.info("\n👋 Программа прервана пользователем")
                    break
                logger.error(f"❌ Неожиданная ошибка: {str(e)}")
                # Убираем дублирование обработки аудио - это уже делается выше
                audio_text = ""

                # 3. Запрашиваем у пользователя текстовый вопрос
                try:
                    user_input = input("\n👤 Ваш вопрос (или Enter для пропуска, либо вставьте ссылку на YouTube): ").strip()
                except EOFError:
                    # Если ввод из файла/pipe, используем пустую строку
                    user_input = ""
                    logger.info("📝 Ввод из файла/pipe, продолжаю...")
                if user_input.lower() in ['exit', 'quit', 'выход']:
                    logger.info("👋 До свидания!")
                    break
                if user_input.lower() in ['stats', 'метрики', 'статистика']:
                    # Показываем метрики производительности
                    stats = self.get_performance_stats()
                    logger.info("\n📊 МЕТРИКИ ПРОИЗВОДИТЕЛЬНОСТИ:")
                    logger.info(f"   Всего действий: {stats['total_actions']}")
                    logger.info(f"   Среднее время ответа: {stats['avg_response_time']} сек")
                    if stats['recent_metrics']:
                        logger.info("   Последние действия:")
                        for metric in stats['recent_metrics'][-5:]:  # Показываем последние 5
                            timestamp = time.strftime("%H:%M:%S", time.localtime(metric['timestamp']))
                            logger.info(f"     [{timestamp}] {metric['action']}: {metric['response_time']:.2f} сек")
                    logger.info(f"   {self.get_context_info()}")
                    continue
                if user_input.lower() in ['reset', 'сброс', 'очистить']:
                    # Сбрасываем метрики и историю
                    self.performance_metrics.clear()
                    self.conversation_history.clear()
                    self.current_context_length = 0
                    logger.info("🔄 Метрики и история сброшены")
                    continue
                if user_input.lower() in ['logs', 'логи']:
                    # Показываем последние записи из лог-файла
                    try:
                        with open("ai_orchestrator.log", "r", encoding="utf-8") as f:
                            lines = f.readlines()
                            logger.info("\n📝 ПОСЛЕДНИЕ ЗАПИСИ В ЛОГЕ:")
                            for line in lines[-10:]:  # Показываем последние 10 строк
                                logger.info(f"   {line.strip()}")
                    except Exception as e:
                        logger.error(f"Ошибка чтения лог-файла: {e}")
                    continue
                if user_input.lower() in ['export', 'экспорт']:
                    # Экспортируем метрики в JSON файл
                    try:
                        stats = self.get_performance_stats()
                        export_data = {
                            "export_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "performance_stats": stats,
                            "context_info": {
                                "current": self.current_context_length,
                                "safe": self.safe_context_length,
                                "max": self.max_context_length
                            }
                        }
                        filename = f"metrics_export_{int(time.time())}.json"
                        with open(filename, "w", encoding="utf-8") as f:
                            json.dump(export_data, f, ensure_ascii=False, indent=2)
                        logger.info(f"📊 Метрики экспортированы в {filename}")
                    except Exception as e:
                        logger.error(f"Ошибка экспорта метрик: {e}")
                    continue
                if not user_input:
                    continue



            except KeyboardInterrupt:
                logger.info("\n�� Программа прервана пользователем")
                break



    def start_telegram_bot(self) -> bool:
        """Запускает Telegram бота. Возвращает True при успешном старте, иначе False."""
        if not self.telegram_bot_token:
            logger.warning("❌ Telegram Bot токен не указан")
            return False
        
        # Предварительная проверка токена через getMe и редактирование токена в логах
        try:
            redacted = self.telegram_bot_token[:10] + "..." if len(self.telegram_bot_token) > 13 else "***"
            logger.info(f"🔐 Проверяю Telegram токен (redacted: {redacted})")
            resp = requests.get(f"https://api.telegram.org/bot{self.telegram_bot_token}/getMe", timeout=5)
            if resp.status_code != 200:
                logger.error("❌ Ошибка проверки Telegram токена: сервер вернул неуспешный статус")
                return False
            data = resp.json()
            if not data.get("ok"):
                # Не логируем сырой токен
                logger.error(f"❌ Telegram токен отклонен сервером (token: {redacted})")
                return False
        except Exception as e:
            logger.error(f"❌ Ошибка проверки Telegram токена: {e}")
            return False
        
        try:
            # Создаем приложение
            self.telegram_app = Application.builder().token(self.telegram_bot_token).build()
            
            # Добавляем обработчики
            self.telegram_app.add_handler(CommandHandler("start", self._telegram_start))
            self.telegram_app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self._telegram_text_message))
            self.telegram_app.add_handler(MessageHandler(filters.PHOTO, self._telegram_photo_message))
            self.telegram_app.add_handler(MessageHandler(filters.AUDIO | filters.VOICE, self._telegram_audio_message))
            
            # Запускаем бота в фоне в отдельном потоке
            import threading
            def run_bot():
                loop = None
                try:
                    # Создаем новый event loop для потока
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    from typing import Any, cast
                    coro = self.telegram_app.run_polling(allowed_updates=Update.ALL_TYPES)
                    if coro is not None:
                        loop.run_until_complete(cast(Any, coro))
                except Exception as e:
                    # В веб-режиме логируем тихо
                    if not getattr(self, 'show_images_locally', True):
                        logger.error(f"❌ Ошибка в Telegram боте: {e}")
                    else:
                        logger.debug(f"Telegram bot polling error: {e}")
                finally:
                    if loop is not None:
                        try:
                            loop.close()
                        except Exception:
                            pass
            
            bot_thread = threading.Thread(target=run_bot, daemon=True)
            bot_thread.start()
            
            # Показываем сообщение только в консольном режиме
            if not getattr(self, 'show_images_locally', True):
                logger.info("🤖 Telegram бот запущен в фоновом режиме")
            return True
            
        except Exception as e:
            # В веб-режиме логируем тихо
            if not getattr(self, 'show_images_locally', True):
                logger.error(f"❌ Ошибка запуска Telegram бота: {e}")
            else:
                logger.debug(f"Telegram bot startup error: {e}")
            return False

    async def _telegram_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик команды /start"""
        if update is None or update.message is None or update.effective_user is None:
            return
        user_id = str(update.effective_user.id)
        if user_id != self.telegram_allowed_user_id:
            await update.message.reply_text("❌ У вас нет доступа к этому боту.")
            return
        
        await update.message.reply_text(
            "🤖 Привет! Я Нейро - AI оркестратор.\n"
            "Я могу:\n"
            "• Обрабатывать текстовые сообщения\n"
            "• Анализировать изображения\n"
            "• Транскрибировать аудио\n"
            "• Генерировать изображения\n"
            "• Выполнять команды PowerShell\n"
            "• Искать информацию в интернете\n\n"
            "Просто отправьте мне сообщение, изображение или аудио!"
        )

    async def _telegram_text_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик текстовых сообщений"""
        if update is None or update.message is None or update.effective_user is None or update.effective_chat is None:
            return
        user_id = str(update.effective_user.id)
        if user_id != self.telegram_allowed_user_id:
            await update.message.reply_text("❌ У вас нет доступа к этому боту.")
            return
        
        text = update.message.text if update.message and update.message.text else ""
        await update.message.reply_text("🔄 Обрабатываю ваше сообщение...")

        try:
            # Отправляем в мозг
            ai_response = self.call_brain_model(text or "")

            # Обрабатываем ответ AI
            continue_dialog = self.process_ai_response(ai_response)

            if not continue_dialog:
                # Если диалог завершен, отправляем финальный ответ
                if hasattr(self, 'last_final_response') and self.last_final_response:
                    await update.message.reply_text(self.last_final_response)

                    # Если есть сгенерированное изображение, отправляем его
                    if hasattr(self, 'last_generated_image_b64') and self.last_generated_image_b64:
                        try:
                            # Конвертируем base64 в bytes
                            img_bytes = base64.b64decode(self.last_generated_image_b64)

                            # Отправляем изображение
                            await context.bot.send_photo(
                                chat_id=update.effective_chat.id,
                                photo=img_bytes,
                                caption="🎨 Сгенерированное изображение"
                            )

                            # Очищаем
                            self.last_generated_image_b64 = None

                        except Exception as e:
                            # В веб-режиме логируем тихо
                            if not getattr(self, 'show_images_locally', True):
                                logger.error(f"❌ Ошибка отправки изображения: {e}")
                            else:
                                logger.debug(f"Telegram image send error: {e}")
                            await update.message.reply_text("❌ Ошибка отправки изображения")
                else:
                    await update.message.reply_text("✅ Задача выполнена!")
            else:
                # Если диалог продолжается, отправляем промежуточный ответ
                await update.message.reply_text("🔄 Обрабатываю... Пожалуйста, подождите.")

        except Exception as e:
            # Веб-режим: логируем тихо
            if getattr(self, 'show_images_locally', True):
                logger.error(f"❌ Ошибка обработки текстового сообщения: {e}")
            else:
                logger.debug(f"Telegram text message error: {e}")
            await update.message.reply_text(f"❌ Произошла ошибка: {str(e)}")

    async def _telegram_photo_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик фотографий"""
        if update is None or update.message is None or update.effective_user is None or update.effective_chat is None:
            return
        user_id = str(update.effective_user.id)
        if user_id != self.telegram_allowed_user_id:
            await update.message.reply_text("❌ У вас нет доступа к этому боту.")
            return
        
        await update.message.reply_text("🖼️ Обрабатываю изображение...")
        
        try:
            # Получаем фото
            photo = update.message.photo[-1]  # Берем самое большое фото
            file = await context.bot.get_file(photo.file_id)
            
            # Скачиваем фото
            photo_bytes = await file.download_as_bytearray()
            photo_b64 = base64.b64encode(photo_bytes).decode('ascii')
            
            # Анализируем изображение
            vision_desc = self.call_vision_model(photo_b64)
            
            # Отправляем описание
            await update.message.reply_text(f"👁️ Описание изображения:\n{vision_desc}")
            
            # Сохраняем для возможного использования в диалоге
            self.last_telegram_image = photo_b64
            
        except Exception as e:
            # В веб-режиме логируем тихо
            if getattr(self, 'show_images_locally', True):
                logger.error(f"❌ Ошибка обработки фото: {e}")
            else:
                logger.debug(f"Telegram photo processing error: {e}")
            await update.message.reply_text(f"❌ Ошибка обработки изображения: {str(e)}")

    async def _telegram_audio_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик аудио сообщений"""
        if update is None or update.message is None or update.effective_user is None or update.effective_chat is None:
            return
        user_id = str(update.effective_user.id)
        if user_id != self.telegram_allowed_user_id:
            await update.message.reply_text("❌ У вас нет доступа к этому бота.")
            return
        
        await update.message.reply_text("🎵 Обрабатываю аудио...")
        
        try:
            # Получаем аудио
            if update.message.audio:
                audio = update.message.audio
            else:
                audio = update.message.voice

            if audio is None:
                await update.message.reply_text("❌ В сообщении нет аудиофайла")
                return
            
            file = await context.bot.get_file(audio.file_id)
            
            # Скачиваем аудио
            audio_bytes = await file.download_as_bytearray()
            
            # Сохраняем во временный файл
            temp_dir = os.path.join(os.path.dirname(__file__), "temp_audio")
            os.makedirs(temp_dir, exist_ok=True)
            temp_file = os.path.join(temp_dir, f"telegram_audio_{int(time.time())}.ogg")
            
            with open(temp_file, 'wb') as f:
                f.write(audio_bytes)
            
            # Транскрибируем аудио
            transcript = self.transcribe_audio_whisper(temp_file, use_separator=False)
            
            if transcript and not transcript.startswith("[Whisper error]"):
                await update.message.reply_text(f"🎤 Транскрипция аудио:\n{transcript}")
                
                # Сохраняем для возможного использования в диалоге
                self.last_telegram_audio_transcript = transcript
            else:
                await update.message.reply_text("❌ Не удалось распознать аудио")
            
            # Удаляем временный файл
            try:
                os.remove(temp_file)
            except Exception:
                pass
                
        except Exception as e:
            # В веб-режиме логируем тихо
            if getattr(self, 'show_images_locally', True):
                logger.error(f"❌ Ошибка обработки аудио: {e}")
            else:
                logger.debug(f"Telegram audio processing error: {e}")
            await update.message.reply_text(f"❌ Ошибка обработки аудио: {str(e)}")

    def play_audio_file(self, audio_path: str) -> bool:
        """
        Воспроизводит аудиофайл один раз без зацикливания
        
        Args:
            audio_path: Путь к аудиофайлу
            
        Returns:
            True если воспроизведение запущено успешно, False при ошибке
        """
        try:
            import pygame  # type: ignore
            import time
            
            logger.info(f"🔊 Воспроизводим аудио: {os.path.basename(audio_path)}")
            
            # Инициализируем pygame mixer
            pygame.mixer.init()
            
            # Загружаем и воспроизводим аудио
            try:
                pygame.mixer.music.load(audio_path)
                pygame.mixer.music.play()
                
                # Ждем окончания воспроизведения
                while pygame.mixer.music.get_busy():
                    time.sleep(0.1)
                
                # Останавливаем и закрываем
                pygame.mixer.music.stop()
                pygame.mixer.quit()
                
                logger.info("✅ Аудио воспроизведено через pygame")
                return True
                
            except Exception as e:
                logger.warning(f"⚠️ pygame воспроизведение не удалось: {e}")
                pygame.mixer.quit()
                
                # Fallback: открываем в плеере по умолчанию
                import subprocess
                subprocess.Popen(["start", audio_path], shell=True)
                logger.info("🔄 Открыт в плеере по умолчанию")
                return True
                
        except ImportError:
            logger.warning("⚠️ pygame не установлен, используем fallback")
            # Fallback: открываем в плеере по умолчанию
            import subprocess
            subprocess.Popen(["start", audio_path], shell=True)
            logger.info("🔄 Открыт в плеере по умолчанию")
            return True
            
        except Exception as e:
            logger.error(f"❌ Ошибка воспроизведения аудио: {e}")
            return False

    def text_to_speech(self, text: str, voice: str = "male", language: str = "ru", auto_play: bool = True) -> str:
        """
        Озвучивает текст с помощью gTTS (Google Text-to-Speech)
        
        Args:
            text: Текст для озвучки
            voice: Тип голоса ("male" или "female") - пока не используется в gTTS
            language: Язык текста ("ru", "en", etc.)
            auto_play: Автоматически воспроизвести после создания файла
            
        Returns:
            Путь к созданному аудиофайлу или пустая строка при ошибке
        """
        try:
            from gtts import gTTS
            
            # Создаем папку для сгенерированной речи
            output_dir = os.path.join(os.path.dirname(__file__), "Audio", "generated_speech")
            os.makedirs(output_dir, exist_ok=True)
            
            # Генерируем уникальное имя файла
            timestamp = int(time.time())
            filename = f"tts_{voice}_{language}_{timestamp}.mp3"
            output_path = os.path.join(output_dir, filename)
            
            logger.info(f"🎤 Озвучиваю текст: {text[:100]}...")
            logger.info(f"🔊 Голос: {voice}, Язык: {language}")
            logger.info(f"🌐 Использую Google TTS API")
            
            # Создаем TTS объект
            tts = gTTS(text=text, lang=language, slow=False)
            
            # Сохраняем аудиофайл
            tts.save(output_path)
            
            logger.info(f"✅ Аудиофайл сохранен: {output_path}")
            
            # Автоматически воспроизводим, если включено
            if auto_play:
                self.play_audio_file(output_path)
            
            return output_path
            
        except ImportError:
            logger.error("❌ gTTS не установлен. Установите: pip install gTTS")
            return ""
        except Exception as e:
            logger.error(f"❌ Ошибка озвучки текста: {e}")
            logger.error(f"🔍 Тип ошибки: {type(e).__name__}")
            import traceback
            logger.error(f"📋 Traceback: {traceback.format_exc()}")
            return ""

    def enhance_prompt_with_memory(self, user_message: str, system_prompt: str = "") -> str:
        """
        Улучшает промпт с помощью контекста из памяти
        
        Args:
            user_message: Сообщение пользователя
            system_prompt: Системный промпт
            
        Returns:
            Улучшенный промпт с контекстом
        """
        try:
            # Получаем релевантный контекст
            context = self.get_relevant_context(user_message, max_context_length=1500)
            
            # Получаем предпочтения пользователя
            preferences = self.get_user_preferences(user_message)
            
            # Формируем улучшенный промпт
            enhanced_prompt = system_prompt
            added_content = ""
            
            if context:
                context_section = f"\n\nРЕЛЕВАНТНЫЙ КОНТЕКСТ ИЗ ПРЕДЫДУЩИХ ДИАЛОГОВ:\n{context}\n\nИНСТРУКЦИЯ: Используйте эту информацию только если она релевантна текущему запросу. Не упоминайте источник информации явно."
                enhanced_prompt += context_section
                added_content += context_section
            
            if preferences:
                preferences_section = f"\n\nПРЕДПОЧТЕНИЯ ПОЛЬЗОВАТЕЛЯ:\n{preferences}\n\nИНСТРУКЦИЯ: Учитывайте эти предпочтения при формировании ответа, но не упоминайте их явно."
                enhanced_prompt += preferences_section
                added_content += preferences_section
            
            enhanced_prompt += f"\n\nТЕКУЩИЙ ЗАПРОС ПОЛЬЗОВАТЕЛЯ: {user_message}"
            
            # Подсчитываем только добавленный контент (без системного промпта и запроса пользователя)
            added_length = len(added_content)
            if added_length > 0:
                logger.info(f"📚 Промпт улучшен с помощью памяти (добавлено: {added_length} символов)")
            else:
                logger.info(f"📚 Промпт не улучшен (память пуста)")
            
            # Логируем общую длину промпта для сравнения с токенами
            total_prompt_length = len(enhanced_prompt)
            logger.info(f"📝 Общая длина промпта: {total_prompt_length} символов")
            
            return enhanced_prompt
            
        except Exception as e:
            logger.error(f"❌ Ошибка улучшения промпта: {e}")
            return f"{system_prompt}\n\nТЕКУЩИЙ ЗАПРОС ПОЛЬЗОВАТЕЛЯ: {user_message}"
    
    def auto_save_conversation(self, user_message: str, ai_response: str, 
                              context: str = "", metadata: Optional[Dict[str, Any]] = None):
        """
        Автоматически сохраняет диалог в память
        
        Args:
            user_message: Сообщение пользователя
            ai_response: Ответ ИИ
            context: Дополнительный контекст
            metadata: Дополнительные метаданные
        """
        try:
            # Добавляем базовые метаданные
            if metadata is None:
                metadata = {}
            
            metadata.update({
                "auto_saved": True,
                "response_length": len(ai_response),
                "user_message_length": len(user_message)
            })
            
            # Сохраняем в память
            success = self.add_to_memory(user_message, ai_response, context, metadata)
            
            if success:
                logger.info("💾 Диалог автоматически сохранен в память")
            else:
                logger.warning("⚠️ Не удалось сохранить диалог в память")
                
        except Exception as e:
            logger.error(f"❌ Ошибка автоматического сохранения: {e}")
    
    def extract_preferences_from_response(self, user_message: str, ai_response: str):
        """
        Извлекает предпочтения пользователя из диалога и сохраняет их
        
        Args:
            user_message: Сообщение пользователя
            ai_response: Ответ ИИ
        """
        try:
            # Простые паттерны для извлечения предпочтений
            preference_patterns = [
                r"мне нравится (.+?)(?:\.|$)",
                r"я предпочитаю (.+?)(?:\.|$)",
                r"лучше всего (.+?)(?:\.|$)",
                r"хочу (.+?)(?:\.|$)",
                r"нужно (.+?)(?:\.|$)",
                r"важно (.+?)(?:\.|$)"
            ]
            
            # Ищем предпочтения в сообщении пользователя
            for pattern in preference_patterns:
                matches = re.findall(pattern, user_message.lower())
                for match in matches:
                    if len(match) > 10:  # Минимальная длина предпочтения
                        self.add_user_preference(
                            match.strip(),
                            category="user_preference",
                            metadata={"source": "extracted", "pattern": pattern}
                        )
            
            # Также ищем в ответе ИИ, если там есть подтверждения
            confirmation_patterns = [
                r"понял(?:а)?, что вам нравится (.+?)(?:\.|$)",
                r"запомню, что вы предпочитаете (.+?)(?:\.|$)",
                r"учту ваше предпочтение (.+?)(?:\.|$)"
            ]
            
            for pattern in confirmation_patterns:
                matches = re.findall(pattern, ai_response.lower())
                for match in matches:
                    if len(match) > 10:
                        self.add_user_preference(
                            match.strip(),
                            category="confirmed_preference",
                            metadata={"source": "ai_confirmation", "pattern": pattern}
                        )
                        
        except Exception as e:
            logger.error(f"❌ Ошибка извлечения предпочтений: {e}")

    ### МЕТОДЫ ДЛЯ РАБОТЫ С CHROMADB ###
    
    def add_to_memory(self, user_message: str, ai_response: str, context: str = "", 
                     metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Добавляет диалог в векторное хранилище памяти
        
        Args:
            user_message: Сообщение пользователя
            ai_response: Ответ ИИ
            context: Дополнительный контекст
            metadata: Дополнительные метаданные
            
        Returns:
            True если успешно добавлено
        """
        try:
            return self.chromadb_manager.add_conversation_memory(
                user_message, ai_response, context, metadata
            )
        except Exception as e:
            logger.error(f"❌ Ошибка добавления в память: {e}")
            return False
    
    def add_user_preference(self, preference_text: str, category: str = "general", 
                           metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Добавляет предпочтение пользователя в векторное хранилище
        
        Args:
            preference_text: Текст предпочтения
            category: Категория предпочтения
            metadata: Дополнительные метаданные
            
        Returns:
            True если успешно добавлено
        """
        try:
            return self.chromadb_manager.add_user_preference(
                preference_text, category, metadata
            )
        except Exception as e:
            logger.error(f"❌ Ошибка добавления предпочтения: {e}")
            return False
    
    def get_relevant_context(self, query: str, max_context_length: int = 2000) -> str:
        """
        Получает релевантный контекст из предыдущих диалогов
        
        Args:
            query: Текущий запрос пользователя
            max_context_length: Максимальная длина контекста
            
        Returns:
            Строка с релевантным контекстом
        """
        try:
            return self.chromadb_manager.get_conversation_context(query, max_context_length)
        except Exception as e:
            logger.error(f"❌ Ошибка получения контекста: {e}")
            return ""
    
    def get_user_preferences(self, query: Optional[str] = None) -> str:
        """
        Получает предпочтения пользователя
        
        Args:
            query: Контекстный запрос (опционально)
            
        Returns:
            Строка с предпочтениями пользователя
        """
        try:
            return self.chromadb_manager.get_user_preferences_summary(query)
        except Exception as e:
            logger.error(f"❌ Ошибка получения предпочтений: {e}")
            return ""
    
    def search_similar_conversations(self, query: str, n_results: int = 5, 
                                   similarity_threshold: float = 0.7) -> List[Dict[str, Any]]:
        """
        Ищет похожие диалоги в векторном хранилище
        
        Args:
            query: Поисковый запрос
            n_results: Количество результатов
            similarity_threshold: Порог схожести (0-1)
            
        Returns:
            Список найденных диалогов
        """
        try:
            return self.chromadb_manager.search_similar_conversations(
                query, n_results, similarity_threshold
            )
        except Exception as e:
            logger.error(f"❌ Ошибка поиска похожих диалогов: {e}")
            return []
    
    def cleanup_old_memory(self, days_to_keep: int = 30) -> int:
        """
        Удаляет старые записи из памяти
        
        Args:
            days_to_keep: Количество дней для хранения записей
            
        Returns:
            Количество удаленных записей
        """
        try:
            return self.chromadb_manager.cleanup_old_records(days_to_keep)
        except Exception as e:
            logger.error(f"❌ Ошибка очистки памяти: {e}")
            return 0
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Получает статистику памяти
        
        Returns:
            Словарь со статистикой
        """
        try:
            return self.chromadb_manager.get_database_stats()
        except Exception as e:
            logger.error(f"❌ Ошибка получения статистики памяти: {e}")
            return {"error": str(e)}
    
    def get_gpu_info(self) -> Dict[str, Any]:
        """
        Получает информацию о GPU для ChromaDB
        
        Returns:
            Словарь с информацией о GPU
        """
        try:
            if hasattr(self, 'chromadb_manager') and self.chromadb_manager:
                return self.chromadb_manager.get_gpu_info()
            else:
                return {"error": "ChromaDB не инициализирован"}
        except Exception as e:
            logger.error(f"❌ Ошибка получения информации о GPU: {e}")
            return {"error": str(e)}

def ensure_wav(audio_path: str) -> Optional[str]:
    """Конвертирует аудиофайл в WAV формат если он не WAV"""
    try:
        if audio_path.lower().endswith('.wav'):
            return audio_path
        
        # Создаем временный файл
        temp_dir = os.path.join(os.path.dirname(__file__), "Audio", "temp_convert")
        os.makedirs(temp_dir, exist_ok=True)
        wav_path = os.path.join(temp_dir, f"converted_{int(time.time())}.wav")
        
        # Конвертируем через ffmpeg
        cmd = [
            'ffmpeg', '-i', audio_path, '-acodec', 'pcm_s16le', 
            '-ar', '16000', '-ac', '1', wav_path, '-y'
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        
        return wav_path
    except Exception as e:
        logger.error(f"Ошибка конвертации в WAV: {e}")
        return audio_path

def main():
    """Главная функция"""
    parser = argparse.ArgumentParser(description='AI PowerShell Оркестратор')
    parser.add_argument('--web', action='store_true', help='Запустить веб-интерфейс')
    args = parser.parse_args()
    
    start_web = args.web
    
    # Настройка логирования для внешних библиотек при веб-интерфейсе
    if not start_web:
        logging.getLogger('httpx').setLevel(logging.WARNING)
        logging.getLogger('telegram').setLevel(logging.WARNING)
        logging.getLogger('telegram.ext').setLevel(logging.WARNING)
    
    logger.info("Настройка AI PowerShell Оркестратора")
    logger.info("="*50)
    
    # Настройки (можно вынести в конфиг файл)
    LM_STUDIO_URL = "http://localhost:1234"  # URL вашего LM Studio сервера
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "").strip()  # Ваш Google API ключ
    GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID", "").strip()   # Ваш Google CSE ID
    
    # Telegram Bot настройки
    TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()  # Введите токен вашего бота
    TELEGRAM_ALLOWED_USER_ID = os.getenv("TELEGRAM_ALLOWED_USER_ID", "").strip()  # ID пользователя, которому разрешено использовать бота

    # --- Автоматическое управление инструментами ---
    # Все инструменты по умолчанию выключены для экономии ресурсов
    # Они будут автоматически включаться при необходимости
    use_image_generation = False  # Включается автоматически при генерации изображений
    use_vision = False           # Включается автоматически при анализе изображений
    use_audio = False            # Включается автоматически при обработке аудио
    use_separator = True         # Всегда включен при использовании Whisper (как вы просили)

    # Мозг по умолчанию - используем указанную вами модель
    brain_model = "J:/models-LM Studio/mradermacher/Huihui-Qwen3-4B-Thinking-2507-abliterated-GGUF/Huihui-Qwen3-4B-Thinking-2507-abliterated.Q4_K_S.gguf"
    logger.info(f"🧠 Используется модель мозга: {os.path.basename(brain_model)}")
    logger.info("🔧 Инструменты будут автоматически включаться по требованию для экономии ресурсов")

    # Пути к моделям (можно вынести в конфиг)
    vision_model = "moondream2-llamafile"  # Имя vision-модели всегда фиксировано
    whisper_model = "ggerganov/whisper-large-v3-GGUF"

    # Проверяем и запускаем нужные модели
    orchestrator = AIOrchestrator(
        lm_studio_url=LM_STUDIO_URL,
        google_api_key=GOOGLE_API_KEY,
        google_cse_id=GOOGLE_CSE_ID
    )

    # Проверяем, заданы ли Google API настройки
    if not GOOGLE_API_KEY or not GOOGLE_CSE_ID:
        logger.warning("⚠️  ВНИМАНИЕ: Google API ключ или CSE ID не настроены!")
        logger.info("   Поиск в интернете будет недоступен.")
        logger.info("   Для настройки отредактируйте переменные в начале main()")
        logger.info("")

    # Передаем brain_model и настройки в оркестратор
    orchestrator.brain_model = brain_model
    orchestrator.use_separator = use_separator
    orchestrator.use_image_generation = use_image_generation
    orchestrator.use_vision = use_vision
    orchestrator.use_audio = use_audio
    
    # Передаем настройки Telegram
    orchestrator.telegram_bot_token = TELEGRAM_BOT_TOKEN
    orchestrator.telegram_allowed_user_id = TELEGRAM_ALLOWED_USER_ID
    
    # Запускаем веб-интерфейс если указан флаг --web
    if start_web:
        try:
            # Отключаем локальный показ изображений во всплывающих окнах при веб-режиме
            orchestrator.show_images_locally = False
        except Exception:
            pass
        # Запускаем uvicorn сервер в фоне
        try:
            # subprocess, sys, os уже импортированы в начале файла
            repo_root = os.path.dirname(os.path.abspath(__file__))
            cmd = [
                _sys.executable, "-m", "uvicorn", "webui.server:app",
                "--host", "127.0.0.1", "--port", "8001", "--app-dir", repo_root
            ]
            logger.info(f"🌐 Стартую веб-сервер: {' '.join(cmd)}")
            subprocess.Popen(cmd, cwd=repo_root)
            logger.info("Откройте в браузере: http://127.0.0.1:8001/")
        except Exception as e:
            logger.warning(f"⚠️ Не удалось запустить веб-интерфейс автоматически: {e}")

    # Запускаем Telegram бота если указан токен
    if TELEGRAM_BOT_TOKEN:
        try:
            if start_web:
                logger.info("🤖 Запускаю Telegram бота...")
            tg_started = orchestrator.start_telegram_bot()
            if start_web:
                if tg_started:
                    logger.info("✅ Telegram бот запущен")
                else:
                    logger.info("ℹ️ Telegram бот не запущен (проверьте токен)")
        except Exception as e:
            if start_web:
                logger.error(f"❌ Ошибка запуска Telegram бота: {e}")
            else:
                # В веб-режиме логируем тихо
                logger.debug(f"Telegram bot error: {e}")
    
    # Запускаем интерактивный режим
    orchestrator.run_interactive()


if __name__ == "__main__":
    main()
