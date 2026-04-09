#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI PowerShell Orchestrator Entry Point
"""
import os
import sys
import time
import logging
import argparse
import requests
from typing import Dict, Any

# Import configuration
from config import (
    DEFAULT_LM_STUDIO_URL,
    LLAMA_CPP_MODEL_PATH,
)

# Import core orchestrator
try:
    from orchestrator_core import AIOrchestrator
except ImportError:
    # Fallback if file is not yet renamed (during transition)
    try:
        from orchestrator_core import AIOrchestrator
    except ImportError:
        print("Error: orchestrator_core.py not found. Please ensure 1.py is renamed to orchestrator_core.py")
        sys.exit(1)

# Import resource manager
from resource_manager import get_background_loader

# Setup logging
from logging_setup import setup_logging

logger = logging.getLogger(__name__)


def _resolve_brain_model_path() -> str:
    """Возвращает путь к модели мозга с приоритетом .env -> config.py."""
    env_model_path = os.getenv("BRAIN_MODEL_PATH", "").strip()
    if env_model_path:
        return env_model_path
    return LLAMA_CPP_MODEL_PATH


def _cleanup_plugins(orchestrator: "AIOrchestrator") -> None:
    """Аккуратно выгружает плагины при завершении приложения."""
    try:
        plugin_manager = getattr(orchestrator, 'plugin_manager', None)
        if not plugin_manager:
            return

        loaded_plugins = getattr(plugin_manager, 'loaded_plugins', {})
        for plugin_name in list(loaded_plugins.keys()):
            plugin_manager.unload_plugin(plugin_name)
        logger.info("🔌 Плагины очищены")
    except Exception as e:
        logger.error(f"Ошибка очистки плагинов: {e}")

def test_startup_initialization():
    """Тестирует инициализацию всех компонентов системы"""
    print("\n" + "="*60)
    print("🧪 ТЕСТ ИНИЦИАЛИЗАЦИИ AI ORCHESTRATOR")
    print("="*60)
    
    total_start_time = time.time()
    
    # Инициализация компонентов
    component_times = {}
    
    # 1. Основной оркестратор
    print("\n📦 Инициализация основного оркестратора...")
    start_time = time.time()
    
    LM_STUDIO_URL = DEFAULT_LM_STUDIO_URL
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "").strip()
    GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID", "").strip()
    
    try:
        orchestrator = AIOrchestrator(
            lm_studio_url=LM_STUDIO_URL,
            google_api_key=GOOGLE_API_KEY,
            google_cse_id=GOOGLE_CSE_ID
        )
        component_times["orchestrator"] = time.time() - start_time
        print(f"   ✅ Основной оркестратор: {component_times['orchestrator']:.2f}с")
    except Exception as e:
        component_times["orchestrator"] = time.time() - start_time
        print(f"   ❌ Основной оркестратор: {component_times['orchestrator']:.2f}с - {e}")
        return
    
    # 2. Тестируем ChromaDB
    print("\n🗃️ Тестирование ChromaDB...")
    start_time = time.time()
    
    try:
        # Инициализация ChromaDB через оркестратор
        orchestrator._ensure_chromadb_initialized()
        component_times["chromadb"] = time.time() - start_time
        print(f"   ✅ ChromaDB: {component_times['chromadb']:.2f}с")
        
        # Проверяем работу ChromaDB
        test_memory = orchestrator.add_to_memory(
            "Тестовое сообщение", "Тестовый ответ", "Контекст теста"
        )
        if test_memory:
            print("   ✅ ChromaDB функциональность: OK")
        else:
            print("   ⚠️ ChromaDB функциональность: Ошибка")
            
        component_times["chromadb"] = time.time() - start_time
    except Exception as e:
        component_times["chromadb"] = time.time() - start_time
        print(f"   ❌ ChromaDB: {component_times['chromadb']:.2f}с - {e}")
    
    # 3. Тестируем EasyOCR
    print("\n👁️ Тестирование EasyOCR...")
    start_time = time.time()
    
    try:
        # Сначала проверим, доступен ли EasyOCR как модуль
        try:
            import easyocr  # type: ignore
            easyocr_available = True
        except ImportError:
            easyocr_available = False
        
        if not easyocr_available:
            component_times["easyocr"] = time.time() - start_time
            print(f"   ❌ EasyOCR: {component_times['easyocr']:.2f}с - Модуль не установлен")
            print("   💡 Установите: pip install easyocr")
        elif orchestrator._ensure_ocr_initialized():
            component_times["easyocr"] = time.time() - start_time
            print(f"   ✅ EasyOCR: {component_times['easyocr']:.2f}с")
            
            # Проверим, что OCR reader действительно создан
            if orchestrator.ocr_reader is not None:
                print("   ✅ EasyOCR функциональность: OK")
            else:
                print("   ⚠️ EasyOCR функциональность: Reader не создан")
        else:
            component_times["easyocr"] = time.time() - start_time
            print(f"   ❌ EasyOCR: {component_times['easyocr']:.2f}с - Ошибка инициализации")
    except Exception as e:
        component_times["easyocr"] = time.time() - start_time
        print(f"   ❌ EasyOCR: {component_times['easyocr']:.2f}с - {e}")
    
    # 4. Тестируем модель мозга
    print("\n🧠 Тестирование модели мозга...")
    start_time = time.time()
    
    try:
        brain_model = _resolve_brain_model_path()
        orchestrator.brain_model = brain_model
        
        # Проверяем доступность LM Studio
        response = requests.get(f"{LM_STUDIO_URL}/v1/models", timeout=10)
        if response.status_code == 200:
            models = response.json().get("data", [])
            print(f"   📊 Всего моделей в LM Studio: {len(models)}")
            
            # Ищем любые модели, не только загруженные
            loaded_models = [m for m in models if m.get("isLoaded", False)]
            available_models = [m.get("id", "unknown") for m in models]
            
            print(f"   📊 Доступные модели: {available_models}")
            print(f"   📊 Загруженных моделей: {len(loaded_models)}")
            
            if models:  # Если есть любые модели
                component_times["brain_model"] = time.time() - start_time
                print(f"   ✅ Модель мозга: {component_times['brain_model']:.2f}с")
                
                # Тестируем запрос к модели (даже если модель не показывается как загруженная)
                test_response = orchestrator.call_brain_model("Привет! Это тест.")
                if test_response and not test_response.startswith("[Brain error]"):
                    print("   ✅ Тестовый запрос: OK")
                    print(f"   📝 Ответ модели: {test_response[:100]}...")
                else:
                    print(f"   ⚠️ Тестовый запрос: {test_response}")
            else:
                component_times["brain_model"] = time.time() - start_time
                print(f"   ⚠️ Модель мозга: {component_times['brain_model']:.2f}с - Нет моделей в LM Studio")
        else:
            component_times["brain_model"] = time.time() - start_time
            print(f"   ❌ Модель мозга: {component_times['brain_model']:.2f}с - LM Studio недоступен")
    except Exception as e:
        component_times["brain_model"] = time.time() - start_time
        print(f"   ❌ Модель мозга: {component_times['brain_model']:.2f}с - {e}")
    
    # 5. Проверяем фоновый загрузчик
    print("\n🚀 Состояние фонового загрузчика...")
    try:
        loader = get_background_loader()
        loaded = list(loader.loaded_components.keys())
        loading_tasks = list(loader.loading_tasks.keys())
        
        # Показываем только те компоненты, которые еще загружаются
        still_loading = [task for task in loading_tasks if task not in loaded]
        
        print(f"   📦 Загруженные компоненты: {loaded}")
        if still_loading:
            print(f"   🔄 Еще загружаются: {still_loading}")
        else:
            print(f"   ✅ Все компоненты загружены")
    except Exception as e:
        print(f"   ❌ Фоновый загрузчик: {e}")
    
    # 6. Проверяем плагины
    print("\n🔌 Проверка системы плагинов...")
    try:
        if orchestrator.plugin_manager:
            # Просто проверяем наличие плагинов без обращения к конкретному атрибуту
            print(f"   ✅ Система плагинов: Инициализирована")
        else:
            print("   ⚠️ Система плагинов: Не инициализирована")
    except Exception as e:
        print(f"   ❌ Система плагинов: {e}")
    
    # Итоговая статистика
    total_time = time.time() - total_start_time
    print("\n" + "="*60)
    print("📊 ИТОГОВАЯ СТАТИСТИКА ИНИЦИАЛИЗАЦИИ")
    print("="*60)
    
    # Показываем время каждого компонента с процентом от общего времени
    for component, duration in component_times.items():
        percentage = (duration / total_time * 100) if total_time > 0 else 0
        status = "✅" if duration < 30 else "⚠️" if duration < 60 else "❌"
        print(f"{status} {component:20}: {duration:6.2f}с ({percentage:5.1f}%)")
    
    print(f"\n🕐 Общее время инициализации: {total_time:.2f}с")
    
    if total_time < 10:
        print("🚀 Отлично! Быстрая инициализация")
    elif total_time < 30:
        print("✅ Хорошо! Приемлемое время инициализации")
    elif total_time < 60:
        print("⚠️ Медленно! Требуется оптимизация")
    else:
        print("❌ Очень медленно! Критические проблемы производительности")
    
    print("\n✅ Тест инициализации завершен")

def main():
    """Главная функция"""
    parser = argparse.ArgumentParser(description='AI PowerShell Оркестратор')
    parser.add_argument('--web', action='store_true', help='Запустить веб-интерфейс')
    parser.add_argument('--test-startup', action='store_true', help='Тестировать только инициализацию системы')
    args = parser.parse_args()
    
    start_web = args.web
    test_startup = args.test_startup
    
    # Если запущен тест инициализации - выполняем его и выходим
    if test_startup:
        test_startup_initialization()
        return
    
    # Настройка логирования для внешних библиотек при веб-интерфейсе
    if not start_web:
        logging.getLogger('httpx').setLevel(logging.WARNING)
        logging.getLogger('telegram').setLevel(logging.WARNING)
        logging.getLogger('telegram.ext').setLevel(logging.WARNING)
    
    logger.info("Настройка AI PowerShell Оркестратора")
    logger.info("="*50)
    
    # Настройки (можно вынести в конфиг файл)
    LM_STUDIO_URL = DEFAULT_LM_STUDIO_URL  # URL вашего LM Studio сервера
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
    use_ocr = False              # Включается автоматически при извлечении текста из изображений

    # Мозг по умолчанию - используем указанную вами модель
    brain_model = _resolve_brain_model_path()
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
    orchestrator.use_ocr = use_ocr
    
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
        from web_server_launcher import launch_web_server
        launch_web_server()

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
    try:
        orchestrator.run_interactive()
    finally:
        _cleanup_plugins(orchestrator)

if __name__ == "__main__":
    main()