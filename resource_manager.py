import threading
import time
import logging
import concurrent.futures
from typing import Dict, Any, Callable, Optional

logger = logging.getLogger(__name__)

# Глобальные переменные для фоновой загрузки
_background_loader = None
_initialization_lock = threading.Lock()

class BackgroundInitializer:
    """Класс для фоновой инициализации тяжелых компонентов"""
    
    def __init__(self):
        self.loaded_components = {}
        self.loading_tasks = {}
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
        self._chromadb_manager = None
        self._easyocr_reader = None
        self._is_loading = set()
        self.logger = logging.getLogger(__name__)
        
    def start_loading(self, component_name, loader_func, *args, **kwargs):
        """Запускает фоновую загрузку компонента"""
        if component_name not in self._is_loading and component_name not in self.loaded_components:
            self._is_loading.add(component_name)
            future = self.executor.submit(self._safe_load, component_name, loader_func, *args, **kwargs)
            self.loading_tasks[component_name] = future
            return future
        return None
    
    def _safe_load(self, component_name, loader_func, *args, **kwargs):
        """Безопасная загрузка компонента с обработкой ошибок"""
        try:
            result = loader_func(*args, **kwargs)
            self.loaded_components[component_name] = result
            self._is_loading.discard(component_name)
            return result
        except Exception as e:
            self.logger.error(f"Ошибка загрузки {component_name}: {e}")
            self._is_loading.discard(component_name)
            return None
    
    def get_component(self, component_name, timeout=30):
        """Получает компонент, ждет завершения загрузки если нужно"""
        if component_name in self.loaded_components:
            return self.loaded_components[component_name]
        
        if component_name in self.loading_tasks:
            try:
                result = self.loading_tasks[component_name].result(timeout=timeout)
                return result
            except concurrent.futures.TimeoutError:
                self.logger.warning(f"Таймаут загрузки {component_name}")
                return None
        
        return None
    
    def is_loaded(self, component_name):
        """Проверяет, загружен ли компонент"""
        return component_name in self.loaded_components
    
    def shutdown(self):
        """Завершает работу загрузчика"""
        self.executor.shutdown(wait=True)

def get_background_loader():
    """Получает глобальный экземпляр фонового загрузчика"""
    global _background_loader
    with _initialization_lock:
        if _background_loader is None:
            _background_loader = BackgroundInitializer()
        return _background_loader

def load_easyocr():
    """Загружает EasyOCR"""
    try:
        logger.info("Загружаем EasyOCR...")
        import easyocr  # type: ignore
        reader = easyocr.Reader(['ru', 'en'])
        return reader
    except Exception as e:
        logger.error(f"Ошибка загрузки EasyOCR: {e}")
        return None

def load_torch():
    """Загружает PyTorch"""
    try:
        logger.info("Загружаем PyTorch...")
        import torch
        return torch
    except Exception as e:
        logger.error(f"Ошибка загрузки PyTorch: {e}")
        return None
