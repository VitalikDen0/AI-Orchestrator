"""
Базовый класс для плагинов AI Orchestrator
Base class for AI Orchestrator plugins

Этот файл содержит базовую структуру для создания плагинов.
This file contains the base structure for creating plugins.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)


class BasePlugin(ABC):
    """
    Базовый класс для всех плагинов AI Orchestrator.
    Base class for all AI Orchestrator plugins.
    
    Наследуйте от этого класса для создания своих плагинов.
    Inherit from this class to create your plugins.
    """
    
    def __init__(self):
        """Инициализация плагина / Plugin initialization"""
        self.name = self.__class__.__name__
        self.version = "1.0.0"
        self.description = "Базовый плагин / Base plugin"
        self.author = "Unknown"
        self.enabled = True
        
        # Логгер для плагина
        # Logger for the plugin
        self.logger = logging.getLogger(f"plugin.{self.name}")
    
    @abstractmethod
    def get_plugin_info(self) -> Dict[str, Any]:
        """
        Возвращает информацию о плагине.
        Returns plugin information.
        
        Returns:
            Dict с полями: name, version, description, author, actions
            Dict with fields: name, version, description, author, actions
        """
        pass
    
    @abstractmethod
    def get_available_actions(self) -> List[str]:
        """
        Возвращает список доступных действий плагина.
        Returns list of available plugin actions.
        
        Returns:
            List[str]: Список названий действий
            List[str]: List of action names
        """
        pass
    
    @abstractmethod
    def execute_action(self, action: str, data: Dict[str, Any], orchestrator) -> Any:
        """
        Выполняет действие плагина.
        Executes plugin action.
        
        Args:
            action: Название действия / Action name
            data: Данные для действия / Action data
            orchestrator: Ссылка на основной класс оркестратора / Reference to main orchestrator
            
        Returns:
            Any: Результат выполнения / Execution result
        """
        pass
    
    def initialize(self, orchestrator) -> bool:
        """
        Инициализация плагина при загрузке.
        Plugin initialization on load.
        
        Args:
            orchestrator: Ссылка на основной класс оркестратора
            
        Returns:
            bool: True если инициализация успешна
        """
        self.logger.info(f"Инициализация плагина {self.name}")
        return True
    
    def cleanup(self) -> None:
        """
        Очистка ресурсов при выгрузке плагина.
        Cleanup resources on plugin unload.
        """
        self.logger.info(f"Очистка плагина {self.name}")
    
    def on_message_received(self, message: str, orchestrator) -> Optional[str]:
        """
        Обработчик получения сообщения (вызывается до основной обработки).
        Message received handler (called before main processing).
        
        Args:
            message: Полученное сообщение / Received message
            orchestrator: Ссылка на оркестратор / Orchestrator reference
            
        Returns:
            Optional[str]: Измененное сообщение или None для продолжения обычной обработки
            Optional[str]: Modified message or None to continue normal processing
        """
        return None
    
    def on_response_generated(self, response: str, orchestrator) -> Optional[str]:
        """
        Обработчик генерации ответа (вызывается после основной обработки).
        Response generated handler (called after main processing).
        
        Args:
            response: Сгенерированный ответ / Generated response
            orchestrator: Ссылка на оркестратор / Orchestrator reference
            
        Returns:
            Optional[str]: Измененный ответ или None для сохранения оригинала
            Optional[str]: Modified response or None to keep original
        """
        return None


class PluginError(Exception):
    """Исключение для ошибок плагинов / Exception for plugin errors"""
    pass
