"""
Пример плагина для демонстрации системы плагинов AI Orchestrator
Example plugin to demonstrate AI Orchestrator plugin system

Этот плагин показывает базовую структуру и возможности.
This plugin shows basic structure and capabilities.
"""

from plugins.base_plugin import BasePlugin, PluginError
from typing import Dict, Any, List, Optional
import time
import random


class ExamplePlugin(BasePlugin):
    """
    Пример плагина с демонстрацией различных возможностей.
    Example plugin demonstrating various capabilities.
    """
    
    def __init__(self):
        super().__init__()
        self.name = "ExamplePlugin"
        self.version = "1.0.0"
        self.description = "Демонстрационный плагин с примерами действий"
        self.author = "AI Orchestrator Team"
        
        # Внутреннее состояние плагина
        # Plugin internal state
        self.counter = 0
        self.messages_processed = 0
    
    def get_plugin_info(self) -> Dict[str, Any]:
        """Возвращает информацию о плагине"""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "author": self.author,
            "actions": self.get_available_actions(),
            "status": {
                "counter": self.counter,
                "messages_processed": self.messages_processed
            }
        }
    
    def get_available_actions(self) -> List[str]:
        """Возвращает список доступных действий"""
        return [
            "hello",
            "count",
            "random_number",
            "echo",
            "status",
            "time"
        ]
    
    def execute_action(self, action: str, data: Dict[str, Any], orchestrator) -> Any:
        """Выполняет действие плагина"""
        self.logger.info(f"Выполняется действие: {action}")
        
        if action == "hello":
            return self.handle_hello(data, orchestrator)
        elif action == "count":
            return self.handle_count(data, orchestrator)
        elif action == "random_number":
            return self.handle_random_number(data, orchestrator)
        elif action == "echo":
            return self.handle_echo(data, orchestrator)
        elif action == "status":
            return self.handle_status(data, orchestrator)
        elif action == "time":
            return self.handle_time(data, orchestrator)
        else:
            raise PluginError(f"Неизвестное действие: {action}")
    
    def handle_hello(self, data: Dict[str, Any], orchestrator) -> str:
        """Приветствие с опциональным именем"""
        name = data.get("name", "Пользователь")
        return f"Привет, {name}! Это пример плагина AI Orchestrator."
    
    def handle_count(self, data: Dict[str, Any], orchestrator) -> str:
        """Увеличивает счетчик и возвращает значение"""
        increment = data.get("increment", 1)
        self.counter += increment
        return f"Счетчик увеличен на {increment}. Текущее значение: {self.counter}"
    
    def handle_random_number(self, data: Dict[str, Any], orchestrator) -> str:
        """Генерирует случайное число в диапазоне"""
        min_val = data.get("min", 1)
        max_val = data.get("max", 100)
        
        if min_val >= max_val:
            raise PluginError("Минимальное значение должно быть меньше максимального")
        
        number = random.randint(min_val, max_val)
        return f"Случайное число от {min_val} до {max_val}: {number}"
    
    def handle_echo(self, data: Dict[str, Any], orchestrator) -> str:
        """Повторяет переданное сообщение"""
        message = data.get("message", "")
        if not message:
            raise PluginError("Не указано сообщение для повтора")
        
        return f"Эхо: {message}"
    
    def handle_status(self, data: Dict[str, Any], orchestrator) -> str:
        """Возвращает статус плагина"""
        return (f"Статус плагина {self.name}:\n"
                f"- Версия: {self.version}\n"
                f"- Счетчик: {self.counter}\n"
                f"- Обработано сообщений: {self.messages_processed}\n"
                f"- Включен: {self.enabled}")
    
    def handle_time(self, data: Dict[str, Any], orchestrator) -> str:
        """Возвращает текущее время"""
        format_str = data.get("format", "%Y-%m-%d %H:%M:%S")
        current_time = time.strftime(format_str)
        return f"Текущее время: {current_time}"
    
    def initialize(self, orchestrator) -> bool:
        """Инициализация плагина"""
        self.logger.info(f"Инициализация {self.name}")
        
        # Пример инициализации ресурсов
        self.counter = 0
        self.messages_processed = 0
        
        return True
    
    def cleanup(self) -> None:
        """Очистка ресурсов"""
        self.logger.info(f"Очистка {self.name}")
        
        # Здесь можно освободить ресурсы
        pass
    
    def on_message_received(self, message: str, orchestrator) -> Optional[str]:
        """Обработчик получения сообщения"""
        self.messages_processed += 1
        self.logger.debug(f"Обработано сообщений: {self.messages_processed}")
        
        # Пример: добавляем префикс к сообщениям, содержащим "example"
        if "example" in message.lower():
            return f"[ExamplePlugin] {message}"
        
        return None  # Не изменяем сообщение
    
    def on_response_generated(self, response: str, orchestrator) -> Optional[str]:
        """Обработчик генерации ответа"""
        # Пример: добавляем подпись к ответам
        if len(response) > 100:  # Только для длинных ответов
            return f"{response}\n\n---\nОбработано плагином {self.name}"
        
        return None  # Не изменяем ответ
