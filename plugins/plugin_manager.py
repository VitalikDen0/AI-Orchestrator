"""
Менеджер плагинов AI Orchestrator
Plugin Manager for AI Orchestrator

Управляет загрузкой, выгрузкой и выполнением плагинов.
Manages loading, unloading and execution of plugins.
"""

import os
import sys
import importlib
import configparser
from typing import Dict, List, Any, Optional, Type
import logging
from pathlib import Path

from .base_plugin import BasePlugin, PluginError

logger = logging.getLogger(__name__)


class PluginManager:
    """
    Менеджер плагинов для AI Orchestrator.
    Plugin manager for AI Orchestrator.
    """
    
    def __init__(self, plugins_dir: str = "plugins"):
        """
        Инициализация менеджера плагинов.
        Plugin manager initialization.
        
        Args:
            plugins_dir: Путь к папке с плагинами / Path to plugins directory
        """
        self.plugins_dir = Path(plugins_dir)
        self.config_file = self.plugins_dir / "config.ini"
        self.loaded_plugins: Dict[str, BasePlugin] = {}
        self.plugin_config = configparser.ConfigParser()
        
        # Создаем папку plugins если её нет
        # Create plugins directory if it doesn't exist
        self.plugins_dir.mkdir(exist_ok=True)
        
        # Загружаем конфигурацию
        # Load configuration
        self.load_config()
    
    def load_config(self) -> None:
        """
        Загружает конфигурацию плагинов.
        Loads plugin configuration.
        """
        if self.config_file.exists():
            self.plugin_config.read(self.config_file, encoding='utf-8')
        else:
            # Создаем базовую конфигурацию
            # Create basic configuration
            self.plugin_config.add_section('plugins')
            self.save_config()
        
        # Убеждаемся что секция plugins существует
        # Ensure plugins section exists
        if not self.plugin_config.has_section('plugins'):
            self.plugin_config.add_section('plugins')
    
    def save_config(self) -> None:
        """
        Сохраняет конфигурацию плагинов.
        Saves plugin configuration.
        """
        with open(self.config_file, 'w', encoding='utf-8') as f:
            self.plugin_config.write(f)
    
    def discover_plugins(self) -> List[str]:
        """
        Находит все .py файлы в папке plugins (кроме служебных).
        Discovers all .py files in plugins directory (except service files).
        
        Returns:
            List[str]: Список названий плагинов
        """
        plugins = []
        
        for file_path in self.plugins_dir.glob("*.py"):
            # Пропускаем служебные файлы
            # Skip service files
            if file_path.name.startswith('_') or file_path.name in ['base_plugin.py', 'plugin_manager.py']:
                continue
            
            plugin_name = file_path.stem
            plugins.append(plugin_name)
        
        return plugins
    
    def is_plugin_enabled(self, plugin_name: str) -> bool:
        """
        Проверяет, включен ли плагин в конфигурации.
        Checks if plugin is enabled in configuration.
        
        Args:
            plugin_name: Название плагина / Plugin name
            
        Returns:
            bool: True если плагин включен
        """
        return self.plugin_config.getboolean('plugins', plugin_name, fallback=True)
    
    def set_plugin_enabled(self, plugin_name: str, enabled: bool) -> None:
        """
        Включает или выключает плагин в конфигурации.
        Enables or disables plugin in configuration.
        
        Args:
            plugin_name: Название плагина / Plugin name
            enabled: Включить (True) или выключить (False) / Enable (True) or disable (False)
        """
        self.plugin_config.set('plugins', plugin_name, str(enabled).lower())
        self.save_config()
    
    def load_plugin(self, plugin_name: str, orchestrator=None) -> bool:
        """
        Загружает и инициализирует плагин.
        Loads and initializes plugin.
        
        Args:
            plugin_name: Название плагина / Plugin name
            orchestrator: Ссылка на основной оркестратор / Main orchestrator reference
            
        Returns:
            bool: True если плагин успешно загружен
        """
        try:
            # Проверяем, не загружен ли уже плагин
            # Check if plugin is already loaded
            if plugin_name in self.loaded_plugins:
                logger.warning(f"Плагин {plugin_name} уже загружен")
                return True
            
            # Проверяем, включен ли плагин
            # Check if plugin is enabled
            if not self.is_plugin_enabled(plugin_name):
                logger.info(f"Плагин {plugin_name} отключен в конфигурации")
                return False
            
            # Добавляем папку plugins в sys.path
            # Add plugins directory to sys.path
            plugins_path = str(self.plugins_dir.absolute())
            if plugins_path not in sys.path:
                sys.path.insert(0, plugins_path)
            
            # Импортируем модуль плагина
            # Import plugin module
            module = importlib.import_module(plugin_name)
            
            # Ищем класс плагина (должен наследоваться от BasePlugin)
            # Find plugin class (must inherit from BasePlugin)
            plugin_class = None
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (isinstance(attr, type) and 
                    issubclass(attr, BasePlugin) and 
                    attr != BasePlugin):
                    plugin_class = attr
                    break
            
            if plugin_class is None:
                raise PluginError(f"В модуле {plugin_name} не найден класс плагина")
            
            # Создаем экземпляр плагина
            # Create plugin instance
            plugin_instance = plugin_class()
            
            # Инициализируем плагин
            # Initialize plugin
            if plugin_instance.initialize(orchestrator):
                self.loaded_plugins[plugin_name] = plugin_instance
                
                # Добавляем плагин в конфигурацию если его там нет
                # Add plugin to config if not present
                if not self.plugin_config.has_option('plugins', plugin_name):
                    self.set_plugin_enabled(plugin_name, True)
                
                logger.info(f"✅ Плагин {plugin_name} успешно загружен")
                return True
            else:
                raise PluginError(f"Ошибка инициализации плагина {plugin_name}")
                
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки плагина {plugin_name}: {e}")
            return False
    
    def unload_plugin(self, plugin_name: str) -> bool:
        """
        Выгружает плагин.
        Unloads plugin.
        
        Args:
            plugin_name: Название плагина / Plugin name
            
        Returns:
            bool: True если плагин успешно выгружен
        """
        if plugin_name not in self.loaded_plugins:
            logger.warning(f"Плагин {plugin_name} не загружен")
            return False
        
        try:
            # Вызываем cleanup у плагина
            # Call plugin cleanup
            self.loaded_plugins[plugin_name].cleanup()
            
            # Удаляем из загруженных
            # Remove from loaded plugins
            del self.loaded_plugins[plugin_name]
            
            logger.info(f"✅ Плагин {plugin_name} выгружен")
            return True
            
        except Exception as e:
            logger.error(f"❌ Ошибка выгрузки плагина {plugin_name}: {e}")
            return False
    
    def load_all_plugins(self, orchestrator=None) -> None:
        """
        Загружает все найденные плагины.
        Loads all discovered plugins.
        
        Args:
            orchestrator: Ссылка на основной оркестратор / Main orchestrator reference
        """
        discovered_plugins = self.discover_plugins()
        
        for plugin_name in discovered_plugins:
            self.load_plugin(plugin_name, orchestrator)
    
    def get_loaded_plugins(self) -> Dict[str, BasePlugin]:
        """
        Возвращает словарь загруженных плагинов.
        Returns dictionary of loaded plugins.
        
        Returns:
            Dict[str, BasePlugin]: Загруженные плагины
        """
        return self.loaded_plugins.copy()
    
    def get_plugin_actions(self) -> Dict[str, List[str]]:
        """
        Возвращает доступные действия всех загруженных плагинов.
        Returns available actions of all loaded plugins.
        
        Returns:
            Dict[str, List[str]]: Словарь {plugin_name: [actions]}
        """
        actions = {}
        for plugin_name, plugin in self.loaded_plugins.items():
            try:
                actions[plugin_name] = plugin.get_available_actions()
            except Exception as e:
                logger.error(f"Ошибка получения действий плагина {plugin_name}: {e}")
                actions[plugin_name] = []
        
        return actions
    
    def execute_plugin_action(self, plugin_name: str, action: str, data: Dict[str, Any], orchestrator=None) -> Any:
        """
        Выполняет действие плагина.
        Executes plugin action.
        
        Args:
            plugin_name: Название плагина / Plugin name
            action: Название действия / Action name
            data: Данные для действия / Action data
            orchestrator: Ссылка на оркестратор / Orchestrator reference
            
        Returns:
            Any: Результат выполнения / Execution result
        """
        if plugin_name not in self.loaded_plugins:
            raise PluginError(f"Плагин {plugin_name} не загружен")
        
        plugin = self.loaded_plugins[plugin_name]
        
        try:
            return plugin.execute_action(action, data, orchestrator)
        except Exception as e:
            logger.error(f"Ошибка выполнения действия {action} плагина {plugin_name}: {e}")
            raise PluginError(f"Ошибка выполнения действия {action}: {e}")
    
    def call_hook_message_received(self, message: str, orchestrator=None) -> str:
        """
        Вызывает hook on_message_received у всех плагинов.
        Calls on_message_received hook for all plugins.
        
        Args:
            message: Исходное сообщение / Original message
            orchestrator: Ссылка на оркестратор / Orchestrator reference
            
        Returns:
            str: Обработанное сообщение / Processed message
        """
        current_message = message
        
        for plugin_name, plugin in self.loaded_plugins.items():
            try:
                result = plugin.on_message_received(current_message, orchestrator)
                if result is not None:
                    current_message = result
                    logger.debug(f"Плагин {plugin_name} изменил сообщение")
            except Exception as e:
                logger.error(f"Ошибка в hook on_message_received плагина {plugin_name}: {e}")
        
        return current_message
    
    def call_hook_response_generated(self, response: str, orchestrator=None) -> str:
        """
        Вызывает hook on_response_generated у всех плагинов.
        Calls on_response_generated hook for all plugins.
        
        Args:
            response: Исходный ответ / Original response
            orchestrator: Ссылка на оркестратор / Orchestrator reference
            
        Returns:
            str: Обработанный ответ / Processed response
        """
        current_response = response
        
        for plugin_name, plugin in self.loaded_plugins.items():
            try:
                result = plugin.on_response_generated(current_response, orchestrator)
                if result is not None:
                    current_response = result
                    logger.debug(f"Плагин {plugin_name} изменил ответ")
            except Exception as e:
                logger.error(f"Ошибка в hook on_response_generated плагина {plugin_name}: {e}")
        
        return current_response
    
    def get_plugins_info(self) -> Dict[str, Dict[str, Any]]:
        """
        Возвращает информацию о всех загруженных плагинах.
        Returns information about all loaded plugins.
        
        Returns:
            Dict[str, Dict[str, Any]]: Информация о плагинах
        """
        info = {}
        
        for plugin_name, plugin in self.loaded_plugins.items():
            try:
                info[plugin_name] = plugin.get_plugin_info()
            except Exception as e:
                logger.error(f"Ошибка получения информации о плагине {plugin_name}: {e}")
                info[plugin_name] = {"error": str(e)}
        
        return info
