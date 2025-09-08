"""
Пакет плагинов AI Orchestrator
AI Orchestrator Plugins Package

Содержит систему плагинов и базовые классы.
Contains plugin system and base classes.
"""

from .base_plugin import BasePlugin, PluginError
from .plugin_manager import PluginManager

__all__ = ['BasePlugin', 'PluginError', 'PluginManager']
