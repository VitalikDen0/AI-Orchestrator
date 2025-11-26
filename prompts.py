"""Prompt loading and caching utilities for AI system prompts."""

from __future__ import annotations

import logging
import os
from typing import List

from config import PROMPTS_DIR_NAME, VISION_PROMPT_FILENAME, VISION_FALLBACK_PROMPT

logger = logging.getLogger(__name__)


class PromptLoader:
    """Dynamically loads system prompts and modules from .md files."""

    def __init__(self, base_dir: str | None = None) -> None:
        """Initialize prompt loader with base directory.
        
        Args:
            base_dir: Base directory path. If None, uses script location.
        """
        if base_dir is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
        
        self.prompts_dir = os.path.join(base_dir, PROMPTS_DIR_NAME)
        self.base_prompt_file = os.path.join(self.prompts_dir, "PROMPT_SYSTEM.md")
        self.vision_prompt_file = os.path.join(self.prompts_dir, VISION_PROMPT_FILENAME)
        
        # Mapping of commands to module files
        self.module_commands = {
            "get_image_generation_help": "image_generation_module.md",
            "get_email_module_help": "email_module.md",
            "get_pc_control_help": "pc_control_module.md",
            "get_file_processing_help": "file_processing_module.md",
            "get_error_handling_help": "error_handling_module.md",
            "get_additional_modules_help": "additional_modules.md",
            "get_search_help": "additional_modules.md",
            "get_media_analysis_help": "additional_modules.md",
            "get_plugins_help": "additional_modules.md",
            "get_memory_help": "additional_modules.md",
            "get_speech_help": "additional_modules.md",
            "get_workflow_help": "additional_modules.md",
            "get_strategy_help": "additional_modules.md",
            "get_ui_automation_help": "ui_automation_module.md",
        }
        
        # Caches for loaded content
        self._module_cache: dict[str, str] = {}
        self._base_prompt_cache: str | None = None
        self._vision_prompt_cache: str | None = None
        
        logger.debug("PromptLoader initialized: prompts_dir=%s", self.prompts_dir)

    def load_base_prompt(self) -> str:
        """Load base system prompt from PROMPT_SYSTEM.md."""
        if self._base_prompt_cache is not None:
            logger.debug("Returning cached base prompt")
            return self._base_prompt_cache
        
        try:
            if not os.path.exists(self.base_prompt_file):
                logger.error("Base prompt file not found: %s", self.base_prompt_file)
                return self._get_fallback_prompt()
            
            with open(self.base_prompt_file, "r", encoding="utf-8") as f:
                content = f.read()
            
            # Extract only the base section
            base_section = self._extract_base_section(content)
            self._base_prompt_cache = base_section
            logger.info("Base prompt loaded from %s (%d chars)", self.base_prompt_file, len(base_section))
            return base_section
        
        except Exception as e:
            logger.error("Error loading base prompt: %s", e, exc_info=True)
            return self._get_fallback_prompt()

    def load_vision_prompt(self) -> str:
        """Load specialized vision prompt."""
        if self._vision_prompt_cache is not None:
            logger.debug("Returning cached vision prompt")
            return self._vision_prompt_cache
        
        try:
            if not os.path.exists(self.vision_prompt_file):
                logger.warning("Vision prompt file not found, using fallback")
                self._vision_prompt_cache = self._get_default_vision_prompt()
                return self._vision_prompt_cache
            
            with open(self.vision_prompt_file, "r", encoding="utf-8") as f:
                content = f.read().strip()
            
            if not content:
                logger.warning("Vision prompt file is empty, using fallback")
                self._vision_prompt_cache = self._get_default_vision_prompt()
            else:
                self._vision_prompt_cache = content
                logger.info("Vision prompt loaded from %s (%d chars)", self.vision_prompt_file, len(content))
            
            return self._vision_prompt_cache
        
        except Exception as e:
            logger.error("Error loading vision prompt: %s", e, exc_info=True)
            self._vision_prompt_cache = self._get_default_vision_prompt()
            return self._vision_prompt_cache

    def _extract_base_section(self, content: str) -> str:
        """Extract base prompt section from full file content."""
        lines = content.split("\n")
        base_lines: list[str] = []
        in_base_section = False
        
        for line in lines:
            # Start of base section (Russian header)
            if "## БАЗОВЫЙ УНИВЕРСАЛЬНЫЙ ПРОМПТ" in line:
                in_base_section = True
                logger.debug("Found base section marker")
                continue
            
            # End of base section (next level-2 header)
            if in_base_section and line.startswith("## ") and "БАЗОВЫЙ" not in line:
                logger.debug("Base section ended at next header")
                break
            
            if in_base_section:
                base_lines.append(line)
        
        if base_lines:
            return "\n".join(base_lines).strip()
        
        # Fallback: return everything before first module
        logger.warning("Base section not found, extracting until first module")
        for i, line in enumerate(lines):
            if line.startswith("## МОДУЛЬ:"):
                return "\n".join(lines[:i]).strip()
        
        return content.strip()

    def load_module(self, command: str) -> str:
        """Load module by command name."""
        if command in self._module_cache:
            logger.debug("Returning cached module for command: %s", command)
            return self._module_cache[command]
        
        if command not in self.module_commands:
            logger.warning("Unknown module command: %s", command)
            return f"Module for command '{command}' not found."
        
        module_file = self.module_commands[command]
        module_path = os.path.join(self.prompts_dir, module_file)
        
        try:
            if not os.path.exists(module_path):
                logger.error("Module file not found: %s", module_path)
                return f"Module file {module_file} not found."
            
            with open(module_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            # Extract specific section for additional_modules.md
            if module_file == "additional_modules.md":
                content = self._extract_specific_module(content, command)
            
            self._module_cache[command] = content
            logger.info("Module loaded: command=%s, file=%s (%d chars)", command, module_file, len(content))
            return content
        
        except Exception as e:
            logger.error("Error loading module %s: %s", module_file, e, exc_info=True)
            return f"Error loading module: {e}"

    def _extract_specific_module(self, content: str, command: str) -> str:
        """Extract specific section from additional_modules.md."""
        # Map commands to section headers (Russian)
        section_map = {
            "get_additional_modules_help": "ВСЕ МОДУЛИ",
            "get_search_help": "МОДУЛЬ: ИНТЕРНЕТ ПОИСК",
            "get_media_analysis_help": "МОДУЛЬ: ВИДЕО И АУДИО АНАЛИЗ",
            "get_plugins_help": "МОДУЛЬ: ПЛАГИНЫ",
            "get_memory_help": "МОДУЛЬ: ВЕКТОРНАЯ ПАМЯТЬ CHROMADB",
            "get_speech_help": "МОДУЛЬ: ОЗВУЧКА",
            "get_workflow_help": "МОДУЛЬ: ЦЕПОЧКИ ДЕЙСТВИЙ",
            "get_strategy_help": "МОДУЛЬ: СТРАТЕГИЧЕСКОЕ МЫШЛЕНИЕ",
        }
        
        target_section = section_map.get(command)
        if not target_section:
            logger.warning("No section mapping for command: %s", command)
            return content
        
        # Return full file for "all modules" command
        if command == "get_additional_modules_help":
            logger.debug("Returning full additional_modules.md")
            return content
        
        lines = content.split("\n")
        section_lines: list[str] = []
        in_target_section = False
        
        for line in lines:
            if target_section in line:
                in_target_section = True
                section_lines.append(line)
                logger.debug("Found target section: %s", target_section)
                continue
            
            if in_target_section:
                # End of section at next module header
                if line.startswith("## МОДУЛЬ:") and target_section not in line:
                    logger.debug("Section ended at next module header")
                    break
                section_lines.append(line)
        
        if section_lines:
            return "\n".join(section_lines).strip()
        
        logger.warning("Target section not found: %s", target_section)
        return content

    def _get_fallback_prompt(self) -> str:
        """Return fallback prompt when files are missing."""
        logger.warning("Using fallback base prompt")
        return """Ты — Нейро, интеллектуальный программный оркестратор.

ВСЕГДА отвечай в формате JSON с одним из доступных действий:
- "powershell" — выполнение команд PowerShell
- "search" — поиск в интернете
- "send_email" — отправка письма
- "get_emails" — получение писем
- "reply_email" — ответ на письмо
- "search_emails" — поиск писем
- "generate_image" — генерация изображения
- "response" — финальный ответ пользователю
- "move_mouse", "left_click", "right_click", "scroll_up", "scroll_down" — управление мышью
- "type_text" — ввод текста
- "take_screenshot" — создание скриншота

КРИТИЧЕСКИ ВАЖНЫЕ ПРАВИЛА:
1. НЕ ВЫДУМЫВАЙ результаты операций
2. ВСЕГДА ЭКРАНИРУЙ обратные слэши в путях (\\\\)
3. ИСПОЛЬЗУЙ UTF-8 для русского текста
4. СТРОЙ ЦЕПОЧКИ действий для сложных задач

Если нужна дополнительная информация о конкретных инструментах, используй команды:
- get_image_generation_help
- get_email_module_help
- get_pc_control_help
- get_file_processing_help
- get_error_handling_help"""

    def _get_default_vision_prompt(self) -> str:
        """Return fallback vision prompt from config."""
        return VISION_FALLBACK_PROMPT

    def is_module_command(self, message: str) -> bool:
        """Check if message is a module loading command."""
        return message.strip() in self.module_commands

    def get_available_commands(self) -> List[str]:
        """Return list of available module commands."""
        return list(self.module_commands.keys())


__all__ = ["PromptLoader"]
