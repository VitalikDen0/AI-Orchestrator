"""Automation utilities for system interaction.

Handles:
- PowerShell command execution
- Mouse and keyboard control (pyautogui)
"""

from __future__ import annotations

import logging
import subprocess
from typing import Any, Dict, Optional, cast

logger = logging.getLogger(__name__)

# Try to import pyautogui for automation
try:
    import pyautogui
    PYAUTOGUI_AVAILABLE = True
    # Set default settings
    pyautogui.FAILSAFE = True
except ImportError:
    PYAUTOGUI_AVAILABLE = False
    pyautogui = None



# ============================================================================
# POWERSHELL UTILITIES
# ============================================================================

def execute_powershell(command: str, logger_instance: Optional[logging.Logger] = None) -> Dict[str, Any]:
    """
    Выполнение PowerShell команды.
    
    Args:
        command: PowerShell команда
        logger_instance: Опциональный логгер
    
    Returns:
        Словарь с результатом выполнения:
        {
            "success": bool,
            "returncode": int,
            "output": str,
            "error": str
        }
    """
    log = logger_instance or logger
    try:
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
            log.info(f"PowerShell: автоисправлен '&&' -> ';' или Push-Location: {command}")
            
        log.info(f"Выполняю PowerShell: {command}")
        
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
        
        log.info(f"PowerShell результат (код: {result.returncode}): {output[:200]}...")
        
        return {
            "success": success,
            "returncode": result.returncode,
            "output": output,
            "error": (result.stderr or "") if not success else ""
        }
        
    except subprocess.TimeoutExpired:
        error_msg = "Команда превысила лимит времени выполнения (60 сек)"
        log.error(error_msg)
        return {"success": False, "returncode": -1, "output": "", "error": error_msg}
    except Exception as e:
        error_msg = f"Ошибка выполнения PowerShell: {str(e)}"
        log.error(error_msg)
        return {"success": False, "returncode": -1, "output": "", "error": error_msg}


# ============================================================================
# MOUSE & KEYBOARD UTILITIES
# ============================================================================

def move_mouse(x: int, y: int) -> Dict[str, Any]:
    """Перемещение мыши в координаты (x, y)"""
    try:
        if not PYAUTOGUI_AVAILABLE:
            return {"success": False, "error": "pyautogui не установлен"}
            
        pyautogui.moveTo(x, y, duration=0.2)  # type: ignore
        return {"success": True, "message": f"Мышь перемещена в ({x}, {y})"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def left_click(x: int, y: int) -> Dict[str, Any]:
    """Клик левой кнопкой мыши по координатам (x, y)"""
    try:
        if not PYAUTOGUI_AVAILABLE:
            return {"success": False, "error": "pyautogui не установлен"}
            
        pyautogui.click(x, y)  # type: ignore
        return {"success": True, "message": f"ЛКМ клик в ({x}, {y})"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def right_click(x: int, y: int) -> Dict[str, Any]:
    """Клик правой кнопкой мыши по координатам (x, y)"""
    try:
        if not PYAUTOGUI_AVAILABLE:
            return {"success": False, "error": "pyautogui не установлен"}
            
        pyautogui.rightClick(x, y)  # type: ignore
        return {"success": True, "message": f"ПКМ клик в ({x}, {y})"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def scroll(pixels: int) -> Dict[str, Any]:
    """Прокрутка колесиком мыши. Положительные значения - вверх, отрицательные - вниз"""
    try:
        if not PYAUTOGUI_AVAILABLE:
            return {"success": False, "error": "pyautogui не установлен"}
            
        pyautogui.scroll(pixels)  # type: ignore
        direction = "вверх" if pixels > 0 else "вниз"
        return {"success": True, "message": f"Прокрутка {direction} на {abs(pixels)} пикселей"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def mouse_down(x: int, y: int) -> Dict[str, Any]:
    """Зажать левую кнопку мыши в координатах (x, y)"""
    try:
        if not PYAUTOGUI_AVAILABLE:
            return {"success": False, "error": "pyautogui не установлен"}
            
        pyautogui.moveTo(x, y)  # type: ignore
        pyautogui.mouseDown(button='left')  # type: ignore
        return {"success": True, "message": f"ЛКМ зажата в ({x}, {y})"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def mouse_up(x: int, y: int) -> Dict[str, Any]:
    """Отпустить левую кнопку мыши в координатах (x, y)"""
    try:
        if not PYAUTOGUI_AVAILABLE:
            return {"success": False, "error": "pyautogui не установлен"}
            
        pyautogui.moveTo(x, y)  # type: ignore
        pyautogui.mouseUp(button='left')  # type: ignore
        return {"success": True, "message": f"ЛКМ отпущена в ({x}, {y})"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def drag_and_drop(x1: int, y1: int, x2: int, y2: int) -> Dict[str, Any]:
    """Перетащить мышью из (x1, y1) в (x2, y2)"""
    try:
        if not PYAUTOGUI_AVAILABLE:
            return {"success": False, "error": "pyautogui не установлен"}
            
        pyautogui.dragTo(x2, y2, duration=0.5, button='left')  # type: ignore
        return {"success": True, "message": f"Перетаскивание из ({x1}, {y1}) в ({x2}, {y2})"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def type_text(text: str) -> Dict[str, Any]:
    """Ввести текст"""
    try:
        if not PYAUTOGUI_AVAILABLE:
            return {"success": False, "error": "pyautogui не установлен"}
            
        pyautogui.typewrite(text, interval=0.05)  # type: ignore
        return {"success": True, "message": f"Введён текст: {text}"}
    except Exception as e:
        return {"success": False, "error": str(e)}


__all__ = [
    "execute_powershell",
    "move_mouse",
    "left_click",
    "right_click",
    "scroll",
    "mouse_down",
    "mouse_up",
    "drag_and_drop",
    "type_text",
]
