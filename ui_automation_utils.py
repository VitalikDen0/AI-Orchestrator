"""
Модуль утилит UI автоматизации.

Этот модуль предоставляет функциональность для взаимодействия с Windows UI Automation API.
Он позволяет находить окна и извлекать иерархию их элементов управления в виде текста,
позволяя ИИ "видеть" графический интерфейс приложений.
"""

import logging
import os
import sys
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)

try:
    import uiautomation as auto
    UI_AUTOMATION_AVAILABLE = True
except ImportError:
    UI_AUTOMATION_AVAILABLE = False
    auto: Any = None

def check_ui_automation_available() -> bool:
    """Проверяет, доступна ли библиотека uiautomation."""
    return UI_AUTOMATION_AVAILABLE

def find_window(window_name: str = "", class_name: str = "", process_id: int = 0) -> Optional[Any]:
    """
    Находит окно по имени, имени класса или ID процесса.
    
    Args:
        window_name: Заголовок окна (допускается частичное совпадение).
        class_name: Имя класса окна.
        process_id: ID процесса приложения.
        
    Returns:
        Объект WindowControl или None, если окно не найдено.
    """
    if not UI_AUTOMATION_AVAILABLE:
        logger.error("Библиотека uiautomation не установлена.")
        return None

    try:
        window = None
        if process_id > 0:
            window = auto.WindowControl(searchDepth=1, ProcessId=process_id)
        elif window_name:
            # Сначала пробуем точное совпадение
            window = auto.WindowControl(searchDepth=1, Name=window_name)
            if not window.Exists(0, 0):
                # Пробуем частичное совпадение
                window = auto.WindowControl(searchDepth=1, RegexName=f".*{window_name}.*")
        elif class_name:
            window = auto.WindowControl(searchDepth=1, ClassName=class_name)
            
        if window and window.Exists(0, 0):
            return window
        
        # Если не найдено на верхнем уровне, пробуем искать на рабочем столе
        desktop = auto.GetRootControl()
        if window_name:
            window = desktop.WindowControl(searchDepth=2, RegexName=f".*{window_name}.*")
            if window.Exists(0, 0):
                return window
                
        return None
    except Exception as e:
        logger.error(f"Ошибка при поиске окна: {e}")
        return None

def get_ui_tree_as_text(window_name: str = "", max_depth: int = 5) -> str:
    """
    Получает иерархию UI окна в виде отформатированной текстовой строки.
    
    Args:
        window_name: Имя окна для инспекции. Если пусто, инспектирует активное окно.
        max_depth: Максимальная глубина обхода дерева UI.
        
    Returns:
        Строковое представление дерева UI.
    """
    if not UI_AUTOMATION_AVAILABLE:
        return "Ошибка: библиотека uiautomation не установлена. Пожалуйста, установите её с помощью 'pip install uiautomation'."

    try:
        target_window = None
        if window_name:
            target_window = find_window(window_name=window_name)
            if not target_window:
                return f"Ошибка: Окно '{window_name}' не найдено."
        else:
            target_window = auto.GetForegroundControl().GetTopLevelControl()
            if not target_window:
                return "Ошибка: Не удалось определить активное окно."

        # Выводим окно на передний план, чтобы элементы были видимы/доступны
        try:
            if target_window.NativeWindowHandle:
                target_window.SetActive()
        except Exception:
            pass

        output = []
        output.append(f"Дерево UI для окна: '{target_window.Name}' (Класс: {target_window.ClassName})")
        output.append("-" * 50)

        def walk_control(control, depth):
            if depth > max_depth:
                return
            
            indent = "  " * depth
            try:
                name = control.Name
                control_type = control.ControlTypeName
                automation_id = control.AutomationId
                
                # Получаем значение/текст, если доступно
                value = ""
                patterns = control.GetSupportedPatterns()
                
                # Пробуем ValuePattern
                if auto.PatternId.ValuePattern in patterns:
                    try:
                        val_pattern = control.GetValuePattern()
                        if val_pattern:
                            value = val_pattern.Value
                    except:
                        pass
                
                # Пробуем TextPattern (для документов/полей редактирования)
                if not value and auto.PatternId.TextPattern in patterns:
                    try:
                        txt_pattern = control.GetTextPattern()
                        if txt_pattern:
                            value = txt_pattern.DocumentRange.GetText(-1).strip()
                            if len(value) > 50:
                                value = value[:47] + "..."
                    except:
                        pass

                # Пробуем LegacyIAccessiblePattern (часто содержит Value или Description)
                if not value and auto.PatternId.LegacyIAccessiblePattern in patterns:
                    try:
                        legacy = control.GetLegacyIAccessiblePattern()
                        if legacy:
                            value = legacy.Value or legacy.Description or ""
                    except:
                        pass

                # Форматируем строку
                line = f"{indent}[{control_type}]"
                if name:
                    line += f" Name='{name}'"
                if automation_id:
                    line += f" ID='{automation_id}'"
                if value:
                    line += f" Value='{value}'"
                
                # Добавляем координаты (BoundingRectangle)
                rect = control.BoundingRectangle
                if rect:
                    line += f" Rect=({rect.left}, {rect.top}, {rect.right}, {rect.bottom})"

                output.append(line)

                for child in control.GetChildren():
                    walk_control(child, depth + 1)
            except Exception as e:
                output.append(f"{indent}[Ошибка чтения элемента: {e}]")

        walk_control(target_window, 0)
        return "\n".join(output)

    except Exception as e:
        logger.error(f"Ошибка получения дерева UI: {e}")
        return f"Ошибка получения дерева UI: {str(e)}"

def click_element(window_name: str, element_name: str, element_type: str = "") -> str:
    """
    Находит и кликает элемент внутри окна.
    
    Args:
        window_name: Имя родительского окна.
        element_name: Имя элемента для клика.
        element_type: Опциональный тип элемента (например, "Button", "Hyperlink").
        
    Returns:
        Сообщение о результате.
    """
    if not UI_AUTOMATION_AVAILABLE:
        return "Ошибка: библиотека uiautomation не установлена."

    try:
        window = find_window(window_name=window_name)
        if not window:
            return f"Ошибка: Окно '{window_name}' не найдено."

        window.SetActive()
        
        # Поиск элемента
        criteria = {"Name": element_name}
        if element_type:
            criteria["ControlTypeName"] = element_type
            
        # Используем рекурсивный поиск или специфичный поиск
        # uiautomation позволяет искать по критериям
        element = None
        
        # Пробуем прямой поиск сначала
        if element_type:
            # Сопоставление строкового типа с ControlType если нужно, или полагаемся на строковое совпадение в цикле
            pass

        # Вспомогательная функция для рекурсивного поиска
        def find_recursive(control):
            if control.Name == element_name:
                if not element_type or control.ControlTypeName == element_type:
                    return control
            for child in control.GetChildren():
                res = find_recursive(child)
                if res:
                    return res
            return None

        element = find_recursive(window)
        
        if element:
            try:
                element.Click()
                return f"Успешно кликнул элемент '{element_name}'."
            except Exception as e:
                # Пробуем паттерн Invoke
                try:
                    invoke = element.GetInvokePattern()
                    if invoke:
                        invoke.Invoke()
                        return f"Успешно активировал элемент '{element_name}'."
                except:
                    pass
                return f"Нашел элемент '{element_name}', но не смог кликнуть: {e}"
        else:
            return f"Элемент '{element_name}' не найден в окне '{window_name}'."

    except Exception as e:
        return f"Ошибка при клике элемента: {str(e)}"
