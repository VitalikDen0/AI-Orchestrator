# Документация по созданию плагинов AI Orchestrator

## Введение

Система плагинов AI Orchestrator позволяет расширять функциональность программы без изменения основного кода. Плагины - это Python модули, которые могут добавлять новые действия, обрабатывать сообщения и изменять поведение системы.

## Структура плагина

Каждый плагин должен:
1. Быть Python файлом (.py) в папке `plugins/`
2. Содержать класс, наследующийся от `BasePlugin`
3. Реализовывать обязательные методы

## Базовая структура плагина

```python
from plugins.base_plugin import BasePlugin
from typing import Dict, Any, List

class MyPlugin(BasePlugin):
    def __init__(self):
        super().__init__()
        self.name = "MyPlugin"
        self.version = "1.0.0"
        self.description = "Мой первый плагин"
        self.author = "Ваше имя"
    
    def get_plugin_info(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "author": self.author,
            "actions": self.get_available_actions()
        }
    
    def get_available_actions(self) -> List[str]:
        return ["my_action"]
    
    def execute_action(self, action: str, data: Dict[str, Any], orchestrator) -> Any:
        if action == "my_action":
            return self.handle_my_action(data, orchestrator)
        else:
            raise ValueError(f"Неизвестное действие: {action}")
    
    def handle_my_action(self, data: Dict[str, Any], orchestrator) -> str:
        # Ваша логика здесь
        return "Действие выполнено!"
```

## Обязательные методы

### `get_plugin_info(self) -> Dict[str, Any]`
Возвращает информацию о плагине.

**Возвращает:**
- `name`: Название плагина
- `version`: Версия плагина
- `description`: Описание функциональности
- `author`: Автор плагина
- `actions`: Список доступных действий

### `get_available_actions(self) -> List[str]`
Возвращает список названий действий, которые может выполнять плагин.

### `execute_action(self, action: str, data: Dict[str, Any], orchestrator) -> Any`
Выполняет указанное действие плагина.

**Параметры:**
- `action`: Название действия
- `data`: Данные для действия (из JSON запроса)
- `orchestrator`: Ссылка на основной класс оркестратора

## Необязательные методы (хуки)

### `initialize(self, orchestrator) -> bool`
Вызывается при загрузке плагина. Используйте для инициализации ресурсов.

### `cleanup(self) -> None`
Вызывается при выгрузке плагина. Используйте для очистки ресурсов.

### `on_message_received(self, message: str, orchestrator) -> Optional[str]`
Вызывается при получении сообщения от пользователя (до основной обработки).
Может изменить сообщение или вернуть `None` для продолжения обычной обработки.

### `on_response_generated(self, response: str, orchestrator) -> Optional[str]`
Вызывается после генерации ответа AI.
Может изменить ответ или вернуть `None` для сохранения оригинального ответа.

## Доступ к функциональности оркестратора

Через параметр `orchestrator` вы можете получить доступ к:
- `orchestrator.call_brain_model(text)` - Вызов AI модели
- `orchestrator.analyze_image_with_vision(path)` - Анализ изображений
- `orchestrator.extract_text_from_image(path)` - OCR
- `orchestrator.telegram_bot` - Telegram бот (если включен)
- `orchestrator.logger` - Логгер
- И другим методам основного класса

## Конфигурация плагинов

Плагины настраиваются через файл `plugins/config.ini`:

```ini
[plugins]
my_plugin=true
another_plugin=false
```

- `true` - плагин включен
- `false` - плагин отключен

## Использование плагинов в JSON действиях

Для вызова действия плагина используйте формат:

```json
{
  "action": "plugin:plugin_name:action_name",
  "data": {
    "param1": "value1",
    "param2": "value2"
  }
}
```

Где:
- `plugin_name` - название плагина (имя .py файла)
- `action_name` - название действия из `get_available_actions()`

## Логирование

Каждый плагин имеет собственный логгер:

```python
self.logger.info("Информационное сообщение")
self.logger.warning("Предупреждение")
self.logger.error("Ошибка")
```

## Обработка ошибок

Используйте `PluginError` для специфичных ошибок плагина:

```python
from plugins.base_plugin import PluginError

if not data.get("required_param"):
    raise PluginError("Отсутствует обязательный параметр")
```

## Примеры использования

### Простой плагин-калькулятор

```python
from plugins.base_plugin import BasePlugin
from typing import Dict, Any, List

class CalculatorPlugin(BasePlugin):
    def __init__(self):
        super().__init__()
        self.name = "Calculator"
        self.version = "1.0.0"
        self.description = "Простой калькулятор"
        self.author = "AI Orchestrator"
    
    def get_plugin_info(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "author": self.author,
            "actions": self.get_available_actions()
        }
    
    def get_available_actions(self) -> List[str]:
        return ["calculate"]
    
    def execute_action(self, action: str, data: Dict[str, Any], orchestrator) -> Any:
        if action == "calculate":
            expression = data.get("expression", "")
            try:
                result = eval(expression)  # В реальности используйте безопасный парсер
                return f"Результат: {expression} = {result}"
            except Exception as e:
                return f"Ошибка вычисления: {e}"
        else:
            raise ValueError(f"Неизвестное действие: {action}")
```

### JSON для вызова:

```json
{
  "action": "plugin:calculator:calculate",
  "expression": "2 + 2 * 3"
}
```

## Лучшие практики

1. **Именование**: Используйте понятные имена для плагинов и действий
2. **Документация**: Добавляйте docstring к методам
3. **Обработка ошибок**: Всегда обрабатывайте возможные исключения
4. **Логирование**: Используйте логи для отладки и мониторинга
5. **Ресурсы**: Освобождайте ресурсы в методе `cleanup()`
6. **Производительность**: Избегайте тяжелых операций в `initialize()`

## Отладка плагинов

1. Проверьте логи в консоли при загрузке плагина
2. Используйте `self.logger` для отладочных сообщений
3. Убедитесь что класс плагина наследуется от `BasePlugin`
4. Проверьте синтаксис Python в файле плагина
5. Убедитесь что плагин включен в `config.ini`

## Ограничения

1. Имя файла плагина должно быть валидным Python идентификатором
2. В одном файле может быть только один класс плагина
3. Плагины не могут напрямую изменять конфигурацию других плагинов
4. Избегайте конфликтов имен действий между плагинами
