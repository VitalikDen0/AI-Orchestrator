# Система плагинов AI Orchestrator

Эта папка содержит систему плагинов для расширения функциональности AI Orchestrator.

## Структура файлов

### Основные файлы системы
- `base_plugin.py` - Базовый класс для всех плагинов
- `plugin_manager.py` - Менеджер загрузки и управления плагинами
- `config.ini` - Конфигурация включения/отключения плагинов

### Документация
- `PLUGIN_GUIDE_RU.md` - Подробная документация на русском
- `PLUGIN_GUIDE_EN.md` - Подробная документация на английском

### Доступные плагины

#### 🔌 Example Plugin (`example_plugin.py`)
Демонстрационный плагин с примерами действий:
- Приветствие, счетчик, случайные числа
- Эхо сообщений, статус, время

#### 🚀 Google Colab Plugin (`google_colab_plugin.py`)  
Интеграция с Google Colab для удаленных вычислений:
- Аутентификация в Google аккаунт
- Создание временных Colab блокнотов
- Запуск AI моделей на GPU T4/V100
- Автоматическая очистка ресурсов

**Документация Colab плагина:**
- `COLAB_QUICKSTART.md` - Быстрый старт
- `COLAB_GUIDE_RU.md` - Подробное руководство
- `colab_requirements.txt` - Дополнительные зависимости

## Быстрый старт

1. Скопируйте `example_plugin.py` с новым именем
2. Измените класс и методы под ваши потребности
3. Плагин автоматически загрузится при следующем запуске
4. Используйте формат `plugin:имя_файла:действие` в JSON

## Примеры использования

### Example Plugin
```json
{
  "action": "plugin:example_plugin:hello",
  "data": {
    "name": "Пользователь"
  }
}
```

### Google Colab Plugin
```json
{
  "action": "plugin:google_colab_plugin:create_session",
  "data": {
    "model": "qwen3-4b",
    "gpu": "T4"
  }
}
```

## Конфигурация

Отредактируйте `config.ini` для включения/отключения плагинов:

```ini
[plugins]
example_plugin=true
google_colab_plugin=true
my_plugin=false
```

## Установка зависимостей

### Для Google Colab плагина:
```bash
pip install -r plugins/colab_requirements.txt
```

## Документация

Для получения полной документации по созданию плагинов смотрите:
- `PLUGIN_GUIDE_RU.md` - Руководство разработчика (русский)
- `PLUGIN_GUIDE_EN.md` - Developer guide (English)
