# 🎭 Система LoRA для Stable Diffusion

## 📁 Структура папок

```
stable_diff/
├── checkpoints/           # Checkpoint модели Stable Diffusion (.safetensors, .ckpt)
├── lora/
│   ├── sd/               # LoRA для SD 1.5 моделей
│   ├── sdxl/             # LoRA для SDXL моделей
│   └── lora_config.json  # Конфигурационный файл LoRA
```

## ⚙️ Конфигурация в .env

```bash
# Приоритет 1: Путь к модели (если указан, используется вместо автопоиска)
STABLE_DIFFUSION_MODEL_PATH=J:\\ComfyUI\\models\\checkpoints\\model.safetensors

# Приоритет 2: Если не указан STABLE_DIFFUSION_MODEL_PATH, 
# система автоматически найдет модели в stable_diff/checkpoints/

# Дополнительные настройки (опционально):
# LORA_PATH=J:\\ComfyUI\\models\\loras
# CHECKPOINTS_PATH=J:\\ComfyUI\\models\\checkpoints
# LORA_CONFIG_PATH=stable_diff/lora/lora_config.json
```

## 📝 Формат lora_config.json

```json
{
  "loras": {
    "my_character_lora": {
      "filename": "character_style.safetensors",
      "model_type": "sdxl",
      "enabled": true,
      "strength": 0.8,
      "triggers": ["character_name", "style_trigger"],
      "description": "LoRA для стиля персонажа",
      "base_model": "SDXL",
      "resolution": "1024x1024",
      "author": "ArtistName",
      "metadata_analyzed": true
    },
    "anime_style": {
      "filename": "anime_style.safetensors",
      "model_type": "sd", 
      "enabled": true,
      "strength": 1.0,
      "triggers": ["anime", "detailed"],
      "description": "Аниме стиль для SD 1.5",
      "base_model": "SD 1.5",
      "resolution": "512x512",
      "author": "",
      "metadata_analyzed": true
    }
  }
}
```

## 🔧 Параметры LoRA

### Основные параметры:
- **filename**: Имя файла LoRA в соответствующей папке (sd/ или sdxl/)
- **model_type**: Тип модели - "sd" для SD 1.5, "sdxl" для SDXL
- **enabled**: Включена ли LoRA (true/false)
- **strength**: Сила LoRA от 0.0 до 2.0 (обычно 0.5-1.0)
- **triggers**: Массив триггер-слов, автоматически добавляемых к промпту
- **description**: Описание LoRA для удобства

### Метаданные (автоматически извлекаемые):
- **base_model**: Базовая модель из метаданных ("SD 1.5", "SDXL")
- **resolution**: Разрешение обучения ("512x512", "1024x1024")
- **author**: Автор LoRA (если указан в метаданных)
- **metadata_analyzed**: Флаг успешного анализа метаданных

## 🚀 Автоматическое определение моделей

### Приоритет загрузки checkpoint:
1. **STABLE_DIFFUSION_MODEL_PATH** из .env файла
2. **Автопоиск** в папке `stable_diff/checkpoints/`

### Определение типа модели:
- **SDXL**: если в имени файла есть "sdxl", "xl", "illustrious", "pony"
- **SD 1.5**: все остальные модели

### 🔍 Автоматический анализ метаданных LoRA:
Система автоматически анализирует метаданные `.safetensors` файлов и извлекает:
- **Базовую модель** (SD 1.5 vs SDXL) - более точно чем по имени файла
- **Разрешение обучения** - для оптимальных настроек
- **Триггер-слова** - автоматически из тегов обучения
- **Автора и описание** - если доступны в метаданных
- **Предпочитаемую силу** - рекомендуемый weight

Если LoRA находится в неправильной папке (например, SDXL LoRA в папке sd/), система автоматически определит правильный тип по метаданным и выдаст предупреждение.

## ⚡ Горячая перезагрузка конфигурации

Система автоматически проверяет изменения в `lora_config.json` перед каждой генерацией.
Вы можете изменять настройки LoRA "на лету":

1. Отредактируйте `lora_config.json`
2. Сохраните файл
3. Следующая генерация использует новые настройки

## 📋 Автогенерация конфигурации

При первом запуске или обнаружении новых LoRA файлов система автоматически:

1. Сканирует папки `stable_diff/lora/sd/` и `stable_diff/lora/sdxl/`
2. Создает базовую конфигурацию для найденных файлов
3. Устанавливает параметры по умолчанию:
   - enabled: true
   - strength: 1.0
   - triggers: [] (пустой массив)

## 🎯 Триггер-слова

Триггер-слова автоматически добавляются к промпту перед генерацией:

**Исходный промпт:** `"beautiful girl, detailed"`
**С триггерами:** `"beautiful girl, detailed, anime, character_name"`

## 🔍 Пример использования

1. Поместите LoRA файлы в соответствующие папки:
   - `stable_diff/lora/sd/anime_style.safetensors`
   - `stable_diff/lora/sdxl/character_lora.safetensors`

2. Запустите систему - конфигурация создастся автоматически

3. Отредактируйте `lora_config.json` по необходимости

4. Генерируйте изображения - LoRA применятся автоматически!

## ⚠️ Важные замечания

- LoRA файлы должны быть совместимы с типом используемой модели
- Сила LoRA выше 1.5 может привести к артефактам
- Слишком много активных LoRA может замедлить генерацию
- Система поддерживает форматы: .safetensors, .ckpt, .pt

## 🐛 Отладка

Логи системы содержат информацию о:
- Обнаруженных моделях и их типах
- Загруженных LoRA и их параметрах  
- Применённых триггер-словах
- Ошибках загрузки LoRA
- Результатах анализа метаданных

### 🔍 Команды для анализа LoRA:

В программе доступны команды для анализа LoRA:

1. **Автоматический анализ при запуске**: Система автоматически сканирует новые LoRA и анализирует их метаданные

2. **Ручной анализ всех LoRA**: Можно вызвать полный анализ всех файлов через ModelManager:
   ```python
   model_manager = ModelManager()
   results = model_manager.analyze_all_loras()
   ```

3. **Анализ конкретного LoRA**:
   ```python
   model_manager = ModelManager()
   metadata = model_manager.analyze_lora_metadata("path/to/lora.safetensors")
   ```

### 🔧 Диагностика проблем:
- **LoRA в неправильной папке**: Система предупредит если SDXL LoRA находится в папке sd/
- **Отсутствие метаданных**: Для .ckpt/.pt файлов используется анализ по имени
- **Несовместимые LoRA**: Логи покажут несоответствие типов моделей

Проверьте логи при возникновении проблем!