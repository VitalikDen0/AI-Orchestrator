# 🚀 Google Colab Plugin - Быстрый старт

Google Colab Plugin позволяет использовать мощности Google Colab для вычислительно сложных задач ИИ.

## ⚡ Быстрая настройка

### 1. Установка зависимостей
```bash
pip install -r plugins/colab_requirements.txt
```

### 2. Настройка Google Cloud
1. Перейдите в [Google Cloud Console](https://console.cloud.google.com/)
2. Создайте проект и включите Google Drive API
3. Создайте OAuth 2.0 credentials (Desktop Application)

### 3. Настройка credentials
```json
{
  "action": "plugin:google_colab_plugin:setup_credentials"
}
```

### 4. Аутентификация
```json
{
  "action": "plugin:google_colab_plugin:authenticate"
}
```

## 🎯 Использование

### Создание сессии
```json
{
  "action": "plugin:google_colab_plugin:create_session",
  "data": {
    "model": "qwen3-4b",
    "gpu": "T4"
  }
}
```

### Запуск модели
```json
{
  "action": "plugin:google_colab_plugin:run_model",
  "data": {
    "prompt": "Объясни квантовую физику",
    "max_tokens": 1000,
    "temperature": 0.7
  }
}
```

### Просмотр статуса
```json
{
  "action": "plugin:google_colab_plugin:status"
}
```

### Закрытие сессии
```json
{
  "action": "plugin:google_colab_plugin:close_session"
}
```

## 📋 Доступные модели

- **qwen3-4b** - Быстрая модель для общих задач (T4 GPU)
- **qwen3-7b** - Более качественная модель (T4/V100 GPU) 
- **llama3-8b** - Высокое качество генерации (T4/V100 GPU)

## 🔧 Функции

- ✅ Автоматическое создание Colab блокнотов
- ✅ Безопасная аутентификация через Google OAuth2
- ✅ Поддержка различных GPU (T4, V100)
- ✅ Автоматическая очистка временных ресурсов
- ✅ Мониторинг состояния и ресурсов
- ✅ API интерфейс для удаленного доступа

## 📖 Полная документация

Подробное руководство: [COLAB_GUIDE_RU.md](COLAB_GUIDE_RU.md)

## ⚠️ Важно

- Требуется Google аккаунт
- Colab имеет ограничения по времени использования GPU
- Рекомендуется Colab Pro для интенсивного использования
- Временные блокноты автоматически удаляются
