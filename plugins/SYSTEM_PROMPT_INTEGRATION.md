# 🔗 Интеграция системного промпта между AI Orchestrator и Google Colab

## 📖 Обзор

Система автоматически передает системный промпт из основного AI Orchestrator в удаленные Colab ноутбуки, обеспечивая консистентное поведение ИИ модели независимо от места выполнения.

## 🔄 Как это работает

### 1. Извлечение системного промпта
```python
# В google_colab_plugin.py
def handle_create_session(self, params):
    # Извлекаем системный промпт из параметров оркестратора
    system_prompt = params.get('orchestrator', {}).get('system_prompt', '')
    
    # Передаем промпт при создании ноутбука
    notebook_file = self._create_colab_notebook(
        template_name=template_name,
        system_prompt=system_prompt
    )
```

### 2. Внедрение в шаблон
```python
# В методе _create_colab_notebook
def _create_colab_notebook(self, template_name, system_prompt=""):
    # Добавляем ячейку с системным промптом
    system_prompt_cell = nbformat.v4.new_code_cell(
        source=f"""# 🧠 Системный промпт из AI Orchestrator
SYSTEM_PROMPT = '''{system_prompt}'''

print('📋 Системный промпт загружен:')
if not SYSTEM_PROMPT.strip():
    print('⚠️ Системный промпт пуст')
else:
    print(f'Содержимое: {{SYSTEM_PROMPT}}')"""
    )
```

### 3. Использование в Colab
```python
# В шаблоне ноутбука
def generate_response(prompt, max_tokens=2048, temperature=0.7, use_system_prompt=True):
    if use_system_prompt and SYSTEM_PROMPT:
        full_prompt = f"{SYSTEM_PROMPT}\\n\\nПользователь: {prompt}\\n\\nАссистент:"
    else:
        full_prompt = f"Human: {prompt}\\n\\nAssistant:"
    
    # ... остальная логика генерации
```

## 🛠️ Настройка и использование

### Передача системного промпта при создании сессии
```python
# В AI Orchestrator
plugin_action = {
    'action': 'create_session',
    'params': {
        'template': 'qwen3_4b',
        'orchestrator': {
            'system_prompt': self.system_prompt  # Автоматически передается
        }
    }
}
```

### API endpoint в Colab
```python
@app.route('/generate', methods=['POST'])
def generate_text():
    data = request.get_json()
    
    use_system_prompt = data.get('use_system_prompt', True)  # По умолчанию используем
    
    result = generate_response(
        prompt=data['prompt'],
        use_system_prompt=use_system_prompt
    )
    
    return jsonify(result)
```

## 📊 Структура данных

### Параметры создания сессии
```json
{
    "action": "create_session",
    "params": {
        "template": "qwen3_4b",
        "orchestrator": {
            "system_prompt": "Ты умный ИИ ассистент...",
            "model_name": "qwen",
            "temperature": 0.7
        }
    }
}
```

### Ответ с информацией о системном промпте
```json
{
    "status": "success",
    "response": "Ответ модели...",
    "tokens_used": 150,
    "model": "qwen3-4b-colab",
    "system_prompt_used": true
}
```

## 🔍 Преимущества интеграции

### ✅ Консистентность
- Одинаковое поведение модели локально и в Colab
- Сохранение контекста и стиля общения
- Единые инструкции для ИИ

### ✅ Гибкость
- Возможность отключить системный промпт (use_system_prompt=false)
- Динамическое обновление промпта при необходимости
- Поддержка различных шаблонов

### ✅ Прозрачность
- Явное указание использования системного промпта в ответе
- Логирование передачи промпта
- Возможность проверки содержимого

## 🐛 Отладка

### Проверка передачи промпта
```python
# В Colab ноутбуке
print(f"Системный промпт: {SYSTEM_PROMPT[:100]}...")
print(f"Длина промпта: {len(SYSTEM_PROMPT)} символов")
```

### Тестирование без системного промпта
```python
# Запрос без системного промпта
response = requests.post(f"{colab_url}/generate", json={
    "prompt": "Тест",
    "use_system_prompt": false
})
```

### Проверка в ответе API
```python
result = response.json()
if result.get('system_prompt_used'):
    print("✅ Системный промпт использован")
else:
    print("⚠️ Системный промпт не использован")
```

## 📋 Файлы в системе

### Основные компоненты
- `plugins/google_colab_plugin.py` - логика передачи промпта
- `plugins/colab_templates/qwen3_4b_template.ipynb` - шаблон с интеграцией
- `1.py` - основной оркестратор с системным промптом

### Конфигурация
- `plugins/plugin_config.ini` - настройки плагинов
- `plugins/colab_templates/` - папка с шаблонами ноутбуков

## 🚀 Расширения

### Кастомные промпты для разных моделей
```python
# Можно добавить специализированные промпты
model_specific_prompts = {
    'qwen3_4b': 'Специальный промпт для Qwen3...',
    'llama2': 'Промпт для LLaMA...'
}
```

### Многоуровневые промпты
```python
# Комбинирование системного и пользовательского промптов
final_prompt = f"{SYSTEM_PROMPT}\\n\\nДополнительные инструкции: {user_instructions}\\n\\nПользователь: {prompt}"
```

---

*Документация актуальна для версии плагина Google Colab 1.0*
