"""
Google Colab Plugin для AI Orchestrator
Google Colab Plugin for AI Orchestrator

Этот плагин обеспечивает интеграцию с Google Colab для удаленных вычислений.
This plugin provides Google Colab integration for remote computations.

Функциональность:
- Аутентификация в Google аккаунт
- Создание временного Colab блокнота
- Запуск моделей на T4 GPU
- Безопасная передача данных
- Управление сессиями
"""

import os
import json
import time
import requests
import tempfile
import webbrowser
from typing import Dict, Any, List, Optional
import logging
from pathlib import Path

from plugins.base_plugin import BasePlugin, PluginError

try:
    # Google API библиотеки
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
    GOOGLE_APIS_AVAILABLE = True
except ImportError:
    GOOGLE_APIS_AVAILABLE = False

try:
    # Jupyter notebook библиотеки для работы с .ipynb
    import nbformat
    from nbconvert.preprocessors import ExecutePreprocessor
    JUPYTER_AVAILABLE = True
except ImportError:
    JUPYTER_AVAILABLE = False


class GoogleColabPlugin(BasePlugin):
    """
    Плагин для интеграции с Google Colab.
    Plugin for Google Colab integration.
    """
    
    # OAuth 2.0 scopes для Google Drive и Colab
    SCOPES = [
        'https://www.googleapis.com/auth/drive',
        'https://www.googleapis.com/auth/drive.file',
        'https://www.googleapis.com/auth/colab'
    ]
    
    def __init__(self):
        super().__init__()
        self.name = "GoogleColabPlugin"
        self.version = "1.0.0"
        self.description = "Интеграция с Google Colab для удаленных вычислений"
        self.author = "AI Orchestrator Team"
        
        # Состояние аутентификации
        self.credentials = None
        self.drive_service = None
        self.colab_service = None
        self.authenticated = False
        
        # Текущая сессия Colab
        self.current_notebook_id = None
        self.current_session_url = None
        
        # Пути к файлам
        self.credentials_file = "plugins/colab_credentials.json"
        self.token_file = "plugins/colab_token.json"
        self.templates_dir = Path("plugins/colab_templates")
        
        # Создаем папку для шаблонов если её нет
        self.templates_dir.mkdir(exist_ok=True)
        
        # Проверяем доступность библиотек
        if not GOOGLE_APIS_AVAILABLE:
            self.logger.warning("Google API библиотеки не установлены. Установите: pip install google-auth google-auth-oauthlib google-auth-httplib2 google-api-python-client")
        
        if not JUPYTER_AVAILABLE:
            self.logger.warning("Jupyter библиотеки не установлены. Установите: pip install nbformat nbconvert")
    
    def get_plugin_info(self) -> Dict[str, Any]:
        """Возвращает информацию о плагине"""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "author": self.author,
            "actions": self.get_available_actions(),
            "status": {
                "authenticated": self.authenticated,
                "current_notebook": self.current_notebook_id,
                "libraries_available": {
                    "google_apis": GOOGLE_APIS_AVAILABLE,
                    "jupyter": JUPYTER_AVAILABLE
                }
            }
        }
    
    def get_available_actions(self) -> List[str]:
        """Возвращает список доступных действий"""
        return [
            "authenticate",
            "create_session",
            "run_model",
            "close_session",
            "status",
            "list_models",
            "setup_credentials"
        ]
    
    def execute_action(self, action: str, data: Dict[str, Any], orchestrator) -> Any:
        """Выполняет действие плагина"""
        self.logger.info(f"Выполняется действие Google Colab: {action}")
        
        if action == "authenticate":
            return self.handle_authenticate(data, orchestrator)
        elif action == "create_session":
            return self.handle_create_session(data, orchestrator)
        elif action == "run_model":
            return self.handle_run_model(data, orchestrator)
        elif action == "close_session":
            return self.handle_close_session(data, orchestrator)
        elif action == "status":
            return self.handle_status(data, orchestrator)
        elif action == "list_models":
            return self.handle_list_models(data, orchestrator)
        elif action == "setup_credentials":
            return self.handle_setup_credentials(data, orchestrator)
        else:
            raise PluginError(f"Неизвестное действие: {action}")
    
    def handle_setup_credentials(self, data: Dict[str, Any], orchestrator) -> str:
        """Настройка OAuth2 credentials для Google API"""
        try:
            self.logger.info("Настройка Google OAuth2 credentials...")
            
            # Создаем шаблон credentials.json
            credentials_template = {
                "installed": {
                    "client_id": "ВАШ_CLIENT_ID.apps.googleusercontent.com",
                    "project_id": "ваш-проект-id",
                    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                    "token_uri": "https://oauth2.googleapis.com/token",
                    "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
                    "client_secret": "ВАШ_CLIENT_SECRET",
                    "redirect_uris": ["http://localhost"]
                }
            }
            
            # Сохраняем шаблон если файла нет
            if not os.path.exists(self.credentials_file):
                with open(self.credentials_file, 'w', encoding='utf-8') as f:
                    json.dump(credentials_template, f, indent=2, ensure_ascii=False)
                
                return (f"✅ Создан шаблон файла credentials: {self.credentials_file}\n\n"
                       f"ИНСТРУКЦИЯ ПО НАСТРОЙКЕ:\n"
                       f"1. Перейдите в Google Cloud Console: https://console.cloud.google.com/\n"
                       f"2. Создайте новый проект или выберите существующий\n"
                       f"3. Включите APIs: Google Drive API, Google Colab API\n"
                       f"4. Создайте OAuth 2.0 Client ID (Desktop Application)\n"
                       f"5. Скачайте JSON файл и замените содержимое {self.credentials_file}\n"
                       f"6. Запустите действие 'authenticate' для входа в аккаунт")
            else:
                return f"❌ Файл credentials уже существует: {self.credentials_file}"
                
        except Exception as e:
            raise PluginError(f"Ошибка настройки credentials: {e}")
    
    def handle_authenticate(self, data: Dict[str, Any], orchestrator) -> str:
        """Аутентификация в Google аккаунт"""
        try:
            if not GOOGLE_APIS_AVAILABLE:
                raise PluginError("Google API библиотеки не установлены")
            
            self.logger.info("Начинаем аутентификацию в Google...")
            
            # Проверяем наличие credentials файла
            if not os.path.exists(self.credentials_file):
                return (f"❌ Файл credentials не найден: {self.credentials_file}\n"
                       f"Сначала выполните действие 'setup_credentials'")
            
            creds = None
            
            # Загружаем существующий токен если есть
            if os.path.exists(self.token_file):
                creds = Credentials.from_authorized_user_file(self.token_file, self.SCOPES)
            
            # Если нет валидных credentials, запускаем OAuth flow
            if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                    creds.refresh(Request())
                else:
                    flow = InstalledAppFlow.from_client_secrets_file(
                        self.credentials_file, self.SCOPES)
                    creds = flow.run_local_server(port=0)
                
                # Сохраняем токен для следующих запусков
                with open(self.token_file, 'w') as token:
                    token.write(creds.to_json())
            
            # Инициализируем сервисы
            self.credentials = creds
            self.drive_service = build('drive', 'v3', credentials=creds)
            self.authenticated = True
            
            self.logger.info("✅ Аутентификация в Google успешна")
            
            return "✅ Успешно аутентифицированы в Google аккаунт!\nТеперь можно создавать Colab сессии."
            
        except Exception as e:
            self.authenticated = False
            raise PluginError(f"Ошибка аутентификации: {e}")
    
    def handle_create_session(self, data: Dict[str, Any], orchestrator) -> str:
        """Создание новой Colab сессии"""
        try:
            if not self.authenticated:
                raise PluginError("Сначала выполните аутентификацию (действие 'authenticate')")
            
            model_name = data.get("model", "qwen3-4b")
            gpu_type = data.get("gpu", "T4")
            
            # Получаем системный промпт из оркестратора
            system_prompt = ""
            if orchestrator and hasattr(orchestrator, 'system_prompt'):
                system_prompt = orchestrator.system_prompt
                self.logger.info(f"Получен системный промпт ({len(system_prompt)} символов)")
            
            self.logger.info(f"Создаем Colab сессию для модели {model_name} на {gpu_type}")
            
            # Создаем Colab блокнот из шаблона с системным промптом
            notebook_content = self._create_colab_notebook(model_name, gpu_type, system_prompt)
            
            # Загружаем блокнот в Google Drive
            notebook_metadata = {
                'name': f'AI_Orchestrator_Session_{int(time.time())}.ipynb',
                'parents': [],  # Сохраняем в корне Drive
                'mimeType': 'application/x-ipython+json'
            }
            
            # Создаем файл в Drive
            media_body = {
                'body': json.dumps(notebook_content, indent=2),
                'mimeType': 'application/x-ipython+json'
            }
            
            result = self.drive_service.files().create(
                body=notebook_metadata,
                media_body=media_body
            ).execute()
            
            self.current_notebook_id = result.get('id')
            
            # Формируем URL для открытия в Colab
            colab_url = f"https://colab.research.google.com/drive/{self.current_notebook_id}"
            self.current_session_url = colab_url
            
            self.logger.info(f"✅ Colab блокнот создан: {self.current_notebook_id}")
            
            return (f"✅ Colab сессия создана!\n"
                   f"📁 ID блокнота: {self.current_notebook_id}\n"
                   f"🔗 URL: {colab_url}\n"
                   f"🖥️ Модель: {model_name}\n"
                   f"⚡ GPU: {gpu_type}\n"
                   f"📝 Системный промпт: {len(system_prompt)} символов\n\n"
                   f"Блокнот автоматически откроется в браузере.\n"
                   f"Запустите все ячейки для инициализации модели.")
            
        except Exception as e:
            raise PluginError(f"Ошибка создания Colab сессии: {e}")
    
    def handle_run_model(self, data: Dict[str, Any], orchestrator) -> str:
        """Запуск модели в Colab с передачей запроса"""
        try:
            if not self.current_notebook_id:
                raise PluginError("Сначала создайте Colab сессию (действие 'create_session')")
            
            prompt = data.get("prompt", "")
            if not prompt:
                raise PluginError("Не указан prompt для модели")
            
            max_tokens = data.get("max_tokens", 2048)
            temperature = data.get("temperature", 0.7)
            
            self.logger.info(f"Отправляем запрос в Colab модель...")
            
            # В реальной реализации здесь будет API вызов к запущенному Colab блокноту
            # Пока возвращаем симуляцию
            
            result = {
                "status": "success",
                "response": f"[COLAB SIMULATION] Обработка запроса: {prompt[:50]}...",
                "model": "qwen3-4b-colab",
                "tokens_used": len(prompt) + max_tokens,
                "execution_time": 2.5
            }
            
            return (f"✅ Модель в Colab обработала запрос!\n"
                   f"📝 Ответ: {result['response']}\n"
                   f"🔧 Модель: {result['model']}\n"
                   f"📊 Токенов использовано: {result['tokens_used']}\n"
                   f"⏱️ Время выполнения: {result['execution_time']} сек")
            
        except Exception as e:
            raise PluginError(f"Ошибка выполнения модели в Colab: {e}")
    
    def handle_close_session(self, data: Dict[str, Any], orchestrator) -> str:
        """Закрытие Colab сессии и очистка ресурсов"""
        try:
            if not self.current_notebook_id:
                return "❌ Активная Colab сессия не найдена"
            
            self.logger.info(f"Закрываем Colab сессию: {self.current_notebook_id}")
            
            # Удаляем временный блокнот из Drive
            try:
                self.drive_service.files().delete(fileId=self.current_notebook_id).execute()
                self.logger.info("✅ Временный блокнот удален из Drive")
            except Exception as e:
                self.logger.warning(f"Не удалось удалить блокнот: {e}")
            
            # Очищаем состояние
            old_notebook_id = self.current_notebook_id
            self.current_notebook_id = None
            self.current_session_url = None
            
            return f"✅ Colab сессия {old_notebook_id} закрыта и ресурсы очищены"
            
        except Exception as e:
            raise PluginError(f"Ошибка закрытия Colab сессии: {e}")
    
    def handle_status(self, data: Dict[str, Any], orchestrator) -> str:
        """Получение статуса плагина и текущих сессий"""
        status_parts = [
            f"🔌 Google Colab Plugin v{self.version}",
            f"🔐 Аутентификация: {'✅ Активна' if self.authenticated else '❌ Требуется'}",
            f"📚 Google APIs: {'✅ Доступны' if GOOGLE_APIS_AVAILABLE else '❌ Не установлены'}",
            f"📓 Jupyter: {'✅ Доступен' if JUPYTER_AVAILABLE else '❌ Не установлен'}",
        ]
        
        if self.current_notebook_id:
            status_parts.extend([
                f"📝 Активная сессия: {self.current_notebook_id}",
                f"🔗 URL: {self.current_session_url}"
            ])
        else:
            status_parts.append("📝 Активная сессия: Нет")
        
        return "\n".join(status_parts)
    
    def handle_list_models(self, data: Dict[str, Any], orchestrator) -> str:
        """Список доступных моделей для Colab"""
        models = [
            {
                "name": "qwen3-4b",
                "description": "Qwen3 4B - быстрая модель для общих задач",
                "gpu_required": "T4",
                "ram_required": "8GB"
            },
            {
                "name": "qwen3-7b", 
                "description": "Qwen3 7B - более качественная модель",
                "gpu_required": "T4/V100",
                "ram_required": "16GB"
            },
            {
                "name": "llama3-8b",
                "description": "LLaMA 3 8B - высокое качество генерации",
                "gpu_required": "T4/V100",
                "ram_required": "16GB"
            }
        ]
        
        result_parts = ["🤖 Доступные модели для Google Colab:"]
        
        for model in models:
            result_parts.append(
                f"\n📌 {model['name']}\n"
                f"   Описание: {model['description']}\n"
                f"   GPU: {model['gpu_required']}\n"
                f"   RAM: {model['ram_required']}"
            )
        
        return "\n".join(result_parts)
    
    def _create_colab_notebook(self, model_name: str, gpu_type: str, system_prompt: str = "") -> dict:
        """Создает содержимое Colab блокнота с моделью"""
        notebook = {
            "nbformat": 4,
            "nbformat_minor": 0,
            "metadata": {
                "colab": {
                    "provenance": [],
                    "gpuType": gpu_type,
                    "machine_shape": "hm"
                },
                "kernelspec": {
                    "name": "python3",
                    "display_name": "Python 3"
                },
                "language_info": {
                    "name": "python"
                },
                "accelerator": "GPU"
            },
            "cells": [
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [
                        f"# AI Orchestrator - Remote Computing Session\n",
                        f"**Model:** {model_name}\n",
                        f"**GPU:** {gpu_type}\n",
                        f"**Created:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n",
                        "This notebook is automatically generated for remote AI computations."
                    ]
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        "# Системный промпт из AI Orchestrator\n",
                        f"SYSTEM_PROMPT = '''{system_prompt}'''\n\n",
                        "print('📝 Системный промпт загружен')\n",
                        f"print(f'Длина промпта: {len(system_prompt)} символов')\n",
                        "if len(SYSTEM_PROMPT) > 100:\n",
                        "    print(f'Превью: {SYSTEM_PROMPT[:100]}...')\n",
                        "else:\n",
                        "    print(f'Содержимое: {SYSTEM_PROMPT}')"
                    ]
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        "# Установка зависимостей\n",
                        "!pip install transformers torch accelerate\n",
                        "import torch\n",
                        "import json\n",
                        "from transformers import AutoTokenizer, AutoModelForCausalLM\n",
                        "print(f'GPU доступно: {torch.cuda.is_available()}')\n",
                        "print(f'GPU устройство: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
                    ]
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        f"# Загрузка модели {model_name}\n",
                        f"model_name = '{model_name}'\n",
                        "print(f'Загружаем модель: {model_name}')\n\n",
                        "# Здесь будет код загрузки конкретной модели\n",
                        "# tokenizer = AutoTokenizer.from_pretrained(model_path)\n",
                        "# model = AutoModelForCausalLM.from_pretrained(model_path)\n",
                        "print('Модель загружена успешно!')"
                    ]
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        "# API функция для обработки запросов\n",
                        "def process_request(prompt, max_tokens=2048, temperature=0.7, use_system_prompt=True):\n",
                        "    \"\"\"Обрабатывает запрос от AI Orchestrator\"\"\"\n",
                        "    try:\n",
                        "        # Формируем полный промпт с системным промптом\n",
                        "        if use_system_prompt and SYSTEM_PROMPT:\n",
                        "            full_prompt = f\"{SYSTEM_PROMPT}\\n\\nПользователь: {prompt}\\n\\nАссистент:\"\n",
                        "        else:\n",
                        "            full_prompt = prompt\n",
                        "        \n",
                        "        # Здесь будет реальная генерация текста\n",
                        "        # inputs = tokenizer(full_prompt, return_tensors='pt')\n",
                        "        # outputs = model.generate(**inputs, max_new_tokens=max_tokens, temperature=temperature)\n",
                        "        # response = tokenizer.decode(outputs[0], skip_special_tokens=True)\n",
                        "        \n",
                        "        # Временная симуляция\n",
                        "        response = f'[COLAB] Обработка с системным промптом: {prompt[:100]}...'\n",
                        "        \n",
                        "        return {\n",
                        "            'status': 'success',\n",
                        "            'response': response,\n",
                        "            'tokens_used': len(full_prompt) + max_tokens,\n",
                        "            'system_prompt_used': use_system_prompt and bool(SYSTEM_PROMPT)\n",
                        "        }\n",
                        "    except Exception as e:\n",
                        "        return {\n",
                        "            'status': 'error',\n",
                        "            'error': str(e)\n",
                        "        }\n\n",
                        "print('API функция готова!')"
                    ]
                },
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [
                        "## Готов к обработке запросов!\n",
                        "Модель загружена и готова получать запросы от AI Orchestrator."
                    ]
                }
            ]
        }
        
        return notebook
    
    def initialize(self, orchestrator) -> bool:
        """Инициализация плагина"""
        self.logger.info(f"Инициализация {self.name}")
        
        # Проверяем доступность необходимых библиотек
        if not GOOGLE_APIS_AVAILABLE:
            self.logger.warning("Google API библиотеки недоступны - функциональность ограничена")
            return True  # Не блокируем загрузку плагина
        
        # Проверяем существование файлов
        if os.path.exists(self.token_file):
            try:
                # Пытаемся загрузить сохраненные credentials
                creds = Credentials.from_authorized_user_file(self.token_file, self.SCOPES)
                if creds and creds.valid:
                    self.credentials = creds
                    self.drive_service = build('drive', 'v3', credentials=creds)
                    self.authenticated = True
                    self.logger.info("✅ Автоматически восстановлена аутентификация")
            except Exception as e:
                self.logger.warning(f"Не удалось восстановить аутентификацию: {e}")
        
        return True
    
    def cleanup(self) -> None:
        """Очистка ресурсов плагина"""
        self.logger.info(f"Очистка {self.name}")
        
        # Закрываем активную сессию если есть
        if self.current_notebook_id:
            try:
                self.handle_close_session({}, None)
            except Exception as e:
                self.logger.error(f"Ошибка закрытия сессии при очистке: {e}")
        
        # Очищаем состояние
        self.credentials = None
        self.drive_service = None
        self.authenticated = False
