"""Communications module for Email and Telegram integration.

Handles:
- Email sending and receiving (SMTP/IMAP)
- Telegram bot integration (python-telegram-bot)
"""

from __future__ import annotations

import asyncio
import base64
import email
import imaplib
import io
import logging
import os
import smtplib
import threading
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any, Dict, List, Optional, cast

# Try to import telegram bot library
try:
    from telegram import Update  # type: ignore
    from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters  # type: ignore
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False
    # Define dummy classes for type hinting if library is missing
    class Update:  # type: ignore
        ALL_TYPES = []
        message = None
        effective_user = None
        effective_chat = None
    
    class Application:  # type: ignore
        @staticmethod
        def builder(): return Application()
        def token(self, t): return self
        def build(self): return self
        def add_handler(self, h): pass
        def run_polling(self, allowed_updates=None): return None
    
    class ContextTypes:  # type: ignore
        DEFAULT_TYPE = Any
    
    class CommandHandler:  # type: ignore
        def __init__(self, cmd, callback): pass
    
    class MessageHandler:  # type: ignore
        def __init__(self, filters, callback): pass
    
    class filters:  # type: ignore
        TEXT = 1
        COMMAND = 2
        PHOTO = 4
        AUDIO = 8
        VOICE = 16
        class Document:
            ALL = 32
        
        @classmethod
        def __and__(cls, other): return cls
        @classmethod
        def __invert__(cls): return cls
        @classmethod
        def __or__(cls, other): return cls

logger = logging.getLogger(__name__)


# ============================================================================
# EMAIL UTILITIES
# ============================================================================

class EmailManager:
    """Manages email operations (SMTP/IMAP)."""
    
    def __init__(self, email_config: Dict[str, Dict[str, Any]]):
        """
        Initialize EmailManager.
        
        Args:
            email_config: Dictionary with configuration for each provider
                          {'provider_name': {'email':..., 'app_password':..., 'smtp_server':..., ...}}
        """
        self.email_config = email_config
        self.available_providers = list(email_config.keys())
        
        if self.available_providers:
            logger.info(f"📧 Почтовая система инициализирована. Доступные провайдеры: {', '.join(self.available_providers)}")
        else:
            logger.warning("⚠️ Почтовые провайдеры не настроены.")

    def send_email(self, provider: str, to_email: str, subject: str, body: str, 
                  attachments: Optional[List[str]] = None, reply_to: Optional[str] = None) -> str:
        """
        Отправляет email через указанного провайдера.
        
        Args:
            provider: провайдер (gmail, outlook, yandex, mail_ru)
            to_email: получатель
            subject: тема письма
            body: текст письма
            attachments: список файлов для прикрепления
            reply_to: ID письма для ответа (опционально)
            
        Returns:
            Строка с результатом операции
        """
        try:
            if provider not in self.available_providers:
                return f"❌ Провайдер {provider} не настроен или недоступен"
            
            config = self.email_config[provider]
            
            # Создаем сообщение
            msg = MIMEMultipart()
            msg['From'] = config['email']
            msg['To'] = to_email
            msg['Subject'] = subject
            
            if reply_to:
                msg['In-Reply-To'] = reply_to
                msg['References'] = reply_to
            
            # Добавляем текст
            msg.attach(MIMEText(body, 'plain', 'utf-8'))
            
            # Добавляем вложения
            if attachments:
                for file_path in attachments:
                    if os.path.exists(file_path):
                        with open(file_path, 'rb') as attachment:
                            part = MIMEApplication(attachment.read(), Name=os.path.basename(file_path))
                            part['Content-Disposition'] = f'attachment; filename="{os.path.basename(file_path)}"'
                            msg.attach(part)
            
            # Отправляем письмо
            with smtplib.SMTP(config['smtp_server'], config['smtp_port']) as server:
                server.starttls()
                server.login(config['email'], config['app_password'])
                server.send_message(msg)
            
            logger.info(f"✅ Письмо отправлено через {provider} на {to_email}")
            return f"✅ Письмо успешно отправлено на {to_email}"
            
        except Exception as e:
            error_msg = f"❌ Ошибка отправки письма через {provider}: {e}"
            logger.error(error_msg)
            return error_msg

    def get_emails(self, provider: str, folder: str = 'INBOX', limit: int = 10, search_criteria: str = 'ALL') -> Any:
        """
        Получает список писем из почтового ящика.
        
        Args:
            provider: провайдер (gmail, outlook, yandex, mail_ru)
            folder: папка (INBOX, SENT, DRAFT и т.д.)
            limit: количество писем для получения
            search_criteria: критерии поиска (ALL, UNSEEN, FROM "email", SUBJECT "text" и т.д.)
            
        Returns:
            Список словарей с информацией о письмах или сообщение об ошибке
        """
        try:
            if provider not in self.available_providers:
                return f"❌ Провайдер {provider} не настроен или недоступен"
            
            config = self.email_config[provider]
            emails_list = []
            
            with imaplib.IMAP4_SSL(config['imap_server'], config['imap_port']) as imap:
                imap.login(config['email'], config['app_password'])
                imap.select(folder)
                
                # Поиск писем
                status, messages = imap.search(None, search_criteria)
                if status != 'OK':
                    return f"❌ Ошибка поиска писем: {status}"
                
                message_ids = messages[0].split()
                
                # Ограничиваем количество и берем самые новые
                message_ids = message_ids[-limit:] if len(message_ids) > limit else message_ids
                message_ids.reverse()  # Самые новые сначала
                
                for msg_id in message_ids:
                    status, msg_data = imap.fetch(msg_id, '(RFC822)')
                    if status == 'OK' and msg_data and msg_data[0] and len(msg_data[0]) > 1:
                        email_body = msg_data[0][1]
                        if isinstance(email_body, bytes):
                            email_message = email.message_from_bytes(email_body)
                        
                            # Извлекаем основную информацию
                            email_info = {
                                'id': msg_id.decode(),
                                'from': email_message.get('From'),
                                'to': email_message.get('To'),
                                'subject': email_message.get('Subject'),
                                'date': email_message.get('Date'),
                                'body': self._extract_email_body(email_message),
                                'attachments': self._get_email_attachments_info(email_message)
                            }
                            emails_list.append(email_info)
            
            logger.info(f"📧 Получено {len(emails_list)} писем из {folder} ({provider})")
            return emails_list
            
        except Exception as e:
            error_msg = f"❌ Ошибка получения писем из {provider}: {e}"
            logger.error(error_msg)
            return error_msg

    def reply_to_email(self, provider: str, original_email_id: str, reply_text: str, attachments: Optional[List[str]] = None) -> str:
        """
        Отвечает на письмо.
        
        Args:
            provider: провайдер
            original_email_id: ID оригинального письма
            reply_text: текст ответа
            attachments: вложения
            
        Returns:
            Строка с результатом операции
        """
        try:
            if provider not in self.available_providers:
                return f"❌ Провайдер {provider} не настроен или недоступен"
            
            config = self.email_config[provider]
            
            # Получаем оригинальное письмо
            with imaplib.IMAP4_SSL(config['imap_server'], config['imap_port']) as imap:
                imap.login(config['email'], config['app_password'])
                imap.select('INBOX')
                
                status, msg_data = imap.fetch(original_email_id, '(RFC822)')
                if status == 'OK' and msg_data and msg_data[0] and len(msg_data[0]) > 1:
                    email_body = msg_data[0][1]
                    if isinstance(email_body, bytes):
                        original_message = email.message_from_bytes(email_body)
                        
                        # Формируем ответ
                        original_from = original_message.get('From')
                        original_subject = original_message.get('Subject', '')
                        reply_subject = f"Re: {original_subject}" if not original_subject.startswith('Re:') else original_subject
                        message_id = original_message.get('Message-ID')
                        
                        # Проверяем, что у нас есть адрес получателя
                        if not original_from:
                            return "❌ Не удалось получить адрес отправителя оригинального письма"
                        
                        # Отправляем ответ
                        return self.send_email(
                            provider=provider,
                            to_email=original_from,
                            subject=reply_subject,
                            body=reply_text,
                            attachments=attachments,
                            reply_to=message_id
                        )
                else:
                    return f"❌ Не удалось получить оригинальное письмо с ID {original_email_id}"
            
            return "❌ Неизвестная ошибка при ответе на письмо"
            
        except Exception as e:
            error_msg = f"❌ Ошибка ответа на письмо: {e}"
            logger.error(error_msg)
            return error_msg

    def search_emails(self, provider: str, query: str, folder: str = 'INBOX', limit: int = 20) -> Any:
        """
        Поиск писем по различным критериям.
        
        Args:
            provider: провайдер
            query: поисковый запрос (может быть текстом для поиска в теме/тексте)
            folder: папка для поиска
            limit: максимальное количество результатов
            
        Returns:
            Список писем или сообщение об ошибке
        """
        try:
            # Формируем IMAP критерии поиска
            search_criteria = f'(OR SUBJECT "{query}" BODY "{query}")'
            
            emails = self.get_emails(provider, folder, limit, search_criteria)
            
            if isinstance(emails, list):
                logger.info(f"🔍 Найдено {len(emails)} писем по запросу '{query}'")
                return emails
            else:
                return emails  # Возвращаем сообщение об ошибке
                
        except Exception as e:
            error_msg = f"❌ Ошибка поиска писем: {e}"
            logger.error(error_msg)
            return error_msg

    def _extract_email_body(self, email_message: Any) -> str:
        """Извлекает текст письма из объекта email."""
        try:
            if email_message.is_multipart():
                for part in email_message.walk():
                    content_type = part.get_content_type()
                    if content_type == "text/plain":
                        body = part.get_payload(decode=True)
                        if body:
                            return body.decode('utf-8', errors='ignore')
            else:
                body = email_message.get_payload(decode=True)
                if body:
                    return body.decode('utf-8', errors='ignore')
            return "Не удалось извлечь текст письма"
        except Exception as e:
            return f"Ошибка извлечения текста: {e}"

    def _get_email_attachments_info(self, email_message: Any) -> List[Dict[str, Any]]:
        """Получает информацию о вложениях письма."""
        attachments = []
        try:
            if email_message.is_multipart():
                for part in email_message.walk():
                    if part.get_content_disposition() == 'attachment':
                        filename = part.get_filename()
                        if filename:
                            attachments.append({
                                'filename': filename,
                                'size': len(part.get_payload(decode=True)) if part.get_payload(decode=True) else 0
                            })
        except Exception as e:
            logger.warning(f"Ошибка получения информации о вложениях: {e}")
        return attachments


# ============================================================================
# TELEGRAM BOT UTILITIES
# ============================================================================

class TelegramBotManager:
    """Manages Telegram Bot lifecycle and handlers."""
    
    def __init__(self, token: str, orchestrator_instance: Any):
        """
        Initialize TelegramBotManager.
        
        Args:
            token: Telegram Bot API token
            orchestrator_instance: Reference to the main orchestrator for callbacks
        """
        self.token = token
        self.orchestrator = orchestrator_instance
        self.app: Optional[Application] = None
        self.bot_thread: Optional[threading.Thread] = None

    def start(self) -> bool:
        """Запускает Telegram бота. Возвращает True при успешном старте."""
        if not TELEGRAM_AVAILABLE:
            logger.error("❌ Библиотека python-telegram-bot не установлена")
            return False
            
        if not self.token:
            logger.warning("❌ Telegram Bot токен не указан")
            return False
        
        try:
            # Создаем приложение
            self.app = Application.builder().token(self.token).build()
            
            # Добавляем обработчики
            self.app.add_handler(CommandHandler("start", self._telegram_start))
            self.app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self._telegram_text_message))
            self.app.add_handler(MessageHandler(filters.PHOTO, self._telegram_photo_message))
            self.app.add_handler(MessageHandler(filters.AUDIO | filters.VOICE, self._telegram_audio_message))
            self.app.add_handler(MessageHandler(filters.Document.ALL, self._telegram_document_message))
            
            # Запускаем бота в фоне
            self.bot_thread = threading.Thread(target=self._run_bot_loop, daemon=True)
            self.bot_thread.start()
            
            logger.info("🤖 Telegram бот запущен в фоновом режиме")
            return True
            
        except Exception as e:
            logger.error(f"❌ Ошибка запуска Telegram бота: {e}")
            return False

    def _run_bot_loop(self):
        """Internal loop for running the bot."""
        loop = None
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            coro = self.app.run_polling(allowed_updates=Update.ALL_TYPES)  # type: ignore
            if coro is not None:
                loop.run_until_complete(cast(Any, coro))
        except Exception as e:
            logger.error(f"❌ Ошибка в Telegram боте: {e}")
        finally:
            if loop is not None:
                try:
                    loop.close()
                except Exception:
                    pass

    async def _safe_reply(self, update: Update, message: str):
        """Безопасная отправка сообщения в Telegram"""
        if update and update.message:
            await update.message.reply_text(message)  # type: ignore

    async def _telegram_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик команды /start"""
        if update is None or update.message is None or update.effective_user is None:
            return
        
        await self._safe_reply(update,
            "🤖 Привет! Я Нейро - AI оркестратор.\n"
            "Я могу:\n"
            "• Обрабатывать текстовые сообщения\n"
            "• Анализировать изображения\n"
            "• Транскрибировать аудио\n"
            "• Генерировать изображения\n"
            "• Выполнять команды PowerShell\n"
            "• Искать информацию в интернете\n\n"
            "Просто отправьте мне сообщение, изображение или аудио!"
        )

    async def _telegram_text_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик текстовых сообщений"""
        if update is None or update.message is None:
            return
        
        text = update.message.text or ""
        
        # Делегируем обработку оркестратору
        # В реальном коде здесь нужно вызывать методы оркестратора
        # Но так как мы переносим код, нам нужно адаптировать логику
        
        # Для простоты пока просто передаем текст в оркестратор, если у него есть метод обработки
        if hasattr(self.orchestrator, 'process_telegram_text'):
            await self.orchestrator.process_telegram_text(update, context, text)
        else:
            # Fallback если метод не реализован (временная заглушка)
            await update.message.reply_text("⚠️ Оркестратор не готов обрабатывать сообщения")  # type: ignore

    async def _telegram_photo_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик фотографий"""
        if update is None or update.message is None:
            return
            
        if hasattr(self.orchestrator, 'process_telegram_photo'):
            await self.orchestrator.process_telegram_photo(update, context)
        else:
            await update.message.reply_text("⚠️ Оркестратор не готов обрабатывать фото")  # type: ignore

    async def _telegram_audio_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик аудио"""
        if update is None or update.message is None:
            return
            
        if hasattr(self.orchestrator, 'process_telegram_audio'):
            await self.orchestrator.process_telegram_audio(update, context)
        else:
            await update.message.reply_text("⚠️ Оркестратор не готов обрабатывать аудио")  # type: ignore

    async def _telegram_document_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик документов"""
        if update is None or update.message is None:
            return
            
        if hasattr(self.orchestrator, 'process_telegram_document'):
            await self.orchestrator.process_telegram_document(update, context)
        else:
            await update.message.reply_text("⚠️ Оркестратор не готов обрабатывать документы")  # type: ignore
