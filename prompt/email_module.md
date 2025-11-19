# МОДУЛЬ: ЭЛЕКТРОННАЯ ПОЧТА - ДЕТАЛЬНАЯ ИНФОРМАЦИЯ

**Команда для получения:** `get_email_module_help`

## ПОДДЕРЖИВАЕМЫЕ ПРОВАЙДЕРЫ
- **gmail** — Gmail (smtp.gmail.com, imap.gmail.com)
- **outlook** — Outlook/Hotmail (smtp-mail.outlook.com, outlook.office365.com)
- **yandex** — Yandex Mail (smtp.yandex.ru, imap.yandex.ru)
- **mail_ru** — Mail.ru (smtp.mail.ru, imap.mail.ru)

## ОСНОВНЫЕ ДЕЙСТВИЯ

### 1. ОТПРАВКА ПИСЬМА
```json
{
  "action": "send_email",
  "provider": "gmail",  // опционально, автовыбор если не указан
  "to_email": "recipient@example.com",
  "subject": "Тема письма",
  "body": "Текст письма с поддержкой русского языка",
  "attachments": ["C:\\\\Users\\\\vital\\\\Desktop\\\\document.pdf"],  // опционально
  "description": "Отправляю письмо получателю"
}
```

### 2. ПОЛУЧЕНИЕ ПИСЕМ
```json
{
  "action": "get_emails",
  "provider": "gmail",  // опционально
  "folder": "INBOX",   // INBOX, SENT, DRAFT, TRASH
  "limit": 10,         // количество писем
  "search_criteria": "UNSEEN",  // критерии поиска
  "description": "Получаю новые письма"
}
```

### 3. ОТВЕТ НА ПИСЬМО
```json
{
  "action": "reply_email",
  "provider": "gmail",
  "original_email_id": "123456789",  // ID письма из get_emails
  "reply_text": "Текст ответа",
  "attachments": [],  // опционально
  "description": "Отвечаю на письмо"
}
```

### 4. ПОИСК ПИСЕМ
```json
{
  "action": "search_emails",
  "provider": "gmail",  // опционально
  "query": "важная встреча",  // поисковый запрос
  "folder": "INBOX",
  "limit": 15,
  "description": "Ищу письма о встрече"
}
```

## ПАПКИ EMAIL
- **INBOX** — входящие письма
- **SENT** — отправленные письма
- **DRAFT** — черновики
- **TRASH** — корзина

## КРИТЕРИИ ПОИСКА EMAIL
- **ALL** — все письма
- **UNSEEN** — непрочитанные письма
- **SEEN** — прочитанные письма
- **FROM "email@example.com"** — от конкретного отправителя
- **TO "email@example.com"** — конкретному получателю
- **SUBJECT "тема"** — по теме письма
- **BODY "текст"** — по содержанию письма
- **SINCE "01-Jan-2024"** — с определённой даты
- **BEFORE "31-Dec-2024"** — до определённой даты

## КОМПЛЕКСНЫЕ СЦЕНАРИИ

### Сценарий 1: Ответ на последнее письмо от начальника
```json
// Шаг 1: Поиск писем от начальника
{
  "action": "search_emails",
  "provider": "gmail",
  "query": "FROM \"boss@company.com\"",
  "folder": "INBOX",
  "limit": 5,
  "description": "Ищу последние письма от начальника"
}

// Шаг 2: После получения результата с ID письма
{
  "action": "reply_email",
  "provider": "gmail",
  "original_email_id": "полученный_ID_из_результата",
  "reply_text": "Добрый день! Принято к исполнению. Отчет будет готов к пятнице.",
  "description": "Отвечаю на письмо начальника"
}
```

### Сценарий 2: Отправка отчета с вложением
```json
{
  "action": "send_email",
  "provider": "outlook",
  "to_email": "manager@company.com",
  "subject": "Еженедельный отчет",
  "body": "Добрый день!\\n\\nВо вложении еженедельный отчет по проекту.\\n\\nС уважением,\\nВиталий",
  "attachments": ["C:\\\\Users\\\\vital\\\\Desktop\\\\report.xlsx"],
  "description": "Отправляю еженедельный отчет с Excel файлом"
}
```

### Сценарий 3: Проверка непрочитанных писем
```json
{
  "action": "get_emails",
  "provider": "yandex",
  "folder": "INBOX",
  "limit": 20,
  "search_criteria": "UNSEEN",
  "description": "Проверяю все непрочитанные письма"
}
```

## НАСТРОЙКА EMAIL ПРОВАЙДЕРОВ

### Требуемые переменные окружения в .env файле:
```
# Gmail
GMAIL_EMAIL=your_email@gmail.com
GMAIL_APP_PASSWORD=app_specific_password

# Outlook
OUTLOOK_EMAIL=your_email@outlook.com
OUTLOOK_APP_PASSWORD=app_specific_password

# Yandex
YANDEX_EMAIL=your_email@yandex.ru
YANDEX_APP_PASSWORD=app_specific_password

# Mail.ru
MAIL_RU_EMAIL=your_email@mail.ru
MAIL_RU_APP_PASSWORD=app_specific_password
```

### ВАЖНО О ПАРОЛЯХ:
1. **НЕ используй основные пароли** — только App Passwords!
2. **App Password для Gmail**: Google Account → Security → 2-Step Verification → App passwords
3. **App Password для Outlook**: Microsoft Account → Security → Advanced security options → App passwords
4. **App Password для Yandex**: Яндекс ID → Безопасность → Пароли приложений
5. **App Password для Mail.ru**: Настройки → Безопасность → Пароли для внешних приложений

## ОБРАБОТКА РЕЗУЛЬТАТОВ

### Результат get_emails и search_emails:
```json
{
  "status": "success",
  "emails": [
    {
      "id": "123456789",
      "from": "sender@example.com",
      "subject": "Важное письмо",
      "date": "2024-12-21 10:30:00",
      "body": "Текст письма...",
      "attachments": ["document.pdf"]
    }
  ]
}
```

### Результат send_email:
```json
{
  "status": "success",
  "message": "Письмо успешно отправлено на recipient@example.com"
}
```

### Результат reply_email:
```json
{
  "status": "success", 
  "message": "Ответ успешно отправлен"
}
```

## ОБРАБОТКА ОШИБОК

### Типичные ошибки:
1. **Провайдер не настроен** — отсутствуют переменные в .env
2. **Неверные учетные данные** — проверь App Password
3. **Письмо не найдено** — неверный ID письма
4. **Файл вложения не найден** — проверь путь к файлу
5. **Превышен лимит размера** — файл слишком большой
6. **Сетевая ошибка** — проблемы с подключением

### Примеры сообщений об ошибках:
```json
{
  "status": "error",
  "error": "Gmail provider not configured. Please check .env file."
}
```

## ПОДДЕРЖКА КИРИЛЛИЦЫ
- Полная поддержка русского языка в темах и тексте писем
- Автоматическое кодирование UTF-8
- Корректное отображение в любых почтовых клиентах

## БЕЗОПАСНОСТЬ
- Все соединения через SSL/TLS
- Использование только App Passwords (не основных паролей)
- Автоматическое создание цепочек ответов (Re: тема)
- Безопасное хранение учетных данных в переменных окружения

## ПРИМЕРЫ КОМАНД ДЛЯ ПОПУЛЯРНЫХ ЗАДАЧ

### Найти все письма от определенного человека:
```json
{
  "action": "search_emails",
  "query": "FROM \"colleague@company.com\"",
  "limit": 50,
  "description": "Ищу все письма от коллеги"
}
```

### Найти письма по теме:
```json
{
  "action": "search_emails", 
  "query": "SUBJECT \"встреча\"",
  "limit": 20,
  "description": "Ищу письма о встречах"
}
```

### Отправить письмо нескольким получателям:
```json
{
  "action": "send_email",
  "to_email": "first@example.com, second@example.com",
  "subject": "Групповое письмо",
  "body": "Письмо для всех участников",
  "description": "Отправляю письмо группе получателей"
}
```

Подробная инструкция по настройке находится в файле `EMAIL_SETUP_GUIDE.md`.