# 📧 Руководство по настройке работы с электронной почтой

## 📋 Обзор

Этот модуль обеспечивает полную интеграцию с Gmail и другими почтовыми сервисами через IMAP/SMTP. Поддерживает:
- ✅ Чтение писем (входящие, отправленные, папки)
- ✅ Отправка писем с вложениями
- ✅ Ответы на письма с сохранением цепочки
- ✅ Поиск по письмам
- ✅ Управление папками
- ✅ Массовые операции
- ✅ HTML и текстовые письма

## 🔧 Настройка Gmail

### 1. Включение двухфакторной аутентификации
1. Перейдите в [Google Account Security](https://myaccount.google.com/security)
2. Включите **2-Step Verification** (двухфакторная аутентификация)

### 2. Создание пароля приложения
1. В разделе "2-Step Verification" найдите **App passwords**
2. Выберите **Mail** и **Windows Computer** (или Other)
3. Нажмите **Generate**
4. **Сохраните сгенерированный 16-символьный пароль!**

### 3. Настройка .env файла
Добавьте в ваш `.env` файл:

```env
# Gmail Configuration
GMAIL_EMAIL=your-email@gmail.com
GMAIL_APP_PASSWORD=abcd efgh ijkl mnop  # 16-символьный пароль приложения
GMAIL_IMAP_SERVER=imap.gmail.com
GMAIL_IMAP_PORT=993
GMAIL_SMTP_SERVER=smtp.gmail.com
GMAIL_SMTP_PORT=587

# Опциональные настройки
EMAIL_CHECK_INTERVAL=300  # Интервал проверки новых писем (секунды)
EMAIL_MAX_MESSAGES=50     # Максимум писем для загрузки за раз
```

## 🔧 Настройка других почтовых сервисов

### Outlook/Hotmail
```env
OUTLOOK_EMAIL=your-email@outlook.com
OUTLOOK_APP_PASSWORD=your-app-password
OUTLOOK_IMAP_SERVER=outlook.office365.com
OUTLOOK_IMAP_PORT=993
OUTLOOK_SMTP_SERVER=smtp-mail.outlook.com
OUTLOOK_SMTP_PORT=587
```

### Yandex
```env
YANDEX_EMAIL=your-email@yandex.ru
YANDEX_APP_PASSWORD=your-app-password
YANDEX_IMAP_SERVER=imap.yandex.ru
YANDEX_IMAP_PORT=993
YANDEX_SMTP_SERVER=smtp.yandex.ru
YANDEX_SMTP_PORT=587
```

### Mail.ru
```env
MAILRU_EMAIL=your-email@mail.ru
MAILRU_PASSWORD=your-password
MAILRU_IMAP_SERVER=imap.mail.ru
MAILRU_IMAP_PORT=993
MAILRU_SMTP_SERVER=smtp.mail.ru
MAILRU_SMTP_PORT=587
```
