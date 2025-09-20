# МОДУЛЬ: УПРАВЛЕНИЕ ПК - ДЕТАЛЬНАЯ ИНФОРМАЦИЯ

**Команда для получения:** `get_pc_control_help`

## ДЕЙСТВИЯ УПРАВЛЕНИЯ МЫШЬЮ

### 1. ПЕРЕМЕЩЕНИЕ МЫШИ
```json
{
  "action": "move_mouse",
  "x": 123,
  "y": 456,
  "description": "Переместить мышь на кнопку 'ОК'"
}
```

### 2. КЛИКИ МЫШЬЮ
```json
{
  "action": "left_click",
  "x": 123,
  "y": 456,
  "description": "Кликнуть левой кнопкой по кнопке"
}

{
  "action": "right_click",
  "x": 123,
  "y": 456,
  "description": "Кликнуть правой кнопкой для контекстного меню"
}
```

### 3. ПРОКРУТКА
```json
{
  "action": "scroll_up",
  "pixels": 100,
  "description": "Прокрутить вверх на 100 пикселей"
}

{
  "action": "scroll_down",
  "pixels": 150,
  "description": "Прокрутить вниз на 150 пикселей"
}
```

### 4. РАСШИРЕННЫЕ ДЕЙСТВИЯ МЫШИ
```json
{
  "action": "mouse_down",
  "x": 100,
  "y": 200,
  "description": "Зажать левую кнопку мыши для выделения"
}

{
  "action": "mouse_up",
  "x": 200,
  "y": 250,
  "description": "Отпустить левую кнопку мыши после выделения"
}

{
  "action": "drag_and_drop",
  "x1": 100,
  "y1": 200,
  "x2": 300,
  "y2": 400,
  "description": "Перетащить объект из одной точки в другую"
}
```

## ДЕЙСТВИЯ КЛАВИАТУРЫ

### ВВОД ТЕКСТА
```json
{
  "action": "type_text",
  "text": "Привет, мир!",
  "description": "Ввести текст в активное поле"
}
```

### СПЕЦИАЛЬНЫЕ КЛАВИШИ (через PowerShell)
```json
{
  "action": "powershell",
  "command": "Add-Type -AssemblyName System.Windows.Forms; [System.Windows.Forms.SendKeys]::SendWait('{ENTER}')",
  "description": "Нажать клавишу Enter"
}

{
  "action": "powershell", 
  "command": "Add-Type -AssemblyName System.Windows.Forms; [System.Windows.Forms.SendKeys]::SendWait('^c')",
  "description": "Нажать Ctrl+C для копирования"
}
```

## СОЗДАНИЕ СКРИНШОТОВ

```json
{
  "action": "take_screenshot",
  "description": "Сделать скриншот экрана для анализа"
}
```

## ВЫПОЛНЕНИЕ КОМАНД POWERSHELL

### Базовый формат:
```json
{
  "action": "powershell",
  "command": "команда PowerShell",
  "description": "описание действия"
}
```

### Примеры команд PowerShell:

#### Работа с файлами и папками:
```json
{
  "action": "powershell",
  "command": "New-Item -Path 'C:\\\\Users\\\\vital\\\\Desktop\\\\НоваяПапка' -ItemType Directory -Force",
  "description": "Создать новую папку на рабочем столе"
}

{
  "action": "powershell",
  "command": "Get-ChildItem -Path 'C:\\\\Users\\\\vital\\\\Desktop' -Name",
  "description": "Получить список файлов и папок на рабочем столе"
}

{
  "action": "powershell",
  "command": "Copy-Item -Path 'C:\\\\source\\\\file.txt' -Destination 'C:\\\\target\\\\file.txt'",
  "description": "Скопировать файл"
}
```

#### Работа с текстовыми файлами:
```json
{
  "action": "powershell",
  "command": "'Содержимое файла' | Out-File -FilePath 'C:\\\\Users\\\\vital\\\\Desktop\\\\test.txt' -Encoding UTF8",
  "description": "Создать текстовый файл с содержимым"
}

{
  "action": "powershell",
  "command": "Get-Content -Path 'C:\\\\Users\\\\vital\\\\Desktop\\\\test.txt' -Encoding UTF8",
  "description": "Прочитать содержимое файла"
}
```

#### Системная информация:
```json
{
  "action": "powershell",
  "command": "Get-Process | Sort-Object CPU -Descending | Select-Object -First 10",
  "description": "Получить топ-10 процессов по использованию CPU"
}

{
  "action": "powershell",
  "command": "Get-WmiObject -Class Win32_LogicalDisk | Select-Object DeviceID,Size,FreeSpace",
  "description": "Получить информацию о дисках"
}
```

#### Сетевые операции:
```json
{
  "action": "powershell",
  "command": "Test-NetConnection -ComputerName google.com -Port 80",
  "description": "Проверить сетевое соединение"
}

{
  "action": "powershell",
  "command": "Get-NetAdapter | Where-Object Status -eq 'Up'",
  "description": "Получить активные сетевые адаптеры"
}
```

## СТРАТЕГИЯ РАБОТЫ С УПРАВЛЕНИЕМ ПК

### 1. АНАЛИЗ ТЕКУЩЕГО СОСТОЯНИЯ
```json
{
  "action": "take_screenshot",
  "description": "Сделать скриншот для анализа текущего состояния экрана"
}
```

### 2. ОПРЕДЕЛЕНИЕ КООРДИНАТ
После получения скриншота, vision-модель опишет:
- Расположение кнопок, элементов интерфейса
- Координаты объектов
- Текст на экране
- Состояние приложений

### 3. ВЫПОЛНЕНИЕ ДЕЙСТВИЯ
На основе анализа скриншота:
```json
{
  "action": "left_click",
  "x": 500,
  "y": 300,
  "description": "Кликнуть по кнопке согласно анализу скриншота"
}
```

### 4. ПРОВЕРКА РЕЗУЛЬТАТА
```json
{
  "action": "take_screenshot", 
  "description": "Проверить результат выполненного действия"
}
```

## РАБОТА С КООРДИНАТАМИ

### Система координат:
- Начало координат (0,0) — левый верхний угол экрана
- X увеличивается слева направо
- Y увеличивается сверху вниз

### Типичные разрешения экрана:
- 1920x1080 (Full HD)
- 1366x768 (HD)
- 2560x1440 (2K)
- 3840x2160 (4K)

### Примеры координат для разных разрешений:
- Центр экрана 1920x1080: x=960, y=540
- Центр экрана 1366x768: x=683, y=384

## КОМПЛЕКСНЫЕ СЦЕНАРИИ

### Сценарий 1: Открыть программу и написать текст
```json
// Шаг 1: Скриншот для анализа
{
  "action": "take_screenshot",
  "description": "Анализирую текущее состояние рабочего стола"
}

// Шаг 2: Клик по иконке программы (после анализа координат)
{
  "action": "left_click",
  "x": 100,
  "y": 50,
  "description": "Кликаю по иконке Блокнота"
}

// Шаг 3: Ввод текста
{
  "action": "type_text",
  "text": "Привет, это тестовый текст!",
  "description": "Ввожу тестовый текст"
}
```

### Сценарий 2: Работа с меню
```json
// Шаг 1: Правый клик для вызова контекстного меню
{
  "action": "right_click",
  "x": 500,
  "y": 300,
  "description": "Вызываю контекстное меню"
}

// Шаг 2: Клик по пункту меню
{
  "action": "left_click",
  "x": 520,
  "y": 350,
  "description": "Выбираю пункт меню"
}
```

## СПЕЦИАЛЬНЫЕ КОМАНДЫ КЛАВИАТУРЫ

### Горячие клавиши через PowerShell:
```powershell
# Ctrl+C (копировать)
Add-Type -AssemblyName System.Windows.Forms; [System.Windows.Forms.SendKeys]::SendWait('^c')

# Ctrl+V (вставить)
Add-Type -AssemblyName System.Windows.Forms; [System.Windows.Forms.SendKeys]::SendWait('^v')

# Ctrl+A (выделить все)
Add-Type -AssemblyName System.Windows.Forms; [System.Windows.Forms.SendKeys]::SendWait('^a')

# Alt+Tab (переключение между окнами)
Add-Type -AssemblyName System.Windows.Forms; [System.Windows.Forms.SendKeys]::SendWait('%{TAB}')

# Windows+R (Выполнить)
Add-Type -AssemblyName System.Windows.Forms; [System.Windows.Forms.SendKeys]::SendWait('^{ESC}r')

# Enter
Add-Type -AssemblyName System.Windows.Forms; [System.Windows.Forms.SendKeys]::SendWait('{ENTER}')

# Escape
Add-Type -AssemblyName System.Windows.Forms; [System.Windows.Forms.SendKeys]::SendWait('{ESC}')
```

## ОБРАБОТКА ОШИБОК

### Типичные проблемы:
1. **Неверные координаты** — объект не в указанной позиции
2. **Окно не активно** — нужно сначала активировать окно
3. **Элемент не кликабелен** — элемент заблокирован или скрыт
4. **Задержка отклика** — система медленно реагирует

### Решения:
1. **Всегда делай скриншот** перед действиями
2. **Проверяй результат** после каждого действия
3. **Используй задержки** между быстрыми действиями
4. **Адаптируйся** к изменениям на экране

## ВАЖНЫЕ ПРАВИЛА

1. **Анализируй перед действием** — всегда делай скриншот сначала
2. **Проверяй результат** — делай скриншот после действия
3. **Экранируй пути** — используй двойные обратные слэши (\\\\)
4. **Используй UTF-8** — для русского текста в файлах
5. **Будь точным с координатами** — неточность приведет к ошибкам
6. **Планируй цепочки действий** — разбивай сложные задачи на шаги