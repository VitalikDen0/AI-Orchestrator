# МОДУЛЬ: ГЕНЕРАЦИЯ ИЗОБРАЖЕНИЙ - ДЕТАЛЬНАЯ ИНФОРМАЦИЯ

**Команда для получения:** `get_image_generation_help`

## ОСНОВНОЙ ФОРМАТ
```json
{
  "action": "generate_image",
  "text": "промпт на английском языке с тегами",
  "negative_prompt": "негативный промпт ОБЯЗАТЕЛЬНО",
  "description": "краткое описание что генерируешь"
}
```

## КАТЕГОРИИ И ТЕГИ ДЛЯ ГЕНЕРАЦИИ ИЗОБРАЖЕНИЙ

### [Универсальные] — базовые теги, почти всегда нужны для высокого качества:
- masterpiece [!] — всегда использовать для лучшего качества
- best quality [!] — всегда использовать для лучшего качества
- extremely detailed [!] — всегда использовать для детализации
- high quality [!] — всегда использовать для качества
- 4k / 8k / 16k resolution — высокое разрешение (опционально)
- dynamic pose — динамичная поза (опционально)
- random pose — случайная поза (опционально)
- various pose — разные позы (опционально)
- random composition — случайная композиция (опционально)
- random clothes — случайная одежда (опционально)
- no specific character — без конкретного персонажа (опционально)
- solo — один персонаж (опционально)
- multiple characters / group — группа персонажей (опционально)
- close-up — крупный план (опционально)
- full body — полный рост (опционально)
- upper body — по пояс (опционально)
- cropped to knees / cropped tight / half body — обрезка кадра (опционально)
- view from below / bird's eye view / side view / front view / back view — ракурс (опционально)
- floating / levitating — парящий (опционально)
- random background / abstract background / surreal background — фон (опционально)
- soft lighting / dramatic lighting / natural lighting — освещение (опционально)
- cinematic lighting — кинематографичное освещение (опционально)
- beautifully lit — красиво освещено (опционально)
- natural colors / vibrant colors / muted colors — цвета (опционально)
- atmospheric — атмосферно (опционально)
- detailed background — детализированный фон (опционально)
- intricately detailed — сложная детализация (опционально)
- ornate — украшения (опционально)
- simple background — минималистичный фон (опционально)
- medium breasts / small breasts / large breasts — размер груди (опционально)
- wide hips / slim hips / athletic build / petite — тип фигуры (опционально)
- cute face / beautiful eyes / expressive eyes / smile / neutral expression / serious expression — выражение лица (опционально)

### [NSFW] — для откровенных сцен, использовать только если требуется:
- nude — обнажённая натура
- lewd — пошлость
- explicit — откровенность
- uncensored — без цензуры
- cleavage — декольте
- nipples visible — видны соски
- medium breasts / large breasts / small breasts — размер груди
- wide hips — широкие бёдра
- ass visible — видна попа
- sexy pose — сексуальная поза
- dynamic pose / random pose — динамика
- legs cropped to knees — акцент на ногах
- solo — один персонаж
- 1girl / 1boy / 1person — один персонаж без имени
- multiple girls / multiple boys — группа
- erotic / sensual / seductive pose — эротика
- bed scene / erotic setting / dim lighting — постельная сцена
- soft skin / smooth skin — мягкая кожа
- skin exposed — открытая кожа
- no clothes / minimal clothes / random clothes — одежда
- random background — случайный фон
- random hair color / natural hair color — цвет волос
- messy hair / flowing hair — растрёпанные волосы
- natural lighting / moody lighting / warm lighting — освещение

### [NSFW - negative prompt] — всегда добавлять для фильтрации багов:
- worst quality [!]
- low quality [!]
- blurry [!]
- jpeg artifacts [!]
- watermark [!]
- signature [!]
- disfigured [!]
- malformed limbs [!]
- bad anatomy [!]
- poorly drawn face [!]
- extra limbs [!]
- missing limbs [!]
- out of frame [!]
- mutilated [!]
- mutated hands [!]
- extra fingers [!]
- text [!]
- error [!]
- cropped [!]
- duplicate [!]
- lowres [!]
- bad proportions [!]
- squint [!]
- grainy [!]
- ugly [!]

### [SFW] — для безопасных сцен, без NSFW:
- sfw [!]
- clothed — одет(а)
- random clothes — случайная одежда
- casual clothes / elegant clothes / formal clothes — стиль одежды
- dynamic pose / random pose — динамика
- walking / sitting / standing / running / jumping — поза/движение
- smiling / happy expression / neutral expression — выражение лица
- cute face / beautiful eyes / expressive eyes — лицо
- solo / group — количество персонажей
- wide shot / medium shot / close-up — план
- background: natural / city / forest / abstract / random background — фон
- bright lighting / natural lighting / studio lighting — освещение
- scenic view — пейзаж
- colorful / vibrant colors / pastel colors — цвета
- hair color random / natural hair colors / random hairstyle — волосы
- standing on grass / street / indoors / outdoors — окружение
- hands visible / face visible — видимость частей тела
- wearing hat / scarf / jacket / dress — аксессуары
- full body / half body / cropped — кадрирование

### [SFW - negative prompt] — всегда добавлять для фильтрации артефактов и NSFW:
- nude [!]
- nsfw [!]
- lewd [!]
- explicit [!]
- uncleared skin [!]
- cleavage [!]
- nipples [!]
- bad anatomy [!]
- malformed [!]
- low quality [!]
- jpeg artifacts [!]
- watermark [!]
- signature [!]
- text [!]
- blurry [!]
- distorted [!]
- out of frame [!]
- duplicate [!]
- extra limbs [!]
- missing limbs [!]
- mutated [!]
- squint [!]
- grainy [!]
- ugly [!]

### [Дополнительные теги] — для случайности и вариативности:
- random hair color — случайный цвет волос
- random eye color — случайный цвет глаз
- random skin tone — случайный тон кожи
- random background — случайный фон
- random lighting — случайное освещение
- dynamic lighting — динамичное освещение
- soft shadows — мягкие тени
- motion blur — эффект движения
- motion lines — линии движения
- floating — парящий
- wind blowing hair / wind effect — ветер
- glowing elements / magical atmosphere — магия
- surreal / abstract shapes — сюрреализм
- random accessories — случайные аксессуары
- random pose transitions — смена поз
- random facial expression — выражение лица
- random angle — угол
- random camera position — позиция камеры
- asymmetrical design — асимметрия
- broken pattern — нарушенный паттерн
- glitch effect — глитч-эффект
- pastel colors / neon colors / monochrome — цветовые схемы

## ПРИМЕРЫ ИСПОЛЬЗОВАНИЯ

### Пример NSFW изображения:
```json
{
  "action": "generate_image",
  "text": "masterpiece, best quality, extremely detailed, nude, 1girl, dynamic pose, detailed face, beautiful eyes, soft skin, natural lighting, random background",
  "negative_prompt": "worst quality, low quality, blurry, jpeg artifacts, watermark, signature, disfigured, malformed limbs, bad anatomy, poorly drawn face, extra limbs, missing limbs, text, error",
  "description": "Генерирую художественное изображение обнажённой женской фигуры"
}
```

### Пример SFW изображения:
```json
{
  "action": "generate_image",
  "text": "masterpiece, best quality, extremely detailed, sfw, 1girl, clothed, dynamic pose, smiling, casual clothes, full body, natural background, bright lighting, colorful",
  "negative_prompt": "nude, nsfw, lewd, explicit, cleavage, nipples, bad anatomy, malformed, low quality, jpeg artifacts, watermark, signature, text, blurry",
  "description": "Генерирую портрет девушки в повседневной одежде"
}
```

### Пример абстрактного изображения:
```json
{
  "action": "generate_image",
  "text": "masterpiece, best quality, extremely detailed, abstract art, surreal shapes, vibrant colors, dynamic composition, floating elements, glowing effects, no people",
  "negative_prompt": "worst quality, low quality, blurry, jpeg artifacts, watermark, signature, text, people, faces, realistic",
  "description": "Генерирую абстрактную композицию"
}
```

## ВАЖНЫЕ ПРАВИЛА

1. **Негативный промпт ОБЯЗАТЕЛЕН** — всегда включай его
2. **Промпт только на английском** — никогда не используй русские слова
3. **Теги [!] критически важны** — не пропускай их (только если тег действительно не требуется, можно не добавлять)
4. **Выбирай SFW или NSFW** — не смешивай противоречащие теги
5. **После генерации система завершит диалог** — не пытайся генерировать повторно
6. **Если генерация отключена** — сообщи пользователю об этом

## ТРИГГЕРНЫЕ СЛОВА ДЛЯ ГЕНЕРАЦИИ
Если пользователь использует слова (или подобные этим по смыслу): "сгенерируй", "нарисуй", "создай изображение", "покажи как выглядит", "визуализируй", "изобрази" — используй действие generate_image.

## ОБРАБОТКА ОШИБОК
- Если генерация не удалась — сообщи точную ошибку
- Если генерация отключена — объясни как её включить
- НЕ выдумывай результаты генерации