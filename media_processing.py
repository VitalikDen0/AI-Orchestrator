"""Media processing utilities for video, audio, and images.

Handles:
- Video frame extraction and YouTube downloads
- Audio transcription with Whisper
- Image conversion and base64 encoding  
- YouTube utilities (cookies, VPN checks)
"""

from __future__ import annotations

import base64
import logging
import math
import os
import shutil
import subprocess
import tempfile
import json
import requests
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

logger = logging.getLogger(__name__)


# ============================================================================
# IMAGE UTILITIES
# ============================================================================

def image_to_base64_balanced(image_path: str, max_size=(500, 500), palette_colors=12) -> str:
    """
    Конвертирует изображение в PNG base64 без ч/б и quantize, только ресайз (если нужно).
    
    Args:
        image_path: Путь к изображению
        max_size: Максимальный размер (ширина, высота)
        palette_colors: Не используется, оставлен для совместимости
        
    Returns:
        Base64 строка PNG изображения
    """
    try:
        logger.info(f"🖼️ Конвертирую изображение в base64: {os.path.basename(image_path)}")
        with Image.open(image_path) as img:
            original_size = img.size
            img = img.convert("RGB")
            img.thumbnail(max_size, Image.Resampling.LANCZOS)
            buf = BytesIO()
            img.save(buf, format="PNG", optimize=True)
            result = base64.b64encode(buf.getvalue()).decode("ascii")
            logger.info(f"✅ Изображение сконвертировано: {original_size} -> {img.size}, {len(result)} символов base64")
            return result
    except Exception as e:
        logger.error(f"❌ Ошибка кодирования (balanced) {image_path}: {e}")
        return ""


# ============================================================================
# VIDEO UTILITIES  
# ============================================================================

def extract_video_frames(video_path: str, fps: int = 1, logger_instance: Optional[logging.Logger] = None) -> List[Tuple[str, str]]:
    """
    Извлекает кадры из видео с заданной частотой.
    
    Args:
        video_path: Путь к видео файлу
        fps: Частота кадров (по умолчанию 1 кадр в секунду)
        logger_instance: Опциональный логгер
        
    Returns:
        Список кортежей (таймкод, base64_изображение)
    """
    log = logger_instance or logger
    frames = []
    temp_dir = tempfile.mkdtemp()
    
    try:
        log.info(f"🎬 Извлекаю кадры из видео: {os.path.basename(video_path)}, fps={fps}")
        
        # Получаем длительность видео через ffprobe
        cmd = [
            'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1', video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        duration = float(result.stdout.strip()) if result.returncode == 0 else 0
        
        if duration == 0:
            log.warning("⚠️ Не удалось определить длительность видео")
            return []
        
        log.info(f"📊 Длительность видео: {duration:.2f} секунд")
        
        # Извлекаем кадры с помощью ffmpeg
        frame_pattern = os.path.join(temp_dir, 'frame_%05d.png')
        cmd = [
            'ffmpeg', '-i', video_path, '-vf', f'fps={fps}', '-q:v', '2', 
            frame_pattern, '-hide_banner', '-loglevel', 'error'
        ]
        subprocess.run(cmd, check=True)
        
        # Собираем кадры и таймкоды
        total_frames = int(math.ceil(duration))
        log.info(f"📸 Обрабатываю ~{total_frames} кадров...")
        
        for i in range(1, total_frames + 1):
            frame_path = os.path.join(temp_dir, f'frame_{i:05d}.png')
            if not os.path.exists(frame_path):
                continue
            
            # Таймкод в формате [HH:MM:SS]
            sec = i - 1
            h = sec // 3600
            m = (sec % 3600) // 60
            s = sec % 60
            timecode = f"[{h:02}:{m:02}:{s:02}]"
            
            # base64 через функцию конвертации
            b64 = image_to_base64_balanced(frame_path)
            if b64:
                frames.append((timecode, b64))
        
        log.info(f"✅ Извлечено {len(frames)} кадров из видео")
        return frames
        
    except Exception as e:
        log.error(f"❌ Ошибка извлечения кадров: {e}")
        return []
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


# ============================================================================
# YOUTUBE UTILITIES
# ============================================================================

def get_youtube_cookies_path() -> Optional[str]:
    """
    Получает путь к файлу cookies для YouTube.
    
    Returns:
        Путь к cookies файлу или None если не найден
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    cookies_path = os.path.join(base_dir, "youtube_cookies.txt")
    
    if os.path.exists(cookies_path):
        logger.info(f"🍪 Найден файл cookies: {cookies_path}")
        return cookies_path
    
    logger.info("ℹ️ Файл cookies не найден")
    return None


def check_cookies_validity(cookies_path: str) -> bool:
    """
    Проверяет валидность cookies файла.
    
    Args:
        cookies_path: Путь к cookies файлу
        
    Returns:
        True если cookies валидны
    """
    try:
        if not os.path.exists(cookies_path):
            logger.warning(f"⚠️ Файл cookies не существует: {cookies_path}")
            return False
        
        # Проверяем размер файла
        file_size = os.path.getsize(cookies_path)
        if file_size < 100:
            logger.warning(f"⚠️ Файл cookies слишком маленький: {file_size} байт")
            return False
        
        # Читаем первые строки для базовой проверки формата
        with open(cookies_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        if len(lines) < 5:
            logger.warning("⚠️ Файл cookies содержит слишком мало строк")
            return False
        
        # Проверяем формат Netscape cookies
        valid_lines = 0
        for line in lines[:10]:
            if line.startswith('#') or line.strip() == '':
                continue
            if '\t' in line:
                valid_lines += 1
        
        if valid_lines == 0:
            logger.warning("⚠️ Файл cookies не содержит валидных записей (отсутствуют табуляции)")
            return False
        
        logger.info("✅ Файл cookies валиден")
        return True
        
    except Exception as e:
        logger.error(f"❌ Ошибка проверки cookies: {e}")
        return False


def suggest_cookies_update():
    """Предлагает пользователю обновить cookies."""
    logger.info("💡 Для улучшения работы с YouTube рекомендуется:")
    logger.info("   1. Запустить: python extract_chrome_cookies.py")
    logger.info("   2. Закрыть Chrome перед извлечением")
    logger.info("   3. Войти в YouTube через VPN")
    logger.info("   4. Cookies обновляются каждые 2-3 месяца")


def download_youtube_video(
    url: str, 
    out_dir: Optional[str] = None,
    logger_instance: Optional[logging.Logger] = None
) -> Optional[str]:
    """
    Скачивает видео с YouTube по ссылке (использует yt-dlp).
    
    Args:
        url: URL YouTube видео
        out_dir: Директория для сохранения (по умолчанию ./Video)
        logger_instance: Опциональный логгер
        
    Returns:
        Путь к скачанному mp4 файлу или None при ошибке
    """
    log = logger_instance or logger
    
    if out_dir is None:
        out_dir = os.path.join(os.path.dirname(__file__), "Video")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "yt_video.%(ext)s")
    
    # Проверяем наличие cookies
    cookies_path = get_youtube_cookies_path()
    
    if cookies_path and check_cookies_validity(cookies_path):
        log.info("🍪 Использую cookies для аутентификации YouTube")
    else:
        log.info("ℹ️ Cookies не найдены или невалидны, использую базовые параметры")
        if not cookies_path:
            suggest_cookies_update()
    
    # Базовые параметры для yt-dlp
    base_cmd = [
        "yt-dlp",
        "--force-ipv4",
        "--user-agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "--extractor-args", "youtube:player_client=android",
        "--no-check-certificate",
        "--prefer-insecure",
        "--geo-bypass",
        "--geo-bypass-country", "US",
        "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4/best[ext=mp4]/best",
        "-o", out_path
    ]
    
    # Добавляем cookies если доступны
    if cookies_path:
        base_cmd.extend(["--cookies", str(cookies_path)])
    
    cmd = base_cmd + [url]
    
    try:
        log.info(f"📥 Скачиваю видео с YouTube: {url}")
        log.debug(f"Команда: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=300)
        
        if result.stdout:
            log.debug(f"yt-dlp stdout: {result.stdout}")
        if result.stderr:
            log.warning(f"yt-dlp stderr: {result.stderr}")
        
        # Найти скачанный файл
        for fname in os.listdir(out_dir):
            if fname.startswith("yt_video") and fname.endswith('.mp4'):
                video_path = os.path.join(out_dir, fname)
                log.info(f"✅ Видео успешно скачано: {fname}")
                return video_path
        
        log.warning("⚠️ Файл не найден после скачивания")
        return None
        
    except subprocess.TimeoutExpired:
        log.error("❌ Таймаут скачивания видео (5 минут)")
        return None
    except subprocess.CalledProcessError as e:
        log.error(f"❌ Ошибка yt-dlp: {e}")
        if e.stderr:
            log.error(f"stderr: {e.stderr}")
        
        # Пробуем альтернативный метод
        log.info("🔄 Пробую альтернативный метод...")
        return _try_alternative_download(url, out_dir, cookies_path, log)
    except Exception as e:
        log.error(f"❌ Неожиданная ошибка: {e}")
        return _try_alternative_download(url, out_dir, cookies_path, log)


def _try_alternative_download(
    url: str, 
    out_dir: str, 
    cookies_path: Optional[str],
    log: logging.Logger
) -> Optional[str]:
    """Пробует альтернативные методы скачивания YouTube видео."""
    out_path = os.path.join(out_dir, "yt_video.%(ext)s")
    
    # Метод 2: Web client
    try:
        alt_cmd = [
            "yt-dlp",
            "--force-ipv4",
            "--user-agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "--extractor-args", "youtube:player_client=web",
            "--no-check-certificate",
            "--geo-bypass",
            "-f", "best[ext=mp4]/best",
            "-o", out_path
        ]
        
        if cookies_path:
            alt_cmd.extend(["--cookies", str(cookies_path)])
        alt_cmd.append(url)
        
        log.debug(f"Альтернативная команда: {' '.join(alt_cmd)}")
        subprocess.run(alt_cmd, check=True, capture_output=True, text=True, timeout=300)
        
        for fname in os.listdir(out_dir):
            if fname.startswith("yt_video") and fname.endswith('.mp4'):
                log.info(f"✅ Видео скачано альтернативным методом: {fname}")
                return os.path.join(out_dir, fname)
                
    except Exception as alt_e:
        log.error(f"❌ Альтернативный метод не сработал: {alt_e}")
        
        # Метод 3: Минимальные параметры
        try:
            simple_cmd = [
                "yt-dlp",
                "--force-ipv4",
                "--no-check-certificate",
                "-f", "best",
                "-o", out_path
            ]
            
            if cookies_path:
                simple_cmd.extend(["--cookies", str(cookies_path)])
            simple_cmd.append(url)
            
            log.info("🔄 Пробую третий метод (минимальные параметры)...")
            subprocess.run(simple_cmd, check=True, capture_output=True, text=True, timeout=300)
            
            for fname in os.listdir(out_dir):
                if fname.startswith("yt_video") and fname.endswith('.mp4'):
                    log.info(f"✅ Видео скачано третьим методом: {fname}")
                    return os.path.join(out_dir, fname)
                    
        except Exception as simple_e:
            log.error(f"❌ Третий метод также не сработал: {simple_e}")
    
    return None


def check_vpn_status(logger_instance: Optional[logging.Logger] = None) -> bool:
    """
    Проверяет, изменился ли IP-адрес (эмуляция проверки работы VPN).
    
    Args:
        logger_instance: Опциональный логгер
        
    Returns:
        True если IP не из РФ (VPN работает)
    """
    log = logger_instance or logger
    try:
        response = requests.get("https://ifconfig.me", timeout=10)
        if response.status_code == 200:
            ip = response.text.strip()
            log.info(f"🌐 Текущий IP адрес: {ip}")

            ru_ips = ["185.", "31.", "46.", "37.", "95.", "178.", "79.", "5.", "176.", "195."]
            if any(ip.startswith(prefix) for prefix in ru_ips):
                log.warning("⚠️ IP адрес похож на российский. VPN может не работать корректно.")
                return False

            log.info("✅ IP адрес не из РФ. VPN работает.")
            return True

        log.warning(f"⚠️ Не удалось проверить IP: {response.status_code}")
        return False

    except Exception as e:
        log.error(f"❌ Ошибка проверки VPN: {e}")
        return False


def get_youtube_info(url: str, logger_instance: Optional[logging.Logger] = None) -> dict:
    """
    Получает информацию о YouTube видео без скачивания.
    
    Args:
        url: URL видео
        logger_instance: Опциональный логгер
        
    Returns:
        Словарь с информацией о видео
    """
    log = logger_instance or logger
    try:
        # Проверяем наличие cookies
        cookies_path = get_youtube_cookies_path()
        use_cookies = False
        
        if cookies_path and check_cookies_validity(cookies_path):
            use_cookies = True
            log.info("🍪 Использую cookies для получения информации о видео")
        
        # Базовые параметры для yt-dlp
        base_cmd = [
            "yt-dlp",
            "--force-ipv4",
            "--user-agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "--extractor-args", "youtube:player_client=android",
            "--no-check-certificate",
            "--geo-bypass",
            "--dump-json"
        ]
        
        # Добавляем cookies если доступны
        if use_cookies:
            base_cmd.extend(["--cookies", str(cookies_path)])
        
        # Добавляем URL в конец
        cmd = base_cmd + [url]
        
        log.info("📋 Получаю информацию о YouTube видео...")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0 and result.stdout:
            try:
                info = json.loads(result.stdout)
                title = info.get('title', 'Неизвестное видео')
                duration = info.get('duration', 0)
                uploader = info.get('uploader', 'Неизвестный автор')
                
                log.info(f"✅ Информация получена: {title} ({duration}с) от {uploader}")
                return {
                    'title': title,
                    'duration': duration,
                    'uploader': uploader,
                    'success': True
                }
            except json.JSONDecodeError:
                log.error("❌ Ошибка парсинга JSON информации о видео")
                return {'success': False, 'error': 'JSON parse error'}
        else:
            log.error(f"❌ Не удалось получить информацию: {result.stderr}")
            
            # Пробуем альтернативный метод без Android клиента
            log.info("🔄 Пробую альтернативный метод получения информации...")
            try:
                alt_cmd = [
                    "yt-dlp",
                    "--force-ipv4",
                    "--user-agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                    "--extractor-args", "youtube:player_client=web",
                    "--no-check-certificate",
                    "--geo-bypass",
                    "--dump-json"
                ]
                
                # Добавляем cookies если доступны
                if use_cookies:
                    alt_cmd.extend(["--cookies", str(cookies_path)])
                
                alt_cmd.append(url)
                
                log.info("🔄 Альтернативная команда для получения информации...")
                alt_result = subprocess.run(alt_cmd, capture_output=True, text=True, timeout=60)
                
                if alt_result.returncode == 0 and alt_result.stdout:
                    try:
                        info = json.loads(alt_result.stdout)
                        title = info.get('title', 'Неизвестное видео')
                        duration = info.get('duration', 0)
                        uploader = info.get('uploader', 'Неизвестный автор')
                        
                        log.info(f"✅ Информация получена альтернативным методом: {title} ({duration}с) от {uploader}")
                        return {
                            'title': title,
                            'duration': duration,
                            'uploader': uploader,
                            'success': True
                        }
                    except json.JSONDecodeError:
                        log.error("❌ Ошибка парсинга JSON альтернативным методом")
                        return {'success': False, 'error': 'JSON parse error (alt method)'}
                else:
                    log.error(f"❌ Альтернативный метод также не сработал: {alt_result.stderr}")
                    return {'success': False, 'error': result.stderr}
                    
            except Exception as alt_e:
                log.error(f"❌ Ошибка альтернативного метода: {alt_e}")
                return {'success': False, 'error': result.stderr}
            
    except Exception as e:
        log.error(f"❌ Ошибка получения информации о видео: {e}")
        return {'success': False, 'error': str(e)}


def check_youtube_accessibility(url: str, logger_instance: Optional[logging.Logger] = None) -> bool:
    """
    Проверяет доступность YouTube ссылки различными методами.
    
    Args:
        url: URL видео
        logger_instance: Опциональный логгер
        
    Returns:
        True если видео доступно
    """
    log = logger_instance or logger
    try:
        # Проверяем наличие cookies
        cookies_path = get_youtube_cookies_path()
        use_cookies = False
        
        if cookies_path and check_cookies_validity(cookies_path):
            use_cookies = True
            log.info("🍪 Использую cookies для проверки доступности")
        
        # Базовые параметры для yt-dlp
        base_cmd = [
            "yt-dlp",
            "--force-ipv4",
            "--user-agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "--extractor-args", "youtube:player_client=android",
            "--no-check-certificate",
            "--geo-bypass",
            "--list-formats"
        ]
        
        # Добавляем cookies если доступны
        if use_cookies:
            base_cmd.extend(["--cookies", str(cookies_path)])
        
        # Добавляем URL в конец
        test_cmd = base_cmd + [url]
        
        log.info("🔍 Проверяю доступность YouTube ссылку...")
        result = subprocess.run(test_cmd, capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            log.info("✅ YouTube ссылка доступна")
            return True
        else:
            log.warning(f"⚠️ YouTube ссылка недоступна: {result.stderr}")
            
            # Пробуем альтернативный метод с web клиентом
            log.info("🔄 Пробую альтернативный метод проверки...")
            try:
                alt_test_cmd = [
                    "yt-dlp",
                    "--force-ipv4",
                    "--user-agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                    "--extractor-args", "youtube:player_client=web",
                    "--no-check-certificate",
                    "--geo-bypass",
                    "--list-formats"
                ]
                
                # Добавляем cookies если доступны
                if use_cookies:
                    alt_test_cmd.extend(["--cookies", str(cookies_path)])
                
                alt_test_cmd.append(url)
                
                alt_result = subprocess.run(alt_test_cmd, capture_output=True, text=True, timeout=60)
                
                if alt_result.returncode == 0:
                    log.info("✅ YouTube ссылка доступна через альтернативный метод")
                    return True
                else:
                    log.warning(f"⚠️ YouTube ссылка недоступна и через альтернативный метод: {alt_result.stderr}")
                    return False
                    
            except Exception as alt_e:
                log.error(f"❌ Ошибка альтернативной проверки: {alt_e}")
                return False
            
    except Exception as e:
        log.error(f"❌ Ошибка проверки доступности YouTube: {e}")
        return False


# ============================================================================
# AUDIO UTILITIES
# ============================================================================

def ensure_wav(audio_path: str, logger_instance: Optional[logging.Logger] = None) -> Optional[str]:
    """
    Конвертирует аудио файл в WAV формат если необходимо.
    
    Args:
        audio_path: Путь к исходному аудио файлу
        logger_instance: Опциональный логгер
        
    Returns:
        Путь к WAV файлу или None при ошибке
    """
    log = logger_instance or logger
    
    if audio_path.lower().endswith('.wav'):
        log.info(f"ℹ️ Файл уже в WAV формате: {os.path.basename(audio_path)}")
        return audio_path
    
    try:
        log.info(f"🔄 Конвертирую аудио в WAV: {os.path.basename(audio_path)}")
        
        base_dir = os.path.dirname(audio_path)
        temp_dir = os.path.join(base_dir, "temp_convert")
        os.makedirs(temp_dir, exist_ok=True)
        
        wav_path = os.path.join(temp_dir, f"converted_{int(__import__('time').time())}.wav")
        
        # Конвертируем через ffmpeg
        cmd = [
            'ffmpeg', '-i', audio_path,
            '-acodec', 'pcm_s16le',
            '-ar', '16000',
            '-ac', '1',
            '-y',
            wav_path
        ]
        
        log.debug(f"Команда ffmpeg: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0 and os.path.exists(wav_path):
            log.info(f"✅ Конвертация успешна: {os.path.basename(wav_path)}")
            return wav_path
        else:
            log.error(f"❌ Ошибка конвертации: {result.stderr}")
            return None
            
    except Exception as e:
        log.error(f"❌ Ошибка конвертации аудио в WAV: {e}")
        return None


def download_youtube_audio(url: str, out_dir: Optional[str] = None) -> str:
    """
    Скачивает аудиодорожку с YouTube по ссылке (использует yt-dlp)
    Возвращает путь к аудиофайлу или пустую строку
    """
    if out_dir is None:
        out_dir = os.path.join(os.path.dirname(__file__), "Audio")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "yt_audio.%(ext)s")
    
    # Проверяем наличие cookies
    cookies_path = get_youtube_cookies_path()
    use_cookies = False
    
    if cookies_path and check_cookies_validity(cookies_path):
        use_cookies = True
        logger.info("🍪 Использую cookies для аутентификации YouTube")
    else:
        logger.info("ℹ️ Cookies не найдены или невалидны, использую базовые параметры")
    
    # Базовые параметры для yt-dlp
    base_cmd = [
        "yt-dlp",
        "--force-ipv4",
        "--user-agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "--extractor-args", "youtube:player_client=android",
        "--no-check-certificate",
        "--prefer-insecure",
        "--geo-bypass",
        "--geo-bypass-country", "US",
        "-f", "bestaudio[ext=m4a]/bestaudio/best",
        "--extract-audio", "--audio-format", "wav",
        "-o", out_path
    ]

    if use_cookies:
        base_cmd.extend(["--cookies", str(cookies_path)])
    
    cmd = base_cmd + [url]
    
    try:
        logger.info(f"Скачиваю аудио с YouTube: {url}")
        cmd_str = " ".join(cmd)
        logger.info(f"Команда: {cmd_str}")
        
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=300)
        
        if result.stdout:
            logger.info(f"yt-dlp stdout: {result.stdout}")
        if result.stderr:
            logger.warning(f"yt-dlp stderr: {result.stderr}")
        
        for fname in os.listdir(out_dir):
            if fname.startswith("yt_audio") and fname.endswith(('.wav', '.m4a', '.mp3', '.ogg', '.flac')):
                logger.info(f"✅ Аудио успешно скачано: {fname}")
                return os.path.join(out_dir, fname)
        
        logger.warning("⚠️ Аудиофайл не найден после скачивания")
        return ""
        
    except subprocess.TimeoutExpired:
        logger.error("❌ Таймаут скачивания аудио (5 минут)")
        return ""
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Ошибка yt-dlp: {e}")
        if e.stderr:
            logger.error(f"stderr: {e.stderr}")
        return ""
    except Exception as e:
        logger.error(f"❌ Неожиданная ошибка скачивания аудио: {e}")
        return ""


def download_whisper_model() -> bool:
    """
    Автоматически скачивает модель whisper-large-v3-q8_0.gguf.
    Возвращает True если успешно, False если ошибка.
    """
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.join(base_dir, "models")
        os.makedirs(models_dir, exist_ok=True)
        
        model_name = "whisper-large-v3-q8_0.gguf"
        model_path = os.path.join(models_dir, model_name)
        
        model_url = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-q8_0.bin"
        
        logger.info(f"📥 Скачиваю модель whisper: {model_name}")
        logger.info(f"🔗 URL: {model_url}")
        
        response = requests.get(model_url, stream=True, timeout=300)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(model_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        if downloaded % (1024*1024*10) == 0: # Log every 10MB
                            logger.info(f"📊 Прогресс: {percent:.1f}% ({downloaded}/{total_size} байт)")
        
        logger.info(f"✅ Модель скачана: {model_path}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Ошибка скачивания модели whisper: {e}")
        return False


def check_whisper_setup() -> bool:
    """
    Проверяет настройку Whisper: наличие whisper-cli.exe и модели.
    Возвращает True если всё готово, False если есть проблемы.
    """
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        exe_path = os.path.join(base_dir, "Release", "whisper-cli.exe")
        model_path = os.path.join(base_dir, "models", "whisper-large-v3-q8_0.gguf")
        
        if not os.path.exists(exe_path):
            logger.error(f"❌ Не найден whisper-cli.exe в папке Release: {exe_path}")
            logger.info("💡 Скачайте whisper.cpp с https://github.com/ggerganov/whisper.cpp")
            return False
        
        if not os.path.exists(model_path):
            logger.warning(f"⚠️ Не найдена модель whisper в папке models: {model_path}")
            logger.info("🔄 Пытаюсь автоматически скачать модель...")
            if download_whisper_model():
                logger.info("✅ Модель whisper успешно загружена")
            else:
                logger.error("❌ Не удалось загрузить модель whisper")
                logger.info("💡 Скачайте модель whisper-large-v3-q8_0.gguf вручную")
                return False
        
        try:
            result = subprocess.run([exe_path, "--help"], capture_output=True, text=True, timeout=10)
            if result.returncode != 0:
                logger.warning("⚠️ whisper-cli.exe не может быть запущен")
                return False
        except Exception as e:
            logger.warning(f"⚠️ Ошибка запуска whisper-cli.exe: {e}")
            return False
        
        logger.info("✅ Whisper настройка проверена успешно")
        return True
        
    except Exception as e:
        logger.error(f"❌ Ошибка проверки настройки Whisper: {e}")
        return False


def convert_audio_to_wav(audio_path: str) -> Optional[str]:
    """
    Конвертирует аудиофайл в WAV формат для Whisper.
    Возвращает путь к WAV файлу или None при ошибке.
    """
    try:
        if not audio_path or not os.path.exists(audio_path):
            return None
        
        if audio_path.lower().endswith('.wav'):
            return audio_path
        
        try:
            subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.warning("⚠️ ffmpeg не найден в системе. Установите ffmpeg для конвертации аудио.")
            return None
        
        temp_dir = os.path.join(os.path.dirname(audio_path), "temp_convert")
        os.makedirs(temp_dir, exist_ok=True)
        
        base_name = os.path.splitext(os.path.basename(audio_path))[0]
        wav_path = os.path.join(temp_dir, f"{base_name}.wav")
        
        cmd = [
            'ffmpeg', '-i', audio_path,
            '-acodec', 'pcm_s16le',
            '-ar', '16000',
            '-ac', '1',
            '-y',
            wav_path
        ]
        
        logger.info(f"🔄 Конвертирую аудио в WAV: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0 and os.path.exists(wav_path):
            logger.info(f"✅ Конвертация успешна: {os.path.basename(wav_path)}")
            return wav_path
        else:
            logger.error(f"❌ Ошибка конвертации: {result.stderr}")
            return None
            
    except Exception as e:
        logger.error(f"❌ Ошибка конвертации аудио в WAV: {e}")
        return None


__all__ = [
    "image_to_base64_balanced",
    "extract_video_frames",
    "get_youtube_cookies_path",
    "check_cookies_validity",
    "suggest_cookies_update",
    "download_youtube_video",
    "ensure_wav",
    "check_vpn_status",
    "get_youtube_info",
    "check_youtube_accessibility",
    "download_youtube_audio",
    "convert_audio_to_wav",
    "download_whisper_model",
    "check_whisper_setup",
    "convert_audio_to_wav",
]
