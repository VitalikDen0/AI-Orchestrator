"""
Module for launching the Web UI server.
"""
import logging
import os
import subprocess
import sys

logger = logging.getLogger(__name__)

def launch_web_server(host: str = "127.0.0.1", port: str = "8001") -> bool:
    """
    Запускает uvicorn сервер с веб-интерфейсом в фоновом режиме.
    
    Args:
        host: Хост для запуска (по умолчанию 127.0.0.1)
        port: Порт для запуска (по умолчанию 8001)
        
    Returns:
        True если процесс запущен успешно, иначе False
    """
    try:
        # Определяем корневую директорию репозитория
        # Предполагаем, что этот файл находится в корне или рядом с 1.py
        repo_root = os.path.dirname(os.path.abspath(__file__))
        
        cmd = [
            sys.executable, "-m", "uvicorn", "webui.server:app",
            "--host", host, "--port", port, "--app-dir", repo_root
        ]
        
        logger.info(f"🌐 Стартую веб-сервер: {' '.join(cmd)}")
        
        # Запускаем процесс в фоне
        subprocess.Popen(cmd, cwd=repo_root)
        
        logger.info(f"Откройте в браузере: http://{host}:{port}/")
        return True
        
    except Exception as e:
        logger.warning(f"⚠️ Не удалось запустить веб-интерфейс автоматически: {e}")
        return False
