# FastAPI backend for local web UI
import os
import json
import base64
import time
import logging
from typing import Optional, Any
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
from fastapi.staticfiles import StaticFiles

# Import orchestrator from 1.py
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ONE_PY = ROOT / "1.py"
FRONTEND_DIST = ROOT / "webui" / "frontend" / "dist"

spec = importlib.util.spec_from_file_location("one_module", str(ONE_PY))
if spec is None or spec.loader is None:
    raise RuntimeError("Failed to load 1.py module spec")
one = importlib.util.module_from_spec(spec)
spec.loader.exec_module(one)  # type: ignore

# Настройка логирования в файл
log_file = ROOT / "ai_orchestrator.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, mode='w', encoding='utf-8'),  # Перезаписываем файл
        logging.StreamHandler()  # И выводим в консоль
    ]
)
logger = logging.getLogger(__name__)

# Global orchestrator instance
orchestrator = None  # type: ignore
event_log = []  # simple in-memory log
history = []    # simple in-memory history (last 200 items)
performance_metrics = []  # Метрики производительности

FILE_CATEGORIES = {
    "output": {
        "title": "Генерированные файлы",
        "path": ROOT / "output"
    },
    "images": {
        "title": "Сгенерированные изображения",
        "path": ROOT / "Images" / "generated"
    },
    "audio": {
        "title": "Аудио",
        "path": ROOT / "Audio" / "generated_speech"
    },
    "video": {
        "title": "Видео",
        "path": ROOT / "generated_videos"
    },
    "export": {
        "title": "Прочие результаты",
        "path": ROOT / "test_output"
    }
}

def log(msg: str, level: str = "INFO"):
    """Логирование с временной меткой в файл и память"""
    from time import strftime
    ts = strftime('%H:%M:%S')
    event_log.append(f"[{ts}] {msg}")
    
    # Логируем в файл
    if level == "ERROR":
        logger.error(msg)
    elif level == "WARNING":
        logger.warning(msg)
    else:
        logger.info(msg)

def add_performance_metric(action: str, response_time: float, context_length: int = 0):
    """Добавляет метрику производительности"""
    metric = {
        "timestamp": time.time(),
        "action": action,
        "response_time": response_time,
        "context_length": context_length
    }
    performance_metrics.append(metric)
    
    # Ограничиваем количество метрик
    if len(performance_metrics) > 100:
        performance_metrics.pop(0)

@asynccontextmanager
async def lifespan(app):
    global orchestrator
    orchestrator = one.AIOrchestrator()
    # do not pop up windows in web mode
    if hasattr(orchestrator, 'show_images_locally'):
        orchestrator.show_images_locally = False
    log("Orchestrator started")
    yield

app = FastAPI(title="Local AI Orchestrator Web API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if (FRONTEND_DIST / "assets").exists():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="frontend-assets")

# Модели данных для FastAPI
class AskPayload(BaseModel):
    message: str

class SettingsPayload(BaseModel):
    brain_model: Optional[str] = None  # Только модель мозга, остальные настройки автоматические

# Функция для добавления истории
def add_history(event_type: str, request: str, response: str):
    global history
    history.append({
        "type": event_type,
        "request": request,
        "response": response,
        "timestamp": time.time()
    })
    if len(history) > 200:
        del history[:len(history)-200]


def build_file_catalog() -> list[dict[str, Any]]:
    categories: list[dict[str, Any]] = []
    for category_id, info in FILE_CATEGORIES.items():
        base_path: Path = info["path"]
        if not base_path.exists():
            continue

        files: list[dict[str, Any]] = []
        try:
            sorted_paths = sorted(
                (p for p in base_path.rglob("*") if p.is_file()),
                key=lambda p: p.stat().st_mtime,
                reverse=True
            )
            for file_path in sorted_paths[:200]:
                stat = file_path.stat()
                files.append({
                    "name": file_path.name,
                    "relative_path": file_path.relative_to(base_path).as_posix(),
                    "size": stat.st_size,
                    "modified": stat.st_mtime
                })
        except Exception as exc:
            log(f"Files catalog error for {category_id}: {exc}", "ERROR")

        categories.append({
            "id": category_id,
            "title": info["title"],
            "files": files
        })

    return categories

@app.get("/")
def index():
    frontend_index = FRONTEND_DIST / "index.html"
    if frontend_index.exists():
        return HTMLResponse(frontend_index.read_text(encoding="utf-8"))

    legacy_index = ROOT / "webui" / "static" / "index.html"
    if not legacy_index.exists():
        raise HTTPException(status_code=404, detail="index.html not found")
    return HTMLResponse(legacy_index.read_text(encoding="utf-8"))


@app.get("/favicon.svg")
def favicon():
    favicon_path = FRONTEND_DIST / "favicon.svg"
    if favicon_path.exists():
        return FileResponse(favicon_path)
    raise HTTPException(status_code=404, detail="favicon not found")

@app.post("/api/ask")
def api_ask(payload: AskPayload):
    if orchestrator is None:
        raise HTTPException(status_code=500, detail="Orchestrator not initialized")
    
    start_time = time.time()
    log(f"ASK: {payload.message[:120]}")
    
    # Вызываем мозг и обрабатываем ответ
    ai_response = orchestrator.call_brain_model(payload.message)
    keep = orchestrator.process_ai_response(ai_response)
    
    # Вычисляем время ответа
    response_time = time.time() - start_time
    context_length = getattr(orchestrator, 'current_context_length', 0)
    
    # Добавляем метрику производительности
    add_performance_metric("brain_response", response_time, context_length)
    
    log(f"ASK processed in {response_time:.2f}s, context: {context_length}")
    add_history("ask", payload.message, getattr(orchestrator, 'last_final_response', ''))
    
    return {
        "continue": keep, 
        "ai_raw": ai_response, 
        "final": orchestrator.last_final_response,
        "performance": {
            "response_time": response_time,
            "context_length": context_length
        }
    }

@app.get("/api/state")
def api_state():
    if orchestrator is None:
        raise HTTPException(status_code=500, detail="Orchestrator not initialized")
    
    # Получаем последние метрики производительности
    recent_metrics = performance_metrics[-10:] if performance_metrics else []
    avg_response_time = 0
    if recent_metrics:
        avg_response_time = sum(m["response_time"] for m in recent_metrics) / len(recent_metrics)
    
    return {
        "brain_model": orchestrator.brain_model,
        "use_image_generation": getattr(orchestrator, "use_image_generation", False),
        "use_vision": getattr(orchestrator, "use_vision", False),
        "use_audio": getattr(orchestrator, "use_audio", False),
        "use_separator": getattr(orchestrator, "use_separator", True),
        "last_final_response": getattr(orchestrator, "last_final_response", ""),
        "has_image": orchestrator.last_generated_image_b64 is not None,
        "log": event_log,
        "history_len": len(history),
        "performance": {
            "recent_metrics": recent_metrics,
            "average_response_time": avg_response_time,
            "context_info": getattr(orchestrator, 'get_context_info', lambda: "N/A")()
        }
    }

@app.get("/api/performance")
def api_performance():
    """API для получения метрик производительности"""
    if not performance_metrics:
        return {"metrics": [], "summary": {}}
    
    # Группируем метрики по действиям
    action_stats = {}
    for metric in performance_metrics:
        action = metric["action"]
        if action not in action_stats:
            action_stats[action] = {"times": [], "count": 0}
        action_stats[action]["times"].append(metric["response_time"])
        action_stats[action]["count"] += 1
    
    # Вычисляем статистику для каждого действия
    summary = {}
    for action, stats in action_stats.items():
        times = stats["times"]
        summary[action] = {
            "count": stats["count"],
            "avg_time": sum(times) / len(times),
            "min_time": min(times),
            "max_time": max(times),
            "total_time": sum(times)
        }
    
    return {
        "metrics": performance_metrics[-50:],  # Последние 50 метрик
        "summary": summary
    }

@app.get("/api/last-image")
def api_last_image():
    if orchestrator is None:
        raise HTTPException(status_code=500, detail="Orchestrator not initialized")
    if not getattr(orchestrator, 'last_generated_image_b64', None):
        raise HTTPException(status_code=404, detail="No image yet")
    # Return as JSON containing data URL and description
    return {
        "data_url": f"data:image/png;base64,{orchestrator.last_generated_image_b64}",
        "description": getattr(orchestrator, 'last_final_response', 'Сгенерированное изображение')
    }

@app.post("/api/settings")
def api_settings(payload: SettingsPayload):
    if orchestrator is None:
        raise HTTPException(status_code=500, detail="Orchestrator not initialized")
    changed = []
    
    # Управляем только brain_model, остальные настройки автоматические
    if payload.brain_model is not None:
        # Поддержка кастомных моделей, включая huihui-gpt-oss-20b-abliterated
        orchestrator.brain_model = payload.brain_model.replace('huihui-ai_','')
        changed.append(f"brain_model={orchestrator.brain_model}")
    
    if changed:
        log("SETTINGS: " + ", ".join(changed))
    return {"ok": True, "state": {
        "brain_model": orchestrator.brain_model,
        "use_image_generation": getattr(orchestrator, "use_image_generation", False),
        "use_vision": getattr(orchestrator, "use_vision", False),
        "use_audio": getattr(orchestrator, "use_audio", False),
        "use_separator": getattr(orchestrator, "use_separator", True),
    }}

# Dev helper: health
@app.get("/api/health")
def api_health():
    return {"ok": True}

# Continuous recording controls
@app.post("/api/continuous-recording/start")
def start_continuous_recording():
    try:
        if orchestrator is None:
            raise HTTPException(status_code=500, detail="Orchestrator not initialized")
        if hasattr(orchestrator, 'start_continuous_recording'):
            result = orchestrator.start_continuous_recording()
            log("Continuous recording started")
            return {"started": result, "message": "Непрерывная запись начата" if result else "Не удалось начать запись"}
        else:
            return {"started": False, "message": "Непрерывная запись не поддерживается"}
    except Exception as e:
        log(f"start_continuous_recording: Exception: {e}", "ERROR")
        raise HTTPException(status_code=500, detail=f"Ошибка запуска непрерывной записи: {e}")

@app.post("/api/continuous-recording/stop")
def stop_continuous_recording():
    try:
        if orchestrator is None:
            raise HTTPException(status_code=500, detail="Orchestrator not initialized")
        if hasattr(orchestrator, 'stop_continuous_recording'):
            result = orchestrator.stop_continuous_recording()
            log("Continuous recording stopped")
            return {"stopped": result, "message": "Непрерывная запись остановлена" if result else "Не удалось остановить запись"}
        else:
            return {"stopped": False, "message": "Непрерывная запись не поддерживается"}
    except Exception as e:
        log(f"stop_continuous_recording: Exception: {e}", "ERROR")
        raise HTTPException(status_code=500, detail=f"Ошибка остановки непрерывной записи: {e}")

@app.get("/api/continuous-recording/status")
def get_continuous_recording_status():
    try:
        if orchestrator is None:
            raise HTTPException(status_code=500, detail="Orchestrator not initialized")
        if hasattr(orchestrator, 'continuous_recording_active'):
            return {"active": getattr(orchestrator, 'continuous_recording_active', False)}
        else:
            return {"active": False, "message": "Непрерывная запись не поддерживается"}
    except Exception as e:
        log(f"get_continuous_recording_status: Exception: {e}", "ERROR")
        raise HTTPException(status_code=500, detail=f"Ошибка получения статуса записи: {e}")

@app.get("/api/history")
def api_history():
    return {"items": history}

@app.post("/api/reset")
def api_reset():
    global history, event_log, performance_metrics
    if orchestrator is None:
        raise HTTPException(status_code=500, detail="Orchestrator not initialized")
    # Clear orchestrator state
    if hasattr(orchestrator, 'conversation_history'):
        orchestrator.conversation_history.clear()
    orchestrator.last_generated_image_b64 = None
    orchestrator.last_final_response = ''
    # Clear logs/history
    history = []
    event_log = []
    performance_metrics = []
    log("RESET: new chat started")
    return {"ok": True}


@app.get("/api/files")
def api_files():
    return {"categories": build_file_catalog()}


@app.get("/api/files/download")
def api_files_download(category: str, file_path: str):
    info = FILE_CATEGORIES.get(category)
    if info is None:
        raise HTTPException(status_code=404, detail="Неизвестная категория")

    base_path: Path = info["path"]
    if not base_path.exists():
        raise HTTPException(status_code=404, detail="Каталог отсутствует")

    try:
        target_path = (base_path / file_path).resolve()
        base_resolved = base_path.resolve()
    except Exception:
        raise HTTPException(status_code=400, detail="Некорректный путь")

    if not str(target_path).startswith(str(base_resolved)):
        raise HTTPException(status_code=400, detail="Недопустимый путь файла")

    if not target_path.exists() or not target_path.is_file():
        raise HTTPException(status_code=404, detail="Файл не найден")

    return FileResponse(target_path, filename=target_path.name)

@app.post("/api/upload/photo")
async def upload_photo(file: UploadFile = File(...), context: str = Form("")):
    try:
        if orchestrator is None:
            log("upload_photo: Orchestrator not initialized")
            raise HTTPException(status_code=500, detail="Orchestrator not initialized")
        
        start_time = time.time()
        photos_dir = os.path.join(ROOT, 'Photos')
        os.makedirs(photos_dir, exist_ok=True)
        fname = file.filename or "upload.png"
        out_path = os.path.join(photos_dir, fname)
        with open(out_path, 'wb') as f:
            f.write(await file.read())
        log(f"Photo uploaded: {file.filename}")
        
        img_b64 = one.image_to_base64_balanced(out_path)
        if not img_b64:
            log(f"upload_photo: image_to_base64_balanced вернул пустую строку для {out_path}")
            raise HTTPException(status_code=400, detail="Ошибка кодирования изображения (base64 пустой)")
        
        desc = ""
        vision_time = 0.0
        if getattr(orchestrator, 'use_vision', False) and img_b64:
            vision_start = time.time()
            desc = orchestrator.call_vision_model(img_b64)
            vision_time = time.time() - vision_start
            add_performance_metric("vision_processing", vision_time)
            log(f"Vision processed photo in {vision_time:.2f}s")
        
        brain_input = ''
        if desc:
            brain_input += f"[Описание изображения]:\n{desc}\n"
        if context:
            brain_input += context
        
        brain_start = time.time()
        ai_response = orchestrator.call_brain_model(brain_input)
        brain_time = time.time() - brain_start
        add_performance_metric("brain_response", brain_time, getattr(orchestrator, 'current_context_length', 0))
        
        cont = orchestrator.process_ai_response(ai_response)
        total_time = time.time() - start_time
        
        log(f"Photo processing completed in {total_time:.2f}s")
        add_history("photo", f"{fname} | ctx: {context[:80]}", getattr(orchestrator, 'last_final_response', ''))
        
        return {
            "continue": cont, 
            "final": orchestrator.last_final_response, 
            "has_image": orchestrator.last_generated_image_b64 is not None,
            "performance": {
                "total_time": total_time,
                "vision_time": vision_time,
                "brain_time": brain_time
            }
        }
    except Exception as e:
        log(f"upload_photo: Exception: {e}", "ERROR")
        raise HTTPException(status_code=500, detail=f"Ошибка загрузки фото: {e}")

@app.post("/api/upload/audio")
async def upload_audio(file: UploadFile = File(...), context: str = Form(""), use_separator: Optional[bool] = Form(None), lang: str = Form("ru"), continuous: bool = Form(False)):
    try:
        if orchestrator is None:
            log("upload_audio: Orchestrator not initialized")
            raise HTTPException(status_code=500, detail="Orchestrator not initialized")
        
        start_time = time.time()
        audio_dir = os.path.join(ROOT, 'Audio')
        os.makedirs(audio_dir, exist_ok=True)
        fname = file.filename or "upload.m4a"
        out_path = os.path.join(audio_dir, fname)
        with open(out_path, 'wb') as f:
            f.write(await file.read())
        
        # Для непрерывной записи используем специальную обработку
        if continuous:
            log(f"Continuous audio chunk uploaded: {file.filename}")
            if hasattr(orchestrator, '_process_audio_chunk'):
                result = await orchestrator._process_audio_chunk(out_path, lang=lang)
                if result:
                    log(f"Continuous audio processed: {result['text'][:50]}...")
                    add_history("continuous", f"chunk | {result['text'][:80]}", result.get('response', ''))
                    return {"continue": result.get('continue', False), "final": result.get('response', ''), "processed": True}
                else:
                    return {"continue": False, "final": "", "processed": False}
            else:
                log("Continuous recording not supported - missing _process_audio_chunk method")
                return {"continue": False, "final": "Непрерывная запись не поддерживается", "processed": False}
        
        # Обычная обработка аудио
        log(f"Audio uploaded: {file.filename}")
        if not getattr(orchestrator, 'use_audio', False):
            brain_start = time.time()
            brain_input = context or ""
            ai_response = orchestrator.call_brain_model(brain_input)
            brain_time = time.time() - brain_start
            add_performance_metric("brain_response", brain_time, getattr(orchestrator, 'current_context_length', 0))
            
            cont = orchestrator.process_ai_response(ai_response)
            total_time = time.time() - start_time
            
            log(f"Audio processing (no ASR) completed in {total_time:.2f}s")
            add_history("audio", f"{fname} | SKIP ASR | ctx: {context[:80]}", getattr(orchestrator, 'last_final_response', ''))
            
            return {
                "continue": cont, 
                "final": orchestrator.last_final_response, 
                "processed": True,
                "performance": {
                    "total_time": total_time,
                    "brain_time": brain_time
                }
            }
        
        # Если use_audio=True, выполняем ASR и отправляем результат в мозг
        sep = getattr(orchestrator, 'use_separator', True) if use_separator is None else use_separator
        
        whisper_start = time.time()
        text = orchestrator.transcribe_audio_whisper(out_path, lang=lang, use_separator=sep)
        whisper_time = time.time() - whisper_start
        add_performance_metric("whisper_transcription", whisper_time)
        
        log("Audio transcribed")
        brain_input = ''
        if context:
            brain_input += context + "\n"
        if text:
            brain_input += f"[Текст из аудио]:\n{text}"
        
        brain_start = time.time()
        ai_response = orchestrator.call_brain_model(brain_input)
        brain_time = time.time() - brain_start
        add_performance_metric("brain_response", brain_time, getattr(orchestrator, 'current_context_length', 0))
        
        cont = orchestrator.process_ai_response(ai_response)
        total_time = time.time() - start_time
        
        log(f"Audio processing completed in {total_time:.2f}s")
        add_history("audio", f"{fname} | lang={lang} | sep={sep} | ctx: {context[:80]}", getattr(orchestrator, 'last_final_response', ''))
        
        return {
            "continue": cont, 
            "final": orchestrator.last_final_response, 
            "processed": True,
            "performance": {
                "total_time": total_time,
                "whisper_time": whisper_time,
                "brain_time": brain_time
            }
        }
    except Exception as e:
        log(f"upload_audio: Exception: {e}", "ERROR")
        raise HTTPException(status_code=500, detail=f"Ошибка загрузки аудио: {e}")

@app.post("/api/upload/video")
async def upload_video(file: UploadFile = File(...), context: str = Form("")):
    try:
        if orchestrator is None:
            log("upload_video: Orchestrator not initialized")
            raise HTTPException(status_code=500, detail="Orchestrator not initialized")
        
        start_time = time.time()
        video_dir = os.path.join(ROOT, 'Video')
        os.makedirs(video_dir, exist_ok=True)
        fname = file.filename or "upload.mp4"
        out_path = os.path.join(video_dir, fname)
        with open(out_path, 'wb') as f:
            f.write(await file.read())
        log(f"Video uploaded: {file.filename}")
        
        frames_desc = ''
        vision_time = 0.0
        if getattr(orchestrator, 'use_vision', False):
            vision_start = time.time()
            frames = orchestrator.extract_video_frames(out_path, fps=1)
            parts = []
            for (tc, b64) in frames:
                if not b64:
                    continue
                frame_start = time.time()
                d = orchestrator.call_vision_model(b64)
                frame_time = time.time() - frame_start
                add_performance_metric("vision_frame", frame_time)
                parts.append(f"{tc}: {d}")
            frames_desc = "\n".join(parts)
            vision_time = time.time() - vision_start
            add_performance_metric("vision_processing", vision_time)
            log(f"Vision processed video frames in {vision_time:.2f}s")
        
        brain_input = ''
        if frames_desc:
            brain_input += f"[Покадровое описание видео]:\n{frames_desc}\n"
        if context:
            brain_input += context
        
        brain_start = time.time()
        ai_response = orchestrator.call_brain_model(brain_input)
        brain_time = time.time() - brain_start
        add_performance_metric("brain_response", brain_time, getattr(orchestrator, 'current_context_length', 0))
        
        cont = orchestrator.process_ai_response(ai_response)
        total_time = time.time() - start_time
        
        log(f"Video processing completed in {total_time:.2f}s")
        add_history("video", f"{fname} | ctx: {context[:80]}", getattr(orchestrator, 'last_final_response', ''))
        
        return {
            "continue": cont, 
            "final": orchestrator.last_final_response,
            "performance": {
                "total_time": total_time,
                "vision_time": vision_time,
                "brain_time": brain_time
            }
        }
    except Exception as e:
        log(f"upload_video: Exception: {e}", "ERROR")
        raise HTTPException(status_code=500, detail=f"Ошибка загрузки видео: {e}")

# Новый API для поиска с увеличенным количеством результатов
@app.post("/api/search")
def api_search(query: str):
    """Выполняет поиск в Google с 10 результатами и полной обработкой"""
    if orchestrator is None:
        raise HTTPException(status_code=500, detail="Orchestrator not initialized")
    
    if not hasattr(orchestrator, 'google_search'):
        raise HTTPException(status_code=500, detail="Google search not available")
    
    start_time = time.time()
    log(f"Search request: {query}")
    
    try:
        # Выполняем поиск с 10 результатами
        search_results = orchestrator.google_search(query, num_results=10)
        
        # Полная обработка результатов
        processed_results = []
        for i, result in enumerate(search_results):
            if "error" in result:
                processed_results.append({
                    "index": i + 1,
                    "error": result["error"]
                })
            else:
                # Получаем полное содержимое страницы
                try:
                    import requests
                    page_response = requests.get(result["url"], timeout=10, headers={
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                    })
                    if page_response.status_code == 200:
                        # Извлекаем основной текст (убираем HTML теги)
                        import re
                        text = re.sub(r'<[^>]+>', '', page_response.text)
                        text = re.sub(r'\s+', ' ', text).strip()
                        # Берем первые 2000 символов
                        content = text[:2000]
                        result["full_content"] = content
                    else:
                        result["full_content"] = f"Не удалось получить содержимое (HTTP {page_response.status_code})"
                except Exception as e:
                    result["full_content"] = f"Ошибка получения содержимого: {str(e)}"
                
                processed_results.append({
                    "index": i + 1,
                    "title": result.get("title", ""),
                    "url": result.get("url", ""),
                    "snippet": result.get("snippet", ""),
                    "content": result.get("content", ""),
                    "full_content": result.get("full_content", "")
                })
        
        search_time = time.time() - start_time
        add_performance_metric("google_search", search_time)
        
        log(f"Search completed in {search_time:.2f}s, found {len(processed_results)} results")
        
        return {
            "query": query,
            "results": processed_results,
            "total_results": len(processed_results),
            "performance": {
                "search_time": search_time
            }
        }
        
    except Exception as e:
        log(f"Search error: {e}", "ERROR")
        raise HTTPException(status_code=500, detail=f"Ошибка поиска: {e}")
