import { FormEvent, useEffect, useMemo, useState } from "react";
import {
  HistoryItem,
  StateResponse,
  AskResponse,
  FileCategory
} from "./types";
import {
  API_BASE,
  extractThinkBlocks,
  fetchJson,
  formatFileSize,
  formatTimestamp
} from "./utils";

type UploadKey = "photo" | "audio" | "video";

interface UploadStatus {
  loading: boolean;
  message: string;
}

const initialUploadStatus: Record<UploadKey, UploadStatus> = {
  photo: { loading: false, message: "" },
  audio: { loading: false, message: "" },
  video: { loading: false, message: "" }
};

function App() {
  const [state, setState] = useState<StateResponse | null>(null);
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [inputValue, setInputValue] = useState("");
  const [isSending, setIsSending] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [lastAnswer, setLastAnswer] = useState<AskResponse | null>(null);
  const [files, setFiles] = useState<FileCategory[]>([]);
  const [uploadStatus, setUploadStatus] = useState(initialUploadStatus);
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [imageDescription, setImageDescription] = useState<string>("");
  const [isLoadingImage, setIsLoadingImage] = useState(false);

  const apiBase = API_BASE.replace(/\/$/, "");

  const refreshState = async () => {
    const data = await fetchJson<StateResponse>(`${apiBase}/api/state`);
    setState(data);
  };

  const refreshHistory = async () => {
    const data = await fetchJson<{ items: HistoryItem[] }>(`${apiBase}/api/history`);
    setHistory(data.items);
  };

  const refreshFiles = async () => {
    const data = await fetchJson<{ categories: FileCategory[] }>(`${apiBase}/api/files`);
    setFiles(data.categories);
  };

  const refreshAll = async () => {
    await Promise.all([refreshState(), refreshHistory(), refreshFiles()]);
  };

  useEffect(() => {
    refreshAll().catch((error) => {
      setErrorMessage(error instanceof Error ? error.message : String(error));
    });
  }, []);

  const sendMessage = async () => {
    if (!inputValue.trim()) {
      return;
    }
    setIsSending(true);
    setErrorMessage(null);
    try {
      const payload = await fetchJson<AskResponse>(`${apiBase}/api/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: inputValue.trim() })
      });
      setLastAnswer(payload);
      setInputValue("");
      await Promise.all([refreshHistory(), refreshState(), refreshFiles()]);
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : String(error));
    } finally {
      setIsSending(false);
    }
  };

  const handleSubmit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    sendMessage();
  };

  const updateUploadStatus = (key: UploadKey, status: Partial<UploadStatus>) => {
    setUploadStatus((prev: Record<UploadKey, UploadStatus>) => ({
      ...prev,
      [key]: { ...prev[key], ...status }
    }));
  };

  const uploadForm = async (key: UploadKey, formData: FormData, endpoint: string) => {
    updateUploadStatus(key, { loading: true, message: "Идёт загрузка..." });
    try {
      await fetchJson(`${apiBase}${endpoint}`, {
        method: "POST",
        body: formData
      });
      updateUploadStatus(key, { loading: false, message: "Готово" });
      await Promise.all([refreshHistory(), refreshState(), refreshFiles()]);
    } catch (error) {
      updateUploadStatus(key, {
        loading: false,
        message: error instanceof Error ? error.message : String(error)
      });
    }
  };

  const submitPhoto = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const form = event.currentTarget;
    const fileInput = form.elements.namedItem("photo-file") as HTMLInputElement | null;
    const contextInput = form.elements.namedItem("photo-context") as HTMLInputElement | null;

    if (!fileInput?.files?.length) {
      updateUploadStatus("photo", { loading: false, message: "Добавьте изображение" });
      return;
    }

    const formData = new FormData();
    formData.append("file", fileInput.files[0]);
    formData.append("context", contextInput?.value ?? "");
    await uploadForm("photo", formData, "/api/upload/photo");
    form.reset();
  };

  const submitAudio = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const form = event.currentTarget;
    const fileInput = form.elements.namedItem("audio-file") as HTMLInputElement | null;
    const contextInput = form.elements.namedItem("audio-context") as HTMLInputElement | null;
    const langInput = form.elements.namedItem("audio-lang") as HTMLSelectElement | null;
    const separatorInput = form.elements.namedItem("audio-separator") as HTMLInputElement | null;

    if (!fileInput?.files?.length) {
      updateUploadStatus("audio", { loading: false, message: "Добавьте аудио" });
      return;
    }

    const formData = new FormData();
    formData.append("file", fileInput.files[0]);
    formData.append("context", contextInput?.value ?? "");
    if (langInput?.value) {
      formData.append("lang", langInput.value);
    }
    formData.append("use_separator", separatorInput?.checked ? "true" : "false");
    await uploadForm("audio", formData, "/api/upload/audio");
    form.reset();
  };

  const submitVideo = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const form = event.currentTarget;
    const fileInput = form.elements.namedItem("video-file") as HTMLInputElement | null;
    const contextInput = form.elements.namedItem("video-context") as HTMLInputElement | null;

    if (!fileInput?.files?.length) {
      updateUploadStatus("video", { loading: false, message: "Добавьте видео" });
      return;
    }

    const formData = new FormData();
    formData.append("file", fileInput.files[0]);
    formData.append("context", contextInput?.value ?? "");
    await uploadForm("video", formData, "/api/upload/video");
    form.reset();
  };

  const loadImagePreview = async () => {
    setIsLoadingImage(true);
    setErrorMessage(null);
    try {
      const data = await fetchJson<{ data_url: string; description: string }>(`${apiBase}/api/last-image`);
      setImageUrl(data.data_url);
      setImageDescription(data.description);
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : String(error));
    } finally {
      setIsLoadingImage(false);
    }
  };

  const thinkInfo = useMemo(() => {
    if (!lastAnswer) {
      return null;
    }
    return extractThinkBlocks(lastAnswer.ai_raw);
  }, [lastAnswer]);

  const averageResponseTime = state?.performance.average_response_time ?? 0;

  return (
    <div className="app-shell">
      <header className="header">
        <div>
          <h1>AI PowerShell Оркестратор</h1>
          <div className="badge">
            <span>{state?.brain_model ?? "Модель не выбрана"}</span>
          </div>
        </div>
        <div className="metrics">
          <span>Среднее время ответа: {averageResponseTime.toFixed(2)} сек</span>
          <span>{state?.performance.context_info ?? "Контекст неизвестен"}</span>
        </div>
      </header>

      {errorMessage && (
        <div className="card" style={{ borderColor: "rgba(248,113,113,0.45)" }}>
          <strong>Ошибка:</strong> {errorMessage}
        </div>
      )}

      <div className="layout">
        <div className="card" style={{ gap: "1.5rem" }}>
          <section className="input-panel">
            <h2>Диалог</h2>
            <form onSubmit={handleSubmit} className="input-panel">
              <textarea
                placeholder="Введите вопрос..."
                value={inputValue}
                onChange={(event) => setInputValue(event.target.value)}
                disabled={isSending}
              />
              <div className="actions-row">
                <button type="submit" className="primary" disabled={isSending}>
                  {isSending ? "Отправляю..." : "Отправить"}
                </button>
                <button
                  type="button"
                  className="ghost"
                  onClick={() => refreshAll().catch(() => undefined)}
                  disabled={isSending}
                >
                  Обновить данные
                </button>
              </div>
            </form>
            {lastAnswer && (
              <div className="status-chip">
                <span>Ответ получен за {lastAnswer.performance.response_time.toFixed(2)} сек</span>
              </div>
            )}
          </section>

          <section className="chat-feed">
            {history.length === 0 && (
              <div className="history-empty">История пока пуста. Задайте первый вопрос!</div>
            )}
            {history.map((item, index) => (
              <article key={`${item.timestamp}-${index}`} className="message ai">
                <div className="meta">
                  <span>Запрос</span>
                  <span>{formatTimestamp(item.timestamp)}</span>
                </div>
                <pre>{item.request}</pre>
                <div className="meta">
                  <span>Ответ</span>
                </div>
                <pre>{item.response || "(пусто)"}</pre>
              </article>
            ))}
          </section>

          {thinkInfo && (thinkInfo.think || thinkInfo.visible !== lastAnswer?.final) && (
            <section>
              <h2>Внутренние рассуждения</h2>
              {thinkInfo.think ? (
                <div className="think-block">
                  <strong>THINK:</strong>
                  <pre>{thinkInfo.think}</pre>
                </div>
              ) : (
                <div className="think-block">Модель не предоставила отдельный think-блок.</div>
              )}
              <div className="think-block">
                <strong>Сырые данные:</strong>
                <pre>{thinkInfo.visible || lastAnswer?.ai_raw}</pre>
              </div>
            </section>
          )}
        </div>

        <div className="info-panels">
          <div className="card">
            <div className="section-title">
              <h2>Загрузки</h2>
            </div>
            <div className="grid-two">
              <form className="upload-card" onSubmit={submitPhoto}>
                <h3>Изображение</h3>
                <label>
                  Файл
                  <input type="file" name="photo-file" accept="image/*" />
                </label>
                <label>
                  Контекст (необязательно)
                  <input type="text" name="photo-context" placeholder="Подпись или инструкция" />
                </label>
                <button type="submit" className="primary" disabled={uploadStatus.photo.loading}>
                  {uploadStatus.photo.loading ? "Загружаю..." : "Отправить"}
                </button>
                {uploadStatus.photo.message && (
                  <span className="small">{uploadStatus.photo.message}</span>
                )}
              </form>

              <form className="upload-card" onSubmit={submitAudio}>
                <h3>Аудио</h3>
                <label>
                  Файл
                  <input type="file" name="audio-file" accept="audio/*" />
                </label>
                <label>
                  Контекст
                  <input type="text" name="audio-context" placeholder="Подсказка для интерпретации" />
                </label>
                <label>
                  Язык распознавания
                  <select name="audio-lang" defaultValue="ru">
                    <option value="ru">Русский</option>
                    <option value="en">Английский</option>
                    <option value="de">Немецкий</option>
                    <option value="es">Испанский</option>
                  </select>
                </label>
                <label>
                  <span>Выделять голос separator</span>
                  <input type="checkbox" name="audio-separator" defaultChecked />
                </label>
                <button type="submit" className="primary" disabled={uploadStatus.audio.loading}>
                  {uploadStatus.audio.loading ? "Загружаю..." : "Отправить"}
                </button>
                {uploadStatus.audio.message && (
                  <span className="small">{uploadStatus.audio.message}</span>
                )}
              </form>

              <form className="upload-card" onSubmit={submitVideo}>
                <h3>Видео</h3>
                <label>
                  Файл
                  <input type="file" name="video-file" accept="video/*" />
                </label>
                <label>
                  Контекст
                  <input type="text" name="video-context" placeholder="Комментарий" />
                </label>
                <button type="submit" className="primary" disabled={uploadStatus.video.loading}>
                  {uploadStatus.video.loading ? "Загружаю..." : "Отправить"}
                </button>
                {uploadStatus.video.message && (
                  <span className="small">{uploadStatus.video.message}</span>
                )}
              </form>
            </div>
          </div>

          <div className="card">
            <div className="section-title">
              <h2>Последнее изображение</h2>
              <button className="ghost" onClick={loadImagePreview} disabled={isLoadingImage}>
                {isLoadingImage ? "Загрузка..." : "Обновить"}
              </button>
            </div>
            {imageUrl ? (
              <div>
                <img src={imageUrl} alt={imageDescription} className="generated-image" />
                <p className="small">{imageDescription}</p>
              </div>
            ) : (
              <p className="small">Пока нет изображения или требуется загрузка.</p>
            )}
          </div>

          <div className="card">
            <h2>Логи</h2>
            <div className="log-list">
              {state?.log.map((entry, index) => (
                <span className="log-entry" key={`${entry}-${index}`}>
                  {entry}
                </span>
              ))}
              {!state?.log?.length && <span className="small">Логи пока пусты.</span>}
            </div>
          </div>

          <div className="card">
            <h2>Файлы</h2>
            {files.length === 0 && <span className="small">Пока нет файлов для отображения.</span>}
            {files.map((category) => (
              <section key={category.id}>
                <h3>{category.title}</h3>
                <div className="file-list">
                  {category.files.length === 0 && (
                    <span className="small">Нет данных.</span>
                  )}
                  {category.files.map((file) => (
                    <div className="file-item" key={`${category.id}-${file.relative_path}`}>
                      <div>
                        <span>{file.relative_path}</span>
                        <div className="small">
                          {formatFileSize(file.size)} · {formatTimestamp(file.modified)}
                        </div>
                      </div>
                      <a
                        className="ghost"
                        href={`${apiBase}/api/files/download?category=${encodeURIComponent(category.id)}&file_path=${encodeURIComponent(file.relative_path)}`}
                        target="_blank"
                        rel="noopener noreferrer"
                      >
                        Скачать
                      </a>
                    </div>
                  ))}
                </div>
              </section>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
