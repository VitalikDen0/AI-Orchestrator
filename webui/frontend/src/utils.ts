export const API_BASE = import.meta.env.VITE_API_BASE ?? "";

export async function fetchJson<T>(input: RequestInfo | URL, init?: RequestInit): Promise<T> {
  const response = await fetch(input, init);
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `Ошибка запроса (${response.status})`);
  }
  return response.json() as Promise<T>;
}

export function formatTimestamp(ts: number): string {
  const date = new Date(ts * 1000);
  return date.toLocaleString("ru-RU", {
    hour12: false,
    hour: "2-digit",
    minute: "2-digit",
    day: "2-digit",
    month: "2-digit",
    year: "numeric"
  });
}

export function formatFileSize(size: number): string {
  if (size < 1024) {
    return `${size} Б`;
  }
  const units = ["КБ", "МБ", "ГБ", "ТБ"];
  let value = size / 1024;
  let unitIndex = 0;
  while (value >= 1024 && unitIndex < units.length - 1) {
    value /= 1024;
    unitIndex += 1;
  }
  return `${value.toFixed(value >= 10 ? 0 : 1)} ${units[unitIndex]}`;
}

export function extractThinkBlocks(aiRaw: string): { think: string; visible: string } {
  const thinkMatch = aiRaw.match(/<think>([\s\S]*?)<\/think>/i);
  if (!thinkMatch) {
    return { think: "", visible: aiRaw };
  }
  const think = thinkMatch[1].trim();
  const visible = aiRaw.replace(thinkMatch[0], "").trim();
  return { think, visible };
}
