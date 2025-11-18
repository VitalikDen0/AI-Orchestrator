export interface HistoryItem {
  type: string;
  request: string;
  response: string;
  timestamp: number;
}

export interface PerformanceMetric {
  timestamp: number;
  action: string;
  response_time: number;
  context_length: number;
}

export interface StateResponse {
  brain_model: string;
  use_image_generation: boolean;
  use_vision: boolean;
  use_audio: boolean;
  use_separator: boolean;
  last_final_response: string;
  has_image: boolean;
  log: string[];
  history_len: number;
  performance: {
    recent_metrics: PerformanceMetric[];
    average_response_time: number;
    context_info: string;
  };
}

export interface AskResponse {
  continue: boolean;
  ai_raw: string;
  final: string;
  performance: {
    response_time: number;
    context_length: number;
  };
}

export interface FileEntry {
  name: string;
  relative_path: string;
  size: number;
  modified: number;
}

export interface FileCategory {
  id: string;
  title: string;
  files: FileEntry[];
}

export interface FilesResponse {
  categories: FileCategory[];
}
