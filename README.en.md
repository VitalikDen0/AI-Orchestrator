# 🤖 AI-Orchestrator

**Intelligent Multimodal AI Assistant with System Integration**

[🇷🇺 Russian](README.md) | [🇬🇧 English](README.en.md)

A comprehensive solution for task automation that combines large language models, computer vision, speech recognition, image generation, and system management through a unified interface.

## Overview

AI-Orchestrator is designed for Windows systems and provides:

- **LLM Integration** via LM Studio (local models with dynamic context management)
- **Vision Analysis** (image understanding, screen analysis)
- **Speech Recognition** (Whisper with support for audio preprocessing)
- **Image Generation** (Stable Diffusion with LoRA support)
- **System Automation** (PowerShell, mouse/keyboard control, UI automation)
- **Communications** (Google Search, YouTube processing, Telegram bot, Email IMAP/SMTP)
- **Vector Memory** (ChromaDB with embeddings for context retention)
- **Smart Resource Management** (automatic tool activation/deactivation, VRAM optimization)

## Quick Start

### Requirements

- **Python** 3.10+
- **LM Studio** 0.3.23+ (with models)
- **ffmpeg** (audio/video processing)
- **Windows** (PowerShell required)
- **RAM** 16GB minimum, 32GB recommended
- **VRAM** 8GB minimum (RTX 4070/4080 recommended)

### Installation

```bash
# Create virtual environment
python -m venv venv
venv\\Scripts\\activate

# Install dependencies
pip install -r requirements.txt
```

### Configuration

1. Copy `.env.example` to `.env` and configure:
   - LM Studio API settings
   - Google Search credentials (optional)
   - Telegram bot token (optional)
   - Model paths

2. Update model paths in `config.py`

3. Create required directories:
   ```
   Photos/ Audio/ Video/ Images/ Videos/ 
   output/ prompt/ plugins/ webui/static/
   stable_diff/checkpoints/ stable_diff/lora/
   ```

### Usage

**Console mode:**
```bash
python main.py
```

**Web interface:**
```bash
python main.py --web
# Open http://127.0.0.1:8001/
```

**Telegram bot:**
Configure `TELEGRAM_BOT_TOKEN` in `.env` - bot starts automatically

## Architecture

```
Orchestrator Core
├── LLM Services (LM Studio API / llama-cpp-python)
├── Media Processing (Vision, Audio, Video)
├── Image Generation (Stable Diffusion)
├── System Automation (PowerShell, UI Automation)
├── Communications (Email, Telegram, Search, YouTube)
├── Resource Manager (Memory, VRAM, Tool Lifecycle)
└── ChromaDB Memory (Vector embeddings, preferences)

WebUI / CLI Interface
```

## Key Features

### Multimodal AI
- **LLM**: Qwen 3-4B or Orchestrator-8B (configurable)
- **Vision**: moondream2 or llama-joycaption
- **Audio**: Whisper large-v3 (Q8 quantization)
- **Image Gen**: Stable Diffusion 1.5/SDXL with LoRA support

### Smart Resource Management
- Dynamic context window optimization (3 trimming levels)
- Token tracking from LM Studio responses
- Auto-enable/disable tools based on usage
- VRAM cleanup with garbage collection

### Communications & Integration
- Email: Gmail, Outlook, Yandex, Mail.ru
- YouTube: Download, extract audio, frame-by-frame analysis
- Google Search: Web queries with snippet extraction
- Telegram: Text, voice, images, documents

### Memory & Personalization
- Vector embeddings with ChromaDB
- Automatic preference extraction
- Conversation history retention (configurable)
- RAG for large documents

## File Reading & Generation

**Read:** DOCX, Excel, PDF, CSV, Markdown  
**Write:** DOCX, Excel, PDF, Markdown  
**RAG:** Sentence-based chunking with context preservation

## Requirements (Detailed)

### Software
- Python 3.10.11 (tested)
- LM Studio 0.3.23+
- ffmpeg
- yt-dlp
- PowerShell (Windows)

### Recommended Models
- **LLM**: `huihui-qwen3-4b-thinking-2507-abliterated` (Q4_K_S, ~4GB)
- **Audio**: `whisper-large-v3-q8_0` (~4GB)
- **Vision**: `moondream2-050824-q8` (~2GB)
- **Image Gen**: `novaAnime_v20` or similar SD 1.5 model (~4-8GB)

### Hardware
- **Minimum**: 16GB RAM, RTX 3060 (12GB VRAM)
- **Recommended**: 32GB RAM, RTX 4070+ (16GB VRAM)
- **OS**: Windows 11

## Configuration

See `docs/email-setup.md` for email integration guide.

## License

**Dual License (Dual-NCAL v1.0 + Commercial)**

- **Non-Commercial**: Free for personal use (GPL-3.0 style)
- **Commercial**: Requires separate license

This is a **source-available** license, not OSI-compliant open source.

For commercial use, contact: dlaproektov96@gmail.com

## Development Status

- ✅ Core multimodal AI integration
- ✅ Email and communications
- ✅ UI automation (beta)
- ⚠️ Video generation (experimental)
- 🔄 Linux support (planned)
- 🔄 Docker support (planned)

## Contributing

Project created by Russian developer using AI tools (Claude, GPT) for code generation and optimization. All AI-generated code has been tested and adapted by human developer.

## Acknowledgments

- LM Studio for local LLM serving
- Hugging Face for model hosting
- OpenCV, PyTorch, and open-source community

---

**[Return to Russian README →](README.md)**
