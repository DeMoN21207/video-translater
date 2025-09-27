# Video Transcription Script

Транскрибация видео в текст с использованием Whisper от OpenAI.

## Установка

```bash
# GPU версия (рекомендуется)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# или CPU версия
pip install torch torchvision torchaudio

# Основные зависимости
pip install openai-whisper moviepy==1.0.3 tqdm
```

> **Важно:** проект больше не использует Docker. Скрипт запускается напрямую в вашей локальной среде Python, поэтому никакие Dockerfile, docker-compose или связанные артефакты в репозитории не требуются.

## Запуск

```bash
python transcribe_video.py "путь/к/видео.mp4"
```

## Результат

Создается папка `translates/` рядом со скриптом. Файлы сохраняются как `video_name.txt`.

**С временными метками:**
```
[00:05 → 00:12] Первая фраза
[00:13 → 00:25] Вторая фраза  
```

**Без временных меток:**
```
Первая фраза. Вторая фраза.
```

## Конфигурация

Файл `config.json`:
```json
{
  "save_path": "./",
  "include_timestamps": true,
  "use_cuda": "auto",
  "model_size": "auto", 
  "delete_audio": true,
  "audio_format": "mp3",
  "language": "russian",
  "show_progress": true,
  "verbose": true
}
```

### Основные параметры

- `save_path` - `"./"` (папка translates) или полный путь
- `include_timestamps` - временные метки (true/false)
- `use_cuda` - GPU: "auto", true, false  
- `model_size` - модель: "auto", "tiny", "base", "small", "medium", "large-v2"
- `delete_audio` - удалять аудио после обработки
- `language` - "russian", "english", "auto"

## Производительность

| Устройство | Модель | 5 мин видео |
|------------|--------|-------------|
| RTX 4070 Ti | large-v2 | ~30 сек |
| RTX 3080 | large-v2 | ~45 сек |
| CPU i7 | base | ~2 мин |

## Устранение проблем

**PyTorch CUDA не найден:**
```bash
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Не хватает VRAM - уменьшите модель:**
```json
"model_size": "medium"
```

**Медленно - используйте меньшую модель:**
```json
"model_size": "tiny"
```
