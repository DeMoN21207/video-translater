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
После установки зависимостей запустите CLI-скрипт:

```bash
python transcribe_video.py "путь/к/видео.mp4"
```

> ℹ️ В проекте нет веб-интерфейса на Flask: используйте CLI-скрипт либо настольное приложение на Qt.

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
  "verbose": true,
  "input_type": "video",
  "dialogue_mode": "segments",
  "pyannote_token": null,
  "speaker_label_prefix": "Голос"
}
```

### Основные параметры

- `save_path` - `"./"` (папка translates) или полный путь
- `include_timestamps` - временные метки (true/false)
- `use_cuda` - режим ускорения: "auto", "cuda"/"gpu", "cpu"
- `model_size` - модель: "auto", "tiny", "base", "small", "medium", "large-v1", "large-v2", "large-v3"
- `delete_audio` - удалять аудио после обработки
- `language` - "russian", "english", "auto"
- `input_type` - тип входного файла: "video" или "audio"
- `dialogue_mode` - режим вывода диалогов: "segments", "speakers", "quotes"
- `pyannote_token` - токен Hugging Face для диаризации (можно оставить `null`, чтобы брать из окружения)
- `speaker_label_prefix` - префикс для отображения голосов (по умолчанию «Голос»)

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

