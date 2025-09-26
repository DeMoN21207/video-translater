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

## Веб-интерфейс через Docker

1. Соберите образ со всеми зависимостями Whisper:

   ```bash
   docker build -t video-transcriber .
   ```

2. Запустите контейнер (порт можно переопределить через переменную `PORT`):

   ```bash
   docker run --rm -p 5000:5000 video-transcriber
   ```

3. Откройте браузер и перейдите на <http://localhost:5000>.

Во вкладке загрузки теперь можно:

- выбрать тип загружаемого файла — **видео** (MP4/MKV и т. п.) или **аудио** (MP3/WAV и т. д.);
- указать режим аппаратного ускорения: автоматический выбор, принудительное использование GPU (если доступно) или только CPU;
- выбрать нужную модель Whisper (скачивается только при первом запуске выбранной модели);
- настроить формат диалогов: оставить сегменты как есть, сгруппировать по голосам (Voice 1, Voice 2 и т. д.) или разбить длинные реплики на отдельные цитаты;
- отслеживать прогресс загрузки файла и стадии обработки прямо в браузере;
- получить таблицу с диалогами и временными метками, а также полный текст расшифровки.

> Контейнер стартует Flask-приложение, позволяющее загружать ролики или аудиодорожки и получать диалоги в табличном виде с указанием времени начала и окончания фразы.

> 🗣 Для режима «По голосам» требуется библиотека `pyannote.audio` и персональный токен Hugging Face (`PYANNOTE_AUTH_TOKEN`). Подробности — ниже.

## Определение голосов (диаризация)

Режим «По голосам» использует `pyannote.audio` для автоматической разметки спикеров (Голос 1, Голос 2 и т. д.).

1. Получите персональный токен Hugging Face и сохраните его в переменную окружения:

   ```bash
   export PYANNOTE_AUTH_TOKEN="hf_xxx..."
   ```

2. Убедитесь, что установлена зависимость `pyannote.audio` (она уже указана в `requirements.txt`).

3. При запуске Docker-проекта передайте токен внутрь контейнера:

   ```bash
   docker run --rm -p 5000:5000 \
     -e PYANNOTE_AUTH_TOKEN=$PYANNOTE_AUTH_TOKEN \
     video-transcriber
   ```

4. В веб-интерфейсе выберите «По голосам (beta)» в поле «Формат диалогов».

Если зависимости или токена нет, приложение продолжит работу, но покажет предупреждение и оставит сегменты без меток спикеров.

## Запуск скрипта напрямую

```bash
python transcribe_video.py "путь/к/видео.mp4"
```

### Использование CLI внутри Docker

```bash
# Сборка образа
docker build -t video-transcriber .

# Транскрибация файла с монтированием папки с видео и результатом
docker run --rm \
  -v "$PWD:/data" \
  video-transcriber \
  python transcribe_video.py /data/путь/к/видео.mp4
```

> Результаты сохраняются в смонтированную директорию `/data` (по умолчанию в подпапку `translates`).

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

## Предзагрузка моделей Whisper в Docker

По умолчанию образ ничего не скачивает заранее — нужная модель загружается при первом запуске.
Чтобы избежать ожидания в рантайме, можно на этапе сборки указать, какие модели сохранить в кэше:

```bash
docker build -t video-transcriber \
  --build-arg WHISPER_MODELS="small,medium" \
  .
```

Допустимые значения: `tiny`, `base`, `small`, `medium`, `large-v1`, `large-v2`, `large-v3`.
Чтобы полностью отключить предзагрузку, передайте `--build-arg WHISPER_MODELS=none`.
