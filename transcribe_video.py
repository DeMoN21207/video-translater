import os
import re
import argparse
import json
import tempfile
from typing import Dict, List

import moviepy.editor as mp
import torch
import whisper


def _prepare_config(config):
    """Возвращает копию конфигурации, чтобы избежать изменения исходного словаря."""
    if config is None:
        config = load_config()
    else:
        # Создаем поверхностную копию, чтобы не изменять исходный словарь вызывающего кода
        config = dict(config)

    return config

# Глобальная переменная для кэширования модели
_cached_model = None
_cached_model_size = None
_cached_device = None

try:
    from pyannote.audio import Pipeline as PyannotePipeline
except ImportError:  # pragma: no cover - optional dependency
    PyannotePipeline = None

_cached_diarization_pipeline = None
_cached_diarization_token = None

def load_config():
    """
    Загружает конфигурацию из config.json
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "config.json")
    
    # Значения по умолчанию
    default_config = {
        "save_path": "./",
        "include_timestamps": True,
        "use_cuda": "auto",
        "model_size": "auto",
        "delete_audio": True,
        "audio_format": "mp3",
        "language": "russian",
        "verbose": True,
        "input_type": "video",
        "dialogue_mode": "segments",
        "pyannote_token": None,
        "speaker_label_prefix": "Голос",
    }
    
    try:
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                user_config = json.load(f)
                
            # Обновляем только настройки (игнорируем комментарии с _)
            for key, value in user_config.items():
                if not key.startswith('_') and key in default_config:
                    default_config[key] = value
                    
            return default_config
        else:
            print(f"⚠️ Конфиг не найден: {config_path}")
            print("📝 Используются настройки по умолчанию")
            return default_config
            
    except Exception as e:
        print(f"❌ Ошибка загрузки конфига: {e}")
        print("📝 Используются настройки по умолчанию")
        return default_config

def load_cached_model(model_size, device):
    """
    Загружает модель Whisper с кэшированием в памяти.
    """
    global _cached_model, _cached_model_size, _cached_device
    
    # Проверяем, нужно ли загружать новую модель
    if (_cached_model is None or 
        _cached_model_size != model_size or 
        _cached_device != device):
        
        print(f"Загрузка модели Whisper ({model_size})...")
        
        # Показываем путь к кэшу
        cache_dir = os.path.expanduser("~/.cache/whisper")
        if os.path.exists(cache_dir):
            print(f"Кэш моделей: {cache_dir}")
        
        _cached_model = whisper.load_model(model_size, device=device)
        _cached_model_size = model_size
        _cached_device = device
    else:
        print(f"Использую кэшированную модель ({model_size})")
    
    return _cached_model


def load_diarization_pipeline(config):
    """Возвращает кэшированный пайплайн диаризации (если доступен)."""

    global _cached_diarization_pipeline, _cached_diarization_token

    if PyannotePipeline is None:
        raise RuntimeError(
            "Библиотека pyannote.audio не установлена. Добавьте 'pyannote.audio' "
            "в зависимости или установите её вручную, чтобы включить разметку голосов."
        )

    token = (
        (config or {}).get("pyannote_token")
        or os.environ.get("PYANNOTE_AUTH_TOKEN")
    )

    if not token:
        raise RuntimeError(
            "Не указан токен Hugging Face (PYANNOTE_AUTH_TOKEN) для pyannote.audio. "
            "Получите персональный токен и задайте его через переменную окружения "
            "или параметр 'pyannote_token' в config.json."
        )

    if _cached_diarization_pipeline is None or token != _cached_diarization_token:
        if (config or {}).get("verbose"):
            print("Загрузка пайплайна диаризации pyannote...")

        _cached_diarization_pipeline = PyannotePipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=token,
        )
        _cached_diarization_token = token

    return _cached_diarization_pipeline


def run_speaker_diarization(audio_path, config):
    """Запускает диаризацию и возвращает список сегментов с метками спикеров."""

    pipeline = load_diarization_pipeline(config)
    diarization = pipeline(audio_path)

    speaker_segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        speaker_segments.append(
            {
                "start": float(turn.start),
                "end": float(turn.end),
                "speaker": speaker,
            }
        )

    return speaker_segments


def assign_speaker_labels(
    whisper_segments: List[Dict],
    diarization_segments: List[Dict],
    label_prefix: str = "Голос",
):
    """Сопоставляет сегментам Whisper наиболее подходящие метки спикеров."""

    speaker_mapping: Dict[str, str] = {}

    for segment in whisper_segments:
        start = segment.get("start", 0.0)
        end = segment.get("end", start)

        overlaps: Dict[str, float] = {}
        for diar in diarization_segments:
            overlap = max(0.0, min(end, diar["end"]) - max(start, diar["start"]))
            if overlap > 0:
                overlaps[diar["speaker"]] = overlaps.get(diar["speaker"], 0.0) + overlap

        if not overlaps:
            continue

        best_speaker = max(overlaps, key=overlaps.get)

        if best_speaker not in speaker_mapping:
            speaker_mapping[best_speaker] = f"{label_prefix} {len(speaker_mapping) + 1}"

        segment["speaker"] = best_speaker
        segment["speaker_display"] = speaker_mapping[best_speaker]


def split_segment_into_quotes(text: str) -> List[str]:
    """Разбивает текст сегмента на цитаты или предложения."""

    if not text:
        return []

    cleaned = text.strip()
    if not cleaned:
        return []

    quotes = re.findall(r'«[^»]+»|"[^"]+"|\'[^\']+\'', cleaned)
    if quotes:
        return [q.strip() for q in quotes]

    sentences = re.split(r"(?<=[.!?])\s+", cleaned)
    return [sentence.strip() for sentence in sentences if sentence.strip()]


def prepare_dialogues(
    segments: List[Dict],
    mode: str = "segments",
) -> List[Dict[str, str]]:
    """Готовит список диалогов для отображения в UI."""

    dialogues: List[Dict[str, str]] = []

    if not segments:
        return dialogues

    mode = (mode or "segments").lower()

    if mode == "quotes":
        for segment in segments:
            quotes = split_segment_into_quotes(segment.get("text", ""))
            if not quotes:
                dialogues.append(
                    {
                        "start": format_timestamp(segment.get("start", 0)),
                        "end": format_timestamp(segment.get("end", 0)),
                        "text": segment.get("text", "").strip(),
                        "speaker": segment.get("speaker_display"),
                    }
                )
                continue

            duration = max(segment.get("end", 0) - segment.get("start", 0), 0.0)
            chunk = duration / len(quotes) if quotes else 0.0

            for idx, quote in enumerate(quotes):
                start = segment.get("start", 0) + idx * chunk
                end = start + chunk if chunk else segment.get("end", start)

                dialogues.append(
                    {
                        "start": format_timestamp(start),
                        "end": format_timestamp(end),
                        "text": quote.strip(),
                        "speaker": segment.get("speaker_display"),
                    }
                )
    else:
        for segment in segments:
            dialogues.append(
                {
                    "start": format_timestamp(segment.get("start", 0)),
                    "end": format_timestamp(segment.get("end", 0)),
                    "text": segment.get("text", "").strip(),
                    "speaker": segment.get("speaker_display"),
                }
            )

    if mode == "speakers":
        merged: List[Dict[str, str]] = []
        for entry in dialogues:
            if not merged:
                merged.append(entry)
                continue

            last = merged[-1]
            if entry.get("speaker") and entry.get("speaker") == last.get("speaker"):
                last["end"] = entry["end"]
                last["text"] = f"{last['text']} {entry['text']}".strip()
            else:
                merged.append(entry)

        dialogues = merged

    return dialogues


def _transcribe_internal(media_path, config=None):
    """
    Общая логика транскрибации медиа-файла.

    :param media_path: Путь к видео- или аудиофайлу.
    :param config: Конфигурация приложения.
    :return: Кортеж из форматированного текста и исходного результата модели.
    """
    config = _prepare_config(config)

    if not os.path.exists(media_path):
        return "Ошибка: Файл не найден.", None, []

    should_cleanup_audio = False
    warnings: List[str] = []

    try:
        input_type = config.get("input_type", "video")
        should_cleanup_audio = config.get("delete_audio", True) and input_type != "audio"

        if input_type == "audio":
            if config["verbose"]:
                print("Обнаружен аудиофайл. Извлечение дорожки не требуется.")

            temp_audio_path = media_path

            # Считываем длительность аудио при необходимости
            duration = None
            try:
                audio_clip = mp.AudioFileClip(media_path)
                duration = audio_clip.duration
            finally:
                if 'audio_clip' in locals():
                    audio_clip.close()

        else:
            # Извлекаем аудио из видео
            if config["verbose"]:
                print("Извлечение аудио из видео...")

            video = mp.VideoFileClip(media_path)

            # Определяем путь для аудио
            video_name = os.path.splitext(os.path.basename(media_path))[0]
            save_path = config["save_path"]
            if save_path == "./":
                # Создаем папку translates рядом со скриптом
                script_dir = os.path.dirname(os.path.abspath(__file__))
                save_path = os.path.join(script_dir, "translates")
                os.makedirs(save_path, exist_ok=True)
            else:
                save_path = os.path.abspath(save_path)
                os.makedirs(save_path, exist_ok=True)

            audio_format = config["audio_format"]
            temp_audio_path = os.path.join(save_path, f"{video_name}.{audio_format}")

            if config["verbose"]:
                print(f"📁 Рабочая папка: {save_path}")
                print(f"🎵 Создается аудио: {video_name}.{audio_format}")

            # Получаем длительность видео для прогрессбара
            duration = video.duration
            if config["verbose"]:
                print(f"Длительность видео: {format_timestamp(duration)}")

            # Извлекаем аудио
            codec = 'mp3' if audio_format == 'mp3' else None
            video.audio.write_audiofile(temp_audio_path, codec=codec, verbose=False, logger=None)
        
        # Определяем устройство
        use_cuda = config["use_cuda"]
        if isinstance(use_cuda, str):
            normalized = use_cuda.lower()
        else:
            normalized = use_cuda

        if normalized == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        elif normalized in ("cuda", "gpu", True):
            device = "cuda" if torch.cuda.is_available() else "cpu"
        elif normalized in ("cpu", False):
            device = "cpu"
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        if config["verbose"]:
            print(f"Используется устройство: {device}")
            if device == "cuda" and torch.cuda.is_available():
                print(f"GPU: {torch.cuda.get_device_name(0)}")
                print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

        # Определяем модель
        model_size = config["model_size"]
        if model_size == "auto":
            model_size = "large-v2" if device == "cuda" else "base"
            
        model = load_cached_model(model_size, device)

        # Транскрибируем аудио
        if config["verbose"]:
            print("Транскрибация аудио...")
        
        # Запускаем транскрибацию
        result = transcribe_simple(model, temp_audio_path, duration, device, config)

        diarization_segments: List[Dict] = []
        dialogue_mode = config.get("dialogue_mode", "segments")
        if dialogue_mode == "speakers":
            try:
                diarization_segments = run_speaker_diarization(temp_audio_path, config)
            except RuntimeError as diarization_error:
                message = str(diarization_error)
                warnings.append(message)
                if config.get("verbose"):
                    print(f"⚠️ {message}")
            except Exception as diarization_error:  # pragma: no cover - защитное логгирование
                message = f"Не удалось выполнить диаризацию: {diarization_error}"
                warnings.append(message)
                if config.get("verbose"):
                    print(f"⚠️ {message}")

        if result.get("segments") and diarization_segments:
            assign_speaker_labels(
                result["segments"],
                diarization_segments,
                config.get("speaker_label_prefix", "Голос"),
            )

        # Форматируем текст
        if config["include_timestamps"]:
            formatted_text = format_transcription_with_timestamps(result)
        else:
            formatted_text = format_transcription_clean(result)

        text = formatted_text

    except Exception as e:
        warnings.append(str(e))
        return f"Произошла ошибка: {e}", None, warnings
    finally:
        # Закрываем видеофайл
        if 'video' in locals():
            video.close()
        # Удаляем временный аудиофайл после обработки (если настроено)
        if (
            config
            and should_cleanup_audio
            and 'temp_audio_path' in locals()
            and os.path.exists(temp_audio_path)
        ):
            os.remove(temp_audio_path)
            if config["verbose"]:
                print(f"🗑️ Аудиофайл удален: {os.path.basename(temp_audio_path)}")

    return text, result, warnings


def transcribe_video(video_path, config=None):
    """
    Транскрибирует видеофайл и возвращает форматированный текст.
    """
    text, _, warnings = _transcribe_internal(video_path, config)

    if config and config.get("verbose") and warnings and not text.startswith("Произошла ошибка"):
        for warning in warnings:
            print(f"⚠️ {warning}")

    return text


def transcribe_video_with_segments(video_path, config=None):
    """
    Транскрибирует видео и возвращает текст вместе с сегментами диалога.

    :return: Кортеж (текст, список сегментов, предупреждения)
    """
    text, result, warnings = _transcribe_internal(video_path, config)

    if result and 'segments' in result:
        segments = result['segments']
    else:
        segments = []

    return text, segments, warnings

def format_transcription_with_timestamps(result):
    """
    Форматирует результат транскрибации с временными метками.
    """
    formatted_lines = []
    formatted_lines.append("=" * 60)
    formatted_lines.append("РАСШИФРОВКА ВИДЕО С ВРЕМЕННЫМИ МЕТКАМИ")
    formatted_lines.append("=" * 60)
    formatted_lines.append("")
    
    segments = result.get('segments', []) if result else []

    for segment in segments:
        start_time = format_timestamp(segment['start'])
        end_time = format_timestamp(segment['end'])
        text = segment.get('text', '').strip()
        speaker_label = segment.get('speaker_display')
        prefix = f"{speaker_label}: " if speaker_label else ""

        formatted_lines.append(f"[{start_time} → {end_time}] {prefix}{text}")
    
    formatted_lines.append("")
    formatted_lines.append("=" * 60)
    
    return "\n".join(formatted_lines)

def format_transcription_clean(result):
    """
    Форматирует результат транскрибации без временных меток.
    """
    formatted_lines = []
    formatted_lines.append("=" * 60)
    formatted_lines.append("РАСШИФРОВКА ВИДЕО")
    formatted_lines.append("=" * 60)
    formatted_lines.append("")
    segments = result.get('segments', []) if result else []

    if segments and any(segment.get('speaker_display') for segment in segments):
        for segment in segments:
            text = segment.get('text', '').strip()
            speaker_label = segment.get('speaker_display')
            if speaker_label:
                formatted_lines.append(f"{speaker_label}: {text}")
            elif text:
                formatted_lines.append(text)
    else:
        formatted_lines.append(result.get('text', '').strip())
    formatted_lines.append("")
    formatted_lines.append("=" * 60)
    
    return "\n".join(formatted_lines)

def format_timestamp(seconds):
    """
    Конвертирует секунды в формат MM:SS или HH:MM:SS.
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    else:
        return f"{minutes:02d}:{secs:02d}"

def estimate_transcription_time(audio_duration, device):
    """
    Оценивает время транскрибации на основе длительности аудио и устройства.
    """
    if device == "cuda":
        # GPU примерно в 10-15 раз быстрее
        ratio = 0.1  # 10% от длительности аудио
    else:
        # CPU медленнее
        ratio = 0.3  # 30% от длительности аудио
    
    estimated_seconds = audio_duration * ratio
    return format_timestamp(estimated_seconds)

def transcribe_simple(model, audio_path, duration, device, config):
    """
    Транскрибирует аудио без прогрессбара.
    """
    # Запускаем транскрибацию
    result = model.transcribe(
        audio_path,
        language=config["language"],
        fp16=(device == "cuda"),
        beam_size=5 if device == "cuda" else 1,
        best_of=5 if device == "cuda" else 1,
        word_timestamps=config["include_timestamps"],
        verbose=config["verbose"]
    )
    
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Транскрибация MP4 видео в текст.")
    parser.add_argument("video_file", help="Путь к вашему MP4 файлу.")
    parser.add_argument("--config", help="Путь к файлу конфигурации (по умолчанию config.json)")
    args = parser.parse_args()

    # Загружаем конфигурацию
    config = load_config()
    
    video_file_path = args.video_file
    
    if config["verbose"]:
        print(f"🎬 Начинаю транскрибацию файла: {video_file_path}")
        print(f"⚙️ Конфигурация загружена")
        
    transcribed_text = transcribe_video(video_file_path, config)
    
    if transcribed_text.startswith("Ошибка:") or transcribed_text.startswith("Произошла ошибка:"):
        print(transcribed_text)
    else:
        if not config["verbose"]:
            print(transcribed_text)
        
        # Определяем путь для сохранения
        save_path = config["save_path"]
        if save_path == "./":
            # Создаем папку translates рядом со скриптом
            script_dir = os.path.dirname(os.path.abspath(__file__))
            save_path = os.path.join(script_dir, "translates")
        else:
            save_path = os.path.abspath(save_path)
            
        # Создаем папку если не существует
        os.makedirs(save_path, exist_ok=True)
        
        # Формируем имя файла
        video_name = os.path.splitext(os.path.basename(video_file_path))[0]
        output_filename = os.path.join(save_path, f"{video_name}.txt")
        
        # Сохраняем файл
        with open(output_filename, "w", encoding="utf-8") as f:
            f.write(transcribed_text)
            
        if config["verbose"]:
            print(f"\n📄 Результат сохранен: {os.path.basename(output_filename)}")
            print(f"📁 Расположение: {save_path}")
