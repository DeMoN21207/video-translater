import os
import whisper
import moviepy.editor as mp
import argparse
import tempfile
import torch
from tqdm import tqdm
import time
import threading
import json

# Глобальная переменная для кэширования модели
_cached_model = None
_cached_model_size = None
_cached_device = None

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
        "verbose": True
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

def transcribe_video(video_path, config=None):
    """
    Транскрибирует видеофайл в текст на русском языке.

    :param video_path: Путь к видеофайлу.
    :param config: Конфигурация приложения.
    :return: Распознанный текст.
    """
    if config is None:
        config = load_config()
        
    if not os.path.exists(video_path):
        return "Ошибка: Видеофайл не найден."

    try:
        # Извлекаем аудио из видео
        if config["verbose"]:
            print("Извлечение аудио из видео...")
            
        video = mp.VideoFileClip(video_path)
        
        # Определяем путь для аудио
        video_name = os.path.splitext(os.path.basename(video_path))[0]
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
        if use_cuda == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        elif use_cuda:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = "cpu"
            
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
        
        # Форматируем текст
        if config["include_timestamps"]:
            formatted_text = format_transcription_with_timestamps(result)
        else:
            formatted_text = format_transcription_clean(result)
            
        text = formatted_text

    except Exception as e:
        return f"Произошла ошибка: {e}"
    finally:
        # Закрываем видеофайл
        if 'video' in locals():
            video.close()
        # Удаляем временный аудиофайл после обработки (если настроено)
        if (config and config["delete_audio"] and 
            'temp_audio_path' in locals() and os.path.exists(temp_audio_path)):
            os.remove(temp_audio_path)
            if config["verbose"]:
                print(f"🗑️ Аудиофайл удален: {os.path.basename(temp_audio_path)}")

    return text

def format_transcription_with_timestamps(result):
    """
    Форматирует результат транскрибации с временными метками.
    """
    formatted_lines = []
    formatted_lines.append("=" * 60)
    formatted_lines.append("РАСШИФРОВКА ВИДЕО С ВРЕМЕННЫМИ МЕТКАМИ")
    formatted_lines.append("=" * 60)
    formatted_lines.append("")
    
    # Добавляем сегменты с временными метками
    for segment in result['segments']:
        start_time = format_timestamp(segment['start'])
        end_time = format_timestamp(segment['end'])
        text = segment['text'].strip()
        
        formatted_lines.append(f"[{start_time} → {end_time}] {text}")
    
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
    formatted_lines.append(result['text'].strip())
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
