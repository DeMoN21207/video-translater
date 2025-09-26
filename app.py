import os
import tempfile
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

from transcribe_video import (
    load_config,
    transcribe_video_with_segments,
    prepare_dialogues,
)

app = Flask(__name__)

# Настройки приложения
app.config["SECRET_KEY"] = os.environ.get("FLASK_SECRET_KEY", "video-translater-secret")
app.config["UPLOAD_ROOT"] = os.environ.get(
    "UPLOAD_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
)
os.makedirs(app.config["UPLOAD_ROOT"], exist_ok=True)


@app.route("/", methods=["GET", "POST"])
def index():
    error_message = None
    transcript_text = None
    dialogues = []
    runtime_warnings = []

    base_config = load_config()

    def _normalize_acceleration_choice(config_value):
        if isinstance(config_value, str):
            normalized = config_value.lower()
        else:
            normalized = config_value

        if normalized in ("cuda", "gpu", True):
            return "gpu"
        if normalized in ("cpu", False):
            return "cpu"
        return "auto"

    allowed_model_sizes = {
        "auto",
        "tiny",
        "base",
        "small",
        "medium",
        "large-v1",
        "large-v2",
        "large-v3",
    }

    selected_media_type = base_config.get("input_type", "video")
    selected_acceleration = _normalize_acceleration_choice(base_config.get("use_cuda", "auto"))
    selected_model_size = base_config.get("model_size", "auto")
    selected_dialogue_mode = base_config.get("dialogue_mode", "segments")
    allowed_dialogue_modes = {
        "segments",
        "speakers",
        "quotes",
    }
    if selected_model_size not in allowed_model_sizes:
        selected_model_size = "auto"

    if selected_dialogue_mode not in allowed_dialogue_modes:
        selected_dialogue_mode = "segments"

    dialogue_mode_options = [
        ("segments", "По фразам"),
        ("speakers", "По голосам (beta)"),
        ("quotes", "Разбивать на цитаты"),
    ]

    if request.method == "POST":
        selected_media_type = request.form.get("media_type", selected_media_type)
        selected_acceleration = request.form.get("acceleration", selected_acceleration)
        candidate_model = request.form.get("model_size", selected_model_size)
        if candidate_model in allowed_model_sizes:
            selected_model_size = candidate_model
        candidate_mode = request.form.get("dialogue_mode", selected_dialogue_mode)
        if candidate_mode in allowed_dialogue_modes:
            selected_dialogue_mode = candidate_mode

        uploaded_file = request.files.get("media")

        if not uploaded_file or uploaded_file.filename == "":
            error_message = "Пожалуйста, выберите файл для загрузки."
        else:
            filename = secure_filename(uploaded_file.filename)

            with tempfile.TemporaryDirectory(dir=app.config["UPLOAD_ROOT"]) as temp_dir:
                media_path = os.path.join(temp_dir, filename)
                uploaded_file.save(media_path)

                config = dict(base_config)
                config["verbose"] = False
                config["save_path"] = temp_dir
                config["delete_audio"] = True
                config["input_type"] = selected_media_type
                config["dialogue_mode"] = selected_dialogue_mode

                if selected_acceleration == "gpu":
                    config["use_cuda"] = "cuda"
                elif selected_acceleration == "cpu":
                    config["use_cuda"] = "cpu"
                else:
                    config["use_cuda"] = "auto"

                config["model_size"] = selected_model_size or base_config.get("model_size", "auto")
                if not config.get("pyannote_token"):
                    config["pyannote_token"] = os.environ.get("PYANNOTE_AUTH_TOKEN")

                transcript_text, segments, warnings = transcribe_video_with_segments(media_path, config)
                runtime_warnings.extend(warnings)

                if transcript_text.startswith("Ошибка") or transcript_text.startswith("Произошла ошибка"):
                    error_message = transcript_text
                else:
                    dialogues = prepare_dialogues(segments, selected_dialogue_mode)

    if runtime_warnings:
        # Удаляем дубликаты, сохраняя порядок
        seen = set()
        runtime_warnings = [w for w in runtime_warnings if not (w in seen or seen.add(w))]

    return render_template(
        "index.html",
        error_message=error_message,
        transcript=transcript_text,
        dialogues=dialogues,
        warnings=runtime_warnings,
        selected_media_type=selected_media_type,
        selected_acceleration=selected_acceleration,
        selected_model_size=selected_model_size,
        selected_dialogue_mode=selected_dialogue_mode,
        dialogue_mode_options=dialogue_mode_options,
    )


if __name__ == "__main__":
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", 5000))
    app.run(host=host, port=port)
