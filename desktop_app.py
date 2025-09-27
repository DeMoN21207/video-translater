"""Десктопный интерфейс на Tkinter для скрипта транскрибации видео."""

import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
from typing import Dict, List, Optional

from transcribe_video import (
    load_config,
    transcribe_video_with_segments,
    prepare_dialogues,
)


class DesktopTranscriberApp:
    """Tkinter-интерфейс для запуска транскрибации видео или аудио."""

    _ALLOWED_MODELS = [
        "auto",
        "tiny",
        "base",
        "small",
        "medium",
        "large-v1",
        "large-v2",
        "large-v3",
    ]

    _ACCELERATION_OPTIONS = [
        ("auto", "Авто"),
        ("cpu", "CPU"),
        ("gpu", "GPU"),
    ]

    _INPUT_TYPES = [
        ("video", "Видео"),
        ("audio", "Аудио"),
    ]

    _DIALOGUE_MODES = [
        ("segments", "По фразам"),
        ("speakers", "По голосам"),
        ("quotes", "Цитаты"),
    ]

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Video Translater – Desktop")
        self.root.geometry("900x700")

        self.base_config = load_config()

        self.file_path_var = tk.StringVar()
        self.output_dir_var = tk.StringVar(value=self.base_config.get("save_path", "./"))
        self.model_var = tk.StringVar(value=self._sanitize_model(self.base_config.get("model_size", "auto")))
        self.input_type_var = tk.StringVar(value=self.base_config.get("input_type", "video"))
        self.acceleration_var = tk.StringVar(value=self._normalize_acceleration(self.base_config.get("use_cuda", "auto")))
        self.dialogue_mode_var = tk.StringVar(value=self.base_config.get("dialogue_mode", "segments"))
        self.include_timestamps_var = tk.BooleanVar(value=self.base_config.get("include_timestamps", True))

        self.status_var = tk.StringVar(value="Готов к работе")
        self.warning_var = tk.StringVar(value="")

        self._transcription_thread: Optional[threading.Thread] = None
        self._current_transcript: str = ""
        self._current_dialogues: List[Dict[str, str]] = []

        self._build_ui()

    # region UI helpers
    def _build_ui(self) -> None:
        main_frame = ttk.Frame(self.root, padding=15)
        main_frame.pack(fill=tk.BOTH, expand=True)

        file_frame = ttk.LabelFrame(main_frame, text="Файл")
        file_frame.pack(fill=tk.X, pady=(0, 10))

        file_entry = ttk.Entry(file_frame, textvariable=self.file_path_var)
        file_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(10, 5), pady=10)

        ttk.Button(file_frame, text="Выбрать", command=self._choose_file).pack(
            side=tk.LEFT, padx=(0, 10), pady=10
        )

        output_frame = ttk.LabelFrame(main_frame, text="Папка сохранения")
        output_frame.pack(fill=tk.X, pady=(0, 10))

        output_entry = ttk.Entry(output_frame, textvariable=self.output_dir_var)
        output_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(10, 5), pady=10)

        ttk.Button(output_frame, text="Изменить", command=self._choose_output_dir).pack(
            side=tk.LEFT, padx=(0, 10), pady=10
        )

        options_frame = ttk.Frame(main_frame)
        options_frame.pack(fill=tk.X, pady=(0, 10))

        self._add_labeled_option(
            options_frame,
            row=0,
            label="Тип входного файла",
            widget=self._create_option_menu(options_frame, self.input_type_var, self._INPUT_TYPES),
        )
        self._add_labeled_option(
            options_frame,
            row=1,
            label="Ускорение",
            widget=self._create_option_menu(options_frame, self.acceleration_var, self._ACCELERATION_OPTIONS),
        )
        self._add_labeled_option(
            options_frame,
            row=0,
            column=2,
            label="Размер модели",
            widget=self._create_option_menu(options_frame, self.model_var, [(value, value) for value in self._ALLOWED_MODELS]),
        )
        self._add_labeled_option(
            options_frame,
            row=1,
            column=2,
            label="Режим диалогов",
            widget=self._create_option_menu(options_frame, self.dialogue_mode_var, self._DIALOGUE_MODES),
        )

        timestamps_check = ttk.Checkbutton(
            main_frame,
            text="Добавлять временные метки",
            variable=self.include_timestamps_var,
        )
        timestamps_check.pack(anchor=tk.W, pady=(0, 10))

        buttons_frame = ttk.Frame(main_frame)
        buttons_frame.pack(fill=tk.X, pady=(0, 10))

        self.start_button = ttk.Button(buttons_frame, text="Запустить", command=self._on_start)
        self.start_button.pack(side=tk.LEFT)

        self.save_button = ttk.Button(buttons_frame, text="Сохранить результат", command=self._save_transcript)
        self.save_button.pack(side=tk.LEFT, padx=(10, 0))
        self.save_button.state(["disabled"])

        status_frame = ttk.Frame(main_frame)
        status_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(status_frame, text="Статус:").pack(side=tk.LEFT)
        ttk.Label(status_frame, textvariable=self.status_var).pack(side=tk.LEFT, padx=(5, 0))

        warning_label = ttk.Label(
            main_frame,
            textvariable=self.warning_var,
            foreground="#a15c00",
            wraplength=860,
        )
        warning_label.pack(fill=tk.X)

        notebook = ttk.Notebook(main_frame)
        notebook.pack(fill=tk.BOTH, expand=True, pady=(10, 0))

        transcript_frame = ttk.Frame(notebook)
        dialogues_frame = ttk.Frame(notebook)
        notebook.add(transcript_frame, text="Транскрипт")
        notebook.add(dialogues_frame, text="Диалоги")

        self.transcript_text = tk.Text(transcript_frame, wrap=tk.WORD)
        self.transcript_text.pack(fill=tk.BOTH, expand=True)

        self.dialogues_text = tk.Text(dialogues_frame, wrap=tk.WORD)
        self.dialogues_text.pack(fill=tk.BOTH, expand=True)

    def _add_labeled_option(self, parent: ttk.Frame, row: int, label: str, widget: ttk.Widget, column: int = 0) -> None:
        label_widget = ttk.Label(parent, text=label)
        label_widget.grid(row=row, column=column, sticky=tk.W, padx=10, pady=5)
        widget.grid(row=row, column=column + 1, sticky=tk.W + tk.E, padx=10, pady=5)
        parent.grid_columnconfigure(column + 1, weight=1)

    def _create_option_menu(
        self,
        parent: ttk.Frame,
        variable: tk.StringVar,
        options: List[tuple[str, str]],
    ) -> ttk.Combobox:
        values = [label for _, label in options]
        mapping = {label: value for value, label in options}

        combo = ttk.Combobox(parent, textvariable=tk.StringVar(), values=values, state="readonly")
        current_value = variable.get()
        for value, label in options:
            if value == current_value:
                combo.set(label)
                break
        else:
            combo.set(values[0])
            variable.set(options[0][0])

        def _on_select(event: tk.Event) -> None:  # pragma: no cover - UI callback
            selection = combo.get()
            if selection in mapping:
                variable.set(mapping[selection])

        combo.bind("<<ComboboxSelected>>", _on_select)
        return combo

    # endregion

    # region actions
    def _choose_file(self) -> None:
        if self.input_type_var.get() == "audio":
            filetypes = [
                ("Аудио файлы", "*.mp3 *.wav *.m4a *.flac"),
                ("Все файлы", "*.*"),
            ]
        else:
            filetypes = [
                ("Видео файлы", "*.mp4 *.mov *.mkv *.avi"),
                ("Все файлы", "*.*"),
            ]
        path = filedialog.askopenfilename(title="Выберите файл", filetypes=filetypes)
        if path:
            self.file_path_var.set(path)

    def _choose_output_dir(self) -> None:
        directory = filedialog.askdirectory(title="Выберите папку для сохранения")
        if directory:
            self.output_dir_var.set(directory)

    def _on_start(self) -> None:
        if self._transcription_thread and self._transcription_thread.is_alive():
            messagebox.showinfo("Выполняется", "Дождитесь завершения текущей транскрибации.")
            return

        file_path = self.file_path_var.get().strip()
        if not file_path:
            messagebox.showerror("Нет файла", "Пожалуйста, выберите файл для транскрибации.")
            return

        if not os.path.exists(file_path):
            messagebox.showerror("Файл не найден", "Выбранный файл не существует.")
            return

        self.status_var.set("Запуск транскрибации...")
        self.warning_var.set("")
        self.transcript_text.delete("1.0", tk.END)
        self.dialogues_text.delete("1.0", tk.END)
        self.save_button.state(["disabled"])
        self.start_button.state(["disabled"])

        thread_args = (
            file_path,
            self.input_type_var.get(),
            self.acceleration_var.get(),
            self.model_var.get(),
            self.dialogue_mode_var.get(),
            self.include_timestamps_var.get(),
            self.output_dir_var.get().strip() or None,
        )

        self._transcription_thread = threading.Thread(
            target=self._run_transcription,
            args=thread_args,
            daemon=True,
        )
        self._transcription_thread.start()

    def _run_transcription(
        self,
        file_path: str,
        input_type: str,
        acceleration: str,
        model_size: str,
        dialogue_mode: str,
        include_timestamps: bool,
        output_dir: Optional[str],
    ) -> None:
        try:
            config = dict(self.base_config)
            config["input_type"] = input_type
            config["model_size"] = self._sanitize_model(model_size)
            config["dialogue_mode"] = dialogue_mode
            config["include_timestamps"] = include_timestamps

            if output_dir:
                config["save_path"] = output_dir

            if acceleration == "gpu":
                config["use_cuda"] = "cuda"
            elif acceleration == "cpu":
                config["use_cuda"] = "cpu"
            else:
                config["use_cuda"] = "auto"

            transcript, segments, warnings = transcribe_video_with_segments(file_path, config)
            dialogues = prepare_dialogues(segments, dialogue_mode)

            self.root.after(
                0,
                lambda: self._on_transcription_complete(transcript, dialogues, warnings),
            )
        except Exception as exc:  # pragma: no cover - защитный блок
            self.root.after(
                0,
                lambda: self._on_transcription_error(str(exc)),
            )

    def _on_transcription_complete(
        self,
        transcript: str,
        dialogues: List[Dict[str, str]],
        warnings: List[str],
    ) -> None:
        self.start_button.state(["!disabled"])

        if transcript.startswith("Ошибка") or transcript.startswith("Произошла ошибка"):
            self.status_var.set("Произошла ошибка")
            messagebox.showerror("Ошибка", transcript)
            return

        self.status_var.set("Готово")
        self._current_transcript = transcript
        self._current_dialogues = dialogues

        self.transcript_text.insert(tk.END, transcript)

        if dialogues:
            formatted = []
            for entry in dialogues:
                speaker = entry.get("speaker")
                prefix = f"{speaker}: " if speaker else ""
                formatted.append(
                    f"[{entry.get('start', '')} – {entry.get('end', '')}] {prefix}{entry.get('text', '')}"
                )
            self.dialogues_text.insert(tk.END, "\n".join(formatted))
        else:
            self.dialogues_text.insert(tk.END, "Диалоги отсутствуют.")

        if warnings:
            unique_warnings = []
            seen = set()
            for warning in warnings:
                if warning not in seen:
                    unique_warnings.append(warning)
                    seen.add(warning)
            self.warning_var.set("\n".join(unique_warnings))
        else:
            self.warning_var.set("")

        self.save_button.state(["!disabled"])

    def _on_transcription_error(self, message: str) -> None:
        self.start_button.state(["!disabled"])
        self.status_var.set("Ошибка")
        self.warning_var.set(message)
        messagebox.showerror("Ошибка", message)

    def _save_transcript(self) -> None:
        if not self._current_transcript:
            messagebox.showinfo("Нет данных", "Сначала выполните транскрибацию.")
            return

        default_dir = self.output_dir_var.get().strip() or os.getcwd()
        initial_dir = os.path.abspath(default_dir)
        os.makedirs(initial_dir, exist_ok=True)

        file_path = filedialog.asksaveasfilename(
            title="Сохранить транскрипт",
            defaultextension=".txt",
            initialdir=initial_dir,
            filetypes=[("Текстовые файлы", "*.txt"), ("Все файлы", "*.*")],
        )
        if not file_path:
            return

        try:
            with open(file_path, "w", encoding="utf-8") as handle:
                handle.write(self._current_transcript)
            messagebox.showinfo("Сохранено", f"Файл сохранен: {file_path}")
        except OSError as exc:  # pragma: no cover - файловые ошибки
            messagebox.showerror("Ошибка", f"Не удалось сохранить файл: {exc}")

    # endregion

    # region utils
    def _sanitize_model(self, candidate: str) -> str:
        candidate = (candidate or "auto").lower()
        if candidate not in self._ALLOWED_MODELS:
            return "auto"
        return candidate

    def _normalize_acceleration(self, value) -> str:
        if isinstance(value, str):
            normalized = value.lower()
        elif value:
            normalized = "gpu"
        else:
            normalized = "cpu"

        if normalized in {"gpu", "cuda"}:
            return "gpu"
        if normalized in {"cpu"}:
            return "cpu"
        return "auto"

    # endregion


def main() -> None:
    root = tk.Tk()
    DesktopTranscriberApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
