import os
import sys
from typing import List, Dict

from PyQt6.QtCore import QObject, Qt, QThread, pyqtSignal, pyqtSlot
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QComboBox,
    QWidget,
    QPlainTextEdit,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QSplitter,
    QSizePolicy,
    QVBoxLayout,
)

from transcribe_video import (
    load_config,
    prepare_dialogues,
    transcribe_video_with_segments,
)


class TranscriptionWorker(QObject):
    progress = pyqtSignal(float, str, bool)
    completed = pyqtSignal(str, list, list)
    failed = pyqtSignal(str, list)

    def __init__(
        self,
        file_path: str,
        media_type: str,
        acceleration: str,
        model_size: str,
        dialogue_mode: str,
    ) -> None:
        super().__init__()
        self.file_path = file_path
        self.media_type = media_type
        self.acceleration = acceleration
        self.model_size = model_size
        self.dialogue_mode = dialogue_mode

    @pyqtSlot()
    def run(self) -> None:
        config = load_config()
        config["verbose"] = False
        config["input_type"] = self.media_type
        config["dialogue_mode"] = self.dialogue_mode
        config["save_path"] = config.get("save_path") or "./"
        if self.acceleration == "gpu":
            config["use_cuda"] = "cuda"
        elif self.acceleration == "cpu":
            config["use_cuda"] = "cpu"
        else:
            config["use_cuda"] = "auto"

        if self.model_size:
            config["model_size"] = self.model_size

        try:
            text, segments, warnings = transcribe_video_with_segments(
                self.file_path,
                config,
                progress_callback=self._handle_progress,
            )

            if text.startswith("Ошибка") or text.startswith("Произошла ошибка"):
                self.failed.emit(text, warnings)
                return

            dialogues = prepare_dialogues(segments, self.dialogue_mode)
            self.completed.emit(text, dialogues, warnings)
        except Exception as exc:  # pragma: no cover - предохранительный блок
            message = f"Произошла ошибка: {exc}"
            self.failed.emit(message, [str(exc)])

    def _handle_progress(self, value: float, stage: str, indeterminate: bool = False) -> None:
        self.progress.emit(value, stage, indeterminate)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Video Translater Desktop")
        self.resize(1180, 760)
        self.worker_thread: QThread | None = None
        self.worker: TranscriptionWorker | None = None

        self.config = load_config()

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(24, 24, 24, 24)
        main_layout.setSpacing(18)

        header = QLabel("<h1 style='margin:0'>Видео → Текст с диалогами</h1>")
        subtitle = QLabel(
            "Выберите файл, модель Whisper и параметры диалогов. Прогресс обработки будет отображаться ниже."
        )
        subtitle.setWordWrap(True)

        main_layout.addWidget(header)
        main_layout.addWidget(subtitle)

        controls_group = QGroupBox("Параметры")
        controls_group_layout = QGridLayout(controls_group)
        controls_group_layout.setHorizontalSpacing(16)
        controls_group_layout.setVerticalSpacing(12)

        # File picker
        self.file_label = QLabel("Файл не выбран")
        self.file_label.setObjectName("fileLabel")
        self.file_label.setStyleSheet("color: #556; font-weight: 500;")
        pick_button = QPushButton("Выбрать файл…")
        pick_button.clicked.connect(self.pick_file)

        controls_group_layout.addWidget(QLabel("Файл"), 0, 0)
        file_layout = QHBoxLayout()
        file_layout.addWidget(self.file_label, 1)
        file_layout.addWidget(pick_button, 0)
        controls_group_layout.addLayout(file_layout, 0, 1)

        # Media type
        self.media_combo = QComboBox()
        self.media_combo.addItem("Видео", "video")
        self.media_combo.addItem("Аудио", "audio")
        if self.config.get("input_type") == "audio":
            self.media_combo.setCurrentIndex(1)
        controls_group_layout.addWidget(QLabel("Тип входного файла"), 1, 0)
        controls_group_layout.addWidget(self.media_combo, 1, 1)

        # Model size
        self.model_combo = QComboBox()
        models = [
            ("Авто (рекомендуется)", "auto"),
            ("tiny", "tiny"),
            ("base", "base"),
            ("small", "small"),
            ("medium", "medium"),
            ("large-v2", "large-v2"),
            ("large-v3", "large-v3"),
        ]
        for label, value in models:
            self.model_combo.addItem(label, value)
        current_model = self.config.get("model_size", "auto")
        index = max(0, self.model_combo.findData(current_model))
        self.model_combo.setCurrentIndex(index)
        controls_group_layout.addWidget(QLabel("Модель Whisper"), 2, 0)
        controls_group_layout.addWidget(self.model_combo, 2, 1)

        # Acceleration
        self.acceleration_combo = QComboBox()
        self.acceleration_combo.addItem("Авто", "auto")
        self.acceleration_combo.addItem("Только GPU", "gpu")
        self.acceleration_combo.addItem("Только CPU", "cpu")
        default_acceleration = self._normalize_acceleration(self.config.get("use_cuda", "auto"))
        index = max(0, self.acceleration_combo.findData(default_acceleration))
        self.acceleration_combo.setCurrentIndex(index)
        controls_group_layout.addWidget(QLabel("Ускорение"), 3, 0)
        controls_group_layout.addWidget(self.acceleration_combo, 3, 1)

        # Dialogue mode
        self.dialogue_combo = QComboBox()
        dialogue_options = [
            ("По фразам", "segments"),
            ("По голосам (beta)", "speakers"),
            ("Разбивать на цитаты", "quotes"),
        ]
        for label, value in dialogue_options:
            self.dialogue_combo.addItem(label, value)
        default_dialogue = self.config.get("dialogue_mode", "segments")
        index = max(0, self.dialogue_combo.findData(default_dialogue))
        self.dialogue_combo.setCurrentIndex(index)
        controls_group_layout.addWidget(QLabel("Формат диалога"), 4, 0)
        controls_group_layout.addWidget(self.dialogue_combo, 4, 1)

        main_layout.addWidget(controls_group)

        # Progress section
        progress_frame = QGroupBox("Прогресс")
        progress_layout = QVBoxLayout(progress_frame)
        progress_layout.setContentsMargins(16, 16, 16, 16)
        progress_layout.setSpacing(12)

        self.status_label = QLabel("Готово к обработке")
        self.status_label.setWordWrap(True)
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)

        progress_layout.addWidget(self.status_label)
        progress_layout.addWidget(self.progress_bar)

        self.start_button = QPushButton("Начать транскрибацию")
        self.start_button.setDefault(True)
        self.start_button.clicked.connect(self.start_transcription)
        progress_layout.addWidget(self.start_button)

        self.warning_label = QLabel()
        self.warning_label.setWordWrap(True)
        self.warning_label.setStyleSheet("color: #c47a00; font-weight: 500;")
        self.warning_label.hide()
        progress_layout.addWidget(self.warning_label)

        main_layout.addWidget(progress_frame)

        # Results section
        results_splitter = QSplitter(Qt.Orientation.Vertical)

        # Dialogues table
        self.dialogue_table = QTableWidget(0, 4)
        self.dialogue_table.setHorizontalHeaderLabels(["Начало", "Конец", "Спикер", "Фраза"])
        header = self.dialogue_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        self.dialogue_table.verticalHeader().setVisible(False)
        self.dialogue_table.setAlternatingRowColors(True)
        self.dialogue_table.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self.transcript_view = QPlainTextEdit()
        self.transcript_view.setPlaceholderText("Здесь появится полный текст расшифровки")
        self.transcript_view.setReadOnly(True)
        self.transcript_view.setStyleSheet("font-family: 'SFMono-Regular', 'JetBrains Mono', monospace; font-size: 13px;")

        results_splitter.addWidget(self.dialogue_table)
        results_splitter.addWidget(self.transcript_view)
        results_splitter.setSizes([350, 250])

        main_layout.addWidget(results_splitter, 1)

        self._apply_palette()

    def _apply_palette(self) -> None:
        self.setStyleSheet(
            """
            QWidget {
                background-color: #f4f6fb;
            }
            QGroupBox {
                background-color: #ffffff;
                border: 1px solid #d6deff;
                border-radius: 12px;
                margin-top: 18px;
                font-weight: 600;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 10px;
                color: #2c3e50;
            }
            QLabel {
                color: #2c3e50;
            }
            QPushButton {
                background-color: #4a6cf7;
                border-radius: 8px;
                padding: 10px 16px;
                color: #ffffff;
                font-weight: 600;
            }
            QPushButton:disabled {
                background-color: #aab4dd;
                color: #f5f5f5;
            }
            QPushButton:hover:!disabled {
                background-color: #3f5fd0;
            }
            QProgressBar {
                background-color: #e6ebff;
                border: 1px solid #c6d2ff;
                border-radius: 6px;
                text-align: center;
                height: 24px;
            }
            QProgressBar::chunk {
                background-color: #4a6cf7;
                border-radius: 6px;
            }
            QComboBox {
                border: 1px solid #c6d2ff;
                border-radius: 6px;
                padding: 6px;
                background-color: #fdfdff;
            }
            QTableWidget {
                background: #ffffff;
                border: 1px solid #d6deff;
                border-radius: 12px;
                gridline-color: #dce3ff;
                selection-background-color: #dfe6ff;
            }
            QTableWidget::item {
                padding: 6px;
            }
            QPlainTextEdit {
                border: 1px solid #d6deff;
                border-radius: 12px;
                background: #ffffff;
            }
            """
        )

    def _normalize_acceleration(self, value) -> str:
        if isinstance(value, str):
            normalized = value.lower()
        else:
            normalized = value
        if normalized in ("gpu", "cuda", True):
            return "gpu"
        if normalized in ("cpu", False):
            return "cpu"
        return "auto"

    def pick_file(self) -> None:
        caption = "Выберите видео или аудио файл"
        filters = "Медиа файлы (*.mp4 *.mkv *.mov *.avi *.mp3 *.wav *.flac *.m4a);;Все файлы (*)"
        file_path, _ = QFileDialog.getOpenFileName(self, caption, os.path.expanduser("~"), filters)
        if file_path:
            self.file_label.setText(os.path.basename(file_path))
            self.file_label.setToolTip(file_path)
            self.selected_path = file_path
        else:
            self.selected_path = None
            self.file_label.setText("Файл не выбран")
            self.file_label.setToolTip("")

    def start_transcription(self) -> None:
        file_path = getattr(self, "selected_path", None)
        if not file_path:
            QMessageBox.warning(self, "Файл не выбран", "Пожалуйста, выберите видео или аудиофайл для обработки.")
            return

        if self.worker_thread:
            QMessageBox.information(self, "Обработка", "Транскрибация уже выполняется. Подождите завершения.")
            return

        self.progress_bar.setValue(0)
        self.progress_bar.setMaximum(100)
        self.status_label.setText("Подготовка…")
        self.warning_label.hide()
        self.start_button.setEnabled(False)

        media_type = self.media_combo.currentData()
        acceleration = self.acceleration_combo.currentData()
        model_size = self.model_combo.currentData()
        dialogue_mode = self.dialogue_combo.currentData()

        self.worker = TranscriptionWorker(
            file_path=file_path,
            media_type=media_type,
            acceleration=acceleration,
            model_size=model_size,
            dialogue_mode=dialogue_mode,
        )
        self.worker_thread = QThread()
        self.worker.moveToThread(self.worker_thread)
        self.worker_thread.started.connect(self.worker.run)
        self.worker.progress.connect(self.update_progress)
        self.worker.completed.connect(self.handle_completed)
        self.worker.failed.connect(self.handle_failed)
        self.worker.completed.connect(self.cleanup_worker)
        self.worker.failed.connect(self.cleanup_worker)
        self.worker_thread.start()

    @pyqtSlot(float, str, bool)
    def update_progress(self, value: float, stage: str, indeterminate: bool) -> None:
        if indeterminate:
            self.progress_bar.setRange(0, 0)
        else:
            if self.progress_bar.maximum() == 0:
                self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(int(value))
        if stage:
            self.status_label.setText(stage)

    @pyqtSlot(str, list, list)
    def handle_completed(self, transcript: str, dialogues: List[Dict], warnings: List[str]) -> None:
        self.start_button.setEnabled(True)
        self.status_label.setText("Готово")
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(100)

        self.transcript_view.setPlainText(transcript)
        self.populate_table(dialogues)
        self._show_warnings(warnings)

    @pyqtSlot(str, list)
    def handle_failed(self, message: str, warnings: List[str]) -> None:
        self.start_button.setEnabled(True)
        self.status_label.setText("Ошибка")
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.transcript_view.setPlainText("")
        self.dialogue_table.setRowCount(0)
        self._show_warnings(warnings)
        QMessageBox.critical(self, "Ошибка", message)

    def populate_table(self, dialogues: List[Dict]) -> None:
        self.dialogue_table.setRowCount(len(dialogues))
        for row, entry in enumerate(dialogues):
            self.dialogue_table.setItem(row, 0, QTableWidgetItem(entry.get("start", "")))
            self.dialogue_table.setItem(row, 1, QTableWidgetItem(entry.get("end", "")))
            self.dialogue_table.setItem(row, 2, QTableWidgetItem(entry.get("speaker", "")))
            self.dialogue_table.setItem(row, 3, QTableWidgetItem(entry.get("text", "")))
        self.dialogue_table.resizeRowsToContents()

    def _show_warnings(self, warnings: List[str]) -> None:
        unique = []
        seen = set()
        for warning in warnings:
            if warning and warning not in seen:
                unique.append(warning)
                seen.add(warning)
        if unique:
            self.warning_label.setText("\n".join(f"⚠️ {w}" for w in unique))
            self.warning_label.show()
        else:
            self.warning_label.hide()

    @pyqtSlot()
    def cleanup_worker(self) -> None:
        if self.worker_thread:
            self.worker_thread.quit()
            self.worker_thread.wait()
            self.worker_thread = None
            self.worker = None


def main() -> int:
    os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "1")
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
