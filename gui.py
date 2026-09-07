"""
Spam Detection - PyQt5 GUI

Single-page desktop interface for processing a two-column spam dataset.
"""

import sys
from pathlib import Path

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QApplication,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from logic import DatasetProcessingError, process_dataset


APP_STYLE = """
QMainWindow, QWidget {
    background: #0f172a;
    color: #e2e8f0;
    font-family: "Segoe UI", Arial, sans-serif;
}

QFrame#header {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                stop:0 #1e3a8a, stop:1 #172554);
    border-radius: 16px;
}

QLabel#title {
    color: #f0f9ff;
    font-size: 27px;
    font-weight: 700;
}

QLabel#subtitle {
    color: #93c5fd;
    font-size: 13px;
}

QFrame#panel, QFrame#metric, QFrame#matrix {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 14px;
}

QLabel#sectionTitle {
    font-size: 17px;
    font-weight: 700;
    color: #f1f5f9;
}

QLabel#muted {
    color: #94a3b8;
    font-size: 12px;
}

QLabel#filePath {
    background: #0f172a;
    border: 1px solid #334155;
    border-radius: 9px;
    padding: 11px;
    color: #cbd5e1;
}

QPushButton#browse {
    background: #1e293b;
    color: #cbd5e1;
    border: 1px solid #475569;
    border-radius: 9px;
    padding: 11px 18px;
    font-weight: 600;
}

QPushButton#browse:hover {
    background: #263347;
    border-color: #60a5fa;
    color: #f0f9ff;
}

QPushButton#process {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                stop:0 #2563eb, stop:1 #1d4ed8);
    color: white;
    border: none;
    border-radius: 9px;
    padding: 12px 22px;
    font-weight: 700;
}

QPushButton#process:hover {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                stop:0 #3b82f6, stop:1 #2563eb);
}

QPushButton#process:disabled {
    background: #334155;
    color: #64748b;
}

QProgressBar {
    border: none;
    border-radius: 5px;
    background: #1e293b;
    height: 8px;
    text-align: center;
}

QProgressBar::chunk {
    border-radius: 5px;
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                stop:0 #3b82f6, stop:1 #60a5fa);
}

QLabel#metricValue {
    font-size: 23px;
    font-weight: 700;
    color: #60a5fa;
}

QLabel#metricName {
    color: #94a3b8;
    font-size: 11px;
    font-weight: 600;
}

QLabel#matrixValue {
    font-size: 20px;
    font-weight: 700;
    padding: 8px;
    border: 1px solid #334155;
    border-radius: 8px;
    background: #0f172a;
    color: #e2e8f0;
}

QLabel#status {
    color: #94a3b8;
    font-size: 12px;
}
"""


class ProcessingThread(QThread):
    finished = pyqtSignal(object)
    failed = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(self, file_path: str):
        super().__init__()
        self.file_path = file_path

    def run(self):
        try:
            self.progress.emit("Loading and validating dataset…")
            result = process_dataset(self.file_path)
            self.progress.emit("Training model and calculating evaluation metrics…")
            self.finished.emit(result)
        except DatasetProcessingError as exc:
            self.failed.emit(str(exc))
        except Exception as exc:
            self.failed.emit(
                "An unexpected processing error occurred. "
                f"Please verify the dataset and try again.\n\nDetails: {exc}"
            )


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.selected_file = None
        self.worker = None
        self.setWindowTitle("Spam Detection • Machine Learning")
        self.setMinimumSize(900, 720)
        self.resize(980, 820)
        self._build_ui()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)

        outer = QVBoxLayout(central)
        outer.setContentsMargins(28, 26, 28, 26)
        outer.setSpacing(18)

        header = QFrame()
        header.setObjectName("header")
        header_layout = QVBoxLayout(header)
        header_layout.setContentsMargins(24, 20, 24, 20)
        header_layout.setSpacing(5)

        title = QLabel("Spam Detection")
        title.setObjectName("title")
        subtitle = QLabel(
            "Machine-learning classification with CountVectorizer + Logistic Regression"
        )
        subtitle.setObjectName("subtitle")

        header_layout.addWidget(title)
        header_layout.addWidget(subtitle)
        outer.addWidget(header)

        input_panel = QFrame()
        input_panel.setObjectName("panel")
        input_layout = QVBoxLayout(input_panel)
        input_layout.setContentsMargins(20, 18, 20, 18)
        input_layout.setSpacing(12)

        section = QLabel("Dataset")
        section.setObjectName("sectionTitle")
        hint = QLabel("Select a CSV, XLS, or XLSX file containing exactly two columns.")
        hint.setObjectName("muted")

        file_row = QHBoxLayout()
        file_row.setSpacing(10)
        self.file_path_label = QLabel("No dataset selected")
        self.file_path_label.setObjectName("filePath")
        self.file_path_label.setMinimumHeight(42)
        self.file_path_label.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)
        self.file_path_label.setTextInteractionFlags(Qt.TextSelectableByMouse)

        browse = QPushButton("Choose Dataset")
        browse.setObjectName("browse")
        browse.setCursor(Qt.PointingHandCursor)
        browse.clicked.connect(self.select_file)

        file_row.addWidget(self.file_path_label, 1)
        file_row.addWidget(browse)

        action_row = QHBoxLayout()
        action_row.addStretch()
        self.process_button = QPushButton("Process Dataset")
        self.process_button.setObjectName("process")
        self.process_button.setCursor(Qt.PointingHandCursor)
        self.process_button.clicked.connect(self.process_file)
        self.process_button.setEnabled(False)
        action_row.addWidget(self.process_button)

        input_layout.addWidget(section)
        input_layout.addWidget(hint)
        input_layout.addLayout(file_row)
        input_layout.addLayout(action_row)
        outer.addWidget(input_panel)

        status_panel = QFrame()
        status_panel.setObjectName("panel")
        status_layout = QVBoxLayout(status_panel)
        status_layout.setContentsMargins(20, 15, 20, 15)
        status_layout.setSpacing(8)

        self.status_label = QLabel("Ready — select a dataset to begin.")
        self.status_label.setObjectName("status")
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.hide()

        status_layout.addWidget(self.status_label)
        status_layout.addWidget(self.progress_bar)
        outer.addWidget(status_panel)

        results_title = QLabel("Evaluation Results")
        results_title.setObjectName("sectionTitle")
        outer.addWidget(results_title)

        grid = QGridLayout()
        grid.setSpacing(12)

        self.metric_labels = {}
        metrics = [
            ("accuracy", "Accuracy"),
            ("precision", "Precision"),
            ("recall", "Recall"),
            ("f1_score", "F1 Score"),
        ]

        for index, (key, label_text) in enumerate(metrics):
            card = QFrame()
            card.setObjectName("metric")
            card_layout = QVBoxLayout(card)
            card_layout.setContentsMargins(17, 15, 17, 15)
            card_layout.setSpacing(4)

            value = QLabel("—")
            value.setObjectName("metricValue")
            name = QLabel(label_text)
            name.setObjectName("metricName")

            card_layout.addWidget(value)
            card_layout.addWidget(name)
            self.metric_labels[key] = value
            grid.addWidget(card, 0, index)

        outer.addLayout(grid)

        matrix_panel = QFrame()
        matrix_panel.setObjectName("matrix")
        matrix_layout = QVBoxLayout(matrix_panel)
        matrix_layout.setContentsMargins(18, 16, 18, 18)
        matrix_layout.setSpacing(10)

        matrix_heading = QLabel("Confusion Matrix")
        matrix_heading.setObjectName("sectionTitle")
        self.matrix_info = QLabel("Actual vs predicted class counts")
        self.matrix_info.setObjectName("muted")

        matrix_layout.addWidget(matrix_heading)
        matrix_layout.addWidget(self.matrix_info)

        self.matrix_grid = QGridLayout()
        self.matrix_grid.setSpacing(8)
        matrix_layout.addLayout(self.matrix_grid)

        outer.addWidget(matrix_panel, 1)

        self.detail_label = QLabel("No results yet.")
        self.detail_label.setObjectName("muted")
        outer.addWidget(self.detail_label)

    def select_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Spam Detection Dataset",
            "",
            "Supported Files (*.csv *.xls *.xlsx);;CSV Files (*.csv);;Excel Files (*.xls *.xlsx)",
        )
        if not file_path:
            return

        self.selected_file = file_path
        self.file_path_label.setText(file_path)
        self.status_label.setText("Dataset selected — ready to process.")
        self.process_button.setEnabled(True)

    def process_file(self):
        if not self.selected_file:
            QMessageBox.warning(self, "Dataset Required", "Please select a dataset first.")
            return

        self.process_button.setEnabled(False)
        self.status_label.setText("Starting dataset processing…")
        self.progress_bar.show()

        self.worker = ProcessingThread(self.selected_file)
        self.worker.progress.connect(self.status_label.setText)
        self.worker.finished.connect(self.on_processing_finished)
        self.worker.failed.connect(self.on_processing_failed)
        self.worker.start()

    def on_processing_finished(self, result):
        self.progress_bar.hide()
        self.process_button.setEnabled(True)

        for key in self.metric_labels:
            self.metric_labels[key].setText(f"{result[key] * 100:.2f}%")

        self._display_confusion_matrix(result["confusion_matrix"], result["labels"])

        self.status_label.setText("Processing completed successfully.")
        self.detail_label.setText(
            f"Processed {result['total_rows']:,} valid rows • "
            f"Training: {result['training_rows']:,} • "
            f"Testing: {result['testing_rows']:,} • "
            f"Text: {result['text_column']} • Target: {result['target_column']}"
        )

    def on_processing_failed(self, message: str):
        self.progress_bar.hide()
        self.process_button.setEnabled(True)
        self.status_label.setText("Processing failed — please review the error message.")
        QMessageBox.critical(self, "Unable to Process Dataset", message)

    def _clear_matrix(self):
        while self.matrix_grid.count():
            item = self.matrix_grid.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def _display_confusion_matrix(self, matrix, labels):
        self._clear_matrix()

        self.matrix_grid.addWidget(QLabel("Actual \\ Predicted"), 0, 0)
        for column, label in enumerate(labels, start=1):
            header = QLabel(str(label).upper())
            header.setAlignment(Qt.AlignCenter)
            header.setFont(QFont("Segoe UI", 10, QFont.Bold))
            self.matrix_grid.addWidget(header, 0, column)

        for row, label in enumerate(labels, start=1):
            actual = QLabel(str(label).upper())
            actual.setAlignment(Qt.AlignCenter)
            actual.setFont(QFont("Segoe UI", 10, QFont.Bold))
            self.matrix_grid.addWidget(actual, row, 0)

            for column, value in enumerate(matrix[row - 1], start=1):
                cell = QLabel(str(value))
                cell.setObjectName("matrixValue")
                cell.setAlignment(Qt.AlignCenter)
                self.matrix_grid.addWidget(cell, row, column)


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Spam Detection")
    app.setStyleSheet(APP_STYLE)

    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
