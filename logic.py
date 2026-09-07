"""
Spam Detection - Machine Learning Logic

This module contains all data loading, validation, preprocessing,
model training, prediction, and evaluation functionality.
"""

from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split


SUPPORTED_EXTENSIONS = {".csv", ".xls", ".xlsx"}
RANDOM_STATE = 42
TEST_SIZE = 0.20


class DatasetProcessingError(Exception):
    """Raised when a dataset cannot be processed safely."""


def load_dataset(file_path: str | Path) -> pd.DataFrame:
    """Load a CSV/XLS/XLSX file and validate its basic structure."""
    path = Path(file_path)

    if not path.exists() or not path.is_file():
        raise DatasetProcessingError("The selected file does not exist or cannot be accessed.")

    extension = path.suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise DatasetProcessingError(
            f"Unsupported file type '{extension or 'unknown'}'. "
            f"Please select one of: {supported}."
        )

    try:
        if extension == ".csv":
            dataframe = pd.read_csv(path)
        else:
            excel_file = pd.ExcelFile(path)
            if len(excel_file.sheet_names) != 1:
                raise DatasetProcessingError(
                    "Excel files must contain exactly one worksheet. "
                    f"This workbook contains {len(excel_file.sheet_names)} worksheets."
                )
            dataframe = pd.read_excel(excel_file, sheet_name=excel_file.sheet_names[0])
    except DatasetProcessingError:
        raise
    except Exception as exc:
        raise DatasetProcessingError(
            f"Unable to read the selected file. Please check that it is a valid "
            f"CSV/Excel dataset.\nDetails: {exc}"
        ) from exc

    if dataframe.empty:
        raise DatasetProcessingError("The dataset is empty. Please select a dataset containing data.")

    if dataframe.shape[1] != 2:
        raise DatasetProcessingError(
            f"The dataset must contain exactly two columns, but {dataframe.shape[1]} "
            "columns were found."
        )

    return dataframe


def _clean_series(series: pd.Series) -> pd.Series:
    """Convert values to strings, strip whitespace, and lowercase."""
    return series.astype("string").str.strip().str.lower()


def detect_columns(dataframe: pd.DataFrame) -> tuple[str, str]:
    """
    Detect the binary target and text columns.

    Returns:
        (text_column, target_column)
    """
    if dataframe.shape[1] != 2:
        raise DatasetProcessingError(
            "Column detection requires a dataset containing exactly two columns."
        )

    unique_counts = {}
    for column in dataframe.columns:
        cleaned = _clean_series(dataframe[column]).replace("", pd.NA).dropna()
        unique_counts[column] = cleaned.nunique()

    binary_columns = [column for column, count in unique_counts.items() if count == 2]

    if len(binary_columns) != 1:
        if len(binary_columns) == 0:
            details = ", ".join(
                f"'{column}': {count} unique values"
                for column, count in unique_counts.items()
            )
            raise DatasetProcessingError(
                "Could not identify a suitable binary category/target column. "
                "Exactly one column must contain two unique values. "
                f"Detected counts: {details}."
            )

        raise DatasetProcessingError(
            "Both columns appear to contain exactly two unique values. "
            "The application cannot determine which column is the category/target column automatically."
        )

    target_column = binary_columns[0]
    text_column = next(column for column in dataframe.columns if column != target_column)
    return text_column, target_column


def preprocess_dataset(
    dataframe: pd.DataFrame,
    text_column: str,
    target_column: str,
) -> tuple[pd.Series, pd.Series]:
    """Clean message text and binary target values."""
    text = _clean_series(dataframe[text_column])
    target = _clean_series(dataframe[target_column])

    cleaned = pd.DataFrame({"text": text, "target": target})
    cleaned = cleaned.replace({"": pd.NA}).dropna(subset=["text", "target"])

    if cleaned.empty:
        raise DatasetProcessingError(
            "No valid rows remain after removing missing or blank values."
        )

    if cleaned["target"].nunique() != 2:
        raise DatasetProcessingError(
            "The category/target column must contain exactly two valid classes "
            "after preprocessing."
        )

    class_counts = cleaned["target"].value_counts()
    if class_counts.min() < 2:
        raise DatasetProcessingError(
            "Each category must contain at least two valid records so the data "
            "can be split into training and testing sets."
        )

    return cleaned["text"], cleaned["target"]


def train_and_evaluate(
    texts: pd.Series,
    targets: pd.Series,
    *,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> dict[str, Any]:
    """Train CountVectorizer + Logistic Regression and return evaluation results."""
    if len(texts) < 4:
        raise DatasetProcessingError(
            "The dataset contains too few valid records for machine-learning evaluation."
        )

    if targets.nunique() != 2:
        raise DatasetProcessingError("Machine learning requires exactly two target classes.")

    try:
        x_train, x_test, y_train, y_test = train_test_split(
            texts,
            targets,
            test_size=test_size,
            random_state=random_state,
            stratify=targets,
        )
    except ValueError as exc:
        raise DatasetProcessingError(
            "There is insufficient data in one or more categories for a reliable "
            f"train/test split. {exc}"
        ) from exc

    vectorizer = CountVectorizer(
        lowercase=False,
        strip_accents="unicode",
        token_pattern=r"(?u)\b\w+\b",
    )

    try:
        x_train_vectorized = vectorizer.fit_transform(x_train)
        x_test_vectorized = vectorizer.transform(x_test)

        if x_train_vectorized.shape[1] == 0:
            raise DatasetProcessingError(
                "No usable text features were generated from the training data."
            )

        model = LogisticRegression(
            max_iter=1000,
            random_state=random_state,
        )
        model.fit(x_train_vectorized, y_train)
        predictions = model.predict(x_test_vectorized)
    except DatasetProcessingError:
        raise
    except Exception as exc:
        raise DatasetProcessingError(
            f"Machine-learning processing failed. Details: {exc}"
        ) from exc

    labels = list(model.classes_)
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test, predictions, average="binary", pos_label=labels[1], zero_division=0
    )
    recall = recall_score(
        y_test, predictions, average="binary", pos_label=labels[1], zero_division=0
    )
    f1 = f1_score(
        y_test, predictions, average="binary", pos_label=labels[1], zero_division=0
    )
    matrix = confusion_matrix(y_test, predictions, labels=labels)

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "confusion_matrix": matrix.tolist(),
        "labels": labels,
        "total_rows": int(len(texts)),
        "training_rows": int(len(x_train)),
        "testing_rows": int(len(x_test)),
        "text_column": str(texts.name) if texts.name is not None else "text",
        "target_column": str(targets.name) if targets.name is not None else "target",
        "model": model,
        "vectorizer": vectorizer,
    }


def process_dataset(file_path: str | Path) -> dict[str, Any]:
    """Run the complete end-to-end spam detection workflow."""
    dataframe = load_dataset(file_path)

    # Keep original column names while normalizing their contents.
    text_column, target_column = detect_columns(dataframe)

    texts, targets = preprocess_dataset(
        dataframe,
        text_column,
        target_column,
    )

    result = train_and_evaluate(texts, targets)
    result["text_column"] = text_column
    result["target_column"] = target_column
    return result
