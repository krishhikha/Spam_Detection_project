# 📩 Spam Detection Machine Learning

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![PyQt5](https://img.shields.io/badge/GUI-PyQt5-41CD52?logo=qt)
![scikit--learn](https://img.shields.io/badge/ML-scikit--learn-orange?logo=scikit-learn)
![License](https://img.shields.io/badge/License-MIT-green)

A clean, beginner-friendly desktop application for detecting spam messages using **CountVectorizer** and **Logistic Regression**. The application automatically identifies the binary target column in a two-column dataset and provides model evaluation results through a polished PyQt5 interface.

## ✨ Project Overview

This project accepts CSV, XLS, and XLSX datasets containing exactly two columns:

- One text/message column.
- One binary category/label column containing exactly two unique values.

The system automatically detects which column is the target, preprocesses the data, trains a reproducible Logistic Regression classifier, and displays Accuracy, Precision, Recall, F1 Score, and a Confusion Matrix.

## 🚀 Key Features

- Modern single-page PyQt5 desktop UI.
- CSV, XLS, and XLSX support.
- Excel workbook validation: exactly one worksheet.
- Automatic binary target-column detection.
- Whitespace stripping and lowercase normalization.
- Missing/invalid value handling.
- CountVectorizer NLP feature extraction.
- Logistic Regression classification.
- Reproducible train/test split.
- Professional metric cards and confusion-matrix display.
- User-friendly validation and processing errors.
- ML logic fully separated from GUI code.

## 🧠 How It Works

```text
Dataset
   ↓
File validation
   ↓
Load CSV / Excel
   ↓
Validate exactly 2 columns
   ↓
Clean text + labels
   ↓
Detect binary target column
   ↓
Train / test split
   ↓
CountVectorizer
   ↓
Logistic Regression
   ↓
Predictions
   ↓
Accuracy / Precision / Recall / F1
   ↓
Confusion Matrix
```

## 📂 Supported Input Files

| Format | Supported |
|---|---|
| `.csv` | ✅ |
| `.xls` | ✅ |
| `.xlsx` | ✅ |

Excel files must contain **exactly one worksheet**.

## 📋 Dataset Requirements

The dataset must contain exactly two columns. Column names can be anything; they are detected automatically.

Example:

| message | label |
|---|---|
| Congratulations, you won a prize | spam |
| Are we meeting at 5 today? | ham |
| Claim your free reward now | spam |

The application looks for a column with **exactly two unique non-empty values** and uses it as the target.

### Important

- The target must be binary.
- The other column is treated as message text.
- Rows with missing/blank message or target values are removed.
- The dataset must contain enough valid samples for a stratified train/test split.

## 🛠️ Installation

Clone the repository:

```bash
git clone https://github.com/<your-username>/spam-detection.git
cd spam-detection
```

Create and activate a virtual environment (recommended):

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## ▶️ Run the Application

```bash
python gui.py
```

Select your dataset, click **Process Dataset**, and review the results on the same page.

## 🖥️ User Interface Overview

The single-page application contains:

1. **Header** — project name and short description.
2. **Dataset Selection** — browse for a supported file.
3. **Selected File** — displays the chosen dataset path.
4. **Process Dataset** — starts the complete ML workflow.
5. **Status Area** — communicates loading, processing, success, and errors.
6. **Results** — displays the five evaluation outputs.
7. **Confusion Matrix** — clearly shows actual vs predicted classes.

## 🤖 Model Details

### Text preprocessing

For both columns:

- Values are converted to strings where appropriate.
- Leading/trailing whitespace is removed with `strip()`.
- Text is normalized to lowercase.
- Missing/blank records are removed.
- Labels are encoded to a binary numerical representation.

### Feature extraction

`CountVectorizer` converts the cleaned text into a sparse matrix of token-count features.

### Classifier

The project uses:

```text
LogisticRegression
```

with a fixed random state for reproducibility.

### Train/Test Split

The default split is:

```text
80% training
20% testing
```

with stratification when the dataset supports it.

## 📊 Evaluation Metrics

The application calculates:

| Metric | Meaning |
|---|---|
| Accuracy | Overall percentage of correct predictions |
| Precision | How many predicted positive samples were actually positive |
| Recall | How many actual positive samples were detected |
| F1 Score | Harmonic mean of precision and recall |
| Confusion Matrix | Counts correct and incorrect predictions by class |

Example display:

```text
Accuracy   : 95.20%
Precision  : 94.80%
Recall     : 96.10%
F1 Score   : 95.44%
```

The exact values depend on the supplied dataset.

## 🧯 Error Handling

The application provides clear GUI messages for:

- Unsupported extensions.
- Missing files.
- Empty datasets.
- Datasets with fewer/more than two columns.
- Excel workbooks containing multiple worksheets.
- Missing or unusable values.
- No suitable binary target column.
- Insufficient samples for train/test splitting.
- Invalid or incompatible data.
- Machine-learning processing failures.

Raw Python tracebacks are not shown to normal users.

## 🔄 Example Workflow

1. Start the application with `python gui.py`.
2. Click **Choose Dataset**.
3. Select a `.csv`, `.xls`, or `.xlsx` file.
4. Confirm the selected path.
5. Click **Process Dataset**.
6. The application validates and cleans the data.
7. The binary target column is detected automatically.
8. CountVectorizer creates text features.
9. Logistic Regression is trained.
10. Predictions are evaluated.
11. Metrics and the confusion matrix appear in the Results section.

## 📁 Project Structure

```text
spam-detection/
│
├── README.md
├── requirements.txt
├── logic.py
└── gui.py
```

### `logic.py`

Contains:

- File loading.
- Excel worksheet validation.
- Dataset validation.
- Data preprocessing.
- Binary target detection.
- CountVectorizer.
- Logistic Regression.
- Prediction.
- Evaluation metrics.
- Confusion matrix generation.

### `gui.py`

Contains:

- PyQt5 interface.
- File selection.
- User interaction.
- Processing status.
- Results display.
- Error dialogs.

## 🧰 Technologies & Libraries

- **Python** — application language.
- **PyQt5** — desktop graphical interface.
- **pandas** — data loading and preprocessing.
- **scikit-learn** — NLP feature extraction, classification, and metrics.
- **openpyxl** — XLSX workbook support.
- **xlrd** — legacy XLS workbook support.

## 🔮 Future Improvements

- Add a configurable train/test split.
- Support additional classifiers such as Naive Bayes and SVM.
- Add model persistence with joblib.
- Add cross-validation.
- Add ROC-AUC and classification-report views.
- Add prediction of individual messages.
- Add drag-and-drop file support.
- Add export of evaluation results.
- Add configurable NLP options such as stop-word removal and n-grams.

## 📜 License

This project is released under the **MIT License**. You may use, modify, and distribute it in accordance with the terms of the license.

---

**Built as a practical, beginner-friendly machine-learning project for spam classification.**
