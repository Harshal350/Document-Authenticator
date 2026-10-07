# DocuGuard – Intelligent Fake Document Detection & Verification System

> **College AIML Mini-Project** | Domain: **Security & Surveillance**  
> An explainable, multi-modal machine learning decision support system for forensic document authenticity analysis.

[![Status](https://img.shields.io/badge/Status-Complete-emerald)](#)
[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](#)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688.svg)](#)
[![React](https://img.shields.io/badge/Frontend-React_19_+_Vite-61dafb.svg)](#)
[![TailwindCSS](https://img.shields.io/badge/Styling-Tailwind_CSS_4-38bdf8.svg)](#)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](#)

---

## 1. Project Overview

**DocuGuard** is an end-to-end artificial intelligence and machine learning forensic platform designed to evaluate uploaded digital and digitized documents (PDF, DOCX, DOC, TXT, PNG, JPG, JPEG, TIFF) to determine whether they appear genuine or fraudulent.

Unlike naive keyword-matching or black-box classifiers, DocuGuard combines:
1. **Multi-Format Document Parsing & OCR Preprocessing** (PyMuPDF, python-docx, OpenCV, Pillow, Tesseract)
2. **Comprehensive Feature Engineering** (32-dimensional text, structural, numerical, metadata, OCR, and visual features)
3. **Multi-Category Forensic Marker Detection** (Textual, Numerical, Date, ID/Reference, Structural, Metadata, Visual)
4. **Machine Learning Model Comparison & Ensembling** (Logistic Regression, Random Forest, and XGBoost / Gradient Boosting)
5. **Explainable AI (XAI)** (Transparent feature contribution scoring, human-readable forensic evidence, and risk decomposition)
6. **Downloadable Forensic PDF Reports** (ReportLab-generated audit reports with cryptographic-style summaries and disclaimers)

> **Important Real-World Disclaimer**:  
> DocuGuard is an AI-assisted decision support system. It provides risk probabilities and forensic anomaly indicators, expressing results as **"Likely Genuine"**, **"Suspicious"**, or **"Likely Fake"**. It does not claim absolute legal authenticity.

---

## 2. Problem Statement & Academic Context

* **Academic Requirement**: *"Design and develop an application or decision support system for the selected problem statement from domains such as Security & Surveillance. Visualize the results and perform a comparative performance analysis of at least three different machine learning models or deep learning models using appropriate evaluation methods."*
* **Selected Domain**: Security & Surveillance.
* **Problem**: Fraudulent invoices, doctored certificates, manipulated salary slips, tampered bank statements, and forged identification documents are increasingly prevalent. Manual forensic analysis is laborious and prone to human error, while standard classifiers fail to provide actionable explanations.
* **Solution**: A unified, automated forensic pipeline that detects structural, semantic, visual, and mathematical inconsistencies across documents, compares three distinct ML classifiers on held-out test data, and delivers an intuitive, cybersecurity-grade interface.

---

## 3. Core Objectives

- [x] Ingest multiple document formats (`.pdf`, `.docx`, `.doc`, `.txt`, `.png`, `.jpg`, `.jpeg`, `.tiff`).
- [x] Extract full document text, page structures, layout properties, and embedded metadata.
- [x] Implement computer vision and OCR preprocessing pipelines (grayscale, denoise, thresholding, deskewing).
- [x] Automatically identify semantic document categories (Invoice, Receipt, Certificate, Identity, Educational, Employment, Financial, Application, Agreement, Letter, Government).
- [x] Detect fine-grained forensic markers across 7 categories (dates, totals, IDs, fonts, margins, metadata stamps).
- [x] Extract a rich 32-dimensional numerical/structural feature vector combined with TF-IDF n-grams.
- [x] Implement, evaluate, and compare at least **three ML classifiers**:
  - **Model 1**: Logistic Regression (Linear baseline & interpretability)
  - **Model 2**: Random Forest Classifier (Non-linear ensemble)
  - **Model 3**: XGBoost / Gradient Boosting (Gradient-boosted decision trees)
- [x] Generate comparative evaluation metrics on held-out test data (Accuracy, Precision, Recall, F1-Score, ROC-AUC, Confusion Matrix).
- [x] Deliver an explainable 0–100 risk score and classification:
  - `0 – 29`: **Likely Genuine**
  - `30 – 59`: **Suspicious**
  - `60 – 100`: **Likely Fake**
- [x] Store scan history and audit logs in an SQLite database with SQLAlchemy ORM.
- [x] Generate downloadable, publication-grade forensic PDF reports using ReportLab.
- [x] Provide a polished, modern cybersecurity-style UI (dark palette `#0A0A0A`, emerald `#10B981` accents, zero gimmicky glassmorphism).

---

## 4. System Architecture

The forensic pipeline is strictly modular, allowing each stage to execute independently and asynchronously:

```text
 ┌────────────────────────────────────────────────────────────────────────┐
 │                      User Interface (React + Vite)                     │
 │      Dark Cybersecurity Dashboard · Recharts · Lucide Icons · Tailwind │
 └───────────────────────────────────┬────────────────────────────────────┘
                                     │ HTTP / REST
 ┌───────────────────────────────────▼────────────────────────────────────┐
 │                       FastAPI Backend (/api/v1)                        │
 └──────┬──────────────┬──────────────┬──────────────┬─────────────┬──────┘
        │              │              │              │             │
 ┌──────▼───────┐┌─────▼────────┐┌────▼────────┐┌────▼───────┐┌───▼──────┐
 │ File Upload  ││ Document     ││ Feature     ││ Marker     ││ ML Engine│
 │ & Validation ││ Processor    ││ Engineering ││ Detector   ││ & Risk   │
 │ (MIME/size)  ││ (PDF/DOCX/   ││ (32-D Text/ ││ (7 anomaly ││ (Ensemble│
 │              ││  Images/OCR) ││  Structure) ││  categories││  + XAI)  │
 └──────┬───────┘└─────┬────────┘└────┬────────┘└────┬───────┘└───┬──────┘
        │              │              │              │            │
        └──────────────┴──────────────┼──────────────┴────────────┘
                                      │
                   ┌──────────────────▼──────────────────┐
                   │    Database (SQLite + SQLAlchemy)   │
                   │    Artifacts (ml_models/ + reports/) │
                   └─────────────────────────────────────┘
```

### Forensic Pipeline Stages

1. **Upload & Validation**: Validates file size (configurable, default 25 MB), checks MIME types, prevents path traversal, and assigns secure unique identifiers.
2. **Document Parsing**: Direct text and layout extraction via PyMuPDF for PDFs, python-docx for DOCX, and raw text decoding for TXT.
3. **OCR Engine**: Fallback/scanned document processing using OpenCV (denoise, threshold, deskew) and Tesseract.
4. **Document Type Detection**: Weighted regex and semantic phrase scoring to classify the document into 11 categories (Invoice, Certificate, etc.), establishing context for expected fields.
5. **Feature Engineering**: Derives 32 quantitative features (vocabulary richness, punctuation ratio, digit ratio, uppercase ratio, sentence length variance, table count, font variance, etc.).
6. **Forensic Marker Engine**: Evaluates rule-based heuristics across 7 categories:
   - *Textual*: Suspicious phrases, excessive capitalization, abnormal spacing, repetitive text.
   - *Numerical*: Subtotal + tax mismatch, mathematical discrepancies, differing repeated figures.
   - *Date*: Impossible calendar dates, issue date occurring after expiration date, future issue dates.
   - *ID / Reference*: Duplicate identifiers, malformed regex patterns, unexpected checksums.
   - *Structural*: Missing essential sections, abnormal page sequence, margin discrepancies.
   - *Visual*: Resolution inconsistencies, compression anomalies, edge artifacts.
   - *Metadata*: Discrepancies between creation and modification timestamps, editing tool footprints.
7. **ML Classification & Ensembling**: Combines TF-IDF vectors and scaled numeric features, obtaining calibrated fraud probabilities from Logistic Regression, Random Forest, and XGBoost.
8. **Ensemble Risk Scoring**: Synthesizes model probabilities, marker penalties, and structural anomalies into a calibrated 0–100 Risk Score.
9. **Explainability Engine**: Computes local feature importances and surfaces plain-English forensic findings.
10. **Reporting**: Assembles a PDF audit report with summary statistics, charts, and verification disclaimers.

---

## 5. Technology Stack

| Layer | Technologies & Libraries |
| :--- | :--- |
| **Backend Framework** | FastAPI, Uvicorn, Pydantic v2, `pydantic-settings`, Python-Multipart |
| **Database & ORM** | SQLite, SQLAlchemy 2.0 |
| **Machine Learning** | scikit-learn, XGBoost, NumPy, pandas, joblib |
| **Document Processing** | PyMuPDF (`fitz`), `python-docx`, Pillow (`PIL`), OpenCV (`cv2`) |
| **Report Generation** | ReportLab, Matplotlib (headless `Agg` backend) |
| **Frontend Framework** | React 19, Vite, React Router DOM 7 |
| **Styling & Icons** | Tailwind CSS 4, Lucide React |
| **Data Visualization** | Recharts (Dashboard & Performance charts) |
| **Testing & QA** | pytest, pytest-asyncio, HTTPX TestClient |

---

## 6. Machine Learning Models & Academic Comparison

To fulfill academic project criteria, DocuGuard implements and compares **three distinct machine learning models** trained on stratified splits (80% training / 20% testing):

### Model Selection Justification

| Model | Why Selected | Advantages | Limitations | Computational Cost | Interpretability |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | Linear statistical benchmark | Fast convergence, well-calibrated probabilities, convex optimization | Cannot capture non-linear feature interactions | Very Low ($O(n \cdot d)$) | **Very High** (Explicit coefficient weights) |
| **Random Forest** | Non-linear ensemble baseline | Handles high-dimensional sparse & dense features, resistant to overfitting | Slower inference with large tree ensembles | Moderate ($O(M \cdot n \cdot d \log n)$) | **Moderate** (Gini impurity / feature importance) |
| **XGBoost / Gradient Boosting** | State-of-the-art boosted trees | Optimizes complex loss functions, superior gradient-driven boundary separation | Sensitive to hyperparameters, needs careful regularization | Moderate to High | **Moderate** (Gain-based tree importance / SHAP) |

### Comparative Evaluation Metrics

Evaluation is carried out on held-out test data. Metrics are persisted to SQLite and displayed on the Model Performance Dashboard:

- **Accuracy**: Overall classification correctness $\frac{TP + TN}{TP + TN + FP + FN}$.
- **Precision**: Positive predictive value $\frac{TP}{TP + FP}$ (minimizing false accusations of forgery).
- **Recall**: Sensitivity / detection rate $\frac{TP}{TP + FN}$ (ensuring fraudulent documents do not slip through).
- **F1-Score**: Harmonic mean of Precision and Recall $\frac{2 \cdot P \cdot R}{P + R}$.
- **ROC-AUC**: Area under the Receiver Operating Characteristic curve, measuring discrimination across all thresholds.
- **Confusion Matrix**: Visual $2 \times 2$ matrix of True Positives, False Positives, True Negatives, False Negatives.

---

## 7. Forensic Feature Engineering

DocuGuard extracts 32 distinct features grouped into 6 forensic dimensions:

```text
Feature Vector (32-D)
├── Textual (12)
│   ├── word_count, char_count, sentence_count, avg_sentence_length
│   ├── vocabulary_richness (Type-Token Ratio)
│   ├── punctuation_ratio, uppercase_ratio, digit_ratio, special_char_ratio
│   └── repeated_phrase_count, spelling_anomaly_count, suspicious_phrase_count
├── Structural (7)
│   ├── page_count, paragraph_count, table_count, heading_count
│   ├── blank_page_count, font_count, font_size_variance
├── Numerical (5)
│   ├── number_count, monetary_value_count, date_count
│   └── total_inconsistencies, id_inconsistencies
├── OCR (3)
│   ├── ocr_used (0 or 1), ocr_confidence, low_confidence_text_ratio
├── Metadata (3)
│   ├── metadata_present, metadata_timestamp_gap_seconds, creator_tool_suspicious
└── Visual (2)
    ├── image_count, suspicious_region_count
```

Numerical features are normalized with `StandardScaler` and combined with a fitted TF-IDF matrix (up to 5,000 unigrams and bigrams).

---

## 8. Dataset Strategy

- **Default Dataset**: Located at `data/training/documents.csv`.
- **Format**:
  ```csv
  document_id,text,document_type,label
  DOC-CERT-001,"CERTIFICATE OF INCORPORATION...",Certificate,0
  DOC-INV-002,"TAX INVOICE Total mismatch...",Invoice,1
  ```
  where `0 = Likely Genuine` and `1 = Likely Fake`.
- **Demo Notice**: The pre-packaged dataset is explicitly labeled:
  > *"Demo dataset – replace with real/collected dataset before final evaluation."*

---

## 9. Installation & Setup

### Prerequisites
- Python 3.9, 3.10, or 3.11
- Node.js 18+ and npm
- (Optional) Tesseract OCR engine for image-only scans:
  - macOS: `brew install tesseract libomp`
  - Ubuntu/Debian: `sudo apt-get install tesseract-ocr libomp-dev`

### Step 1: Clone the Repository

```bash
git clone https://github.com/Harshal350/Document-Authenticator.git
cd Document-Authenticator
```

### Step 2: Backend Setup

```bash
# 1. Create a Python virtual environment
python3 -m venv .venv

# 2. Activate the virtual environment
source .venv/bin/activate       # On Windows: .venv\Scripts\activate

# 3. Install Python dependencies
pip install --upgrade pip
pip install -r backend/requirements.txt
```

### Step 3: Frontend Setup

```bash
cd frontend
npm install
cd ..
```

---

## 10. Running the Application

### 1. Launch the Backend API

From the project root:

```bash
source .venv/bin/activate
uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
```

- API Base URL: `http://localhost:8000`
- Interactive Swagger Documentation: `http://localhost:8000/api/v1/docs`
- Health Check: `http://localhost:8000/api/v1/health`

### 2. Launch the Frontend

In a separate terminal window:

```bash
cd frontend
npm run dev
```

- Frontend Web Application: `http://localhost:3000`

### 3. Train the ML Models

To train the models on the dataset and generate initial evaluation charts:

```bash
# Via cURL:
curl -X POST "http://localhost:8000/api/v1/models/train"

# Or directly in Python:
PYTHONPATH=backend .venv/bin/python -c "from app.ml.train import train_from_dataset; train_from_dataset('data/training/documents.csv')"
```

---

## 11. REST API Specification

| HTTP Method | Route | Description |
| :--- | :--- | :--- |
| `GET` | `/api/v1/health` | System health, database connection, and uptime |
| `POST` | `/api/v1/upload` | Upload a document (`.pdf`, `.docx`, images, etc.) |
| `POST` | `/api/v1/upload/batch` | Upload multiple documents simultaneously |
| `POST` | `/api/v1/analyze/{document_id}` | Execute full forensic pipeline on an uploaded file |
| `GET` | `/api/v1/results/{analysis_id}` | Retrieve comprehensive forensic analysis report |
| `GET` | `/api/v1/history` | Paginated scan history with search and decision filter |
| `DELETE` | `/api/v1/history/{analysis_id}` | Remove a scan record and its artifacts |
| `GET` | `/api/v1/dashboard` | Aggregated forensic metrics and risk distributions |
| `GET` | `/api/v1/models/performance` | Persisted evaluation metrics for all 3 ML models |
| `POST` | `/api/v1/models/train` | Trigger model retraining and chart generation |
| `GET` | `/api/v1/report/{analysis_id}` | Generate and download forensic PDF report |

---

## 12. Viva-Friendly Architecture Guide

For academic project viva presentations, be prepared to explain the complete data flow:

```text
[Uploaded Document]
        │
        ▼
1. Document Parsing (PyMuPDF / python-docx / OpenCV)
   - Extracts plain text, bounding boxes, table count, image metadata.
        │
        ▼
2. OCR Preprocessing (If scanned or image-based)
   - Converts to grayscale, applies Gaussian blur, Otsu thresholding, and deskewing before Tesseract OCR.
        │
        ▼
3. Semantic Type Detection
   - Scores keywords to identify document category (e.g. Invoice expects totals; Certificate expects recipient).
        │
        ▼
4. Feature Extraction
   - Generates 32 quantitative features + TF-IDF matrix.
        │
        ▼
5. Forensic Marker Engine
   - 7 independent heuristic detectors flag mathematical mismatches, inverted dates, and anomalous formatting.
        │
        ▼
6. Multi-Model ML Classification
   - Logistic Regression, Random Forest, and XGBoost output individual fraud probabilities.
        │
        ▼
7. Risk Ensemble & Explainability
   - Weighted combination creates 0–100 Risk Score; XAI engine extracts top contributing factors.
        │
        ▼
8. Visualization & PDF Reporting
   - Renders interactive charts on React dashboard; compiles ReportLab PDF.
```

### Common Viva Questions & Answers

1. **Why not just use a deep learning model or LLM?**  
   *Answer*: LLMs are computationally heavy, non-deterministic, and prone to hallucinations. In document security, institutions require deterministic mathematical consistency checks (e.g., verifying if tax + subtotal = total), explicit date comparisons, and explainable feature attributions that can stand up to audits.
2. **Why evaluate three different models?**  
   *Answer*: Different model architectures offer different trade-offs. Logistic Regression provides a transparent linear baseline; Random Forest captures non-linear tabular interactions; Gradient Boosting (XGBoost) iteratively corrects residual errors. Comparing them on the same stratified test split demonstrates rigorous methodology.
3. **How is data leakage avoided during preprocessing?**  
   *Answer*: The `StandardScaler` and `TfidfVectorizer` are fit **strictly** on the training split ($80\%$) and then applied via `.transform()` to the test split ($20\%$).
4. **What is the meaning of the 0–100 Risk Score?**  
   *Answer*: It is a normalized composite score: $0–29$ indicates Likely Genuine, $30–59$ denotes Suspicious (recommending human review), and $60–100$ indicates Likely Fake.

---

## 13. Project Directory Structure

```text
DocGuard/
├── .env.example                    # Environment variable configuration template
├── .gitignore                      # Git ignore patterns for venv, node_modules, temp files
├── PROMPT DocGuard.pdf             # Project requirements and specification
├── README.md                       # Comprehensive project documentation
├── backend/
│   ├── requirements.txt            # Python dependencies
│   └── app/
│       ├── __init__.py
│       ├── main.py                 # FastAPI application, CORS, routers & lifecycle
│       ├── config.py               # pydantic-settings configuration
│       ├── api/                    # REST API routes
│       │   ├── upload.py           # Document upload & batch processing
│       │   ├── analysis.py         # Forensic analysis execution & results
│       │   ├── history.py          # Scan history & dashboard statistics
│       │   ├── models.py           # Model training & performance metrics
│       │   └── reports.py          # PDF report generation
│       ├── core/                   # Forensic analysis engines
│       │   ├── document_processor.py   # Multi-format parsing & text extraction
│       │   ├── document_type_detector.py # Semantic document classification
│       │   ├── feature_engineering.py  # 32-dimensional feature vector extraction
│       │   ├── marker_detector.py      # Rule-based forensic anomaly engine
│       │   ├── ocr_engine.py           # OpenCV preprocessing & Tesseract OCR
│       │   ├── risk_engine.py          # Ensemble scoring & decision logic
│       │   └── explainability.py       # Plain-language explanation & SHAP-style breakdown
│       ├── ml/                     # Machine learning pipeline
│       │   ├── preprocessing.py    # TF-IDF vectorization & feature scaling
│       │   ├── train.py            # Stratified training for LogReg, RF, XGBoost
│       │   ├── evaluate.py         # Metric calculation & black/green charts
│       │   └── predict.py          # Model inference & probability calibration
│       ├── database/               # Database management
│       │   └── database.py         # SQLAlchemy engine, session factory & tables init
│       ├── models/                 # Schemas & data models
│       │   ├── database_models.py  # SQLAlchemy ORM models
│       │   └── schemas.py          # Pydantic v2 validation schemas
│       └── utils/                  # Utilities
│           ├── file_validation.py  # MIME, size & filename security validation
│           └── logging.py          # Structured logging configuration
├── frontend/
│   ├── package.json                # Frontend dependencies
│   ├── vite.config.js              # Vite build configuration
│   ├── index.html                  # HTML entry point
│   └── src/
│       ├── App.jsx                 # Main React router & layout
│       ├── main.jsx                # React root mount
│       ├── index.css               # Cybersecurity dark theme styles
│       ├── pages/                  # Application views
│       │   ├── Dashboard.jsx       # Overview metrics & distribution charts
│       │   ├── AnalyzeDocument.jsx # File upload & analysis trigger
│       │   ├── AnalysisResult.jsx  # Detailed forensic report view
│       │   ├── ModelPerformance.jsx# ML comparative analysis charts & table
│       │   └── History.jsx         # Scan history table with filtering
│       ├── components/             # Reusable UI widgets
│       │   ├── Sidebar.jsx         # Navigation sidebar
│       │   ├── RiskGauge.jsx       # Circular SVG risk score gauge
│       │   ├── MarkerTable.jsx     # Detected anomaly breakdown
│       │   ├── ModelBar.jsx        # Model prediction bars
│       │   ├── StatCard.jsx        # Summary KPI cards
│       │   ├── DecisionBadge.jsx   # Status indicator badge
│       │   └── ErrorState.jsx      # Graceful error display
│       └── services/
│           └── api.js              # Axios/Fetch API client
├── data/
│   ├── markers.json                # Configurable forensic marker rules
│   ├── training/
│   │   └── documents.csv           # Labelled training dataset
│   ├── raw/                        # Ephemeral upload storage
│   └── processed/                  # Ephemeral processed artifacts
├── ml_models/                      # Serialized model artifacts (.joblib)
├── reports/                        # Output PDF reports & metric comparison charts
└── tests/                          # Automated test suite
    ├── test_core.py                # Core forensic engines test cases
    └── test_api.py                 # FastAPI endpoints integration tests
```

---

## 14. Testing & Quality Assurance

DocuGuard includes a comprehensive test suite using `pytest`:

```bash
# Run all tests
.venv/bin/pytest tests/ -v
```

### Verified Test Coverage
- **Extraction Tests**: PDF, DOCX, and TXT parsing.
- **Forensic Marker Tests**: Date anomalies, numerical inconsistencies, ID pattern checks, repeated sentences.
- **Feature Engineering**: Vector dimensions, zero-division safeguards, scaling behavior.
- **ML Pipeline**: Model prediction outputs, probability calibration, fallback mechanics.
- **Risk Scoring**: Correct decision mapping (`Likely Genuine`, `Suspicious`, `Likely Fake`).
- **API Endpoints**: Health check, single and batch upload, analysis pipeline, model performance, and history endpoints.

---

## 15. Limitations & Future Scope

### Current Limitations
- Best calibrated for English-language commercial and institutional documents.
- OCR relies on underlying image quality; heavily degraded scans may have lower character confidence.
- Machine learning models reflect the characteristics of the training dataset provided.

### Future Scope
- Multi-document cross-comparison mode (diffing two versions of a document).
- Deep learning transformer backbones (DistilBERT / LayoutLM) for visual document understanding.
- Docker & Docker Compose containerization for cloud deployments.

---

## 16. Team Contributions

| Member Name | Role | Key Responsibilities |
| :--- | :--- | :--- |
| **Team Member 1** | ML & NLP Engineer | Feature engineering pipeline, ML model training (LogReg, RF, XGBoost), evaluation metrics, and comparison charts |
| **Team Member 2** | Backend & Forensic Engineer | FastAPI architecture, marker detection rules, document parsing (PyMuPDF, docx), and ReportLab PDF generator |
| **Team Member 3** | Full-Stack & UI/UX Developer | React 19 frontend, Tailwind cybersecurity design system, Recharts integration, and API service layer |
| **Team Member 4** | QA & Security Specialist | Test suites (pytest), upload validation & MIME security, dataset curation, and documentation |

---

## 17. References

1. Scikit-learn Machine Learning Library: https://scikit-learn.org
2. XGBoost Documentation: https://xgboost.readthedocs.io
3. FastAPI Documentation: https://fastapi.tiangolo.com
4. PyMuPDF (fitz) Documentation: https://pymupdf.readthedocs.io
5. SQLAlchemy 2.0 Documentation: https://docs.sqlalchemy.org
6. React Documentation: https://react.dev
7. Tailwind CSS: https://tailwindcss.com
8. Recharts: https://recharts.org