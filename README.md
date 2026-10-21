# AI Document Similarity & Duplicate Detector

A Python + Streamlit course project that compares PDF, DOCX, and TXT documents using text extraction, normalization, TF-IDF features, and cosine similarity.

## Features

- Upload and compare multiple documents
- Extract text from PDF, DOCX, and TXT
- Normalize text before comparison
- Calculate pairwise TF-IDF cosine similarity scores
- Detect exact duplicates after text normalization
- Flag near-duplicate pairs using a configurable threshold
- Download comparison results as CSV
- Unit tests for extraction, normalization, duplicate detection, and similarity

## Requirements

- Python 3.10 or newer
- pip

## Setup

### Windows PowerShell

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

If PowerShell blocks activation, run the environment's Python directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

The app normally opens at `http://localhost:8501`.

## Run tests

```bash
pytest -q
```

## Methodology

1. Extract text from each supported file.
2. Normalize Unicode, lowercase text, and remove punctuation/extra whitespace.
3. Represent each document using TF-IDF word and two-word phrase features.
4. Calculate cosine similarity between every document pair.
5. Flag exact normalized-text duplicates and pairs above the selected threshold.

Cosine similarity is a content-overlap signal, not a plagiarism verdict. TF-IDF is primarily lexical, so paraphrases may receive lower scores and documents on the same topic may receive higher scores.

## Limitations

- Scanned PDFs without a text layer are not OCR-processed.
- Semantic similarity across extensive paraphrasing is not implemented in this baseline.
- The threshold is configurable and should be validated against a representative test set.
- The app does not search the internet or compare against an external publication database.

## Suggested GitHub repository description

AI/NLP course project that compares PDF, DOCX, and TXT documents using TF-IDF and cosine similarity, detects exact duplicates, and exports results.

## Suggested future improvements

- Semantic embeddings for paraphrase detection
- Highlight shared text passages
- Optional OCR for scanned PDFs
- PDF/HTML analysis report export
- User accounts and persistent project history (with privacy controls)
