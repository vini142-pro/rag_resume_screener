# 🔍 RAG Resume Screener


> Coding + AI API → RAG Pipeline

An intelligent resume screening system built on **Retrieval-Augmented Generation (RAG)** using free, local HuggingFace models and FAISS vector search — no paid API required.

---

## 🧠 What is RAG?

**RAG = Retrieval + Augmented + Generation**

| Step | What Happens | Tool Used |
|------|-------------|-----------|
| **Embed** | Convert resumes + JDs into dense vectors | HuggingFace `all-MiniLM-L6-v2` |
| **Store** | Store vectors in a searchable index | FAISS (local, offline) |
| **Retrieve** | Find top-K most semantically similar resumes | FAISS cosine similarity |
| **Generate** | Create natural-language recruiter report | `google/flan-t5-base` LLM |

Traditional keyword search → **Semantic understanding** → **AI-generated summaries**

---

## 📁 Project Structure

```
rag_resume_screener/
├── main.py                          ← Entry point (run this)
├── requirements.txt
├── data/
│   ├── sample_data.py               ← 10 resumes + 3 job descriptions
│   └── sample_resume.txt            ← Custom resume for CLI testing
├── embeddings/
│   ├── embedder.py                  ← HuggingFace sentence transformer
│   ├── vector_store.py              ← FAISS index (build/search/save/load)
│   ├── faiss_index.bin              ← Auto-created after first run
│   └── faiss_meta.pkl               ← Auto-created after first run
├── utils/
│   ├── preprocessor.py              ← Text cleaning + skill extraction
│   ├── rag_generator.py             ← Flan-T5 LLM for report generation
│   └── visualizer.py                ← 4 output charts
└── outputs/
    ├── 01_similarity_rankings.png
    ├── 02_similarity_heatmap.png
    ├── 03_skill_breakdown.png
    └── 04_rag_pipeline.png
```

---

## ⚙️ Setup

### 1. Create virtual environment
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

> **Note:** First run downloads HuggingFace models (~350MB total). Cached locally after that.

---

## 🚀 How to Run

### Full RAG pipeline — all 3 roles
```bash
python main.py
```

### Single role
```bash
python main.py --role "NLP Engineer"
python main.py --role "Data Scientist"
python main.py --role "Machine Learning Engineer"
```

### Skip LLM (faster, rule-based summaries only)
```bash
python main.py --no-llm
```

### Change number of retrieved candidates
```bash
python main.py --top-k 3
```

### Screen a custom resume
```bash
python main.py --role "NLP Engineer" --resume data/sample_resume.txt
```

### Force re-embed resumes (if you add new ones)
```bash
python main.py --rebuild
```

### Skip chart generation
```bash
python main.py --no-plots
```

---

## 📊 Output Charts

| Chart | Description |
|-------|-------------|
| `01_similarity_rankings.png` | Horizontal bars showing semantic similarity per role |
| `02_similarity_heatmap.png`  | Heatmap: all roles × all candidates |
| `03_skill_breakdown.png`     | Matched vs missing skills per top candidate |
| `04_rag_pipeline.png`        | Visual diagram of the RAG architecture |

---

## 🔬 How RAG Works Here

```
Job Description (query)
        │
        ▼
  HuggingFace Embedder          ← all-MiniLM-L6-v2
  (converts text → 384-dim vector)
        │
        ▼
  FAISS Vector Store             ← cosine similarity search
  (finds top-K most similar resumes)
        │
        ▼
  Retrieved Resume Context       ← top-K candidates with scores
        │
        ▼
  Flan-T5 LLM Generator         ← google/flan-t5-base
  (produces recruiter summary report)
        │
        ▼
  📋 Final Recruiter Report
```

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| `sentence-transformers` | Generate semantic embeddings |
| `faiss-cpu` | Fast local vector similarity search |
| `transformers` | Flan-T5 LLM for text generation |
| `torch` | PyTorch backend |
| `scikit-learn` | Skill extraction helpers |
| `matplotlib / seaborn` | Visualization |
| `tabulate` | Terminal table formatting |

---

## ➕ Add Your Own Data

### Add resumes
Edit `data/sample_data.py` → `RESUMES` list:
```python
{"candidate": "Your Name", "text": "...your resume text..."}
```
Then run: `python main.py --rebuild`

### Add a job role
Edit `data/sample_data.py` → `JOB_DESCRIPTIONS` dict:
```python
"Data Engineer": "Looking for Spark, Kafka, Airflow, SQL expert..."
```


