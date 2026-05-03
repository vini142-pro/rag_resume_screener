# utils/preprocessor.py
# ─────────────────────────────────────────────────────────────────────
#  Text Preprocessing + Skill Extraction
# ─────────────────────────────────────────────────────────────────────

import re
import string

# ── Skill Database ────────────────────────────────────────────────────
SKILLS_DB = {
    "languages":    ["python", "r", "java", "c++", "javascript", "sql", "bash", "scala"],
    "ml_dl":        ["scikit-learn", "pytorch", "tensorflow", "keras", "xgboost",
                     "lightgbm", "catboost", "mlflow", "deep learning", "machine learning"],
    "nlp":          ["nlp", "bert", "gpt", "spacy", "nltk", "huggingface", "transformers",
                     "text classification", "ner", "sentiment analysis", "embeddings",
                     "langchain", "rag", "llm", "llama", "fine-tuning"],
    "cv":           ["opencv", "yolo", "cnn", "image segmentation", "object detection",
                     "computer vision"],
    "data":         ["pandas", "numpy", "matplotlib", "seaborn", "tableau", "power bi",
                     "statistics", "a/b testing", "regression", "classification", "clustering"],
    "cloud_devops": ["aws", "gcp", "azure", "docker", "kubernetes", "ci/cd", "terraform",
                     "sagemaker", "airflow", "mlflow"],
    "vector_db":    ["faiss", "chromadb", "pinecone", "vector search", "weaviate"],
    "web":          ["flask", "fastapi", "rest api", "react", "node.js", "html", "css",
                     "mongodb", "postgresql"],
    "big_data":     ["spark", "hadoop", "kafka", "etl", "data warehousing", "dbt"],
}

ALL_SKILLS = [skill for group in SKILLS_DB.values() for skill in group]


def preprocess_text(text: str) -> str:
    """Clean and normalize text for embedding / TF-IDF."""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)           # remove URLs
    text = re.sub(r"\S+@\S+", "", text)                  # remove emails
    text = re.sub(r"[^\w\s\-\/\+\#]", " ", text)        # keep alphanum + tech chars
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_skills(text: str) -> list[str]:
    """Extract known tech skills from resume text."""
    text_lower = text.lower()
    found = []
    for skill in ALL_SKILLS:
        # word-boundary match
        pattern = r"\b" + re.escape(skill) + r"\b"
        if re.search(pattern, text_lower):
            found.append(skill)
    return list(set(found))
