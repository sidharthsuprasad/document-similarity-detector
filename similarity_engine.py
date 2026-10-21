"""TF-IDF and cosine similarity functions."""
from __future__ import annotations

from itertools import combinations
from typing import Iterable

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def compare_documents(documents: list[dict]) -> tuple[list[dict], list[list[float]]]:
    """Return all pairwise TF-IDF cosine similarity scores and a similarity matrix."""
    if len(documents) < 2:
        return [], []

    texts = [doc["text"] for doc in documents]
    # Word n-grams help detect both shared terms and short shared phrases.
    vectorizer = TfidfVectorizer(
        lowercase=False,
        ngram_range=(1, 2),
        sublinear_tf=True,
        token_pattern=r"(?u)\b\w+\b",
    )
    try:
        vectors = vectorizer.fit_transform(texts)
        matrix = cosine_similarity(vectors)
    except ValueError as exc:
        if "empty vocabulary" in str(exc).lower():
            raise ValueError("Not enough readable words were found to compare these documents.") from exc
        raise

    results = []
    for i, j in combinations(range(len(documents)), 2):
        score = float(matrix[i, j]) * 100
        results.append({
            "Document A": documents[i]["name"],
            "Document B": documents[j]["name"],
            "Similarity (%)": round(score, 2),
        })
    return results, matrix.tolist()


def find_exact_duplicates(documents: list[dict]) -> list[tuple[str, str]]:
    """Find pairs with identical normalized text."""
    pairs = []
    for i, j in combinations(range(len(documents)), 2):
        if documents[i]["text"] == documents[j]["text"]:
            pairs.append((documents[i]["name"], documents[j]["name"]))
    return pairs
