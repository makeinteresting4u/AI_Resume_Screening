from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
except ImportError:  # pragma: no cover - optional at runtime
    TfidfVectorizer = None

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - handled by requirements in runtime
    OpenAI = None

try:
    import faiss
except ImportError:  # pragma: no cover - optional at runtime
    faiss = None


@dataclass
class EmbeddingResult:
    vectors: np.ndarray
    model_name: str


class EmbeddingService:
    """
    Produces dense-ish vectors for job descriptions and resumes.

    Priority:
    1. OpenAI embeddings when an API key is configured.
    2. TF-IDF vectors as a reliable local fallback.
    """

    def __init__(self) -> None:
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        self.vectorizer: Optional[object] = None

    @property
    def use_openai(self) -> bool:
        return bool(self.api_key and OpenAI is not None)

    def fit_transform(self, texts: List[str]) -> EmbeddingResult:
        if self.use_openai:
            return EmbeddingResult(
                vectors=self._embed_with_openai(texts),
                model_name=self.embedding_model,
            )

        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=4000,
        ) if TfidfVectorizer is not None else SimpleTfidfVectorizer()
        vectors = self.vectorizer.fit_transform(texts).toarray().astype("float32")
        return EmbeddingResult(vectors=vectors, model_name="tfidf")

    def transform(self, texts: List[str]) -> EmbeddingResult:
        if self.use_openai:
            return EmbeddingResult(
                vectors=self._embed_with_openai(texts),
                model_name=self.embedding_model,
            )

        if self.vectorizer is None:
            raise ValueError("TF-IDF vectorizer is not fitted yet.")
        vectors = self.vectorizer.transform(texts).toarray().astype("float32")
        return EmbeddingResult(vectors=vectors, model_name="tfidf")

    def _embed_with_openai(self, texts: List[str]) -> np.ndarray:
        client = OpenAI(api_key=self.api_key)
        response = client.embeddings.create(model=self.embedding_model, input=texts)
        vectors = [item.embedding for item in response.data]
        return np.array(vectors, dtype="float32")


class ArrayWrapper:
    def __init__(self, array: np.ndarray) -> None:
        self.array = array

    def toarray(self) -> np.ndarray:
        return self.array


class SimpleTfidfVectorizer:
    """
    Minimal fallback vectorizer used when scikit-learn is unavailable.
    It supports the subset of the TfidfVectorizer API used in this project.
    """

    def __init__(self) -> None:
        self.vocabulary_: dict[str, int] = {}
        self.idf_: dict[str, float] = {}

    def fit_transform(self, texts: List[str]) -> ArrayWrapper:
        tokenized = [self._tokenize(text) for text in texts]
        terms = sorted({token for tokens in tokenized for token in tokens})
        self.vocabulary_ = {term: index for index, term in enumerate(terms)}

        document_count = len(texts)
        for term in terms:
            doc_freq = sum(1 for tokens in tokenized if term in tokens)
            self.idf_[term] = np.log((1 + document_count) / (1 + doc_freq)) + 1

        matrix = np.vstack([self._tfidf_vector(tokens) for tokens in tokenized]).astype("float32")
        return ArrayWrapper(matrix)

    def transform(self, texts: List[str]) -> ArrayWrapper:
        matrix = np.vstack([self._tfidf_vector(self._tokenize(text)) for text in texts]).astype("float32")
        return ArrayWrapper(matrix)

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"\b[a-zA-Z][a-zA-Z\-\+\.]{1,}\b", text.lower())

    def _tfidf_vector(self, tokens: List[str]) -> np.ndarray:
        vector = np.zeros(len(self.vocabulary_), dtype="float32")
        if not tokens or not self.vocabulary_:
            return vector
        token_count = len(tokens)
        for token in tokens:
            index = self.vocabulary_.get(token)
            if index is None:
                continue
            vector[index] += 1.0 / token_count
        for term, index in self.vocabulary_.items():
            if vector[index] > 0:
                vector[index] *= float(self.idf_.get(term, 1.0))
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector /= norm
        return vector


class FaissIndexer:
    """Small FAISS wrapper for optional scalable retrieval."""

    def __init__(self) -> None:
        self.index = None

    @property
    def available(self) -> bool:
        return faiss is not None

    def build(self, vectors: np.ndarray) -> None:
        if not self.available:
            return
        if vectors.size == 0:
            return
        normalized = vectors.copy()
        faiss.normalize_L2(normalized)
        self.index = faiss.IndexFlatIP(normalized.shape[1])
        self.index.add(normalized)

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[int]:
        if not self.available or self.index is None:
            return []
        query = query_vector.copy().reshape(1, -1)
        faiss.normalize_L2(query)
        _, indices = self.index.search(query, top_k)
        return indices.flatten().tolist()
