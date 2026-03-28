from __future__ import annotations

import re
from typing import Dict, List

import numpy as np

try:
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:  # pragma: no cover - optional at runtime
    cosine_similarity = None

from models.embeddings import EmbeddingService, FaissIndexer
from utils.llm_analyzer import LLMAnalyzer
from utils.parser import extract_skills


def normalize_score(raw_similarity: float) -> int:
    """Map cosine similarity to a user-friendly 0-100 score."""
    clipped = max(0.0, min(1.0, raw_similarity))
    return int(round(clipped * 100))


def extract_experience_years(text: str) -> int:
    """
    Pull a coarse experience estimate from free text.
    This is intentionally simple and explainable for beginner-friendly code.
    """
    patterns = [
        r"(\d+)\+?\s+years? of experience",
        r"experience[:\s]+(\d+)\+?\s+years?",
        r"(\d+)\+?\s+years? in",
    ]
    lowered = text.lower()
    for pattern in patterns:
        match = re.search(pattern, lowered)
        if match:
            return int(match.group(1))
    return 0


def process_job_description(job_description: str) -> Dict[str, object]:
    skills = extract_skills(job_description)
    experience_years = extract_experience_years(job_description)
    keywords = sorted(set(re.findall(r"\b[a-zA-Z][a-zA-Z\-\+\.]{2,}\b", job_description.lower())))
    return {
        "skills": skills,
        "experience_years": experience_years,
        "keywords": keywords[:75],
    }


def keyword_overlap(job_description: str, resume_text: str) -> float:
    jd_words = set(re.findall(r"\b[a-zA-Z][a-zA-Z\-\+\.]{2,}\b", job_description.lower()))
    resume_words = set(re.findall(r"\b[a-zA-Z][a-zA-Z\-\+\.]{2,}\b", resume_text.lower()))
    if not jd_words:
        return 0.0
    return len(jd_words & resume_words) / len(jd_words)


def skill_match_ratio(jd_skills: List[str], resume_skills: List[str]) -> float:
    if not jd_skills:
        return 0.0
    return len(set(jd_skills) & set(resume_skills)) / len(set(jd_skills))


def score_resumes(job_description: str, resumes: List[Dict[str, str]]) -> List[Dict[str, object]]:
    if not job_description.strip():
        raise ValueError("Job description cannot be empty.")
    if not resumes:
        raise ValueError("At least one resume is required.")

    jd_profile = process_job_description(job_description)
    embedder = EmbeddingService()
    llm_analyzer = LLMAnalyzer()

    corpus = [job_description] + [resume["text"] for resume in resumes]
    embedding_result = embedder.fit_transform(corpus)
    vectors = embedding_result.vectors
    jd_vector = vectors[0].reshape(1, -1)
    resume_vectors = vectors[1:]

    faiss_indexer = FaissIndexer()
    faiss_indexer.build(resume_vectors)

    semantic_scores = compute_cosine_similarity(jd_vector, resume_vectors).flatten()
    results: List[Dict[str, object]] = []

    for index, resume in enumerate(resumes):
        resume_skills = extract_skills(resume["text"])
        keyword_score = keyword_overlap(job_description, resume["text"])
        skill_score = skill_match_ratio(jd_profile["skills"], resume_skills)

        experience_required = int(jd_profile["experience_years"])
        experience_found = extract_experience_years(resume["text"])
        experience_score = 1.0
        if experience_required > 0:
            experience_score = min(experience_found / max(experience_required, 1), 1.0)

        combined_similarity = (
            semantic_scores[index] * 0.55
            + keyword_score * 0.2
            + skill_score * 0.2
            + experience_score * 0.05
        )
        score = normalize_score(float(combined_similarity))

        analysis = llm_analyzer.analyze(job_description, resume["text"], score)
        matched_keywords = _highlight_keywords(jd_profile["skills"], resume_skills)

        results.append(
            {
                "candidate_name": resume["candidate_name"],
                "score": score,
                "strengths": analysis["strengths"],
                "gaps": analysis["gaps"],
                "recommendation": analysis["recommendation"],
                "matched_skills": matched_keywords,
                "resume_excerpt": resume["text"][:500],
                "embedding_model": embedding_result.model_name,
            }
        )

    results.sort(key=lambda item: item["score"], reverse=True)
    for rank, result in enumerate(results, start=1):
        result["rank"] = rank
    return results


def _highlight_keywords(jd_skills: List[str], resume_skills: List[str]) -> List[str]:
    return sorted(set(jd_skills) & set(resume_skills))


def compute_cosine_similarity(jd_vector: np.ndarray, resume_vectors: np.ndarray) -> np.ndarray:
    if cosine_similarity is not None:
        return cosine_similarity(jd_vector, resume_vectors)

    jd_norm = np.linalg.norm(jd_vector, axis=1, keepdims=True)
    resume_norm = np.linalg.norm(resume_vectors, axis=1, keepdims=True).T
    denominator = np.maximum(jd_norm * resume_norm, 1e-12)
    similarities = np.dot(jd_vector, resume_vectors.T) / denominator
    return similarities
