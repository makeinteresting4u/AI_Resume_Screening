from __future__ import annotations

import json
import os
from typing import Dict, List

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - runtime dependency
    OpenAI = None

from utils.parser import extract_skills


class LLMAnalyzer:
    """Generates strengths, gaps, and a hiring recommendation."""

    def __init__(self) -> None:
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")

    @property
    def use_openai(self) -> bool:
        return bool(self.api_key and OpenAI is not None)

    def analyze(self, job_description: str, resume_text: str, score: int) -> Dict[str, List[str] | str]:
        if self.use_openai:
            try:
                return self._analyze_with_openai(job_description, resume_text, score)
            except Exception:
                # Fall back to rules if the API is unavailable or rate-limited.
                pass
        return self._analyze_with_rules(job_description, resume_text, score)

    def _analyze_with_openai(self, job_description: str, resume_text: str, score: int) -> Dict[str, List[str] | str]:
        client = OpenAI(api_key=self.api_key)
        prompt = (
            "Compare this resume with the job description. Identify strengths, gaps, "
            "and give a hiring recommendation.\n\n"
            "Return strict JSON with keys: strengths, gaps, recommendation.\n"
            "strengths and gaps must each contain 2 to 3 concise bullet-style strings.\n"
            "recommendation must be one of: Strong Fit, Moderate Fit, Not Fit.\n\n"
            f"Match score: {score}\n\n"
            f"Job Description:\n{job_description}\n\n"
            f"Resume:\n{resume_text[:12000]}"
        )
        response = client.responses.create(
            model=self.model,
            input=prompt,
            temperature=0.2,
        )
        content = response.output_text.strip()
        parsed = json.loads(content)
        return {
            "strengths": parsed.get("strengths", [])[:3],
            "gaps": parsed.get("gaps", [])[:3],
            "recommendation": parsed.get("recommendation", "Moderate Fit"),
        }

    def _analyze_with_rules(self, job_description: str, resume_text: str, score: int) -> Dict[str, List[str] | str]:
        jd_skills = set(extract_skills(job_description))
        resume_skills = set(extract_skills(resume_text))

        overlap = sorted(jd_skills & resume_skills)
        missing = sorted(jd_skills - resume_skills)

        strengths = [
            f"Matches required skill: {skill}" for skill in overlap[:3]
        ] or ["Resume shows some relevant experience aligned with the role."]

        gaps = [
            f"Missing or unclear evidence for: {skill}" for skill in missing[:3]
        ] or ["No major skill gaps were detected from the provided job description."]

        if score >= 80:
            recommendation = "Strong Fit"
        elif score >= 55:
            recommendation = "Moderate Fit"
        else:
            recommendation = "Not Fit"

        return {
            "strengths": strengths[:3],
            "gaps": gaps[:3],
            "recommendation": recommendation,
        }
