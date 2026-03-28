from __future__ import annotations

import io
from typing import List

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from utils.parser import parse_resume_file
from utils.scorer import score_resumes

app = FastAPI(
    title="AI Resume Screening System",
    description="Analyze resumes against a job description and rank candidates.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check() -> dict:
    return {"status": "ok"}


@app.post("/analyze")
async def analyze_resumes(
    jd: str = Form(...),
    resume_files: List[UploadFile] = File(...),
) -> dict:
    if not jd.strip():
        raise HTTPException(status_code=400, detail="Job description cannot be empty.")
    if not resume_files:
        raise HTTPException(status_code=400, detail="Please upload at least one resume.")

    parsed_resumes = []
    errors = []

    for upload in resume_files:
        try:
            content = await upload.read()
            file_obj = io.BytesIO(content)
            parsed = parse_resume_file(upload.filename, file_obj)
            parsed_resumes.append(parsed)
        except Exception as exc:
            errors.append({"file": upload.filename, "error": str(exc)})

    if not parsed_resumes:
        raise HTTPException(status_code=400, detail={"message": "No valid resumes found.", "errors": errors})

    try:
        results = score_resumes(jd, parsed_resumes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensive API boundary
        raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}") from exc

    return {
        "job_description_length": len(jd),
        "total_candidates": len(results),
        "results": results,
        "errors": errors,
    }
