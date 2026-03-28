# AI Resume Screening System

A production-ready starter project that compares multiple resumes against a job description and returns:

- Match score (0-100)
- Candidate ranking
- Key strengths
- Key gaps
- Final recommendation

The system includes:

- `FastAPI` backend for API-based resume analysis
- `Streamlit` app for a simple recruiter-friendly UI
- Resume parsing for `PDF`, `DOCX`, and `TXT`
- Semantic matching with OpenAI embeddings when configured
- Local `TF-IDF` fallback when no API key is present
- Optional `FAISS` indexing for scalable similarity search

## Project Structure

```text
resume_screening/
├── app.py
├── api.py
├── models/
│   ├── __init__.py
│   └── embeddings.py
├── utils/
│   ├── __init__.py
│   ├── llm_analyzer.py
│   ├── parser.py
│   └── scorer.py
├── data/
│   └── resumes/
├── requirements.txt
└── README.md
```

## Features

### 1. Resume Parsing
- Extracts text from PDF, DOCX, and TXT files
- Normalizes whitespace and removes noisy characters

### 2. JD Processing
- Extracts job-related skills
- Pulls a simple estimate for required years of experience
- Tracks keywords for overlap analysis

### 3. Matching Algorithm
- Uses OpenAI embeddings if `OPENAI_API_KEY` is set
- Falls back to `TF-IDF + cosine similarity` when no API key is available
- Combines semantic similarity, skill overlap, keyword overlap, and experience match into a score from 0-100

### 4. LLM Analysis
- Uses OpenAI to generate strengths, gaps, and recommendation if configured
- Falls back to deterministic rule-based analysis when not configured

### 5. Ranking System
- Sorts candidates by score descending
- Returns ranking and JSON-ready output

## Setup

### 1. Create and activate a virtual environment

```powershell
cd resume_screening
python -m venv .venv
.venv\Scripts\activate
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

### 3. Configure environment variables

Optional but recommended for richer analysis:

```powershell
$env:OPENAI_API_KEY="your_api_key_here"
$env:OPENAI_CHAT_MODEL="gpt-4o-mini"
$env:OPENAI_EMBEDDING_MODEL="text-embedding-3-small"
```

If no API key is set, the app still works using local scoring and rule-based analysis.

## Run the FastAPI Backend

```powershell
cd resume_screening
uvicorn api:app --reload
```

API docs:

- `http://127.0.0.1:8000/docs`

### FastAPI Endpoint

- `POST /analyze`

Form fields:

- `jd`: Job description text
- `resume_files`: One or more uploaded resumes

Example response:

```json
{
  "candidate_name": "John Doe",
  "score": 82,
  "strengths": ["Strong SQL", "Good analytics"],
  "gaps": ["Weak Python"],
  "recommendation": "Strong Fit"
}
```

## Run the Streamlit App

```powershell
cd resume_screening
streamlit run app.py
```

## Sample Test Resumes

Sample text resumes are included in `data/resumes/` for quick testing.

## Notes for Production

- Add authentication and rate limiting before public deployment
- Store uploaded files in object storage if you need auditability
- Replace the rule-based skill dictionary with a richer taxonomy for your domain
- Add structured logging and observability for production traffic
- Consider async batch jobs for large resume volumes

## Beginner-Friendly Extension Ideas

- Add named entity recognition for companies, education, and certifications
- Highlight JD keywords inside resume excerpts
- Persist embeddings for repeated searches across many candidates
- Add CSV export for recruiter workflows
