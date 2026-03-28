from __future__ import annotations

import io

import pandas as pd
import streamlit as st

from utils.parser import parse_resume_file
from utils.scorer import score_resumes

st.set_page_config(page_title="AI Resume Screening System", layout="wide")

st.title("AI Resume Screening System")
st.caption("Screen multiple resumes against a job description and rank candidates.")

with st.sidebar:
    st.header("How It Works")
    st.write(
        "1. Paste the job description.\n"
        "2. Upload PDF, DOCX, or TXT resumes.\n"
        "3. Click Analyze to get ranked candidates, strengths, gaps, and recommendations."
    )
    st.info("If `OPENAI_API_KEY` is set, the app uses OpenAI for richer strengths and gap analysis.")

job_description = st.text_area(
    "Job Description",
    height=220,
    placeholder="Paste the full job description here...",
)

uploaded_files = st.file_uploader(
    "Upload Resumes",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True,
)

analyze_clicked = st.button("Analyze", type="primary", use_container_width=True)

if analyze_clicked:
    if not job_description.strip():
        st.error("Please provide a job description before running analysis.")
        st.stop()
    if not uploaded_files:
        st.error("Please upload at least one resume.")
        st.stop()

    parsed_resumes = []
    file_errors = []

    with st.spinner("Parsing resumes and scoring candidates..."):
        for uploaded_file in uploaded_files:
            try:
                file_obj = io.BytesIO(uploaded_file.getvalue())
                parsed_resumes.append(parse_resume_file(uploaded_file.name, file_obj))
            except Exception as exc:
                file_errors.append(f"{uploaded_file.name}: {exc}")

        if not parsed_resumes:
            st.error("No valid resumes could be parsed.")
            if file_errors:
                st.write(file_errors)
            st.stop()

        results = score_resumes(job_description, parsed_resumes)

    summary_df = pd.DataFrame(
        [
            {
                "Rank": result["rank"],
                "Candidate": result["candidate_name"],
                "Score": result["score"],
                "Recommendation": result["recommendation"],
            }
            for result in results
        ]
    )

    st.subheader("Candidate Ranking")
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    for result in results:
        with st.expander(f"#{result['rank']} {result['candidate_name']} | Score: {result['score']}"):
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Key Strengths**")
                for item in result["strengths"]:
                    st.write(f"- {item}")

                st.markdown("**Matched Skills**")
                if result["matched_skills"]:
                    st.write(", ".join(result["matched_skills"]))
                else:
                    st.write("No direct skill overlaps found.")

            with col2:
                st.markdown("**Key Gaps**")
                for item in result["gaps"]:
                    st.write(f"- {item}")

                st.markdown("**Recommendation**")
                st.write(result["recommendation"])

            st.markdown("**Resume Excerpt**")
            st.code(result["resume_excerpt"], language="text")

    if file_errors:
        st.warning("Some files could not be analyzed.")
        for error in file_errors:
            st.write(f"- {error}")
