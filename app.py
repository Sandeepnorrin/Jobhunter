import streamlit as st
import pandas as pd
from utils import extract_text_from_pdf, extract_keywords_from_resume, search_jobs, evaluate_job_match

st.set_page_config(page_title="AI Job Matcher", layout="wide")

st.title("🎯 AI Job Matcher")
st.write("Upload your resume and find matching jobs across platforms using Groq AI.")

with st.sidebar:
    st.header("Settings")
    groq_api_key = st.text_input("Groq API Key", type="password")
    locations = st.multiselect(
        "Preferred Locations",
        ["Bengaluru", "Hyderabad", "Chennai", "Mumbai", "Pune", "Delhi", "Remote"],
        default=["Bengaluru", "Hyderabad", "Chennai"]
    )
    results_per_location = st.slider("Results per location", 5, 30, 10)
    max_evaluations = st.slider("Max jobs to evaluate with AI", 1, 20, 5)

uploaded_file = st.file_uploader("Upload your Resume (PDF)", type="pdf")

if uploaded_file and groq_api_key:
    # Use the file name and size as a simple key to detect changes
    file_id = f"{uploaded_file.name}_{uploaded_file.size}"

    if 'file_id' not in st.session_state or st.session_state.file_id != file_id:
        with st.spinner("Parsing resume..."):
            st.session_state.resume_text = extract_text_from_pdf(uploaded_file)
            st.session_state.keywords = extract_keywords_from_resume(st.session_state.resume_text, groq_api_key)
            st.session_state.file_id = file_id

    st.subheader("Extracted Keywords")
    keywords_str = st.text_input("Edit keywords for search", value=", ".join(st.session_state.keywords))
    search_keywords = [k.strip() for k in keywords_str.split(",") if k.strip()]

    if st.button("Search and Match Jobs"):
        with st.spinner("Searching for jobs..."):
            jobs_df = search_jobs(search_keywords, locations, results_per_location)

        if not jobs_df.empty:
            # Limit evaluations to avoid long wait times
            jobs_to_eval = jobs_df.head(max_evaluations)
            st.success(f"Found {len(jobs_df)} jobs. Evaluating top {len(jobs_to_eval)} matches...")

            results = []
            progress_bar = st.progress(0)
            for i, (idx, row) in enumerate(jobs_to_eval.iterrows()):
                # Some job descriptions might be short or missing depending on the site
                jd = row.get('description', '')
                if not jd:
                    # Try to use title if description is empty, though less accurate
                    jd = row.get('title', '')

                evaluation = evaluate_job_match(
                    st.session_state.resume_text,
                    jd,
                    row['title'],
                    groq_api_key
                )

                results.append({
                    "Title": row['title'],
                    "Company": row['company'],
                    "Location": row['location'],
                    "Score": evaluation['score'],
                    "Matched Skills": ", ".join(evaluation['matched_skills']),
                    "Missing Skills": ", ".join(evaluation['missing_skills']),
                    "Justification": evaluation['justification'],
                    "Link": row['job_url']
                })
                progress_bar.progress((i + 1) / len(jobs_to_eval))

            results_df = pd.DataFrame(results)
            results_df = results_df.sort_values(by="Score", ascending=False)

            st.subheader("Matching Jobs")
            for _, job in results_df.iterrows():
                with st.expander(f"{job['Title']} @ {job['Company']} - Score: {job['Score']}%"):
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        st.write(f"**Location:** {job['Location']}")
                        st.write(f"**Justification:** {job['Justification']}")
                        st.write(f"**Matched Skills:** {job['Matched Skills']}")
                        st.write(f"**Missing Skills:** {job['Missing Skills']}")
                    with col2:
                        st.link_button("Apply Now", job['Link'])
        else:
            st.warning("No jobs found with the given keywords.")
elif not groq_api_key:
    st.info("Please enter your Groq API Key in the sidebar to get started.")
elif not uploaded_file:
    st.info("Please upload your resume to get started.")
