import pypdf
from groq import Groq
import json
from jobspy import scrape_jobs
import pandas as pd

def extract_text_from_pdf(pdf_file):
    reader = pypdf.PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def extract_keywords_from_resume(resume_text, api_key):
    client = Groq(api_key=api_key)

    prompt = f"""
    Extract the most relevant job-related keywords and skills from the following resume text.
    Focus on technical skills, job titles, and industry-specific terms that can be used to search for jobs on platforms like LinkedIn or Indeed.
    Return the result as a JSON object with a key 'keywords' containing a list of strings.

    Resume Text:
    {resume_text}
    """

    completion = client.chat.completions.create(
        model="llama-3.3-70b-specdec",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that extracts job keywords from resumes. Respond only with JSON."},
            {"role": "user", "content": prompt}
        ],
        response_format={"type": "json_object"}
    )

    try:
        result = json.loads(completion.choices[0].message.content)
        return result.get('keywords', [])
    except Exception as e:
        print(f"Error parsing JSON: {e}")
        return []

def search_jobs(keywords, locations=["Bengaluru", "Hyderabad", "Chennai"], results_per_location=10):
    all_jobs = []

    # join keywords into a search string if it's a list
    search_term = " ".join(keywords) if isinstance(keywords, list) else keywords

    for loc in locations:
        try:
            jobs = scrape_jobs(
                site_name=["linkedin", "indeed", "glassdoor"],
                search_term=search_term,
                location=f"{loc}, India",
                results_wanted=results_per_location,
                hours_old=72,
                country_indeed='india'
            )
            if not jobs.empty:
                all_jobs.append(jobs)
        except Exception as e:
            print(f"Error searching in {loc}: {e}")

    if not all_jobs:
        return pd.DataFrame()

    return pd.concat(all_jobs).drop_duplicates(subset=['job_url'])

def evaluate_job_match(resume_text, job_description, job_title, api_key):
    client = Groq(api_key=api_key)

    prompt = f"""
    Compare the following resume text with the job description for the position of '{job_title}'.

    Resume Text:
    {resume_text}

    Job Description:
    {job_description}

    Evaluate the match and provide:
    1. A matching score from 0 to 100.
    2. Key skills found in both.
    3. Missing skills or gaps.
    4. A brief justification.

    Return the result as a JSON object with the following keys:
    'score' (integer), 'matched_skills' (list), 'missing_skills' (list), 'justification' (string).
    """

    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-specdec",
            messages=[
                {"role": "system", "content": "You are a professional technical recruiter evaluating job matches. Respond only with JSON."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )

        result = json.loads(completion.choices[0].message.content)
        return result
    except Exception as e:
        print(f"Error evaluating match: {e}")
        return {
            "score": 0,
            "matched_skills": [],
            "missing_skills": [],
            "justification": f"Error: {str(e)}"
        }
