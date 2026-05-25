import unittest
import pandas as pd
from unittest.mock import MagicMock, patch
from utils import extract_text_from_pdf, extract_keywords_from_resume, search_jobs, evaluate_job_match

class TestUtils(unittest.TestCase):
    @patch('pypdf.PdfReader')
    def test_extract_text_from_pdf(self, mock_reader):
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "This is a test resume."
        mock_reader.return_value.pages = [mock_page]

        text = extract_text_from_pdf("dummy.pdf")
        self.assertEqual(text, "This is a test resume.")

    @patch('utils.Groq')
    def test_extract_keywords_from_resume(self, mock_groq):
        mock_client = MagicMock()
        mock_groq.return_value = mock_client

        mock_completion = MagicMock()
        mock_completion.choices[0].message.content = '{"keywords": ["Python", "Machine Learning"]}'
        mock_client.chat.completions.create.return_value = mock_completion

        keywords = extract_keywords_from_resume("This is a test resume text.", "fake_key")
        self.assertEqual(keywords, ["Python", "Machine Learning"])

    @patch('utils.scrape_jobs')
    def test_search_jobs(self, mock_scrape):
        mock_df = pd.DataFrame({
            'job_url': ['url1', 'url2'],
            'title': ['Dev', 'DS']
        })
        mock_scrape.return_value = mock_df

        jobs = search_jobs(["Python"])
        self.assertFalse(jobs.empty)
        self.assertEqual(len(jobs), 2) # It will call 3 times (3 locations) but we mock it once for simplicity if it returns the same

    @patch('utils.Groq')
    def test_evaluate_job_match(self, mock_groq):
        # same fix as before for Groq mock location
        mock_client = MagicMock()
        mock_groq.return_value = mock_client

        mock_completion = MagicMock()
        mock_completion.choices[0].message.content = '{"score": 85, "matched_skills": ["Python"], "missing_skills": ["Docker"], "justification": "Good match"}'
        mock_client.chat.completions.create.return_value = mock_completion

        result = evaluate_job_match("resume", "jd", "title", "fake_key")
        self.assertEqual(result['score'], 85)
        self.assertEqual(result['matched_skills'], ["Python"])

if __name__ == '__main__':
    unittest.main()
