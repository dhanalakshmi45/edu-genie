import os

from dotenv import load_dotenv
from google import genai

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("gemini-3.5-flash")

client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None


def answer_question_with_gemini(question: str) -> str:
    """Answer an academic/general-knowledge question using Gemini."""

    if client is None:
        return "Error in QnA: GEMINI_API_KEY is not configured."

    try:
        response = client.models.generate_content(
            model="gemini-3.5-flash",
            contents=question,
        )

        if not response.text:
            return "Error in QnA: Gemini returned an empty response."

        return response.text.strip()

    except Exception as e:
        return f"Error in QnA: {e}"