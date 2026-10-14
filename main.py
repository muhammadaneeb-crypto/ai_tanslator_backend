
import os

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from groq import Groq

load_dotenv()

app = FastAPI(title="AI Translator Backend")

api_key = os.getenv("GROQ_API_KEY")
client = Groq(api_key=api_key) if api_key else None


class TranslationRequest(BaseModel):
    text: str = Field(min_length=1, max_length=10000)
    source_language: str = "Auto-detect"
    target_language: str = "English"


@app.get("/")
def home():
    return {"message": "AI Translator backend is running with Groq"}


@app.post("/translate")
def translate(request: TranslationRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text is empty")

    if client is None:
        raise HTTPException(
            status_code=500,
            detail="GROQ_API_KEY is not configured in backend .env"
        )

    try:
        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a professional translator. "
                        "Translate the user's text from the specified "
                        "source language to the specified target language. "
                        "Preserve the meaning and tone. "
                        "Return only the translation, without explanations."
                    )
                },
                {
                    "role": "user",
                    "content": (
                        f"Source language: {request.source_language}\n"
                        f"Target language: {request.target_language}\n\n"
                        f"Text to translate:\n{request.text}"
                    )
                }
            ],
            max_completion_tokens=2048
        )

        translated_text = response.choices[0].message.content

        if not translated_text:
            raise HTTPException(
                status_code=502,
                detail="Groq returned an empty translation"
            )

        return {"translated_text": translated_text.strip()}

    except HTTPException:
        raise
    except Exception as e:
        print(f"Groq API error: {type(e).__name__}: {e}")
        raise HTTPException(
            status_code=502,
            detail="Groq API request failed. Check the backend terminal."
        )
