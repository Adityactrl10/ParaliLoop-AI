
import os
from pathlib import Path
from dotenv import load_dotenv
import streamlit as st
from google import genai
from google.genai import types

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

# Read credentials from local .env or Streamlit Cloud Secrets
try:
    API_KEY = (
        os.getenv("GEMINI_API_KEY")
        or st.secrets.get("GEMINI_API_KEY", "")
    )
    MODEL_NAME = (
        os.getenv("GEMINI_MODEL")
        or st.secrets.get("GEMINI_MODEL", "gemini-3.5-flash-lite")
    )
except Exception:
    API_KEY = os.getenv("GEMINI_API_KEY", "")
    MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")

SYSTEM_INSTRUCTION = """
You are ParaliLoop AI, an intelligent agricultural assistant
developed by Team TriVortex for Sustainability Hackathon at IIT Indore.

Help farmers, especially in Punjab and Haryana, with:
1. Crop residue and stubble management.
2. Alternatives to stubble burning.
3. Happy Seeder, Super Seeder and mulching.
4. Biomass management.
5. Satellite fire detection and fire-risk information.
6. Sustainable farming practices.

RULES:
- Use simple language.
- Support Hindi, English and Punjabi.
- Give practical step-by-step advice.
- Never claim satellite detections confirm stubble burning.
- Never invent government schemes, subsidies, prices or statistics.
- Do not claim access to live weather or satellite data unless supplied.
- Avoid unsafe pesticide dosage recommendations.
- Clearly mention uncertainty.
- You are an AI assistant, not a government authority.
"""

def get_gemini_client():
    if not API_KEY:
        raise ValueError(
            "Gemini API key missing. Check Streamlit Secrets or root .env file."
        )
    return genai.Client(api_key=API_KEY)


def ask_gemini(user_message, chat_history=None, farm_context=None):
    if not user_message or not user_message.strip():
        return "Please enter your farming question."

    try:
        client = get_gemini_client()

        prompt_parts = []

        if farm_context:
            prompt_parts.append(
                "PARALILOOP PROJECT DATA:\n" + str(farm_context)
            )

        if chat_history:
            prompt_parts.append("RECENT CONVERSATION:")

            for message in chat_history[-10:]:
                role = message.get("role", "user")
                content = message.get("content", "")

                if content:
                    label = "Farmer" if role == "user" else "ParaliLoop AI"
                    prompt_parts.append(f"{label}: {content}")

        prompt_parts.append(
            "CURRENT FARMER QUESTION:\n" + user_message.strip()
        )

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents="\n\n".join(prompt_parts),
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                temperature=0.5,
                max_output_tokens=1000
            )
        )

        answer = response.text

        if not answer:
            return "No response generated. Please try again."

        return answer.strip()

    except ValueError as error:
        return f"Configuration error: {error}"

    except Exception as error:
        print("Gemini API error:", repr(error))
        return (
            "Sorry, ParaliLoop AI could not connect right now. "
            "Check your API key, internet connection and API limits."
        )
