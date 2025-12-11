from google import genai
import json
import os

def load_gemini_config():
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    with open(config_path, "r") as f:
        return json.load(f)

def ask_gemini(question, context):
    cfg = load_gemini_config()
    api_key = cfg.get("gemini_api_key")
    model = cfg.get("gemini_model", "gemini-1.5-pro")

    if not api_key:
        raise ValueError("Gemini API key missing in config.json")

    client = genai.Client(api_key=api_key)

    prompt = f"""
You are a RAG answering system. Answer ONLY from the provided context. 
If context does not contain enough info, say "I don't know."

Context:
{context}

User question:
{question}

Answer:
"""

    resp = client.models.generate_content(
        model=model,
        contents=prompt
    )

    return resp.text
