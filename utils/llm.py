import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.3
)


def ask_gemini(question, context):

    prompt = f"""
You are an AI assistant that answers questions ONLY using the provided document.

Rules:

1. Answer ONLY from the provided context.
2. Do not make up facts.
3. If the answer is missing, reply exactly:
"I couldn't find the answer in the uploaded document."
4. Keep answers concise and clear.

Context:
----------------------------------
{context}
----------------------------------

Question:
{question}

Answer:
"""

    response = llm.invoke(prompt)

    if isinstance(response.content, list):

        texts = []

        for item in response.content:

            if (
                isinstance(item, dict)
                and item.get("type") == "text"
            ):
                texts.append(item["text"])

        return "\n".join(texts)

    return response.content