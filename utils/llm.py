import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables
load_dotenv()

# Read API key from .env
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Initialize Gemini
llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.3
)


def ask_gemini(question, context):
    """
    Generate an answer using the retrieved document context.
    """

    prompt = f"""
You are an AI assistant that answers questions ONLY using the provided context.

If the answer is not present in the context, reply exactly:

"I couldn't find the answer in the uploaded document."

Context:
{context}

Question:
{question}

Answer:
"""

    response = llm.invoke(prompt)

    # Extract only the text from the response
    if isinstance(response.content, list):
        texts = []

        for item in response.content:
            if isinstance(item, dict) and item.get("type") == "text":
                texts.append(item["text"])

        return "\n".join(texts)

    return response.content