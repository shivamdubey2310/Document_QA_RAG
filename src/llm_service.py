from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
load_dotenv()

## LLM 
try :
    llm = ChatGroq(
        model = "llama-3.3-70b-versatile",
        temperature = 0,
        api_key = os.getenv("GROQ_API_KEY")
    )
except Exception as e:
    print(f"Error in loading model : {str(e)} ")

def rewrite_query(query):
    """This function improves retrieval quality when user asks vague questions."""

    prompt = f"""
    Rewrite the user question so it becomes clearer and better for document retrieval.

    User Question:
    {query}

    Improved Question:
    """
    response = llm.invoke(prompt)
    return response.content.strip()


def generate_response(query,context):
    """This function calls the LLM and answers the user query"""

    prompt = f"""
    You are an expert assistant specialized in analyzing multiple documents.

    You are given a context that may contain information from different documents 
    (such as resumes, CVs, identity cards, forms, etc.).

    Your task is to answer the user's question using ALL relevant information from the context.

    Instructions:
    1. Carefully read the entire context. It may contain multiple documents.
    2. Extract and combine relevant information from ALL parts of the context.
    3. Do NOT rely on a single section if multiple sections contain useful data.
    4. If the same information appears in multiple places, consolidate it.
    5. If there is conflicting information, mention the conflict clearly.
    6. Do NOT use any external knowledge. Only use the provided context.
    7. If the answer is not present in the context, say:
    "The answer is not available in the provided documents."
    8. Keep the answer clear, factual, and concise.
    9. When possible, mention which type of document the information came from 
    (e.g., Resume, Aadhar, CV, etc.) based on the context.

    Context : {context}

    User Question : {query}

    Answer:
    """

    response = llm.invoke(prompt)
    return response.content