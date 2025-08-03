import os
import requests
import io
from dotenv import load_dotenv

from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
import google.generativeai as genai

from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from mangum import Mangum

load_dotenv()

app = FastAPI(
    title="LLM RAG Pipeline",
    description="An API to answer questions about a document using a RAG pipeline.",
    version="1.0.0"
)

bearer_scheme = HTTPBearer()
AUTH_API_KEY = os.getenv("AUTH_API_KEY")

try:
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
except Exception as e:
    print(f"Error configuring Google AI: {e}")

EMBEDDER = SentenceTransformer("all-MiniLM-L6-v2")
LLM = genai.GenerativeModel('gemini-1.5-flash')
RAG_CACHE = {} 


class RequestPayload(BaseModel):
    documents: str = Field(..., example="https://hackrx.blob.core.windows.net/assets/policy.pdf?...")
    questions: list[str] = Field(..., example=["What is the grace period?"])

class ResponsePayload(BaseModel):
    answers: list[str]

def _load_pdf_from_url(url: str) -> str:
    """Downloads a PDF from a URL and extracts its text."""
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status() 
        
        pdf_file = io.BytesIO(response.content)
        reader = PdfReader(pdf_file)
        text = "\n".join(page.extract_text() for page in reader.pages if page.extract_text())
        return text
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Could not fetch or read PDF from URL: {e}")

def _build_rag_index(text: str):
    chunks = [text[i:i+1000] for i in range(0, len(text), 800)]
    
    embs = EMBEDDER.encode(chunks, convert_to_numpy=True)
    
    dim = embs.shape[1]
    idx = faiss.IndexFlatL2(dim)
    idx.add(embs)
    
    return {"chunks": chunks, "index": idx}

def get_rag_components(url: str):
    if url not in RAG_CACHE:
        print(f"Building RAG index for new URL: {url[:50]}...")
        text = _load_pdf_from_url(url)
        RAG_CACHE[url] = _build_rag_index(text)
    return RAG_CACHE[url]

def answer_question(query: str, rag_components: dict) -> str:
    chunks = rag_components["chunks"]
    idx = rag_components["index"]
    
    q_emb = EMBEDDER.encode([query], convert_to_numpy=True)
    _, I = idx.search(q_emb, k=3) 
    
    context = "\n---\n".join(chunks[i] for i in I[0])

    prompt = f"""
    You are a helpful Q&A assistant. Your task is to answer the user's question based *only* on the provided text excerpts from a document.
    Do not use any external knowledge. If the answer is not found in the provided text, state that clearly.
    Provide a direct and concise answer.

    Policy Excerpts:
    {context}

    Question:
    {query}

    Answer:
    """
    
    try:
        response = LLM.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"Error generating content from LLM: {e}")
        return "Error: Could not generate an answer."



def api_key_auth(credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)):
    if not credentials or credentials.scheme != "Bearer" or credentials.credentials != AUTH_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )

@app.post("/hackrx/run", response_model=ResponsePayload, dependencies=[Depends(api_key_auth)])
async def run_rag_pipeline(payload: RequestPayload):

    document_url = payload.documents
    questions = payload.questions
    
    rag_components = get_rag_components(document_url)
    
    answers = [answer_question(q, rag_components) for q in questions]
    
    return ResponsePayload(answers=answers)

@app.get("/", include_in_schema=False)
def root():
    return {"message": "API is running. Send POST requests to /hackrx/run"}

handler = Magnum(app)
