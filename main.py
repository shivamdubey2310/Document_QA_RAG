from fastapi import FastAPI
from fastapi import Request
from pydantic import BaseModel
from typing import List
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import time
import logging
from logging.handlers import RotatingFileHandler
import os
import sys
import requests

## allowing imports from src folder
sys.path.append("src")

from embeddings import EmbeddingModel
from vectorstore import VectorStore
from llm_service import generate_response


app = FastAPI(title="Document Q&A")

## logging setup 
logger = logging.getLogger("rag_logger")
logger.setLevel(logging.INFO)

handler = RotatingFileHandler("app.log", maxBytes=5_000_000, backupCount=3)
formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
handler.setFormatter(formatter)

if not logger.handlers:
    logger.addHandler(handler)

## request schema
class QueryRequest(BaseModel):
    query: str


## response schema
class QueryResponse(BaseModel):
    answer: str
    sources: List[str]


## initialize models once at startup
embedder = EmbeddingModel()
vectorstore = VectorStore(persist_dir="vectorDB")

## middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()

    response = await call_next(request)

    process_time = time.time() - start_time

    logger.info(f"{request.method} {request.url} | Time: {process_time:.4f}s")

    return response



@app.get("/")
def home():
    return "Hii"


@app.post("/query", response_model=QueryResponse)
def query_documents(request: QueryRequest):

    start_time = time.time()

    query = request.query

    ## generate query embedding
    query_embedding = embedder.embed([query])

    ## retrieve documents
    results = vectorstore.collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=5
    )

    docs = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    similarity_threshold = 1.2

    filtered_docs = []
    sources = []

    for doc, meta, dist in zip(docs, metadatas, distances):
        if dist < similarity_threshold:
            filtered_docs.append(doc)
            source = f"{meta.get('source')} (page {meta.get('page')})"
            sources.append(source)

    if len(filtered_docs) == 0:
        return QueryResponse(
            answer="No relevant information found in the knowledge base.",
            sources=[]
        )

    ## build context
    context = "\n\n---\n\n".join(filtered_docs)

    ## generate LLM answer
    answer = generate_response(query, context)

    response_time = time.time() - start_time

    logger.info(
        f"QUERY: {query} | "
        f"ANSWER: {answer[:200]} | "   # truncate to avoid huge logs
        f"SOURCES: {list(set(sources))} | "
        f"TIME: {response_time:.4f}s"
    )

    return QueryResponse(
        answer=answer,
        sources=list(set(sources))
    )