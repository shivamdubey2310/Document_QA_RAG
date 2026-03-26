from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import os
import sys

## allowing imports from src folder
sys.path.append("src")

from embeddings import EmbeddingModel
from vectorstore import VectorStore
from llm_service import generate_response


app = FastAPI(title="Policy Intelligence Engine API")
app.mount("/ui", StaticFiles(directory="ui"), name="ui")

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


@app.get("/")
def home():
    return FileResponse("ui/index.html")


@app.post("/query", response_model=QueryResponse)
def query_documents(request: QueryRequest):

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

    return QueryResponse(
        answer=answer,
        sources=list(set(sources))
    )