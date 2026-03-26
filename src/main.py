## importing required libraries
from loader_liteParser import load_files
from chunking import split_documents
from embeddings import EmbeddingModel
from vectorstore import VectorStore
from llm_service import generate_response , rewrite_query
from loader_images import load_images
import os


## set vectorDB path (relative to workspace root)
VECTOR_DB_PATH = "../vectorDB"  # use same folder as main.py

## initialization 
embedder = EmbeddingModel()
vectorstore = VectorStore(persist_dir=VECTOR_DB_PATH)


def ingest_documents(doc_type):
    """Create vector database only if it's empty"""

    try:
        existing_count = vectorstore.collection.count()
    except:
        existing_count = 0

    if existing_count == 0:
        print("Creating vectorDB...")

        if doc_type == 1: ## for pdf's
           docs = load_files("../dataset")
        else: ## for images 
           docs = load_images("../image-dataset")
       
        ## chunking
        chunks = split_documents(docs)

        ## extract text
        texts = [c.page_content for c in chunks]

        ## create embeddings
        embedder = EmbeddingModel()
        embeddings = embedder.embed(texts)

        ## store in DB
        vectorstore.add_documents(chunks, embeddings)

        print("VectorDB created successfully")

    else:
        print(f"VectorDB already exists with {existing_count} documents. Skipping ingestion")


def retrieve_documents(query):
    """Retrieve relevant documents from vectorDB"""

    improved_query = rewrite_query(query)
    query_embedding = embedder.embed([improved_query])

    results = vectorstore.collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=5
    )

    docs = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    filtered_docs = []
    sources = []

    similarity_threshold = 1.2

    for doc, meta, dist in zip(docs , metadatas , distances):

        if dist < similarity_threshold:
            filtered_docs.append(doc)
            source = f"{meta['source']} (page {(meta['page']+1)})"
            sources.append(source)

    return filtered_docs, sources


def build_context(filtered_docs):
    """Create context for the LLM"""

    if len(filtered_docs) == 0:
        return None

    return "\n\n---\n\n".join(filtered_docs)


def answer_query(query):
    """Full RAG pipeline for answering a question"""

    filtered_docs, sources = retrieve_documents(query)

    context = build_context(filtered_docs)

    if context is None:
        print("No relevant documents found.")
        return

    answer = generate_response(query, context)
    
    ## prints answer
    print("\nAnswer:\n")
    print(answer)

    ## prints response
    print("\nSources:\n")
    for s in set(sources):
        print("-", s)

## main function
def main():
    doc_type = input("Enter 1 for PDFs and 2 for image : ")
    ingest_documents(doc_type)
    while True:
        query = input("\nEnter your query : ")
        if query.lower() == "exit":
            break
        else:
            answer_query(query)
            
if __name__ == "__main__":
    main()