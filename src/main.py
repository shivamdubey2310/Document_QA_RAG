## importing required libraries
from loader import load_files
from chunking import split_documents
from embeddings import EmbeddingModel
from vectorstore import VectorStore
from llm_service import generate_response , rewrite_query
import os


## set vectorDB path (relative to workspace root)
VECTOR_DB_PATH = "../vectorDB"  # use same folder as main.py

## initialization 
embedder = EmbeddingModel()
vectorstore = VectorStore(persist_dir=VECTOR_DB_PATH)


# def ingest_documents():
#     """Create vector database if it does not exist"""

#     if not os.path.exists(VECTOR_DB_PATH) or len(os.listdir(VECTOR_DB_PATH)) == 0:

#         print("Creating vectorDB")
#         print("IF codn")
#         # ## load all the files 
#         # docs = load_files("../dataset")
#         # print("docs = ",docs)
        
#         # ## Split document into chunks 
#         # chunks = split_documents(docs)
#         # print("\n chunks = ",chunks)
        
#         # ## fetch page content from chunks
#         # texts = [c.page_content for c in chunks]

#         # ## create embeddings 
#         # embeddings = embedder.embed(texts)
#         # print("embeddings =",embeddings)

#         # ## adds chunks to vectorDB
#         # vectorstore.add_documents(chunks, embeddings)

#     else:
#         print("vectorDB already exists so skipping ingestion")


def ingest_documents():
    """Create vector database only if it's empty"""

    try:
        existing_count = vectorstore.collection.count()
    except:
        existing_count = 0

    if existing_count == 0:
        print("Creating vectorDB...")

        ## load files
        docs = load_files("../dataset")

        ## chunking
        chunks = split_documents(docs)

        ## extract text
        texts = [c.page_content for c in chunks]

        ## create embeddings
        embedder = EmbeddingModel()
        embeddings = embedder.embed(texts)

        ## store in DB
        vectorstore.add_documents(chunks, embeddings)

        ## persist DB (VERY IMPORTANT)
        # vectorstore.persist()

        print("VectorDB created successfully ✅")

    else:
        print(f"VectorDB already exists with {existing_count} documents. Skipping ingestion ✅")


def retrieve_documents():
    """Retrieve relevant documents from vectorDB"""
    query = "Tell me about projects"

    query_embedding = embedder.embed([query])
    print("retrieve documents ")
    results = vectorstore.collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=5
    )
    print(results["documents"][0])

ingest_documents()

retrieve_documents()

#     docs = results["documents"][0]
#     metadatas = results["metadatas"][0]
#     distances = results["distances"][0]

#     filtered_docs = []
#     sources = []

#     similarity_threshold = 1.2

#     for doc, meta, dist in zip(docs , metadatas , distances):

#         if dist < similarity_threshold:
#             filtered_docs.append(doc)
#             source = f"{meta['source']} (page {meta['page']})"
#             sources.append(source)

#     return filtered_docs, sources


# def build_context(filtered_docs):
#     """Create context for the LLM"""

#     if len(filtered_docs) == 0:
#         return None

#     return "\n\n---\n\n".join(filtered_docs)


# def answer_query(query):
#     """Full RAG pipeline for answering a question"""

#     filtered_docs, sources = retrieve_documents(query)

#     context = build_context(filtered_docs)

#     if context is None:
#         print("No relevant documents found.")
#         return

#     answer = generate_response(query, context)
    
#     ## prints answer
#     print("\nAnswer:\n")
#     print(answer)

#     ## prints response
#     print("\nSources:\n")
#     for s in set(sources):
#         print("-", s)

# ## main function
# def main():
# ingest_documents()
#     while True:
#         query = input("\nEnter your query : ")
#         if query.lower() == "exit":
#             break
#         else:
#             answer_query(query)

# if __name__ == "__main__":
#     main()