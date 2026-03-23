# Document Q&A RAG System

## Project Objective
Build a Retrieval-Augmented Generation (RAG) based Document Q&A system that can:
- Ingest PDF documents from the `data/` folder.
- Split document content into semantic chunks.
- Convert chunks into vector embeddings.
- Store and search embeddings using a vector database.
- Retrieve the most relevant context for a user query.
- Generate accurate final answers using an LLM (Groq).

The goal is to improve answer quality by grounding LLM responses in document context instead of relying only on model memory.

## Implementation Steps

### 1. Load Documents
- Read PDF files from the `data/` directory.
- Use `PyMuPDFLoader` (`langchain_community.document_loaders`) to load the document text.
- Current loader implementation: `src/loader.py`.

### 2. Chunk Documents
- Split loaded documents into smaller chunks for better retrieval.
- Use `RecursiveCharacterTextSplitter` (`langchain_text_splitters`).
- Current chunking settings (from `src/chunking.py`):
	- `chunk_size = 300`
	- `chunk_overlap = 50`

### 3. Generate Embeddings
- Convert each chunk into vector embeddings.
- Embedding model planned: `all-MiniLM-L6-v2`.

### 4. Store in Vector Database
- Store chunk embeddings in a vector DB for fast similarity search.
- Planned vector database: `ChromaDB`.

### 5. Build Retriever
- Create a retriever on top of vector DB.
- Retrieve top relevant chunks for a question using cosine similarity.

### 6. Build Prompt
- Construct a prompt that combines:
	- User query
	- Retrieved context
- This ensures the LLM answer is grounded in source documents.

### 7. Call LLM
- Send the built prompt to an LLM endpoint.
- Planned provider: `Groq`.
- Return the final answer to the user.

## Current Code Structure
```text
main.py
Project Implementation.txt
pyproject.toml
requirements.txt
data/
src/
	chunking.py
	loader.py
	main.py
```

---