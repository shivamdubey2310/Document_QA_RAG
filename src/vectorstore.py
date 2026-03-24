import chromadb
from chromadb import PersistentClient
import uuid

class VectorStore:
    """This class created the VectorDB collection"""
    
    def __init__(self,persist_dir="vectorDB",collection_name="documents"):
        self.client = chromadb.PersistentClient(path = persist_dir)
        self.collection = self.client.get_or_create_collection(
            name = collection_name,
            metadata={"hnsw:space": "cosine"} ## used cosine similarity as we have done normalized embeddings
        )

    def add_documents(self,chunks,embeddings):
        """This functions add the documents to vector-store"""

        if len(chunks) != len(embeddings): ## handles edge case 
            raise ValueError("Chunks and embeddings must have the same length")

        texts = [c.page_content for c in chunks]
        metadatas = [c.metadata for c in chunks]
        ids = [str(uuid.uuid4()) for _ in chunks] ## every chunk must have a unique ID.

        ## adding above data into collection
        self.collection.add(
            ids = ids,
            documents = texts,
            metadatas = metadatas,
            embeddings = embeddings.tolist()
        )
        print(f"{len(chunks)} documents stored in vectorDB")