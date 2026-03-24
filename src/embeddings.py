from sentence_transformers import SentenceTransformer

class EmbeddingModel:
    """This class initializes the embedding model and create the embeddings"""
    
    ## initializes the embedding model
    def __init__(self,embedding_model="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(embedding_model)

    ## This function creates the embeddings
    def embed(self,texts):
        embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            normalize_embeddings=True
        )
        return embeddings