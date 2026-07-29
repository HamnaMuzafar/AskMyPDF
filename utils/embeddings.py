from sentence_transformers import SentenceTransformer

# Load the embedding model once
model = SentenceTransformer("all-MiniLM-L6-v2")


def create_embeddings(chunks):
    """
    Convert document chunks into embeddings.
    """
    return model.encode(chunks)


def create_query_embedding(query):
    """
    Convert a user's question into an embedding.
    """
    return model.encode(query)