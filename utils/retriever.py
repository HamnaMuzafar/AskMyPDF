import faiss
import numpy as np


def create_vector_store(embeddings):
    """
    Create a FAISS index from embeddings.
    """

    # Convert embeddings to float32 (required by FAISS)
    embeddings = np.array(embeddings).astype("float32")

    # Get embedding dimension
    dimension = embeddings.shape[1]

    # Create FAISS index
    index = faiss.IndexFlatL2(dimension)

    # Add embeddings
    index.add(embeddings)

    return index