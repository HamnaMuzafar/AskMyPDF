import faiss
import numpy as np


def create_vector_store(embeddings):
    """
    Create a FAISS index from embeddings.
    """

    embeddings = np.array(embeddings).astype("float32")

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(embeddings)

    return index


def search_vector_store(index, query_embedding, chunks, k=3):
    """
    Search the vector store and return the most relevant chunks.
    """

    query_embedding = np.array([query_embedding]).astype("float32")

    distances, indices = index.search(query_embedding, k)

    results = []

    for idx in indices[0]:
        results.append(chunks[idx])

    return results