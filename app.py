import streamlit as st

from utils.pdf_loader import extract_text_from_pdf
from utils.splitter import split_text
from utils.embeddings import (
    create_embeddings,
    create_query_embedding
)
from utils.retriever import (
    create_vector_store,
    search_vector_store
)
from utils.llm import ask_gemini

# ----------------------------------
# Page Configuration
# ----------------------------------
st.set_page_config(
    page_title="AskMyPDF",
    page_icon="📄",
    layout="wide"
)

# ----------------------------------
# Title
# ----------------------------------
st.title("📄 AskMyPDF")
st.subheader("Chat with your PDF using AI")

# ----------------------------------
# Upload PDF
# ----------------------------------
uploaded_file = st.file_uploader(
    "Upload a PDF",
    type=["pdf"]
)

# ----------------------------------
# Process Uploaded PDF
# ----------------------------------
if uploaded_file:

    st.success(f"Uploaded: {uploaded_file.name}")

    # Read PDF
    with st.spinner("Reading PDF..."):
        text = extract_text_from_pdf(uploaded_file)

    # Split into chunks
    chunks = split_text(text)

    # Generate embeddings
    with st.spinner("Generating embeddings..."):
        embeddings = create_embeddings(chunks)

    # Create FAISS vector store
    with st.spinner("Creating vector store..."):
        vector_store = create_vector_store(embeddings)

    # ----------------------------------
    # Document Statistics
    # ----------------------------------
    st.subheader("📊 Document Statistics")

    st.write(f"**Characters:** {len(text)}")
    st.write(f"**Chunks Created:** {len(chunks)}")
    st.write(f"**Embeddings Created:** {len(embeddings)}")
    st.write(f"**Vectors Stored:** {vector_store.ntotal}")

    # ----------------------------------
    # Embedding Preview
    # ----------------------------------
    st.subheader("🧠 Embedding Preview")

    st.write(f"**Embedding Dimension:** {len(embeddings[0])}")

    st.write("**First 10 values of Chunk 1's embedding:**")

    st.code(str(embeddings[0][:10]), language="text")

    # ----------------------------------
    # Vector Store
    # ----------------------------------
    st.subheader("📦 Vector Store")

    st.success("FAISS vector store created successfully!")

    st.write(f"Total vectors stored: {vector_store.ntotal}")

    # ----------------------------------
    # Ask a Question
    # ----------------------------------
    st.subheader("❓ Ask a Question")

    question = st.text_input(
        "Ask something about the uploaded PDF"
    )

    if question:

        # Search document
        with st.spinner("Searching document..."):

            query_embedding = create_query_embedding(question)

            retrieved_chunks = search_vector_store(
                vector_store,
                query_embedding,
                chunks,
                k=3
            )

        # Combine retrieved chunks
        context = "\n\n".join(retrieved_chunks)

        # Ask Gemini
        with st.spinner("Generating answer with Gemini..."):
            answer = ask_gemini(question, context)

        # Display answer
        st.subheader("🤖 AI Answer")
        st.success(answer)

        # Optional: Show retrieved chunks
        with st.expander("🔍 View Retrieved Chunks"):
            for i, chunk in enumerate(retrieved_chunks, start=1):
                st.markdown(f"### Chunk {i}")
                st.write(chunk)

    # ----------------------------------
    # Debug Section
    # ----------------------------------
    st.subheader("📄 First Chunk Preview")

    st.text_area(
        "Chunk Content",
        chunks[0],
        height=300
    )