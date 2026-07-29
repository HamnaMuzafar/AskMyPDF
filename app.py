import streamlit as st

from utils.pdf_loader import extract_text_from_pdf
from utils.splitter import split_text
from utils.embeddings import create_embeddings
from utils.retriever import create_vector_store


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
# Process PDF
# ----------------------------------
if uploaded_file:

    st.success(f"Uploaded: {uploaded_file.name}")

    # Extract text
    with st.spinner("Reading PDF..."):
        text = extract_text_from_pdf(uploaded_file)

    # Split text into chunks
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
    # Vector Store Information
    # ----------------------------------
    st.subheader("📦 Vector Store")

    st.success("FAISS vector store created successfully!")

    st.write(f"Total vectors in FAISS: {vector_store.ntotal}")

    # ----------------------------------
    # Chunk Preview
    # ----------------------------------
    st.subheader("📄 First Chunk Preview")

    st.text_area(
        label="Chunk Content",
        value=chunks[0],
        height=300
    )