import streamlit as st
from utils.pdf_loader import extract_text_from_pdf
from utils.splitter import split_text
from utils.embeddings import create_embeddings

st.set_page_config(
    page_title="AskMyPDF",
    page_icon="📄",
    layout="wide"
)

st.title("📄 AskMyPDF")
st.subheader("Chat with your PDF using AI")

uploaded_file = st.file_uploader(
    "Upload a PDF",
    type=["pdf"]
)

if uploaded_file:

    st.success(f"Uploaded: {uploaded_file.name}")

    with st.spinner("Reading PDF..."):
        text = extract_text_from_pdf(uploaded_file)

    chunks = split_text(text)
    embeddings = create_embeddings(chunks)
    st.subheader("Document Statistics")

    st.write(f"Characters: {len(text)}")
    st.write(f"Chunks Created: {len(chunks)}")
    st.write(f"Embeddings Created: {len(embeddings)}")
    st.subheader("🧠 Embedding Preview")

    st.write(f"**Embedding Dimension:** {len(embeddings[0])}")

    st.write("**First 10 values of Chunk 1's embedding:**")

    st.code(str(embeddings[0][:10]), language="text")
    st.subheader("First Chunk")
    st.text_area(
        "Chunk Preview",
        chunks[0],
        height=300
    )