import streamlit as st
from utils.pdf_loader import extract_text_from_pdf
from utils.splitter import split_text

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

    st.subheader("Document Statistics")
    st.write(f"Characters: {len(text)}")
    st.write(f"Chunks Created: {len(chunks)}")

    st.subheader("First Chunk")
    st.text_area(
        "Chunk Preview",
        chunks[0],
        height=300
    )