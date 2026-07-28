import streamlit as st
from utils.pdf_loader import extract_text_from_pdf

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

    st.subheader("Extracted Text")

    st.text_area(
        "PDF Content",
        text,
        height=400
    )