import time
import streamlit as st
import streamlit.components.v1 as components

from utils.pdf_loader import extract_text_from_pdf
from utils.splitter import split_text
from utils.embeddings import (
    create_embeddings,
    create_query_embedding,
)
from utils.retriever import (
    create_vector_store,
    search_vector_store,
)
from utils.llm import ask_gemini


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="AskMyPDF",
    page_icon="📄",
    layout="wide",
)

st.markdown("""
<style>

/* Chat input */
[data-testid="stChatInput"] {
    border-radius: 14px;
}

[data-testid="stChatInput"] > div {
    border: 1px solid #4a4a4a !important;
    border-radius: 14px !important;
}

/* Remove red outline */
[data-testid="stChatInput"] textarea:focus {
    box-shadow: none !important;
    outline: none !important;
}

/* Normal blue highlight */
[data-testid="stChatInput"]:focus-within {
    border: 1px solid #4F8BF9 !important;
}

</style>
""", unsafe_allow_html=True)

st.title("📄 AskMyPDF")
st.caption("Ask questions, retrieve relevant information, and get AI-powered answers from your documents.")


# --------------------------------------------------
# Session State
# --------------------------------------------------

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "processed" not in st.session_state:
    st.session_state.processed = False

if "file_names" not in st.session_state:
    st.session_state.file_names = []

if "total_characters" not in st.session_state:
    st.session_state.total_characters = 0

if "all_chunks" not in st.session_state:
    st.session_state.all_chunks = None

if "chunk_sources" not in st.session_state:
    st.session_state.chunk_sources = None

if "embeddings" not in st.session_state:
    st.session_state.embeddings = None

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

st.sidebar.title("Upload PDF")

uploaded_files = st.sidebar.file_uploader(
    "Choose PDF(s)",
    type=["pdf"],
    accept_multiple_files=True,
    key=f"uploader_{st.session_state.uploader_key}",
)

reset_col, clear_col = st.sidebar.columns(2)

if reset_col.button("🗑 Reset Everything"):
    st.session_state.chat_history = []
    st.session_state.processed = False
    st.session_state.file_names = []
    st.session_state.total_characters = 0
    st.session_state.all_chunks = None
    st.session_state.chunk_sources = None
    st.session_state.embeddings = None
    st.session_state.vector_store = None
    st.session_state.uploader_key += 1
    st.rerun()

if clear_col.button("💬 Clear Conversation"):
    st.session_state.chat_history = []
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.info("Gemini 3.6 Flash")

# --------------------------------------------------
# Process PDFs (Only When the Uploaded Set Changes)
# --------------------------------------------------

if uploaded_files:

    current_names = [pdf.name for pdf in uploaded_files]

    if (
        not st.session_state.processed
        or st.session_state.file_names != current_names
    ):

        st.session_state.chat_history = []

        all_chunks = []
        chunk_sources = []
        total_characters = 0

        with st.spinner("Reading PDFs..."):

            for pdf in uploaded_files:

                text = extract_text_from_pdf(pdf)

                total_characters += len(text)

                pdf_chunks = split_text(text)

                all_chunks.extend(pdf_chunks)

                chunk_sources.extend(
                    [pdf.name] * len(pdf_chunks)
                )

        with st.spinner("Creating embeddings..."):
            embeddings = create_embeddings(all_chunks)

        with st.spinner("Building vector store..."):
            vector_store = create_vector_store(embeddings)

        st.session_state.all_chunks = all_chunks
        st.session_state.chunk_sources = chunk_sources
        st.session_state.total_characters = total_characters
        st.session_state.embeddings = embeddings
        st.session_state.vector_store = vector_store
        st.session_state.file_names = current_names
        st.session_state.processed = True

    all_chunks = st.session_state.all_chunks
    chunk_sources = st.session_state.chunk_sources
    total_characters = st.session_state.total_characters
    embeddings = st.session_state.embeddings
    vector_store = st.session_state.vector_store

    if len(uploaded_files) == 1:
        st.success(f"{uploaded_files[0].name} uploaded successfully")
    else:
        st.success(f"{len(uploaded_files)} documents uploaded successfully")

    # --------------------------------------------------
    # Sidebar: Uploaded Documents
    # --------------------------------------------------

    st.sidebar.markdown("---")
    st.sidebar.subheader("Uploaded Documents")

    for pdf in uploaded_files:
        st.sidebar.write(f"• {pdf.name}")

    # --------------------------------------------------
    # Sidebar Statistics
    # --------------------------------------------------

    st.sidebar.markdown("---")
    st.sidebar.subheader("Document Statistics")

    st.sidebar.metric("Documents", len(uploaded_files))
    st.sidebar.metric("Characters", f"{total_characters:,}")
    st.sidebar.metric("Chunks", len(all_chunks))
    st.sidebar.metric("Embedding Dimension", len(embeddings[0]))
    st.sidebar.metric("Vectors Stored", vector_store.ntotal)

    # --------------------------------------------------
    # Chat
    # --------------------------------------------------

    st.subheader("Chat with your PDFs")

    question = st.chat_input(
        "Ask a question about the uploaded PDF(s)..."
    )

    if question:

        start_time = time.time()

        with st.chat_message("user", avatar="👩"):
            st.write(question)

        try:

            with st.spinner("Searching relevant information..."):

                query_embedding = create_query_embedding(question)

                retrieved_chunks = search_vector_store(
                    vector_store,
                    query_embedding,
                    all_chunks,
                    chunk_sources,
                    k=6
                )

            context = "\n\n".join(
                chunk["text"]
                for chunk in retrieved_chunks
            )

            with st.spinner("Generating answer..."):

                answer = ask_gemini(
                    question,
                    context
                )

            elapsed = time.time() - start_time

            st.session_state.chat_history.append(
                {
                    "question": question,
                    "answer": answer,
                    "chunks": retrieved_chunks,
                    "time": elapsed,
                }
            )

        except Exception as e:

            st.error(f"❌ Error: {e}")

    # --------------------------------------------------
    # Conversation History
    # --------------------------------------------------

    if st.session_state.chat_history:

        st.markdown("---")
        st.subheader("Conversation")

        for chat in reversed(st.session_state.chat_history):

            with st.chat_message("user", avatar="👩"):
                st.write(chat["question"])

            with st.chat_message("assistant", avatar="🤖"):

                st.write(chat["answer"])

                st.caption(
                    f"{chat['time']:.2f}s response time"
                )

            st.caption("Retrieved Context")

            for i, chunk in enumerate(chat["chunks"], start=1):

                with st.expander(f"{chunk['source']} • Chunk {i}"):
                    st.write(chunk["text"])

        # Auto-scroll to the latest message
        components.html(
            """
            <div id="scroll-anchor"></div>
            <script>
                var anchor = window.parent.document.getElementById("scroll-anchor");
                if (anchor) {
                    anchor.scrollIntoView({behavior: "smooth", block: "end"});
                } else {
                    window.parent.scrollTo(0, window.parent.document.body.scrollHeight);
                }
            </script>
            """,
            height=0,
        )

    # --------------------------------------------------
    # Debug Preview (developer-only)
    # --------------------------------------------------

    with st.sidebar.expander("Developer Tools"):
        st.caption(f"Source: {chunk_sources[0]}")
        st.text_area(
            "Chunk 1 content",
            value=all_chunks[0],
            height=250,
            disabled=True
        )

else:

    st.info("""
Welcome!

Upload one or more PDFs from the sidebar to begin chatting with your documents.
""")

# --------------------------------------------------
# Footer
# --------------------------------------------------

st.caption("Streamlit · Sentence Transformers · FAISS · Gemini 3.6 Flash")