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

if "file_name" not in st.session_state:
    st.session_state.file_name = None

if "file_size" not in st.session_state:
    st.session_state.file_size = None

if "text" not in st.session_state:
    st.session_state.text = None

if "chunks" not in st.session_state:
    st.session_state.chunks = None

if "embeddings" not in st.session_state:
    st.session_state.embeddings = None

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

st.sidebar.title("Upload PDF")

uploaded_file = st.sidebar.file_uploader(
    "Choose a PDF",
    type=["pdf"]
)

reset_col, clear_col = st.sidebar.columns(2)

if reset_col.button("🗑 Reset Everything"):
    st.session_state.chat_history = []
    st.session_state.processed = False
    st.session_state.file_name = None
    st.session_state.file_size = None
    st.session_state.text = None
    st.session_state.chunks = None
    st.session_state.embeddings = None
    st.session_state.vector_store = None
    st.rerun()

if clear_col.button("💬 Clear Conversation"):
    st.session_state.chat_history = []
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.info("Gemini 3.6 Flash")

# --------------------------------------------------
# Process PDF (Only Once)
# --------------------------------------------------

if uploaded_file:

    # Only process if a new file is uploaded
    if (
        not st.session_state.processed
        or st.session_state.file_name != uploaded_file.name
    ):

        st.session_state.chat_history = []

        with st.spinner("Reading PDF..."):
            text = extract_text_from_pdf(uploaded_file)

        with st.spinner("Splitting document..."):
            chunks = split_text(text)

        with st.spinner("Creating embeddings..."):
            embeddings = create_embeddings(chunks)

        with st.spinner("Building vector database..."):
            vector_store = create_vector_store(embeddings)

        st.session_state.text = text
        st.session_state.chunks = chunks
        st.session_state.embeddings = embeddings
        st.session_state.vector_store = vector_store
        st.session_state.file_name = uploaded_file.name
        st.session_state.file_size = uploaded_file.size
        st.session_state.processed = True

    text = st.session_state.text
    chunks = st.session_state.chunks
    embeddings = st.session_state.embeddings
    vector_store = st.session_state.vector_store

    st.success(f"{uploaded_file.name} uploaded successfully")

    # --------------------------------------------------
    # Sidebar Current Document
    # --------------------------------------------------

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Current Document")
    st.sidebar.write(st.session_state.file_name)
    st.sidebar.caption(f"{st.session_state.file_size / 1024:.1f} KB")

    # --------------------------------------------------
    # Sidebar Statistics
    # --------------------------------------------------

    st.sidebar.markdown("---")
    st.sidebar.subheader("Document Statistics")

    st.sidebar.metric("Characters", f"{len(text):,}")
    st.sidebar.metric("Chunks", len(chunks))
    st.sidebar.metric("Embedding Dimension", len(embeddings[0]))
    st.sidebar.metric("Vectors Stored", vector_store.ntotal)

    # --------------------------------------------------
    # Chat
    # --------------------------------------------------

    st.subheader("Chat with your PDF")

    question = st.chat_input(
        "Ask a question about the uploaded PDF..."
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
                    chunks,
                    k=6
                )

            context = "\n\n".join(retrieved_chunks)

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

            with st.expander("Retrieved Context"):

                for i, chunk in enumerate(chat["chunks"], start=1):

                    st.text_area(
                        f"Chunk {i}",
                        chunk,
                        height=180,
                        disabled=True
                    )

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
        st.text_area(
            "Chunk 1 content",
            value=chunks[0],
            height=250,
            disabled=True
        )

else:

    st.info("""
Welcome!

Upload a PDF from the sidebar to begin chatting with your document.
""")

# --------------------------------------------------
# Footer
# --------------------------------------------------

st.caption("Streamlit · Sentence Transformers · FAISS · Gemini 3.6 Flash")