from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_text(text):
    """
    Split text into smaller chunks for vector storage.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        length_function=len
    )

    chunks = splitter.split_text(text)

    return chunks