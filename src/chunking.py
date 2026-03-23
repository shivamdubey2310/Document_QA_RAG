from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_chunks(docs):
    """"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 300,
        chunk_overlap = 50
    )
    chunks = splitter.split_documents(docs)
    return chunks 


