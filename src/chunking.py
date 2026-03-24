from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_documents(documents,chunk_size=800,chunk_overlap=200):
    """This function splits all the loaded documents into chunks"""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap
    )
    chunks = splitter.split_documents(documents)

    print(f"{len(chunks)} chunks created")

    return chunks 
