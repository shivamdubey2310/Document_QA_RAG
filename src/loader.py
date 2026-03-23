## import required libraries 
from langchain_community.document_loaders import PyMuPDFLoader

def load_files(data_path):
    loader = PyMuPDFLoader(str(data_path))
    documents = loader.load()
    return documents 