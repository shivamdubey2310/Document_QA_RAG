from loader import load_files
from chunking import load_chunks

## Step-1 : Loading data 
docs = load_files("../data/Trishansh.pdf")
print(docs[0])

## Step-2 : Chunking 
chunks = load_chunks(docs)
print(f"No.of chunks = {len(chunks)}")