import json
import subprocess
from pathlib import Path
from langchain_core.documents import Document

def parse_with_liteparse(pdf_path):
    result = subprocess.run(
        ["lit", "parse", pdf_path, "--format", "json"],
        capture_output=True,
        text=True,
        shell=True
    )

    # print(json.loads(result.stdout))
    return json.loads(result.stdout)


def load_files(dataset_path):
    documents = []
    pdf_files = list(Path(dataset_path).glob("**/*.pdf"))
    print(f"{len(pdf_files)} files loaded")

    for pdf in pdf_files:
        data = parse_with_liteparse(str(pdf))

        category = pdf.parent.name
        file_name = pdf.name

        for page_id, page in enumerate(data.get("pages", [])):
            
            text = page.get("text", "")

            # blocks = page.get("blocks", [])
            # text = "\n".join([b["text"] for b in blocks if b["text"].strip()])

            doc = Document(
                page_content=text,
                metadata={
                    "source": file_name,
                    "category": category,
                    "page": page_id + 1,
                    "file_path": str(pdf)
                }
            )
            documents.append(doc)

        print(f"Loaded {len(data.get('pages', []))} pages from {pdf.name}")
    
    return documents

