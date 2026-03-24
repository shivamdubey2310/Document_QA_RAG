import json
import subprocess
from pathlib import Path
from langchain_core.documents import Document

def parse_with_liteparse(file_path):
    result = subprocess.run(
        ["lit", "parse", file_path, "--format", "json"],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print("LiteParse Error:", result.stderr)
        return {}

    try:
        return json.loads(result.stdout)
    except Exception as e:
        print("JSON Error:", e)
        return {}


def load_images(dataset_path):
    documents = []

    # Supported image formats
    image_extensions = ["*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp", "*.tiff"]

    image_files = []
    for ext in image_extensions:
        image_files.extend(Path(dataset_path).glob(f"**/{ext}"))

    print(f"{len(image_files)} image files loaded")

    for img in image_files:
        data = parse_with_liteparse(str(img))

        category = img.parent.name
        file_name = img.name

        for page_id, page in enumerate(data.get("pages", [])):

            text_items = page.get("textItems", [])

            text = " ".join([
            item.get("text", "")
            for item in text_items
            if item.get("text", "").strip()
            ])

            doc = Document(
                page_content=text,
                metadata={
                    "source": file_name,
                    "category": category,
                    "page": page_id + 1,
                    "file_path": str(img),
                    "type": "image"
                }
            )

            documents.append(doc)

        print(f"Loaded {len(data.get('pages', []))} pages from {file_name}")

    return documents


docs = load_images("images-dataset")
print(docs[0].page_content)