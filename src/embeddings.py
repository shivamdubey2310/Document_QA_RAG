import os
import requests

class EmbeddingModel:
    def __init__(self, model="sentence-transformers/all-MiniLM-L6-v2"):
        self.api_token = os.getenv("HF_KEY")

        self.url = f"https://router.huggingface.co/hf-inference/models/{model}/pipeline/feature-extraction"

        self.headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json"
        }

    def embed(self, texts):
        response = requests.post(
            self.url,
            headers=self.headers,
            json={"inputs": texts}   # can pass list directly
        )

        if response.status_code != 200:
            raise Exception(f"HF API error: {response.text}")

        return response.json()