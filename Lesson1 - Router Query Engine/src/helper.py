import os
from dotenv import load_dotenv, find_dotenv
from llama_index.core.embeddings import BaseEmbedding
from huggingface_hub import InferenceClient
from typing import List
from pydantic import PrivateAttr

def load_env():
    _ = load_dotenv(find_dotenv())

def get_open_api_key():
    load_env()
    return os.getenv("OPENAI_API_KEY")

def get_hf_token():
    load_env()
    return os.getenv("HF_TOKEN")

class HFEmbedding(BaseEmbedding):
    model_name: str
    _client: InferenceClient = PrivateAttr()

    def __init__(self, model_name: str, api_key: str, **kwargs):
        super().__init__(model_name=model_name, **kwargs)
        self.model_name = model_name
        self._client = InferenceClient(api_key=api_key)

    def _get_text_embedding(self, text: str) -> List[float]:
        return self._client.feature_extraction(
            text=text,
            model=self.model_name
        )
    
    def _aget_query_embedding(self, text: str) -> List[float]:
        return self._get_text_embedding(text)

    def _get_query_embedding(self, query: str) -> List[float]:
        return self._get_text_embedding(query)
