import os
from dotenv import load_dotenv, find_dotenv
import pdfplumber
from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document

def load_env():
    _ = load_dotenv(find_dotenv())

def get_open_api_key():
    load_env()
    return os.getenv("OPENAI_API_KEY")

def get_hf_token():
    load_env()
    return os.getenv("HF_TOKEN")

def get_nvidia_api_key():
    load_env()
    return os.getenv("NVIDIA_API_KEY")

def get_groq_api_key():
    load_env()
    return os.getenv("GROQ_API_KEY")


class API_KEY:
    load_dotenv()
    OPEN_API_KEY = os.getenv("OPENAI_API_KEY")
    NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
    HF_TOKEN = os.getenv("HF_TOKEN")
    HF_URL_BASE = "https://router.huggingface.co/hf-inference/v1"
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

class PDFPlumberReader(BaseReader):
    def lazy_load_data(self, file_path, **kwargs):
        docs = []
        with pdfplumber.open(file_path) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text:
                    docs.append(Document(text=text, metadata={"page": i+1}))
        return docs