# from llama_index.embeddings.nvidia import NVIDIAEmbedding

# from llama_index.core.readers.base import BaseReader
# from llama_index.core.schema import Document
# import pdfplumber
# from llama_index.core import SimpleDirectoryReader

# class PDFPlumberReader(BaseReader):
#     def lazy_load_data(self, file_path, **kwargs):
#         docs = []
#         with pdfplumber.open(file_path) as pdf:
#             for i, page in enumerate(pdf.pages):
#                 text = page.extract_text()
#                 if text:
#                     docs.append(Document(text=text, metadata={"page": i+1}))
#         return docs

# extractors_override = {
#     ".pdf": PDFPlumberReader()
# }

# reader = SimpleDirectoryReader(
#     input_files=["Lesson1 - Router Query Engine/metagpt.pdf"],
#     file_extractor=extractors_override
# )


# # documents = reader.load_data()
# # embed_model = NVIDIAEmbedding(
# #     api_key="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",
# #     model="nvidia/nemotron-3-embed-1b",
# #     input_type="passage",
# # )

# # embedding = embed_model.get_text_embedding(
# #     "LlamaIndex is a framework for building RAG applications."
# # )

# # print("Dimension:", len(embedding))
# # print(embedding[:5])



# # documents = SimpleDirectoryReader(input_files=["Lesson1 - Router Query Engine/resume.pdf"]).load_data()
# # print(len(documents))
# documents = reader.load_data()
# print(type(documents[0]))
# print(documents[0].text[:500])
from src.helper import API_KEY
from llama_index.llms.groq import Groq

print(API_KEY.HF_URL_BASE)

# Đảm bảo có đuôi /v1
llm = Groq(
    model="openai/gpt-oss-20b",
    api_key=API_KEY.GROQ_API_KEY,
    is_function_calling_model=False,
)

# Test kết nối trực tiếp
test_resp = llm.complete("Hello, are you ready?")
print("Test LLM thành công:", test_resp.text)