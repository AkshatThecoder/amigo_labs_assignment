

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
import os

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def create_vector_store(chunks, persist_directory):
    os.makedirs(persist_directory, exist_ok=True)
    texts = [doc.page_content for doc in chunks if doc.page_content.strip()]

    if not texts:
        raise ValueError("No text to embed. Check chunking logic.")

    test_embed = embedding_model.embed_documents(texts[:1])
    if not test_embed or len(test_embed[0]) == 0:
        raise ValueError("HuggingFace embeddings returned empty vectors. Check model or inputs.")

    vectordb = Chroma.from_texts(texts=texts, embedding=embedding_model, persist_directory=persist_directory)
    vectordb.persist()
    return vectordb

def load_vector_store(persist_directory):
    return Chroma(persist_directory=persist_directory, embedding_function=embedding_model)
