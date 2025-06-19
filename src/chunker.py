

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import os

def load_and_chunk(pdf_path, chunk_dir):
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(pages)

    # Filter out empty chunks
    valid_chunks = [chunk for chunk in chunks if chunk.page_content.strip()]
    
    os.makedirs(chunk_dir, exist_ok=True)
    for i, chunk in enumerate(valid_chunks):
        with open(f"{chunk_dir}/chunk_{i}.txt", "w") as f:
            f.write(chunk.page_content)

    print(f"[Chunker] Total chunks created: {len(valid_chunks)}")
    return valid_chunks
