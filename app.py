# === Project Structure ===
# .
# ├── app.py
# ├── data/
# │   └── AI_Training_Document.pdf
# ├── chunks/
# ├── vectordb/
# ├── src/
# │   ├── chunker.py
# │   ├── embedder.py
# │   └── qa.py
# ├── requirements.txt
# └── README.md






# === app.py ===
import streamlit as st
from src.chunker import load_and_chunk
from src.embedder import create_vector_store, load_vector_store
from src.qa import get_qa_chain
import os
from dotenv import load_dotenv

st.set_page_config("RAG Chatbot")
st.title("RAG Chatbot - AI Training Document")

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not os.path.exists("chunks"):
    with st.spinner("Chunking PDF..."):
        chunks = load_and_chunk("data/AI_Training_Document.pdf", "chunks")
else:
    from langchain.docstore.document import Document
    chunks = [Document(page_content=open(f"chunks/{f}").read()) for f in os.listdir("chunks")]

if not os.path.exists("vectordb/index"):
    with st.spinner("Embedding chunks..."):
        create_vector_store(chunks, "vectordb")

vectordb = load_vector_store("vectordb")
qa = get_qa_chain(vectordb, GROQ_API_KEY)

query = st.text_input("Ask a question from the document:")
if query:
    with st.spinner("Thinking..."):
        response = qa(query)
        st.success(response)

