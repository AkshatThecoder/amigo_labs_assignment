from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
import os

# Using Groq-compatible wrapper
class GroqLLM:
    def __init__(self, api_key):
        self.api_key = api_key

    def call(self, prompt):
        import requests
        res = requests.post(
            url="https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": "llama3-70b-8192",
                "messages": [{"role": "user", "content": prompt}],
                "stream": False
            }
        )
        return res.json()['choices'][0]['message']['content']


def get_qa_chain(vectordb, groq_api_key):
    retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    def qa_fn(query):
        docs = retriever.get_relevant_documents(query)
        context = "\n\n".join([doc.page_content for doc in docs])
        prompt = f"Answer the following based on context:\n\n{context}\n\nQuestion: {query}"
        return GroqLLM(groq_api_key).call(prompt)

    return qa_fn