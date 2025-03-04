import os
import json
import streamlit as st
import requests
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFaceHub
from langchain_community.embeddings import HuggingFaceEmbeddings

# Disable CORS/WebSocket issues
os.environ["STREAMLIT_SERVER_ENABLE_CORS"] = "false"
os.environ["STREAMLIT_SERVER_ENABLE_WEBSOCKET_COMPRESSION"] = "false"

# Load environment variables
load_dotenv()

# Set HuggingFace & Gemini API keys
HUGGINGFACE_API_KEY = "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"
GEMINI_API_KEY = "AIzaSyDUtxtmf8pkMHjHbbwzspYogxAtQW9M3xw"
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"

# Function to extract text from PDF
def load_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    return "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])

# Function to create knowledge base from PDF text
def create_knowledge_base(text):
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    embeddings = HuggingFaceEmbeddings()
    return FAISS.from_texts(chunks, embeddings)

# Function to query Gemini API
def query_gemini(query, context=""):
    payload = {"contents": [{"parts": [{"text": f"{context}\n{query}"}]}]}
    response = requests.post(GEMINI_URL, json=payload)
    
    if response.status_code == 200:
        result = response.json()
        return result["candidates"][0]["content"]["parts"][0]["text"]
    else:
        return "Error: Gemini API request failed."

# Streamlit app setup
def main():
    st.set_page_config(page_title="PDF Q&A")
    st.title("📄 Ask Your PDF (Falcon-7B + Gemini) 🤖")

    pdf = st.file_uploader("📤 Upload your PDF", type="pdf")

    if pdf is not None:
        if "knowledge_base" not in st.session_state:
            with st.spinner("🔄 Processing PDF... Please wait."):
                text = load_pdf(pdf)
                st.session_state.knowledge_base = create_knowledge_base(text)
                st.session_state.text = text  # Store extracted text for Gemini
            st.success("✅ PDF processed! You can now ask questions.")

        query = st.text_input("🔎 Ask a question about the PDF:")

        if query:
            os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACE_API_KEY
            llm = HuggingFaceHub(repo_id="tiiuae/falcon-7b-instruct", model_kwargs={"temperature": 0.1, "max_length": 512})

            try:
                with st.spinner("🤖 Generating answer..."):
                    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=st.session_state.knowledge_base.as_retriever())
                    response = qa_chain.run(query)

                if response:
                    st.write(f"**Falcon-7B:** {response}")
                else:
                    raise ValueError("Falcon-7B did not return a response.")

            except Exception as e:
                st.warning(f"⚠️ Falcon-7B failed: {e}")
                st.info("💡 Switching to Gemini AI for faster response...")
                gemini_response = query_gemini(query, st.session_state.text)
                st.write(f"**Gemini AI:** {gemini_response}")

if __name__ == "__main__":
    main()
