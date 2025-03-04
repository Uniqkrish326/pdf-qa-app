import os
import json
import streamlit as st
import requests
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter

# Load environment variables
load_dotenv()

# Set Gemini API key
GEMINI_API_KEY = "AIzaSyDUtxtmf8pkMHjHbbwzspYogxAtQW9M3xw"
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"

# Function to extract text from PDF
def load_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    return "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])

# Function to split text into chunks
def split_text(text):
    text_splitter = CharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    return text_splitter.split_text(text)

# Function to query Gemini API
def query_gemini(query, context=""):
    headers = {"Content-Type": "application/json"}
    payload = {"contents": [{"parts": [{"text": f"Context: {context}\nUser Question: {query}"}]}]}

    try:
        response = requests.post(GEMINI_URL, headers=headers, json=payload)

        if response.status_code == 200:
            result = response.json()
            return result["candidates"][0]["content"]["parts"][0]["text"]
        else:
            return f"⚠️ Error {response.status_code}: {response.text}"

    except requests.exceptions.RequestException as e:
        return f"🚨 Network Error: {e}"

# Streamlit app setup
def main():
    st.set_page_config(page_title="PDF Q&A with Gemini")
    st.title("📄 Ask Your PDF (Powered by Gemini) 🚀")

    pdf = st.file_uploader("📤 Upload your PDF", type="pdf")

    if pdf is not None:
        if "pdf_text" not in st.session_state:
            with st.spinner("🔄 Processing PDF... Please wait."):
                text = load_pdf(pdf)
                st.session_state.pdf_text = text
            st.success("✅ PDF processed! You can now ask questions.")

        query = st.text_input("🔎 Ask a question about the PDF:")

        if query:
            with st.spinner("🤖 Generating answer..."):
                response = query_gemini(query, st.session_state.pdf_text)
            st.write(f"**💡 Gemini AI:** {response}")

if __name__ == "__main__":
    main()
