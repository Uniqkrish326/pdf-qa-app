import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS  # Updated import path
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFaceHub  # Updated import path
from langchain_community.embeddings import HuggingFaceEmbeddings  # Updated import path

# Fix CORS and WebSocket issues
os.environ["STREAMLIT_SERVER_ENABLE_CORS"] = "false"
os.environ["STREAMLIT_SERVER_ENABLE_WEBSOCKET_COMPRESSION"] = "false"

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

# Streamlit app setup
def main():
    load_dotenv()
    st.set_page_config(page_title="PDF Q&A")
    st.title("Ask Your PDF (Falcon-7B-Instruct) 🤓")

    pdf = st.file_uploader("Upload your PDF", type="pdf")
    if pdf is not None:
        text = load_pdf(pdf)
        knowledge_base = create_knowledge_base(text)

        query = st.text_input("Ask a question about the PDF:")
        if query:
            os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
            llm = HuggingFaceHub(repo_id="tiiuae/falcon-7b-instruct", model_kwargs={"temperature": 0.1, "max_length": 512})
            qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=knowledge_base.as_retriever())
            response = qa_chain.run(query)
            st.write(response)

if __name__ == "__main__":
    main()
