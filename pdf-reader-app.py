import streamlit as st
import pandas as pd
import os
from langchain.chat_models import ChatOpenAI
from pypdf import PdfReader
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.callbacks import get_openai_callback

# Configure streamlit page
st.set_page_config(page_title="PDF Q & A", layout="wide")

# App title
st.title("PDF Q & A")

with st.sidebar:
    st.title("How to get started")
    st.markdown("""
    1. Enter your OpenAI API key
    2. Upload a PDF
    3. Ask questions
    """)
    
    api_key = st.text_input("Enter your OpenAI key:", type="password")
    os.environ["OPENAI_API_KEY"] = api_key

# Save api key as an environment variable to authenticate OpenAI API
if api_key:
    os.environ["OPENAI_API_KEY"] = api_key
    st.success("API key set successfully.")
else:
    st.warning("Please enter a valid OpenAI API key.")

# Process text and create vector store
def process_text(text):
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    
    # Create embeddings
    embeddings = OpenAIEmbeddings()
    
    # Create vector store
    knowledge_base = FAISS.from_texts(chunks, embeddings)
    
    return knowledge_base

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

uploaded_file = st.file_uploader("Upload your PDF document", type="pdf")

if uploaded_file is not None:
    st.success("PDF uploaded successfully.")
    
    # Create a spinner while processing 
    with st.spinner("Processing the document..."):
    
        text = extract_text_from_pdf(uploaded_file)
        
        # Process text and create knowledge base
        if api_key and api_key.strip():
            try:
                knowledge_base = process_text(text)
                st.session_state.knowledge_base = knowledge_base
                st.success("Document processed.")
            except Exception as e:
                st.error(f"Error processing document: {e}")
                if "openai" in str(e).lower():
                    st.warning("Please ensure your OpenAI API key is valid.")
        else:
            st.warning("Please add your API key to process the document.")
    
    # User 
    query = st.text_area("Ask a question about your document:")
    
    # Button to submit query
    if st.button("Submit"):
        if not api_key:
            st.warning("Please enter your OpenAI API key.")
        elif not query:
            st.warning("Please enter a question.")
        elif not hasattr(st.session_state, 'knowledge_base'):
            st.warning("Please wait for document processing to complete.")
        else:
            with st.spinner("Generating answer..."):
                try:
                    # Search for relevant documents
                    docs = st.session_state.knowledge_base.similarity_search(query, k=4)
                    
                    # Initialize LLM and QA chain
                    llm = ChatOpenAI(model_name="gpt-4.1", temperature=0)
                    chain = load_qa_chain(llm, chain_type="stuff")
                    
                    # Get answer from the chain
                    with get_openai_callback() as cb:
                        response = chain.run(input_documents=docs, question=query)
                    
                    # Display answer
                    st.header("Answer")
                    st.write(response)
                            
                except Exception as e:
                    st.error(f"Error generating answer: {e}")
else:
    st.info("Please upload a PDF document.")

