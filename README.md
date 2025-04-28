# PDF Q&A Application

This application allows users to upload PDF documents and ask questions about the content. It uses Streamlit for the UI and leverages LangChain, PdfReader, and OpenAI to implement a RAG system that provides answers based on the document's content.

## Features

- Upload and process PDF documents
- Extract text from PDFs
- Split text into manageable chunks
- Create vector embeddings for efficient retrieval
- Ask questions about the document content
- Get AI-generated answers based on document context

## Prerequisites

- Python 3.8+
- OpenAI API key

## Setup

Install the required dependencies:
   ```bash
   pip install streamlit langchain openai pypdf pandas faiss-cpu
   ```

## Usage

1. Run the application:
   ```bash
   streamlit run pdf-reader-app.py
   ```

2. Open the application in your web browser http://localhost:8501

3. Follow the steps:
   - Enter your OpenAI API key
   - Upload a PDF document
   - Wait for it to be processed
   - Submit questions about the document content
