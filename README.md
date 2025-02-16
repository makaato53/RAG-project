# Retrieval-Augmented Generation (RAG) Chatbot

## Overview
This project implements a Retrieval-Augmented Generation (RAG) chatbot that uses a vector search to retrieve relevant context from a knowledge base and then uses an LLM (such as OpenAI's GPT) to generate contextually informed responses. It demonstrates key AI competencies including text preprocessing, embeddings, vector search using FAISS, and web-based UI development with Streamlit.

## Features
- **Knowledge Base Integration:** Reads and processes a text corpus from a CSV file.
- **Text Chunking & Embedding:** Uses Sentence Transformers (e.g., `all-MiniLM-L6-v2`) to create embeddings for text chunks.
- **Vector Search:** Implements FAISS for fast similarity search.
- **Language Model Integration:** Connects to OpenAI's API to generate responses based on retrieved context.
- **User Interface:** A simple, interactive web app built with Streamlit.
- **Deployment Ready:** Easy to deploy locally or via cloud platforms like Streamlit Cloud.

## Project Structure
## Usage
### 1. Running the Application
```bash
streamlit run app.py
```

### 2. Interacting with the Chatbot
- Enter your question in the text input field
- The system will:
  1. Convert your question into an embedding
  2. Search the knowledge base for relevant context
  3. Generate a response using the LLM and retrieved context

## Configuration
- Update OpenAI API key in the application
- Modify model parameters in app.py as needed
- Adjust chunk size and overlap for text preprocessing

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License
MIT
"# RAG-project" 
"# RAG-project" 
