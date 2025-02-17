# CUDA RAG Chatbot

## Overview
This project implements a Retrieval-Augmented Generation (RAG) chatbot specifically focused on the domain of **CUDA**. The chatbot retrieves relevant content from a knowledge base built from CUDA documentation, tutorials, FAQs, and other text-based resources. It then uses this context to generate precise, contextually informed responses. This project is ideal for demonstrating your proficiency with AI techniques and your deep interest in CUDA technology.

## Features
- **Domain-Specific Knowledge Base:**  
  The chatbot is powered by a curated collection of CUDA-related texts (documentation, tutorials, blog posts, etc.), which are preprocessed and chunked into manageable pieces.
- **Text Chunking & Embedding:**  
  The data is processed by splitting long documents into ~200–500 token chunks. These chunks are then embedded using Sentence Transformers to create vector representations.
- **Vector Search with FAISS:**  
  Uses FAISS (or FAISS-GPU if desired) for efficient similarity search, allowing rapid retrieval of relevant CUDA content.
- **LLM Integration for Answer Generation:**  
  Augments the retrieved context by integrating with a language model (e.g., OpenAI’s gpt-3.5-turbo) to produce coherent answers about CUDA topics.
- **User-Friendly Interface:**  
  A web application built with Streamlit that allows users to enter queries and view responses in real time.

## Project Structure
├── README.md # Project overview and instructions ├── app.py # Main Streamlit application ├── knowledge_base.csv # CSV file containing CUDA-related texts (columns: id, text, [optional: source]) ├── embeddings.pkl # (Auto-generated) Precomputed embeddings for the knowledge base ├── requirements.txt # List of Python dependencies └── Dockerfile # (Optional) Docker configuration for containerized deployment

markdown
Copy

## Domain & Knowledge Base Preparation

### Choosing the Domain: CUDA
- **Why CUDA?**  
  CUDA is a popular parallel computing platform and programming model developed by NVIDIA. With extensive documentation and a strong community, it’s an excellent domain to demonstrate technical expertise and showcase AI capabilities in retrieving and processing complex technical information.
  
### Data Gathering
- **Sources:**  
  Gather CUDA documentation, official tutorials, blog posts, and FAQs from trusted sources (e.g., NVIDIA’s official documentation, reputable tech blogs, and academic materials).
  
- **Data Format:**  
  Store the text in a simple CSV file (`knowledge_base.csv`) with at least the following columns:
  - `id`: Unique identifier for each document chunk.
  - `text`: The content chunk (200–500 tokens per chunk).
  - Optionally, `source`: A reference to the original document.
  
- **Chunking:**  
  For lengthy documents, split the text into smaller chunks. This ensures that the retrieval system can locate and return the most relevant parts when a query is made.

