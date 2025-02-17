import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import pickle

# Page config
st.set_page_config(
    page_title="CUDA RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)

@st.cache_resource
def load_models():
    """Load all necessary models and data"""
    # Load embeddings and dataframe
    with open('cuda_embeddings.pkl', 'rb') as f:
        df, embeddings = pickle.load(f)
    
    # Load embedding model
    embed_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    # Load LLM and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
    llm = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
    
    # Setup FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype('float32'))
    
    return df, embeddings, embed_model, tokenizer, llm, index

# Load all models
df, embeddings, embed_model, tokenizer, llm, index = load_models()

def search_knowledge_base(query: str, top_k: int = 3):
    """Search for relevant texts using the query"""
    # Create query embedding
    query_embedding = embed_model.encode([query])[0]
    
    # Search in FAISS
    distances, indices = index.search(np.array([query_embedding]).astype('float32'), top_k)
    
    # Get the text of the most similar chunks
    results = [
        {
            'text': df.iloc[idx]['text'],
            'source': df.iloc[idx]['source'],
            'similarity': 1 - dist/2
        }
        for dist, idx in zip(distances[0], indices[0])
    ]
    
    return results

def generate_response(query: str, relevant_chunks: list):
    """Generate response using FLAN-T5"""
    context = "\n\n".join([chunk['text'] for chunk in relevant_chunks])
    prompt = f"""Use the following CUDA documentation to answer the question.
    
    Documentation:
    {context}

    Question: {query}

    Answer:"""
    
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    outputs = llm.generate(**inputs, max_length=200, num_beams=4, temperature=0.7)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return response

# Streamlit UI
st.title("📚 CUDA Documentation Assistant")
st.markdown("""
This chatbot helps you understand CUDA programming concepts by searching through documentation
and generating responses based on relevant information.
""")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask about CUDA programming..."):
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("assistant"):
        # Search for relevant documentation
        with st.spinner("Searching documentation..."):
            relevant_chunks = search_knowledge_base(prompt)
        
        # Generate response
        with st.spinner("Generating response..."):
            response = generate_response(prompt, relevant_chunks)
            st.markdown(response)
            
            # Show sources
            with st.expander("View Source Documentation"):
                for i, chunk in enumerate(relevant_chunks, 1):
                    st.markdown(f"**Source {i}** (Similarity: {chunk['similarity']:.2f})")
                    st.markdown(chunk['text'])
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

# Sidebar with information
with st.sidebar:
    st.header("About")
    st.markdown("""
    This chatbot uses RAG (Retrieval-Augmented Generation) to:
    1. Search through CUDA documentation
    2. Find relevant information
    3. Generate accurate responses
    
    Built with:
    - Sentence Transformers
    - FAISS
    - FLAN-T5
    - Streamlit
    """)