import streamlit as st
import pandas as pd
import faiss
import pickle
import json
import torch
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import numpy as np
import gdown
import os

# File Paths
FAISS_INDEX_FILE = "financial_faiss.index"
BM25_FILE = "bm25_corpus.pkl"
PROCESSED_DATA_FILE = "financial_data_with_embeddings.csv"

# Google Drive FAISS Index Download
FILE_ID = "1rvFh5LzvIVx-MBPvey6Vf47fVljeKSW6"  # Extracted file ID
GDRIVE_URL = f"https://drive.google.com/uc?id={FILE_ID}"

def download_faiss_from_drive():
    """Download FAISS index if not available locally."""
    if not os.path.exists(FAISS_INDEX_FILE):
        print("Downloading FAISS Index...")
        gdown.download(GDRIVE_URL, FAISS_INDEX_FILE, quiet=False)
        print("✅ FAISS Index downloaded!")

@st.cache_resource
def load_faiss_index():
    """Load FAISS index and cache it."""
    try:
        download_faiss_from_drive()
        index = faiss.read_index(FAISS_INDEX_FILE)
        print(f"✅ FAISS Index Loaded (Entries: {index.ntotal})")
        return index
    except Exception as e:
        st.error(f"❌ Error loading FAISS Index: {e}")
        return None

@st.cache_resource
def load_embedder():
    """Load Sentence Transformer model with caching."""
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    return SentenceTransformer(model_name)

@st.cache_resource
def load_bm25():
    """Load BM25 corpus with caching."""
    try:
        with open(BM25_FILE, "rb") as f:
            bm25 = pickle.load(f)
        return bm25
    except Exception as e:
        st.error(f"❌ Error loading BM25: {e}")
        return None

# Initialize Streamlit App
st.set_page_config(page_title="Financial RAG Chatbot", layout="wide")
st.title("💰 13F-HR FILING RAG CHATBOT")

# Load resources
embedder = load_embedder()
faiss_index = load_faiss_index()
bm25 = load_bm25()

query = st.text_input("🔎 Enter your financial query:")

if query:
    st.write(f"Searching for: `{query}`")
    
    # Validate Query
    query = query.strip().lower()
    allowed_keywords = ["market value", "cusip", "issuer", "quarter", "shares", "investment", "voting authority"]
    if not any(keyword in query for keyword in allowed_keywords):
        st.warning("⚠️ Query does not seem related to financial data. Please refine your request.")
    else:
        # Encode Query
        query_embedding = embedder.encode(query).reshape(1, -1)
        
        # Ensure FAISS Index matches query size
        if query_embedding.shape[1] == faiss_index.d:
            distances, indices = faiss_index.search(query_embedding, 3)
            st.write(f"Retrieved indices: {indices[0]}")
        else:
            st.error(f"❌ Embedding Size Mismatch! Query: {query_embedding.shape[1]}, FAISS: {faiss_index.d}")
