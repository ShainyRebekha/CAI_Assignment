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
DATA_FILE = "financial_data.csv"
FAISS_INDEX_FILE = "financial_faiss.index"
BM25_FILE = "bm25_corpus.pkl"
PROCESSED_DATA_FILE = "financial_data_with_embeddings.csv"
GDRIVE_URL = "https://drive.google.com/uc?id=1rvFh5LzvIVx-MBPvey6Vf47fVljeKSW6"

def download_faiss_from_drive():
    """Download FAISS index from Google Drive if missing."""
    if not os.path.exists(FAISS_INDEX_FILE):
        gdown.download(GDRIVE_URL, FAISS_INDEX_FILE, quiet=False)
        print("✅ FAISS Index downloaded successfully!")

def load_faiss_index():
    """Load FAISS index, ensuring the file exists and is valid."""
    download_faiss_from_drive()
    if not os.path.exists(FAISS_INDEX_FILE):
        st.error("❌ FAISS Index file missing! Check the download link.")
        return None
    try:
        index = faiss.read_index(FAISS_INDEX_FILE)
        print(f"✅ FAISS Index Loaded (Entries: {index.ntotal})")
        return index
    except Exception as e:
        st.error(f"❌ Error loading FAISS Index: {e}")
        return None

def load_bm25():
    """Load BM25 corpus."""
    try:
        with open(BM25_FILE, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"❌ Error loading BM25: {e}")
        return None

@st.cache_resource
def load_embedder():
    """Load the Sentence Transformer model."""
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def validate_query(query):
    """Validate and clean user query."""
    query = query.strip().lower()
    if not query:
        st.error("⚠️ Query cannot be empty!")
        return None
    allowed_keywords = ["market value", "cusip", "issuer", "quarter", "shares", "investment", "voting authority"]
    if not any(keyword in query for keyword in allowed_keywords):
        st.warning("⚠️ Query does not seem related to financial data. Please refine your request.")
        return None
    return query

def retrieve_documents(query, top_k=3):
    """Retrieve relevant financial records using FAISS & BM25."""
    query = validate_query(query)
    if query is None:
        return pd.DataFrame()
    
    df = pd.read_csv(PROCESSED_DATA_FILE)
    query_embedding = embedder.encode(query).reshape(1, -1)
    
    if faiss_index is None:
        st.error("❌ FAISS Index not loaded. Check for errors.")
        return pd.DataFrame()
    
    if query_embedding.shape[1] != faiss_index.d:
        st.error(f"❌ Embedding Size Mismatch! Query: {query_embedding.shape[1]}, FAISS: {faiss_index.d}")
        return pd.DataFrame()
    
    distances, indices = faiss_index.search(query_embedding, top_k)
    confidence_scores = np.exp(-distances[0]) / np.exp(-distances[0]).sum()
    valid_indices = [i for i in indices[0] if i < len(df)]
    
    if not valid_indices:
        return pd.DataFrame()
    
    retrieved_docs = df.iloc[valid_indices].copy()
    retrieved_docs["Confidence"] = confidence_scores[:len(valid_indices)]
    
    if "market value" in query:
        return retrieved_docs[["Quarter", "Value", "Confidence"]]
    if "cusip" in query:
        return retrieved_docs[["Quarter", "CUSIP", "Confidence"]]
    
    return retrieved_docs

# Streamlit UI
st.set_page_config(page_title="Financial RAG Chatbot", layout="wide")
st.title("💰 13F-HR Filing RAG Chatbot")

embedder = load_embedder()
faiss_index = load_faiss_index()
bm25 = load_bm25()

query = st.text_input("🔎 Enter your financial query:")

if query:
    retrieved_docs = retrieve_documents(query)
    if not retrieved_docs.empty:
        st.subheader("📄 Retrieved Documents:")
        st.dataframe(retrieved_docs.sort_values(by="Confidence", ascending=False))
    else:
        st.warning("⚠️ No relevant financial documents found.")
