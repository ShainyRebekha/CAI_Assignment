import streamlit as st
import pandas as pd
import faiss
import pickle
import json
import torch
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import numpy as np

# File Paths
DATA_FILE = "financial_data.csv"
FAISS_INDEX_FILE = "financial_faiss.index"
BM25_FILE = "bm25_corpus.pkl"
METADATA_FILE = "metadata.json"
PROCESSED_DATA_FILE = "financial_data_with_embeddings.csv"
MEMORY_FILE = "chat_memory.json"

# Load FAISS Index
def load_faiss_index():
    try:
        index = faiss.read_index(FAISS_INDEX_FILE)
        #st.success(f"✅ FAISS Index Loaded (Embedding Dim: {index.d}, Entries: {index.ntotal})")
        return index
    except Exception as e:
        st.error(f"❌ Error loading FAISS Index: {e}")
        return None

# Load BM25 Corpus
def load_bm25():
    try:
        with open(BM25_FILE, "rb") as f:
            bm25 = pickle.load(f)
        #st.success("✅ BM25 Model Loaded")
        return bm25
    except Exception as e:
        st.error(f"❌ Error loading BM25: {e}")
        return None

# Load Sentence Transformer Model
@st.cache_resource
def load_embedder():
    model_name = "sentence-transformers/all-MiniLM-L6-v2"  # Ensure consistency
    return SentenceTransformer(model_name)

# Retrieve Documents
import re

import re

def validate_query(query):
    """Guardrail: Validate and sanitize user query."""
    query = query.strip().lower()
    
    # Block empty or irrelevant queries
    if not query:
        st.error("⚠️ Query cannot be empty!")
        return None
    
    # Allow only relevant financial keywords
    allowed_keywords = ["market value", "cusip", "issuer", "quarter", "shares", "investment", "voting authority"]
    if not any(keyword in query for keyword in allowed_keywords):
        st.warning("⚠️ Query does not seem related to financial data. Please refine your request.")
        return None

    # Remove unwanted special characters
    query = re.sub(r"[^\w\s]", "", query)

    return query

def retrieve_documents(query, top_k=3):
    """Retrieve relevant financial records from FAISS & BM25 based on query intent."""
    query = validate_query(query)
    if query is None:
        return pd.DataFrame()

    df = pd.read_csv(PROCESSED_DATA_FILE)
    
    # Query Embedding
    query_embedding = embedder.encode(query).reshape(1, -1)
    
    # Ensure FAISS Index matches query size
    if query_embedding.shape[1] != faiss_index.d:
        st.error(f"❌ Embedding Size Mismatch! Query: {query_embedding.shape[1]}, FAISS: {faiss_index.d}")
        return pd.DataFrame()

    # FAISS Search
    distances, indices = faiss_index.search(query_embedding, top_k)
    
    # Compute Confidence Scores (normalize distances)
    confidence_scores = np.exp(-distances[0])  # Convert distances to similarity-like scores
    confidence_scores = confidence_scores / confidence_scores.sum()  # Normalize

    # Filter valid indices
    valid_indices = [i for i in indices[0] if i < len(df)]
    
    if not valid_indices:
        return pd.DataFrame()

    retrieved_docs = df.iloc[valid_indices].copy()
    retrieved_docs["Confidence"] = confidence_scores[:len(valid_indices)]

    # Extract user intent and quarter
    query_lower = query.lower()
    
    # Extract quarter from query (e.g., Q1, Q2, Q3, Q4)
    quarter_match = re.search(r"(q[1-4])", query_lower)
    requested_quarter = quarter_match.group(1).upper() if quarter_match else None

    if requested_quarter:
        retrieved_docs = retrieved_docs[retrieved_docs["Quarter"] == requested_quarter]

    # Check for specific field requests
    if "market value" in query_lower:
        return retrieved_docs[["Quarter", "Value", "Confidence"]]

    if "cusip" in query_lower:
        return retrieved_docs[["Quarter", "CUSIP", "Confidence"]]

    # Check if query mentions a specific issuer (company name)
    for issuer in df["Name of Issuer"].unique():
        if issuer.lower() in query_lower:
            return retrieved_docs[retrieved_docs["Name of Issuer"].str.lower() == issuer.lower()]

    return retrieved_docs  # Default: Return all retrieved documents



# Initialize Streamlit App
st.set_page_config(page_title="Financial RAG Chatbot", layout="wide")
st.title("💰 13F-HR FILING RAG CHATBOT ")

# Load Resources
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
