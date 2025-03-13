import streamlit as st
import pandas as pd
import faiss
import pickle
import json
import torch
import os
import gdown
import numpy as np
import re
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

# File Paths
DATA_FILE = "financial_data.csv"
FAISS_INDEX_FILE = "financial_faiss.index"
BM25_FILE = "bm25_corpus.pkl"
METADATA_FILE = "metadata.json"
PROCESSED_DATA_FILE = "financial_data_with_embeddings.csv"
MEMORY_FILE = "chat_memory.json"

GDRIVE_FILE_ID = "1rvFh5LzvIVx-MBPvey6Vf47fVljeKSW6"
GDRIVE_URL = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}&export=download"

# Ensure Consistent Model for Embeddings
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# ✅ Download FAISS Index (if missing)
def download_faiss_from_drive():
    if not os.path.exists(FAISS_INDEX_FILE):
        try:
            st.info("📥 Downloading FAISS Index from Google Drive...")
            gdown.download(GDRIVE_URL, FAISS_INDEX_FILE, quiet=False)
            st.success("✅ FAISS Index downloaded successfully!")
        except Exception as e:
            st.error(f"❌ Error downloading FAISS Index: {e}")

# ✅ Load FAISS Index
@st.cache_resource
def load_faiss_index():
    download_faiss_from_drive()
    try:
        index = faiss.read_index(FAISS_INDEX_FILE)
        st.success(f"✅ FAISS Index Loaded (Entries: {index.ntotal})")
        return index
    except Exception as e:
        st.error(f"❌ Error loading FAISS Index: {e}")
        return None

# ✅ Load BM25 Corpus
@st.cache_resource
def load_bm25():
    try:
        with open(BM25_FILE, "rb") as f:
            bm25 = pickle.load(f)
        st.success("✅ BM25 Model Loaded")
        return bm25
    except Exception as e:
        st.error(f"❌ Error loading BM25: {e}")
        return None

# ✅ Load Sentence Transformer Model
@st.cache_resource
def load_embedder():
    return SentenceTransformer(MODEL_NAME)

# ✅ Query Validation
def validate_query(query):
    """Guardrail: Validate and sanitize user query."""
    query = query.strip().lower()

    if not query:
        st.error("⚠️ Query cannot be empty!")
        return None

    allowed_keywords = ["market value", "cusip", "issuer", "quarter", "shares", "investment", "voting authority"]
    if not any(keyword in query for keyword in allowed_keywords):
        st.warning("⚠️ Query does not seem related to financial data. Please refine your request.")
        return None

    return re.sub(r"[^\w\s]", "", query)

# ✅ Retrieve Documents
def retrieve_documents(query, top_k=3):
    query = validate_query(query)
    if query is None:
        return pd.DataFrame()

    df = pd.read_csv(PROCESSED_DATA_FILE)

    # Encode Query
    query_embedding = embedder.encode(query).reshape(1, -1)

    # Validate FAISS Index
    if faiss_index is None or query_embedding.shape[1] != faiss_index.d:
        st.error(f"❌ FAISS Index Error: Embedding Size Mismatch! Query: {query_embedding.shape[1]}, FAISS: {faiss_index.d}")
        return pd.DataFrame()

    # FAISS Search
    distances, indices = faiss_index.search(query_embedding, top_k)
    confidence_scores = np.exp(-distances[0]) / np.exp(-distances[0]).sum()

    # Retrieve Relevant Documents
    valid_indices = [i for i in indices[0] if i < len(df)]
    if not valid_indices:
        return pd.DataFrame()

    retrieved_docs = df.iloc[valid_indices].copy()
    retrieved_docs["Confidence"] = confidence_scores[:len(valid_indices)]

    # Filter by Quarter (if mentioned)
    query_lower = query.lower()
    quarter_match = re.search(r"(q[1-4])", query_lower)
    requested_quarter = quarter_match.group(1).upper() if quarter_match else None
    if requested_quarter:
        retrieved_docs = retrieved_docs[retrieved_docs["Quarter"] == requested_quarter]

    # Field-Specific Responses
    if "market value" in query_lower:
        return retrieved_docs[["Quarter", "Value", "Confidence"]]
    if "cusip" in query_lower:
        return retrieved_docs[["Quarter", "CUSIP", "Confidence"]]

    # Issuer-Specific Retrieval
    for issuer in df["Name of Issuer"].unique():
        if issuer.lower() in query_lower:
            return retrieved_docs[retrieved_docs["Name of Issuer"].str.lower() == issuer.lower()]

    return retrieved_docs

# ✅ Initialize Streamlit App
st.set_page_config(page_title="Financial RAG Chatbot", layout="wide")
st.title("💰 13F-HR FILING RAG CHATBOT")

# Load Resources
embedder = load_embedder()
faiss_index = load_faiss_index()
bm25 = load_bm25()

# ✅ User Input & Retrieval
query = st.text_input("🔎 Enter your financial query:")
if query:
    retrieved_docs = retrieve_documents(query)
    if not retrieved_docs.empty:
        st.subheader("📄 Retrieved Documents:")
        st.dataframe(retrieved_docs.sort_values(by="Confidence", ascending=False))
    else:
        st.warning("⚠️ No relevant financial documents found.")
