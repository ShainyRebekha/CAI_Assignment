import streamlit as st
import pandas as pd
import faiss
import pickle
import json
import torch
import numpy as np
import gdown
import os
import re
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

# ✅ Set File Paths
DATA_FILE = "financial_data.csv"
FAISS_INDEX_FILE = "/tmp/financial_faiss.index"  # Store in /tmp/ for hosted environments
BM25_FILE = "bm25_corpus.pkl"
METADATA_FILE = "metadata.json"
PROCESSED_DATA_FILE = "financial_data_with_embeddings.csv"
MEMORY_FILE = "chat_memory.json"

# ✅ Google Drive Link Fix
GDRIVE_ID = "1rvFh5LzvIVx-MBPvey6Vf47fVljeKSW6"  # Extracted from the original link
GDRIVE_URL = f"https://drive.google.com/uc?id={GDRIVE_ID}"

# ✅ Download FAISS Index from Google Drive
def download_faiss_from_drive():
    if not os.path.exists(FAISS_INDEX_FILE) or os.path.getsize(FAISS_INDEX_FILE) < 1024:  # Ensure valid file
        st.info("📥 Downloading FAISS Index from Google Drive...")
        gdown.download(GDRIVE_URL, FAISS_INDEX_FILE, quiet=False)

        # ✅ Verify if the file downloaded correctly
        if os.path.exists(FAISS_INDEX_FILE) and os.path.getsize(FAISS_INDEX_FILE) > 1024:
            st.success(f"✅ FAISS Index downloaded successfully! Size: {os.path.getsize(FAISS_INDEX_FILE)} bytes")
        else:
            st.error("❌ FAISS file is missing or corrupted. Check Google Drive permissions.")
            return False
    return True

# ✅ Load FAISS Index
def load_faiss_index():
    try:
        if download_faiss_from_drive():
            index = faiss.read_index(FAISS_INDEX_FILE)
            st.success(f"✅ FAISS Index Loaded (Entries: {index.ntotal})")
            return index
    except Exception as e:
        st.error(f"❌ Error loading FAISS Index: {e}")
    return None

# ✅ Load BM25 Corpus
def load_bm25():
    try:
        with open(BM25_FILE, "rb") as f:
            bm25 = pickle.load(f)
        return bm25
    except Exception as e:
        st.error(f"❌ Error loading BM25: {e}")
        return None

# ✅ Load Sentence Transformer Model
@st.cache_resource
def load_embedder():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# ✅ Query Validation Guardrail
def validate_query(query):
    """Sanitize and validate user query."""
    query = query.strip().lower()
    if not query:
        st.error("⚠️ Query cannot be empty!")
        return None
    
    allowed_keywords = ["market value", "cusip", "issuer", "quarter", "shares", "investment", "voting authority"]
    if not any(keyword in query for keyword in allowed_keywords):
        st.warning("⚠️ Query does not seem related to financial data. Please refine your request.")
        return None

    query = re.sub(r"[^\w\s]", "", query)
    return query

# ✅ Retrieve Documents using FAISS & BM25
def retrieve_documents(query, top_k=3):
    """Retrieve relevant financial records based on query intent."""
    query = validate_query(query)
    if query is None:
        return pd.DataFrame()

    df = pd.read_csv(PROCESSED_DATA_FILE)
    query_embedding = embedder.encode(query).reshape(1, -1)

    # ✅ Ensure FAISS Index is loaded
    if faiss_index is None:
        st.error("❌ FAISS Index not loaded. Check for errors.")
        return pd.DataFrame()

    if query_embedding.shape[1] != faiss_index.d:
        st.error(f"❌ Embedding Size Mismatch! Query: {query_embedding.shape[1]}, FAISS: {faiss_index.d}")
        return pd.DataFrame()

    # ✅ FAISS Search
    distances, indices = faiss_index.search(query_embedding, top_k)
    confidence_scores = np.exp(-distances[0])
    confidence_scores /= confidence_scores.sum()

    valid_indices = [i for i in indices[0] if i < len(df)]
    if not valid_indices:
        return pd.DataFrame()

    retrieved_docs = df.iloc[valid_indices].copy()
    retrieved_docs["Confidence"] = confidence_scores[:len(valid_indices)]

    # ✅ Extract Quarter from Query
    quarter_match = re.search(r"(q[1-4])", query.lower())
    requested_quarter = quarter_match.group(1).upper() if quarter_match else None
    if requested_quarter:
        retrieved_docs = retrieved_docs[retrieved_docs["Quarter"] == requested_quarter]

    # ✅ Return Relevant Fields
    if "market value" in query:
        return retrieved_docs[["Quarter", "Value", "Confidence"]]
    if "cusip" in query:
        return retrieved_docs[["Quarter", "CUSIP", "Confidence"]]

    for issuer in df["Name of Issuer"].unique():
        if issuer.lower() in query:
            return retrieved_docs[retrieved_docs["Name of Issuer"].str.lower() == issuer.lower()]

    return retrieved_docs

# ✅ Initialize Streamlit App
st.set_page_config(page_title="Financial RAG Chatbot", layout="wide")
st.title("💰 13F-HR FILING RAG CHATBOT")

# ✅ Load Resources
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
