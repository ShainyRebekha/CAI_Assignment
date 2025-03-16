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

# File Paths
DATA_FILE = "financial_data.csv"
FAISS_INDEX_FILE = "financial_faiss.index"
BM25_FILE = "bm25_corpus.pkl"
METADATA_FILE = "metadata.json"
PROCESSED_DATA_FILE = "financial_data_with_embeddings.csv"
MEMORY_FILE = "chat_memory.json"

GDRIVE_URL = "https://drive.google.com/file/d/1rvFh5LzvIVx-MBPvey6Vf47fVljeKSW6/view?usp=drive_link"  # Replace with your file ID

def download_faiss_from_drive():
    if not os.path.exists(FAISS_INDEX_FILE):  # Download only if not available
        gdown.download(GDRIVE_URL, FAISS_INDEX_FILE, quiet=False)
        print("✅ FAISS Index downloaded successfully!")

def load_faiss_index():
    try:
        download_faiss_from_drive()
        index = faiss.read_index(FAISS_INDEX_FILE)
        print(f"✅ FAISS Index Loaded (Entries: {index.ntotal})")
        return index
    except Exception as e:
        print(f"❌ Error loading FAISS Index: {e}")
        return None

# ✅ Load BM25 Corpus
def load_bm25():
    try:
        with open(BM25_FILE, "rb") as f:
            return pickle.load(f)
    except:
        return None

# ✅ Load Sentence Transformer Model
@st.cache_resource
def load_embedder():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# ✅ Query Validation
def validate_query(query):
    query = query.strip().lower()
    if not query:
        return None
    allowed_keywords = ["market value", "cusip", "figi", "issuer", "quarter", "shares", "investment"]
    if not any(keyword in query for keyword in allowed_keywords):
        return None
    return re.sub(r"[^\w\s]", "", query)

# ✅ Retrieve Documents
def retrieve_documents(query, top_k=3):
    query = validate_query(query)
    if query is None:
        return []

    df = pd.read_csv(PROCESSED_DATA_FILE)
    query_embedding = embedder.encode(query).reshape(1, -1)
    if faiss_index is None:
        return []
    distances, indices = faiss_index.search(query_embedding, top_k)
    confidence_scores = np.exp(-distances[0])
    confidence_scores /= confidence_scores.sum()
    valid_indices = [i for i in indices[0] if i < len(df)]
    if not valid_indices:
        return []
    retrieved_docs = df.iloc[valid_indices].copy()
    retrieved_docs["Confidence"] = confidence_scores[:len(valid_indices)]
    return retrieved_docs.to_dict(orient="records")

# ✅ Initialize Streamlit Chatbot UI
st.set_page_config(page_title="Financial RAG Chatbot", layout="wide")
st.title("💰 Financial Chatbot: Ask About 13F-HR Filings")

# ✅ Load Resources
embedder = load_embedder()
faiss_index = load_faiss_index()
bm25 = load_bm25()

# ✅ Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# ✅ Display Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ✅ User Input
query = st.chat_input("Ask any Nuveen's 13F-HR Filings details ...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)
    
    retrieved_docs = retrieve_documents(query)
    if retrieved_docs:
        df_results = pd.DataFrame(retrieved_docs)
        df_results.rename(columns={"Value": "Market Value"}, inplace=True)
        
        st.write("### 🔍 Financial Data Details")
        if "cusip" in query.lower() or "figi" in query.lower():
            identifier_type = "CUSIP" if "cusip" in query.lower() else "FIGI"
            best_match = df_results.sort_values("Confidence", ascending=False).drop_duplicates(subset=[identifier_type]).head(1)
            st.markdown(f"**Name of Issuer:** {best_match.iloc[0]['Name of Issuer']}\n\n**{identifier_type}:** {best_match.iloc[0][identifier_type]}\n\n**Confidence:** {best_match.iloc[0]['Confidence']:.2f}")
        elif "market value" in query.lower():
            if "quarter" in query.lower():
                match = re.search(r"quarter\s*(\d+)\s*year\s*(\d+)", query)
                if match:
                    quarter, year = match.groups()
                    matched_records = df_results[(df_results["Quarter"] == int(quarter)) & (df_results["Year"] == int(year))]
                else:
                    matched_records = df_results
            else:
                matched_records = df_results
            st.write(matched_records[["Name of Issuer", "Market Value", "Quarter", "Year", "Confidence"]])
    else:
        st.write("❌ No relevant financial records found. Try rephrasing your query.")
    
    st.session_state.messages.append({"role": "assistant", "content": "Response displayed in structured format."})
