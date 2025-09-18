import os
import sys
import streamlit as st
from openai import OpenAI
from PyPDF2 import PdfReader



# --- Fix for ChromaDB on Streamlit Cloud ---
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import chromadb

st.title("IST 688 - LAB - 04 A")
# Initialize ChromaDB
chromaDB_path = "./ChromaDB_for_lab"
chroma_client = chromadb.PersistentClient(path=chromaDB_path)
collection = chroma_client.get_or_create_collection("Lab4Collection")


if "openai_client" not in st.session_state:
    api_key = st.secrets["OPENAI_API_KEY"]
    st.session_state.openai_client = OpenAI(api_key=api_key)


def read_pdf(path: str) -> str:
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text


PDF_DIR = "HWS/File_Folders"

if "Lab4_vectorDB" not in st.session_state:
    pdfs = [f for f in os.listdir(PDF_DIR) if f.endswith(".pdf")]
    docs, ids, metas = [], [], []
    for i, pdf in enumerate(pdfs, start=1):
        text = read_pdf(os.path.join(PDF_DIR, pdf))
        docs.append(text)
        ids.append(f"doc_{i}")
        metas.append({"filename": pdf})
    if docs:
        collection.add(documents=docs, ids=ids, metadatas=metas)
    st.session_state.Lab4_vectorDB = collection
    st.success("A New Vector DB built and cached.")
else:
    st.info("Using cached Vector DB.")

topic = st.selectbox("Pick a test topic", ["Generative AI", "Text Mining", "Data Science Overview"])

if st.button("Run search"):
    openai_client = st.session_state.openai_client
    response = openai_client.embeddings.create(
        input=topic,
        model="text-embedding-3-small", dimensions= 384,
    )
    query_embedding = response.data[0].embedding

    results = collection.query(query_embeddings=[query_embedding], n_results=3)

    st.write("Top 3 results:")
    for i, md in enumerate(results["metadatas"][0], start=1):
        st.write(f"{i}. {md['filename']}")