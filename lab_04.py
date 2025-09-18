import os, sys
import streamlit as st
from openai import OpenAI
from PyPDF2 import PdfReader


__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

st.title("IST 688 - LAB 04B: Course Information Chatbot")


if "openai_client" not in st.session_state:
    api_key = st.secrets["OPENAI_API_KEY"]
    st.session_state.openai_client = OpenAI(api_key=api_key)


if "Lab4_vectorDB" not in st.session_state:
    chromaDB_path = "./ChromaDB_for_lab"
    chroma_client = chromadb.PersistentClient(path=chromaDB_path)
    embed_fn = OpenAIEmbeddingFunction(
        api_key=st.secrets["OPENAI_API_KEY"],
        model_name="text-embedding-3-small"
    )
    collection = chroma_client.get_or_create_collection(
    name="Lab4Collection_openai",
    embedding_function=embed_fn
)




    PDF_DIR = "HWS/File_Folders"
    pdfs = [f for f in os.listdir(PDF_DIR) if f.endswith(".pdf")]
    docs, ids, metas = [], [], []
    for i, pdf in enumerate(pdfs, start=1):
        text = PdfReader(os.path.join(PDF_DIR, pdf))
        content = ""
        for page in text.pages:
            content += page.extract_text() or ""
        if content.strip():
            docs.append(content)
            ids.append(f"doc_{i}")
            metas.append({"filename": pdf})
    if docs:
        collection.add(documents=docs, ids=ids, metadatas=metas)

    st.session_state.Lab4_vectorDB = collection
else:
    collection = st.session_state.Lab4_vectorDB


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


for role, msg in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(msg)

# --- Chat input ---
if user_query := st.chat_input("Ask about the course PDFs…"):
    # 1. show user message
    st.session_state.chat_history.append(("user", user_query))
    with st.chat_message("user"):
        st.markdown(user_query)

    
    results = collection.query(query_texts=[user_query], n_results=3)
    retrieved_chunks = [doc for doc in results["documents"][0]]
    context_text = "\n\n---\n\n".join(retrieved_chunks)

    
    prompt = (
        "You are a helpful course information assistant.\n"
        "If relevant, use the following retrieved context to answer clearly.\n"
        "If you rely on this, say: 'Based on the course material I found...'\n\n"
        f"Context:\n{context_text}\n\n"
        f"User question: {user_query}"
    )


    client = st.session_state.openai_client
    response = client.chat.completions.create(
        model="gpt-4o-mini",  # or gpt-3.5-turbo
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=500
    )
    answer = response.choices[0].message.content

   
    st.session_state.chat_history.append(("assistant", answer))
    with st.chat_message("assistant"):
        st.markdown(answer)
