# --- Fix for SQLite + Chroma ---
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
import pandas as pd
import streamlit as st
from chromadb import Client
from chromadb.config import Settings
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

# -------------------------------
# APP TITLE
# -------------------------------
st.title("HW7 - 📰 News Info Bot")

# -------------------------------
# STEP 1: Load CSV
# -------------------------------
CSV_PATH = "HWS/File_Folders/Example_news_info_for_testing.csv"

if not os.path.exists(CSV_PATH):
    st.error(f"❌ CSV file not found at {CSV_PATH}")
    st.stop()

df = pd.read_csv(CSV_PATH)
st.subheader("📄 Preview of News Dataset")
st.write(df.head())
st.info(f"Total rows loaded: {len(df)}")

# Combine text fields
if "title" in df.columns and "description" in df.columns:
    df["combined_text"] = df["title"].fillna("") + ". " + df["description"].fillna("")
else:
    df["combined_text"] = df.apply(lambda r: " ".join(map(str, r.values)), axis=1)

# -------------------------------
# STEP 2: Vector Search Setup  (for Chroma <= 0.4.x)
# -------------------------------
st.subheader("🔍 Setting up Vector Search")

DB_DIR = "news_vector_db"
os.makedirs(DB_DIR, exist_ok=True)

OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# New API (0.5.x) — simple and correct
vector_db = Chroma.from_texts(
    texts=df["combined_text"].tolist(),
    embedding=embeddings,
    collection_name="news_collection",
    persist_directory="news_vector_db"
)
vector_db.persist()

st.success("✅ Vector database ready!")


# -------------------------------
# STEP 3: Define Helper Function
# -------------------------------
def get_relevant_news(query, k=5):
    """Retrieve top-k relevant news stories for a given query"""
    return vector_db.similarity_search(query, k=k)

# -------------------------------
# STEP 4: Sidebar – Model Selector
# -------------------------------
st.sidebar.header("⚙️ Model Settings")
model_choice = st.sidebar.radio(
    "Select LLM:", ["OpenAI (gpt-4o-mini)", "Gemini (gemini-2.0-flash-lite)"]
)

# -------------------------------
# STEP 5: Query Input
# -------------------------------
query = st.text_input(
    "Ask your question (e.g., 'Find the most interesting news' or 'Find news about AI'):"
)

if st.button("🔎 Get News Insights") and query.strip():
    with st.spinner("Processing..."):
        # Step 1: Retrieve relevant articles
        retrieved_docs = get_relevant_news(query, k=5)
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])

        # Step 2: Build LLM prompt
        prompt = f"""
        You are a news analysis assistant for a global law firm.
        The following are excerpts from recent news articles:
        {context}

        Task:
        - Summarize or rank the most interesting news stories.
        - If user asks for a topic, show relevant stories with reasoning.
        Respond concisely.
        """

        # Step 3: Call selected model
        if "OpenAI" in model_choice:
            llm = ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0.3,
                openai_api_key=OPENAI_API_KEY
            )
            response = llm.invoke(prompt)
            answer = response.content

        else:  # Gemini
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash-lite",
                google_api_key=GEMINI_API_KEY,
                temperature=0.3
            )
            response = llm.invoke(prompt)
            answer = response.content

    st.subheader("🧠 LLM Response")
    st.write(answer)

    st.subheader("📰 Retrieved News Snippets")
    for i, doc in enumerate(retrieved_docs, 1):
        st.markdown(f"**{i}.** {doc.page_content[:300]}...")

