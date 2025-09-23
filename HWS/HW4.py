# --- SQLite patch (must be first, before any Chroma imports) ---
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# -------------------------------
# CONFIG
# -------------------------------
DB_DIR = "vector_db"
HTML_DIR = "HWS/File_Folders/su_orgs"
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

# -------------------------------
# STEP 1: LOAD AND CHUNK DOCUMENTS
# -------------------------------
def create_or_load_db():
    # If DB already exists → just load it
    if os.path.exists(DB_DIR):
        st.sidebar.success("Loaded existing vector DB ✅")
        return Chroma(
            persist_directory=DB_DIR,
            embedding_function=OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        )

    # Load all HTML files
    loader = DirectoryLoader(HTML_DIR, glob="*.html")
    docs = loader.load()

    # Split each HTML into 2 chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, chunk_overlap=50, separators=["\n", ".", " "]
    )
    split_docs = []
    for d in docs:
        chunks = text_splitter.split_documents([d])
        # enforce only 2 mini-docs per file
        if len(chunks) > 2:
            chunks = chunks[:2]
        split_docs.extend(chunks)

    # Build vector DB once
    vectordb = Chroma.from_documents(
        documents=split_docs,
        embedding=OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY),
        persist_directory=DB_DIR
    )
    vectordb.persist()
    st.sidebar.success("Vector DB created ✅")
    return vectordb

# -------------------------------
# STEP 2: STREAMLIT APP
# -------------------------------
def main():
    st.title("🤖 iSchool Student Orgs Chatbot")

    # Vector DB (create or load)
    vectordb = create_or_load_db()

    # Conversation memory (last 5 turns)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # Sidebar: choose model
    st.sidebar.header("⚙️ Model Settings")
    model_choice = st.sidebar.radio(
        "Select LLM:", ["gpt-3.5-turbo", "gpt-4", "gpt-4o-mini"]
    )

    if model_choice == "gpt-3.5-turbo":
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3, openai_api_key=OPENAI_API_KEY)
    elif model_choice == "gpt-4":
        llm = ChatOpenAI(model="gpt-4", temperature=0.3, openai_api_key=OPENAI_API_KEY)
    else:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3, openai_api_key=OPENAI_API_KEY)

    # Conversational retrieval
    qa = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectordb.as_retriever(),
        memory=memory
    )

    # User input
    user_input = st.text_input("💬 Ask me about iSchool student organizations:")

    if user_input:
        response = qa.invoke({"question": user_input})
        st.chat_message("user").write(user_input)
        st.chat_message("assistant").write(response["answer"])

    # Show memory log
    if memory.chat_memory.messages:
        st.subheader("📝 Conversation Memory (last 5 turns)")
        for m in memory.chat_memory.messages[-5:]:
            role = "User" if m.type == "human" else "Bot"
            st.write(f"**{role}:** {m.content}")

if __name__ == "__main__":
    main()
