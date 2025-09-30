import os
import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain


st.title("IST 688 - HW 5: Intelligent Conversation Memory Chatbot ")


OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]


DB_DIR = "vector_db"   # <- same DB folder from HW4/Lab5
if not os.path.exists(DB_DIR):
    st.error("Vector DB not found! Please build it first from your HW4/Lab5 data.")

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
vectorstore = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)


memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)


llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    openai_api_key=OPENAI_API_KEY
)

qa = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    memory=memory
)


st.sidebar.header("Chat Settings")
reset = st.sidebar.button("Clear Chat History")

if reset:
    memory.clear()
    st.sidebar.success("Chat history cleared ")

st.subheader("Ask about your courses/clubs")
query = st.text_input("Enter your question:")

if query:
    response = qa({"question": query})
    st.write("**Assistant:**", response["answer"])
