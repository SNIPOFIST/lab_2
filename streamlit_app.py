# main.py
import streamlit as st


st.set_page_config(
    page_title="Hari Ram Selvaraj - IST688 - Homework",
    page_icon= "📚",
    initial_sidebar_state= "auto"
)

st.sidebar.markdown("Home work Manager")



pages=[
    st.Page("HWS/HW1.py", title = "IST_688_Homework_1"),
    st.Page("HWS/HW2.py", title= "IST_688_Homework_2"),
    st.Page("HWS/HW3.py", title="IST_688_Homework_3"),
    st.Page("HWS/HW4.py", title= "IST_688_Homework_4"),
    st.Page("HWS/HW5.py", title="IST_688_Homework_5"),
    st.Page("HWS/HW7.py", title="IST_688_Homework_7"),
    st.Page("about.py", title= "About page")
]


pg = st.navigation(pages)
pg.run()


st.sidebar.write("")
st.sidebar.write("")
st.sidebar.write("")
st.sidebar.divider()
st.sidebar.caption(body="Note : The LLM's have the capability to hallucinate or might not be right always, user to use it under reminder of the same, and please use the application responsibly.", width="stretch")




# Working library - requirements.txt
'''streamlit>=1.33
openai>=1.50.0
mistralai>=1.1.0
google-generativeai>=0.7.2
requests>=2.31.0
beautifulsoup4>=4.12.2
lxml>=4.9.0
PyPDF2>=3.0.1
# pysqlite3-binary
protobuf
langchain==0.3.27
langchain-community==0.3.29
langchain-openai==0.3.33
tiktoken

unstructured
langchain_google_genai
langchain_mistralai
pysqlite3-binary>=0.5.2
chromadb==0.4.24


# chromadb==0.5.4'''