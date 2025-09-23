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
    st.Page("about.py", title= "About page")
]


pg = st.navigation(pages)
pg.run()


st.sidebar.write("")
st.sidebar.write("")
st.sidebar.write("")
st.sidebar.divider()
st.sidebar.caption(body="Note : The LLM's have the capability to hallucinate or might not be right always, user to use it under reminder of the same, and please use the application responsibly.", width="stretch")
