# main.py
import streamlit as st

pages=[
    st.Page("HWS/HW1.py", title = "IST_688_Homework_1"),
    st.Page("HWS/HW2.py", title= "IST_688_Homework_2")
]

pg = st.navigation(pages)
pg.run()

st.title("Home work manager ")