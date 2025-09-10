import streamlit as st
import openai

# Use the secret key from .streamlit/secrets.toml
client = openai.OpenAI(api_key=st.secrets["api_keys"]["openai_key"])

st.title(" Document Summarizer with GPT-3.5 and GPT.4o (Lab 2C)")

# Sidebar controls
st.sidebar.title("Summary Options")

summary_type = st.sidebar.radio("Choose summary format:", [
    "100-word summary",
    "2-paragraph summary",
    "5 bullet points"
])

use_advanced = st.sidebar.checkbox("Use Advanced Model (GPT-4o)")

# Select model
model = "gpt-4o" if use_advanced else "gpt-3.5-turbo"

# File uploader
uploaded_file = st.file_uploader("Upload a .txt or .md file", type=("txt", "md"))

if uploaded_file:
    document = uploaded_file.read().decode(errors="ignore")

    # Build prompt based on summary type
    if summary_type == "100-word summary":
        instruction = "Summarize the document in approximately 100 words."
    elif summary_type == "2-paragraph summary":
        instruction = "Summarize the document in exactly two connected paragraphs."
    elif summary_type == "5 bullet points":
        instruction = "Summarize the document in 5 clear and concise bullet points."
    else:
        instruction = "Summarize the document."

    # Prepare LLM message
    messages = [
        {
            "role": "user",
            "content": f"Here's a document:\n\n{document}\n\n---\n\n{instruction}"
        }
    ]

    # Call OpenAI API
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True
    )

    # Display output
    st.write(f"🔍 Summary using **{model}** in format: *{summary_type}*")
    st.write_stream(stream)
