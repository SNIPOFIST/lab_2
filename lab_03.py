
import os
from typing import List, Dict

import streamlit as st
import openai

# Optional tokenizer for better token estimates
try:
    import tiktoken
except Exception:
    tiktoken = None


st.set_page_config(page_title="Lab 3 – Chatbot with Memory", page_icon="💬")
st.title("💬 Lab 3: Streaming Chatbot with Conversation Memory")

# Secrets (same pattern as Lab 2)
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
client = openai.OpenAI(api_key=OPENAI_API_KEY)


st.sidebar.title("⚙️ Chat Settings")
use_advanced = st.sidebar.checkbox("Use Advanced Model (GPT-4o)", value=False)
model = "gpt-4o" if use_advanced else "gpt-3.5-turbo"
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.3, 0.1)
max_output_tokens = st.sidebar.slider("Max output tokens (reply length)", 128, 2048, 512, 64)
max_context_tokens = st.sidebar.slider("Max context tokens (cap)", 256, 8192, 2048, 128)
kid_mode = st.sidebar.checkbox("Explain like I'm 10 years old", value=True)

if st.sidebar.button("🧹 Clear chat"):
    st.session_state.clear()
    st.rerun()

st.caption(f"Model: **{model}** • Kid mode: **{kid_mode}**")


def init_state():
    if "history" not in st.session_state:
        st.session_state.history: List[Dict] = []   # for UI rendering (full chat)
    if "pairs" not in st.session_state:
        st.session_state.pairs: List[Dict] = []     # [{"user": str, "assistant": str}, ...]
    if "awaiting_more_info" not in st.session_state:
        st.session_state.awaiting_more_info = False
    if "more_info_context" not in st.session_state:
        st.session_state.more_info_context = None   # {"question": str, "answer": str, "times_provided": int}
    if "last_token_count" not in st.session_state:
        st.session_state.last_token_count = 0

init_state()


def estimate_tokens(text: str) -> int:
    if tiktoken:
        try:
            enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
        except Exception:
            pass
    # Fallback: ~4 chars ≈ 1 token
    return max(1, len(text) // 4)

def count_message_tokens(messages: List[Dict]) -> int:
    total = 0
    for m in messages:
        total += estimate_tokens(f"{m.get('role','')}: {m.get('content','')}")
    return total


def build_system_prompt() -> str:
    base = "You are a helpful, friendly assistant. Keep answers clear and concise."
    if kid_mode:
        base += " Explain like I'm 10 years old. Use simple words and short sentences."
    return base

def last_two_pairs_as_messages() -> List[Dict]:
    msgs: List[Dict] = []
    for pair in st.session_state.pairs[-2:]:
        msgs.append({"role": "user", "content": pair["user"]})
        msgs.append({"role": "assistant", "content": pair["assistant"]})
    return msgs

def trim_context(messages: List[Dict], cap_tokens: int) -> List[Dict]:
    """Remove oldest (after system) until under cap."""
    if not messages or messages[0].get("role") != "system":
        return messages
    while count_message_tokens(messages) > cap_tokens and len(messages) > 1:
        messages.pop(1)
    return messages


def stream_completion(messages: list[dict]) -> str:
    """Streams response to UI and returns the final text."""
    collected = []
    placeholder = st.empty()

    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_output_tokens,
        stream=True,
    )

    for chunk in stream:
        delta = chunk.choices[0].delta.content or ""
        if delta:
            collected.append(delta)
            # Update the placeholder with the growing text
            placeholder.markdown("".join(collected))

    return "".join(collected).strip()


for m in st.session_state.history:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

def ask_more_info():
    with st.chat_message("assistant"):
        st.markdown("**DO YOU WANT MORE INFO?** (yes / no)")


def handle_new_question(user_text: str):
    system = {"role": "system", "content": build_system_prompt()}
    buffer_msgs = last_two_pairs_as_messages()
    messages = [system] + buffer_msgs + [{"role": "user", "content": user_text}]

    # Token diagnostics + trimming
    st.session_state.last_token_count = count_message_tokens(messages)
    messages = trim_context(messages, max_context_tokens)
    st.session_state.last_token_count = count_message_tokens(messages)

    # UI: user bubble
    st.session_state.history.append({"role": "user", "content": user_text})
    with st.chat_message("user"):
        st.markdown(user_text)

    # Stream reply
    reply = stream_completion(messages)
    if not reply:
        reply = "Sorry, I couldn't generate a response right now."

    st.session_state.history.append({"role": "assistant", "content": reply})
    st.session_state.pairs.append({"user": user_text, "assistant": reply})

    # Enter “more info” loop
    st.session_state.awaiting_more_info = True
    st.session_state.more_info_context = {"question": user_text, "answer": reply, "times_provided": 0}
    ask_more_info()

def handle_more_info_reply(user_text: str):
    # Show the user's yes/no
    st.session_state.history.append({"role": "user", "content": user_text})
    with st.chat_message("user"):
        st.markdown(user_text)

    t = user_text.strip().lower()
    is_yes = t in {"y", "yes", "yeah", "yup", "sure", "ok", "okay"}
    is_no  = t in {"n", "no", "nope", "nah"}

    if is_yes:
        ctx = st.session_state.more_info_context or {}
        followup = (
            f"The user asked: {ctx.get('question','')}\n\n"
            f"You answered: {ctx.get('answer','')}\n\n"
            "Provide MORE INFORMATION: extra helpful details, simple examples, and a clear explanation."
        )
        system = {"role": "system", "content": build_system_prompt()}
        buffer_msgs = last_two_pairs_as_messages()
        messages = [system] + buffer_msgs + [{"role": "user", "content": followup}]

        st.session_state.last_token_count = count_message_tokens(messages)
        messages = trim_context(messages, max_context_tokens)
        st.session_state.last_token_count = count_message_tokens(messages)

        more = stream_completion(messages)
        if not more:
            more = "Here's a bit more information."

        st.session_state.history.append({"role": "assistant", "content": more})
        st.session_state.pairs.append({"user": "[more info please]", "assistant": more})
        st.session_state.more_info_context["times_provided"] += 1
        ask_more_info()
        return

    if is_no:
        st.session_state.awaiting_more_info = False
        st.session_state.more_info_context = None
        with st.chat_message("assistant"):
            st.markdown("Got it! What question can I help with next?")
        st.session_state.history.append({"role": "assistant", "content": "Got it! What question can I help with next?"})
        return

    # not clear → re-ask
    with st.chat_message("assistant"):
        st.markdown("Please reply **yes** or **no**. Do you want more info?")
    st.session_state.history.append({"role": "assistant", "content": "Please reply **yes** or **no**. Do you want more info?"})

# ---------------------------
# Chat Input
# ---------------------------
user_input = st.chat_input("Ask me anything…")
if user_input:
    if st.session_state.awaiting_more_info:
        handle_more_info_reply(user_input)
    else:
        handle_new_question(user_input)

# ---------------------------
# Token Diagnostics
# ---------------------------
with st.expander("🔎 Tokens sent in last request (approx.)"):
    st.write(f"Estimated context tokens last request: **{st.session_state.last_token_count}**")
    st.caption("Uses `tiktoken` if available; otherwise a rough estimate (~4 chars ≈ 1 token).")

st.caption(
    "Tip: Only the last 2 turns are passed to the model as context (trimmed further if over the cap). "
)
