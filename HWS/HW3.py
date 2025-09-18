# HW3.py — Streaming Chatbot that discusses up to TWO URLs (OpenAI, Gemini, Mistral)
# Features:
#  • Two URL inputs (sidebar) → content extracted via requests + BeautifulSoup
#  • Vendor + model picker: OpenAI (mini/flagship), Gemini (flash/pro), Mistral (small/large)
#  • Memory strategy picker: (A) last 6 Qs, (B) token buffer (2000), (C) conversation summary
#  • Streaming answers where supported
#  • Token counting + context trimming
#  • Uses both URLs (if provided) + conversation memory to answer the question
import os
from typing import List, Dict, Optional

import streamlit as st
import requests
from bs4 import BeautifulSoup

# --- Optional tokenizer ---
try:
    import tiktoken
except Exception:
    tiktoken = None

# --- Vendor SDKs (import lazily in call sites if you prefer) ---
import openai
try:
    import google.generativeai as genai
except Exception:
    genai = None
try:
    from mistralai import Mistral
except Exception:
    Mistral = None

st.set_page_config(page_title="HW3 – URL Chatbot with Memory", page_icon="🧠")
st.title("🖇️ Home Work 3: Streaming Chatbot that Discusses URLs")

# --- Secrets ---
# OPENAI_API_KEY  = st.secrets["api_keys"].get("OPENAI_API_KEY")
# GEMINI_API_KEY  = st.secrets["api_keys"].get("GEMINI_API_KEY")
# MISTRAL_API_KEY = st.secrets["api_keys"].get("MISTRAL_API_KEY")

OPENAI_API_KEY  = st.secrets.get("OPENAI_API_KEY")
GEMINI_API_KEY  = st.secrets.get("GEMINI_API_KEY")
MISTRAL_API_KEY = st.secrets.get("MISTRAL_API_KEY")
# Create vendor clients (on demand)
def get_openai_client():
    return openai.OpenAI(api_key=OPENAI_API_KEY)

def ensure_gemini():
    if genai is None:
        raise RuntimeError("google-generativeai not installed")
    genai.configure(api_key=GEMINI_API_KEY)

def get_mistral_client():
    if Mistral is None:
        raise RuntimeError("mistralai not installed")
    return Mistral(api_key=MISTRAL_API_KEY)

# --- Sidebar: inputs ---
with st.sidebar:
    st.header("Inputs & Settings")

    url1 = st.text_input("URL 1", placeholder="https://how_to_ace_ist688_course.com")
    url2 = st.text_input("URL 2 ", placeholder="https://why_ischool_is_the_best_school.com")

    st.subheader("Model Vendor")
    vendor = st.selectbox("Provider", ["OpenAI", "Gemini", "Mistral"], index=0)

    st.caption("Pick cheap vs. flagship model")
    model_map = {
        "OpenAI": {"Cheap": "gpt-4o-mini", "Flagship": "gpt-4o"},
        "Gemini": {"Cheap": "gemini-2.5-flash", "Flagship": "gemini-1.5-pro"},
        "Mistral": {"Cheap": "mistral-small-latest", "Flagship": "mistral-large-latest"},
    }
    strength = st.radio("Tier", ["Cheap", "Flagship"], horizontal=True, index=0)
    model = model_map[vendor][strength]

    st.subheader("Memory Strategy")
    mem_strategy = st.selectbox(
        "Conversation memory",
        [
            "Buffer: last 6 questions",
            "Token buffer: 2000 tokens",
            "Conversation summary"
        ],
        index=0
    )

    temperature = st.slider("Temperature", 0.0, 1.0, 0.3, 0.1)
    max_output_tokens = st.slider("Max output tokens (reply)", 128, 2048, 512, 64)

    if st.button("🧹 Clear chat"):
        st.session_state.clear()
        st.rerun()

st.caption(f"Using **{vendor}** → `{model}` | Memory: **{mem_strategy}**")

# --- Session State ---
def init_state():
    if "history" not in st.session_state:
        st.session_state.history: List[Dict] = []  # for UI
    if "pairs" not in st.session_state:
        st.session_state.pairs: List[Dict] = []    # [{"user":..., "assistant":...}]
    if "summary" not in st.session_state:
        st.session_state.summary = ""              # rolling conversation summary
    if "last_token_count" not in st.session_state:
        st.session_state.last_token_count = 0
    if "url_cache" not in st.session_state:
        st.session_state.url_cache = {}            # {url: text}

init_state()

# --- Utilities ---
def estimate_tokens(text: str) -> int:
    if tiktoken:
        try:
            enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
        except Exception:
            pass
    return max(1, len(text) // 4)

def count_message_tokens(messages: List[Dict]) -> int:
    total = 0
    for m in messages:
        total += estimate_tokens(f"{m.get('role','')}: {m.get('content','')}")
    return total

def read_url(url: str) -> Optional[str]:
    if not url:
        return None
    if url in st.session_state.url_cache:
        return st.session_state.url_cache[url]
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers=headers, timeout=20)
        r.raise_for_status()
        soup = BeautifulSoup(r.content, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        main = soup.select_one("#mw-content-text") or soup.select_one("main") or soup.body
        if not main:
            return None
        text = main.get_text(separator="\n", strip=True)
        lines = [ln for ln in (t.strip() for t in text.splitlines()) if ln]
        out = "\n".join(lines) if lines else None
        st.session_state.url_cache[url] = out
        return out
    except Exception as e:
        st.warning(f"Could not read URL: {url} ({e})")
        return None

def build_system_prompt():
    return "You are a careful assistant. Answer using the provided web content and the conversation context. If unsure, say so."

def build_context_messages(user_text: str, vendor_name: str) -> List[Dict]:
    # URL contents
    u1 = read_url(url1) if url1 else None
    u2 = read_url(url2) if url2 else None

    url_block = ""
    if u1:
        url_block += f"\n\n[URL 1 content]\n{u1[:8000]}"
    if u2:
        url_block += f"\n\n[URL 2 content]\n{u2[:8000]}"
    if not url_block:
        url_block = "\n\n[No URL content provided]"

    # Memory strategy
    msgs: List[Dict] = [{"role": "system", "content": build_system_prompt()}]

    if mem_strategy == "Buffer: last 6 questions":
        # last 6 user+assistant turns
        for pair in st.session_state.pairs[-6:]:
            msgs.append({"role": "user", "content": pair["user"]})
            msgs.append({"role": "assistant", "content": pair["assistant"]})

    elif mem_strategy == "Token buffer: 2000 tokens":
        # Add as many past messages as fit under ~2000 tokens (including system + urls + user)
        # We'll add history from the end (most recent) backwards.
        tail: List[Dict] = []
        # flatten into chat-style messages
        for pair in reversed(st.session_state.pairs):
            tail.insert(0, {"role": "user", "content": pair["user"]})
            tail.insert(1, {"role": "assistant", "content": pair["assistant"]})
            temp = msgs + tail + [{"role": "user", "content": f"[URL MATERIAL]{url_block}\n\nUser question: {user_text}"}]
            if count_message_tokens(temp) > 2000:
                # remove the last inserted pair if over limit
                tail = tail[2:]
                break
        msgs += tail

    elif mem_strategy == "Conversation summary":
        # Maintain/update a rolling summary after each turn (below in post-processing)
        if st.session_state.summary:
            msgs.append({"role": "system", "content": f"Conversation summary so far:\n{st.session_state.summary}"})

    # Append current user prompt with URL material inline
    msgs.append({"role": "user", "content": f"[URL MATERIAL]{url_block}\n\nUser question: {user_text}"})
    return msgs

def update_conversation_summary():
    """After appending the newest pair, compress conversation into summary (cheap model)."""
    try:
        if not OPENAI_API_KEY:
            return
        client = get_openai_client()
        # Build a short compression prompt using last ~10 turns
        recent_pairs = st.session_state.pairs[-10:]
        convo = []
        for p in recent_pairs:
            convo.append(f"User: {p['user']}\nAssistant: {p['assistant']}")
        convo_text = "\n\n".join(convo)
        prompt = (
            "Summarize the following chat so the key facts, constraints, and tasks are preserved. "
            "Use 5–8 short bullet points. Keep it neutral and factual.\n\n"
            f"{convo_text}"
        )
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=300,
        )
        st.session_state.summary = resp.choices[0].message.content.strip()
    except Exception:
        # Best-effort; ignore failures
        pass

# --- Vendor-specific streaming ---
def stream_openai(messages: List[Dict]) -> str:
    client = get_openai_client()
    placeholder = st.empty()
    final = []
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
            final.append(delta)
            placeholder.markdown("".join(final))
    return "".join(final).strip()

def stream_gemini(messages: List[Dict]) -> str:
    ensure_gemini()
    # Convert messages to a single prompt (Gemini doesn't use roles the same way)
    sys_texts = [m["content"] for m in messages if m["role"] == "system"]
    user_texts = [m["content"] for m in messages if m["role"] != "system"]
    prompt = ""
    if sys_texts:
        prompt += "System instructions:\n" + "\n".join(sys_texts) + "\n\n"
    prompt += "\n\n".join(user_texts)

    # Streaming
    try:
        gmodel = genai.GenerativeModel(model)
        resp = gmodel.generate_content(prompt, stream=True, generation_config={"temperature": temperature})
        placeholder = st.empty()
        final = []
        for chunk in resp:
            part = getattr(chunk, "text", "") or ""
            if part:
                final.append(part)
                placeholder.markdown("".join(final))
        return "".join(final).strip()
    except Exception:
        # Fallback non-streaming
        gmodel = genai.GenerativeModel(model)
        resp = gmodel.generate_content(prompt)
        st.markdown(resp.text)
        return resp.text or ""

def stream_mistral(messages: List[Dict]) -> str:
    client = get_mistral_client()
    placeholder = st.empty()
    final = []
    try:
        # New mistralai supports streaming with client.chat.stream
        stream = client.chat.stream(model=model, messages=messages, temperature=temperature, max_tokens=max_output_tokens)
        for event in stream:
            if hasattr(event, "data") and event.data and event.data.choices:
                delta = (event.data.choices[0].delta.get("content") or "")
                if delta:
                    final.append(delta)
                    placeholder.markdown("".join(final))
        return "".join(final).strip()
    except Exception:
        # Fallback non-streaming
        resp = client.chat.complete(model=model, messages=messages, temperature=temperature, max_tokens=max_output_tokens)
        text = resp.choices[0].message.content
        st.markdown(text)
        return text

def stream_answer(messages: List[Dict]) -> str:
    # Count tokens pre-trim (rough)
    st.session_state.last_token_count = count_message_tokens(messages)

    if vendor == "OpenAI":
        return stream_openai(messages)
    if vendor == "Gemini":
        return stream_gemini(messages)
    if vendor == "Mistral":
        return stream_mistral(messages)
    raise ValueError("Unsupported vendor")

# --- Render previous chat ---
for m in st.session_state.history:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# --- Chat input + flow ---
user_input = st.chat_input("Ask about the URLs… (e.g., 'Summarize the rules for base running')")
if user_input:
    # Build messages per selected memory strategy and URLs
    messages = build_context_messages(user_input, vendor)
    st.session_state.last_token_count = count_message_tokens(messages)

    # Show user
    st.session_state.history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Stream answer
    with st.chat_message("assistant"):
        answer = stream_answer(messages)

    st.session_state.history.append({"role": "assistant", "content": answer})
    st.session_state.pairs.append({"user": user_input, "assistant": answer})

    # If using conversation summary, update summary
    if mem_strategy == "Conversation summary":
        update_conversation_summary()

# --- Diagnostics ---
with st.expander("🔎 Tokens sent this request (approx.)"):
    st.write(f"Estimated context tokens: **{st.session_state.last_token_count}**")
    st.caption("Uses `tiktoken` when available; otherwise a rough ~4 chars/token estimate.")

st.caption("Tip: Both URLs (if provided) + the selected memory strategy are used to answer each question.")
