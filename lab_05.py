import requests
import streamlit as st
from openai import OpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
import json

st.title("IST 688 - LAB 05C: Travel Weather & Suggestion Bot")

# -------------------------------
# Weather Function
# -------------------------------
def get_current_weather(location: str, api_key: str):
    """Fetch current weather data from OpenWeatherMap API"""
    if "," in location:
        location = location.split(",")[0].strip()

    url = f"https://api.openweathermap.org/data/2.5/weather?q={location}&appid={api_key}"
    response = requests.get(url)
    if response.status_code != 200:
        return None

    data = response.json()
    try:
        # Kelvin → Celsius
        temp = data["main"]["temp"] - 273.15
        feels_like = data["main"]["feels_like"] - 273.15
        description = data["weather"][0]["description"]

        return {
            "location": location,
            "temperature": round(temp, 1),
            "feels_like": round(feels_like, 1),
            "description": description,
        }
    except KeyError:
        return None


# -------------------------------
# API Keys
# -------------------------------
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
WEATHER_API_KEY = st.secrets["WEATHER_API_KEY"]
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]

# -------------------------------
# Sidebar: Model Selector
# -------------------------------
st.sidebar.header("⚙️ Model Settings")
model_choice = st.sidebar.radio(
    "Select LLM:", ["OpenAI (gpt-4o-mini)", "Gemini (gemini-2.0-flash-lite)"]
)

# -------------------------------
# Streamlit UI
# -------------------------------
city = st.text_input("Enter a city:", "Syracuse, NY")

if st.button("Get Travel Weather Suggestion"):
    # Step 1: Retrieve weather (always required)
    if not city.strip():
        city = "Syracuse, NY"
    weather = get_current_weather(city, WEATHER_API_KEY)

    if not weather:
        st.error("Could not fetch weather data. Try another city.")
    else:
        # Step 2: Build weather context
        weather_context = (
            f"The weather in {weather['location']} is {weather['description']}, "
            f"temperature {weather['temperature']}°C, feels like {weather['feels_like']}°C."
        )

        st.subheader(f"🌍 Weather in {weather['location']}")
        st.write(f"☁️ {weather['description'].capitalize()}")
        st.write(f"🌡 {weather['temperature']}°C (feels like {weather['feels_like']}°C)")

        # Step 3: Generate clothing + picnic suggestion
        if "OpenAI" in model_choice:
            client = OpenAI(api_key=OPENAI_API_KEY)
            suggestion = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a travel assistant that suggests clothing and picnic advice."},
                    {"role": "user", "content": weather_context},
                    {"role": "user", "content": "What should I wear today? Is it a good day for a picnic?"}
                ],
                temperature=0.3,
                max_tokens=200,
            )
            advice = suggestion.choices[0].message.content

        else:  # Gemini
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash-lite",
                temperature=0.3,
                google_api_key=GEMINI_API_KEY
            )
            response = llm.invoke(
                f"{weather_context}\n\nPlease suggest appropriate clothing and say if it's a good picnic day."
            )
            advice = response.content

        # Display suggestion
        st.subheader("👕 Travel Suggestion")
        st.write(advice)
