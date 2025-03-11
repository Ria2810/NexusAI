import sys
if sys.platform.startswith("win"):
    import asyncio
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
import nest_asyncio
nest_asyncio.apply()

import os
import streamlit as st
from dotenv import load_dotenv
import openai
import requests
from gtts import gTTS
import string
import random
import numpy as np
import soundfile as sf
import av
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, RTCConfiguration, WebRtcMode

# --- Setup & Environment Variables ---
st.set_page_config(page_title="Agri Chatbot", page_icon="static/images/favicon.ico")
load_dotenv()
hugging_face = os.getenv('HUGGING_FACE')
open_ai_key = os.getenv('OPENAI_API_KEY')
client = openai.OpenAI(api_key=open_ai_key)

# Sidebar with description, features, and example questions
with st.sidebar:
    st.markdown("# KrishiBot")
    st.markdown("## Overview")
    st.write(
         "Krishi Help Bot is a friendly agriculture chatbot designed to assist farmers by providing accurate and timely information on "
         "agricultural practices, crop seasons, and farming tips. It supports both text and voice inputs to ensure a smooth, interactive experience."
    )
    st.markdown("## Features")
    st.write("- **Voice & Text Input:** Engage with the bot by typing or recording your queries.")
    st.write("- **AI-Powered Responses:** Leverages advanced language models via OpenAI and Hugging Face for expert advice.")
    st.write("- **Audio Transcription:** Converts recorded audio into text for further processing.")
    st.write("- **Agricultural Expertise:** Provides detailed insights on agriculture seasons, crop recommendations, and best farming practices.")
    st.markdown("## Example Questions")
    st.write("- What are the main agricultural seasons in India?")
    st.write("- Which crops should I plant during the Kharif season?")
    st.write("- How can I improve soil fertility?")
    st.write("- What are the best practices for pest management?")

# Directories for uploaded audio and generated MP3s
UPLOAD_FOLDER = 'uploads'
AUDIO_FOLDER = os.path.join('static', 'audio')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(AUDIO_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {'webm', 'wav'}

# --- Audio Processor for streamlit-webrtc ---
class AudioProcessor(AudioProcessorBase):
    def __init__(self) -> None:
        self.frames = []
    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        self.frames.append(frame.to_ndarray())
        return frame

# --- Updated OpenAI Function ---
def get_anwer_openai(question):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "I want you to act like a helpful agriculture chatbot and help farmers with their query"},
            {"role": "user", "content": "Give a Brif Of Agriculture Seasons in India"},
            {"role": "system", "content": (
                "In India, the agricultural season consists of three major seasons: the Kharif (monsoon), the Rabi (winter), "
                "and the Zaid (summer) seasons. Each season has its own specific crops and farming practices.\n\n"
                "1. Kharif Season (Monsoon Season):\nThe Kharif season typically starts in June and lasts until September. This season "
                "is characterized by the onset of the monsoon rains, which are crucial for agricultural activities in several parts "
                "of the country. Major crops grown during this season include rice, maize, jowar (sorghum), bajra (pearl millet), "
                "cotton, groundnut, turmeric, and sugarcane. These crops thrive in the rainy conditions and are often referred to as rain-fed crops.\n\n"
                "2. Rabi Season (Winter Season):\nThe Rabi season usually spans from October to March. This season is characterized by cooler "
                "temperatures and lesser or no rainfall. Crops grown during the Rabi season are generally sown in October and harvested in March-April. "
                "The major Rabi crops include wheat, barley, mustard, peas, gram (chickpeas), linseed, and coriander. These crops rely mostly on irrigation "
                "and are well-suited for the drier winter conditions.\n\n"
                "3. Zaid Season (Summer Season):\nThe Zaid season occurs between March and June and is a transitional period between Rabi and Kharif seasons. "
                "This season is marked by warmer temperatures and relatively less rainfall. The Zaid crops are grown during this time and include vegetables "
                "like cucumber, watermelon, muskmelon, bottle gourd, bitter gourd, and leafy greens such as spinach and amaranth. These crops are generally "
                "irrigated and have a shorter growing period compared to Kharif and Rabi crops.\n\n"
                "These three agricultural seasons play a significant role in India's agricultural economy and provide stability to food production throughout "
                "the year. Farmers adapt their farming practices and crop selection accordingly to make the best use of the prevailing climatic conditions in each season."
            )},
            {"role": "user", "content": question}
        ]
    )
    return response.choices[0].message.content

def text_to_audio(text, filename):
    tts = gTTS(text)
    tts.save(os.path.join(AUDIO_FOLDER, f'{filename}.mp3'))

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_audio(filepath):
    API_URL = "https://api-inference.huggingface.co/models/jonatasgrosman/wav2vec2-large-xlsr-53-english"
    headers = {"Authorization": hugging_face}
    with open(filepath, "rb") as f:
        data = f.read()
    response = requests.post(API_URL, headers=headers, data=data)
    data = response.json()
    return data.get('text', 'Transcription failed or returned empty.')

def process_text(text):
    response_text = get_anwer_openai(text)
    random_filename = ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))
    text_to_audio(response_text, random_filename)
    return {"text": response_text, "voice": f"{random_filename}.mp3"}

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.markdown(
    """
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.3.0/font/bootstrap-icons.css">
    <style>
      body { background-color: #f7f7f7; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
      .header-img { width: 100%; max-height: 200px; object-fit: cover; margin-bottom: 20px; }
      .chat-container {
          height: 60vh;
          overflow-y: auto;
          border: 1px solid #ccc;
          padding: 15px;
          border-radius: 8px;
          margin-bottom: 20px;
      }
      .user-msg {
          text-align: right;
          padding: 10px;
          border: 1px solid;
          border-radius: 8px;
          margin: 8px;
          display: inline-block;
      }
      .bot-msg {
          text-align: left;
          padding: 10px;
          border: 1px solid;
          border-radius: 8px;
          margin: 8px;
          display: inline-block;
      }
      .input-row { margin-bottom: 20px; }
      .footer { text-align: center; margin-top: 30px; font-size: 0.9em; color: #555; }
      .center { text-align: center; }
    </style>
    """, unsafe_allow_html=True
)

# --- Centered Title with Emojis ---
st.markdown("<h1 class='center'>🤖 Krishi Help Bot 🧑‍🌾</h1>", unsafe_allow_html=True)
st.markdown("<p class='center'>Disclaimer: please have patience, this might take some time.</p>", unsafe_allow_html=True)

# --- Chat Container ---
if st.session_state.chat_history:
    with st.container():
        for msg in st.session_state.chat_history:
            if msg[0] == "User":
                st.markdown(f"<div class='user-msg'><strong>👤 User:</strong> {msg[1]}</div>", unsafe_allow_html=True)
            elif msg[0] == "Bot":
                st.markdown(f"<div class='bot-msg'><strong>🤖 Bot:</strong> {msg[1]}</div>", unsafe_allow_html=True)
                if len(msg) > 2:
                    audio_file_path = os.path.join(AUDIO_FOLDER, msg[2])
                    if os.path.exists(audio_file_path):
                        with open(audio_file_path, "rb") as af:
                            st.audio(af.read(), format="audio/mp3")
        st.markdown("</div>", unsafe_allow_html=True)
else:
    st.info("No messages yet. Start the conversation below!")

# --- Input Row: Text Input & Recorder Side-by-Side ---
col_text, col_rec = st.columns([4, 1])
with col_text:
    user_input = st.text_input("Type your message...", key="user_input")
with col_rec:
    st.write("Record:")
    RTC_CONFIGURATION = {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    webrtc_ctx = webrtc_streamer(
        key="audio",
        mode=WebRtcMode.SENDONLY,
        rtc_configuration=RTC_CONFIGURATION,
        audio_processor_factory=AudioProcessor,
        media_stream_constraints={"audio": True, "video": False}
    )
    if webrtc_ctx.audio_processor is not None:
        if st.button("Send Recorded"):
            frames = webrtc_ctx.audio_processor.frames
            if frames:
                audio_data = np.concatenate(frames, axis=0)
                rec_filename = ''.join(random.choices(string.ascii_uppercase + string.digits, k=8)) + ".wav"
                rec_filepath = os.path.join(UPLOAD_FOLDER, rec_filename)
                sf.write(rec_filepath, audio_data, samplerate=44100)
                transcription = process_audio(rec_filepath)
                st.text_area("Transcription", transcription, height=100)
                st.session_state.chat_history.append(("User", transcription))
                result = process_text(transcription)
                st.session_state.chat_history.append(("Bot", result["text"], result["voice"]))
                st.rerun()
            else:
                st.warning("No audio recorded.")

if st.button("Send Message"):
    if user_input.strip():
        st.session_state.chat_history.append(("User", user_input))
        result = process_text(user_input)
        st.session_state.chat_history.append(("Bot", result["text"], result["voice"]))
        st.rerun()
    else:
        st.error("Please enter a message.")

st.markdown(
    """
    <div style="position: fixed; bottom: 0; left: 0; width: 100%; background-color: #0E1117; padding: 15px; text-align: center;">
            © <a href="https://github.com/Ria2810" target="_blank">Ria Choudhari</a> | Made with ❤️
    </div>
    """, unsafe_allow_html=True
)
