import os
import time
import streamlit as st
from generating_syllabus import generate_syllabus
from teaching_agent import teaching_agent
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)

# Set up the page configuration with title and icon
st.set_page_config(page_title="Your AI Instructor", page_icon=":robot_face:")

# Sidebar with description, features, and example actions
with st.sidebar:
    st.markdown("# Your AI Instructor")
    st.markdown("## Overview")
    st.write(
        "This platform empowers you to create a custom learning experience. "
        "Use the syllabus generator to build your course outline and interact with an AI-powered instructor for personalized guidance."
    )
    st.markdown("## Features")
    st.write("- **Syllabus Generator:** Automatically generate a detailed course syllabus for the topic you want to learn.")
    st.write("- **AI Instructor Chat:** Engage in real-time conversation with an AI instructor to get explanations and further insights.")
    st.write("- **Interactive Learning:** Experience a dynamic, conversation-based learning interface.")
    st.markdown("## Example Actions")
    st.write("- Generate a course syllabus on Data Science")
    st.write("- Ask: 'Can you explain how gradient descent works?'")
    st.write("- Ask: 'What are the prerequisites for machine learning?'")

# Centered title with emojis for an AI instructor
st.markdown("<h1 style='text-align: center;'>🤖 Your AI Instructor 🤖</h1>", unsafe_allow_html=True)

# Create two tabs: one for building the bot and one for the AI instructor chat
tab1, tab2 = st.tabs(["Input Your Information", "AI Instructor"])

with tab1:
    st.header("Input Your Information")
    # Text input for the topic the user wants to learn
    input_text = st.text_input("State the name of topic you want to learn:")
    if st.button("Build the Bot!!!"):
        if input_text:
            task = "Generate a course syllabus to teach the topic: " + input_text
            # Generate the syllabus using the provided function
            syllabus = generate_syllabus(input_text, task)
            # Seed the teaching agent with the syllabus and task
            teaching_agent.seed_agent(syllabus, task)
            # Display the generated syllabus
            st.text_area("Your syllabus will be shown here:", value=syllabus, height=300)
        else:
            st.warning("Please enter a topic.")

with tab2:
    st.header("AI Instructor")
    
    # Initialize chat history in session_state if not already done
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Container to display chat messages using HTML for colored boxes
    chat_container = st.empty()

    def update_chat():
        """Update the chat container with messages in colored boxes."""
        chat_html = ""
        for user_msg, bot_msg in st.session_state.chat_history:
            user_html = (
                f"<div style='background-color: #333255; padding: 10px; border-radius: 5px; margin-bottom: 10px;'>"
                f"😃 <b>You:</b> {user_msg}</div>"
            )
            bot_html = (
                f"<div style='background-color: #333255; padding: 10px; border-radius: 5px; margin-bottom: 10px;'>"
                f"🤖 <b>Bot:</b> {bot_msg}</div>"
            )
            chat_html += user_html + bot_html
        chat_container.markdown(chat_html, unsafe_allow_html=True)

    update_chat()

    # User input for the chat conversation
    user_input = st.text_input("What do you concern about?", key="user_input")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Send"):
            if user_input:
                # Process the user message
                teaching_agent.human_step(user_input)
                # Append the user's message with an empty bot reply
                st.session_state.chat_history.append((user_input, ""))
                update_chat()
                # Get the bot's reply (simulate typing effect)
                bot_message = teaching_agent.instructor_step()
                reply = ""
                for char in bot_message:
                    reply += char
                    st.session_state.chat_history[-1] = (user_input, reply)
                    update_chat()
                    time.sleep(0.05)
                # Clear the text input after sending
                st.session_state.user_input = ""
            else:
                st.warning("Please enter a message.")
    with col2:
        if st.button("Clear"):
            st.session_state.chat_history = []
            update_chat()
