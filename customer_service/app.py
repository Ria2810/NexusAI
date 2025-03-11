import streamlit as st
from workflow import run_customer_support

# Set up the page configuration
st.set_page_config(page_title="TechNova Solutions Support Chatbot", page_icon=":robot_face:")

# Sidebar with description, features, and example questions
with st.sidebar:
    st.markdown("# TechNova Solutions Chatbot")
    st.markdown("## Overview")
    st.write(
        "This intelligent customer support chatbot is designed to assist with a wide range of inquiries including technical issues, "
        "billing questions, and general support. It leverages advanced language models to ensure you receive quick and accurate responses."
    )
    st.markdown("## Features")
    st.write("- **Query Categorization:** Automatically classifies inquiries into technical, billing, or general categories.")
    st.write("- **Sentiment Analysis:** Assesses the emotional tone of your queries to better understand your needs.")
    st.write("- **Response Generation:** Provides tailored responses based on your inquiry and sentiment.")
    st.write("- **Escalation Mechanism:** Escalates issues to human agents when necessary for further assistance.")
    st.markdown("## Example Questions")
    st.write("- My internet connection keeps dropping. Can you help?")
    st.write("- I need help talking to chatGPT")
    st.write("- Where can I find my receipt?")
    st.write("- What are your business hours?")

# Center aligned title with emojis using HTML
st.markdown("<h1 style='text-align: center;'>🤖 TechNova Solutions Support Chatbot 🤖</h1>", unsafe_allow_html=True)
st.write("Welcome to TechNova Solutions customer support. How can we help you today?")

# Initialize conversation history in session state
if "conversation" not in st.session_state:
    st.session_state.conversation = []

# Form for the user to enter a query
with st.form(key="user_input_form", clear_on_submit=True):
    user_query = st.text_input("Enter your query:", "")
    submit_button = st.form_submit_button(label="Send")

# When the user submits a query, process it and update the conversation history
if submit_button and user_query:
    # Append the user's message to the conversation history
    st.session_state.conversation.append({"role": "user", "message": user_query})
    
    # Run the customer support workflow using LangChain ChatOpenAI
    result = run_customer_support(user_query)
    agent_response = result["response"]
    
    # Append the agent's response to the conversation history
    st.session_state.conversation.append({"role": "agent", "message": agent_response})

# Display the conversation history with emojis and styled message boxes
for chat in st.session_state.conversation:
    if chat["role"] == "user":
        st.markdown(
            f"<div style='background-color: #3A506B; padding: 10px; border-radius: 5px; margin-bottom: 10px; color: white;'>"
            f"<strong>👤 You:</strong> {chat['message']}</div>", 
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"<div style='background-color: #3A506B; padding: 10px; border-radius: 5px; margin-bottom: 10px; color: white;'>"
            f"<strong>🤖 TechNova Agent:</strong> {chat['message']}</div>", 
            unsafe_allow_html=True
        )
