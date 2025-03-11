# pylint: disable = invalid-name
import os
import uuid
import streamlit as st
from langchain_core.messages import HumanMessage

from agents.agent import Agent

def send_email(receiver_email, thread_id):
    try:
        # Set the receiver email dynamically from user input
        os.environ['TO_EMAIL'] = receiver_email
        
        # Ensure default sender and subject are set (from .env or fallback defaults)
        if 'FROM_EMAIL' not in os.environ:
            os.environ['FROM_EMAIL'] = 'riachoudhari9@gmail.com'
        if 'EMAIL_SUBJECT' not in os.environ:
            os.environ['EMAIL_SUBJECT'] = 'Travel Information'
        
        # Prepare a state with the travel information
        state = {'messages': [HumanMessage(content=st.session_state.travel_info)]}
        
        # Directly call the email_sender function to force its execution
        st.session_state.agent.email_sender(state)
        
        st.success('Email sent successfully!')
        # Clear session state
        for key in ['travel_info', 'thread_id']:
            st.session_state.pop(key, None)
    except Exception as e:
        st.error(f'Error sending email: {e}')


def initialize_agent():
    if 'agent' not in st.session_state:
        st.session_state.agent = Agent()


def render_custom_css():
    st.markdown(
        '''
        <style>
        .main-title {
            font-size: 2.5em;
            text-align: center;
            margin-bottom: 0.5em;
            font-weight: bold;
        }
        .sub-title {
            font-size: 1.2em;
            text-align: left;
            margin-bottom: 0.5em;
        }
        .center-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            width: 100%;
        }
        .query-box {
            width: 80%;
            max-width: 600px;
            margin-top: 0.5em;
            margin-bottom: 1em;
        }
        .query-container {
            width: 80%;
            max-width: 600px;
            margin: 0 auto;
        }
        </style>
        ''', unsafe_allow_html=True)


def render_ui():
    st.markdown('<div class="center-container">', unsafe_allow_html=True)
    st.markdown('<div class="main-title">✈️🌍 AI Travel Agent 🏨🗺️</div>', unsafe_allow_html=True)
    st.markdown('<div class="query-container">', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Enter your travel query and get flight and hotel information:</div>', unsafe_allow_html=True)
    user_input = st.text_area(
        'Travel Query',
        height=200,
        key='query',
        placeholder='Type your travel query here...',
    )
    st.markdown('</div>', unsafe_allow_html=True)
    return user_input


def process_query(user_input):
    if user_input:
        try:
            thread_id = str(uuid.uuid4())
            st.session_state.thread_id = thread_id

            messages = [HumanMessage(content=user_input)]
            config = {'configurable': {'thread_id': thread_id}}

            result = st.session_state.agent.graph.invoke({'messages': messages}, config=config)

            st.subheader('Travel Information')
            st.write(result['messages'][-1].content)

            st.session_state.travel_info = result['messages'][-1].content

        except Exception as e:
            st.error(f'Error: {e}')
    else:
        st.error('Please enter a travel query.')


def render_email_form():
    with st.form(key='email_form'):
        receiver_email = st.text_input('Enter your email address')
        submit_button = st.form_submit_button(label='Send Email')

    if submit_button:
        if receiver_email:
            send_email(receiver_email, st.session_state.thread_id)
        else:
            st.error('Please enter your email address.')


def main():
    # Initialize the agent and custom CSS
    initialize_agent()
    render_custom_css()
    
    # Sidebar with description, features, and example queries
    with st.sidebar:
        st.markdown("# AI Travel Agent")
        st.markdown("## Overview")
        st.write(
            "This AI-powered Travel Agent helps you get detailed travel information—including flight and hotel details—based on your queries. "
            "You can also send the travel information directly to your email."
        )
        st.markdown("## Features")
        st.write("- **Instant Travel Info:** Provides comprehensive travel details based on your input.")
        st.write("- **Email Integration:** Easily email the travel details to yourself.")
        st.write("- **User-Friendly Interface:** Simple and interactive design for a seamless experience.")
        st.markdown("## Example Queries")
        st.write("- 'Find me flights and hotels to Paris'")
        st.write("- 'Plan a weekend getaway to Bali'")
        st.write("- 'What are the travel options for a trip to New York?'")
        st.image('travel-agent/images/ai-travel.png', caption='AI Travel Assistant')
    
    # Main UI for travel query input
    user_input_text = render_ui()

    if st.button('Get Travel Information'):
        process_query(user_input_text)

    if 'travel_info' in st.session_state:
        render_email_form()


if __name__ == '__main__':
    main()
