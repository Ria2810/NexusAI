# app.py
import streamlit as st
from career_assistant import (
    run_user_query,
    handle_resume_making,
    handle_learning_resource,
    handle_interview_preparation,
    job_search,
    tutorial_agent,
    ask_query_bot,
    interview_topics_questions,
    mock_interview
)

# -------------------------------
# Initialize Session State
# -------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []  # conversation history
if "current_route" not in st.session_state:
    st.session_state.current_route = None  # e.g. "handle_resume_making"

# -------------------------------
# Custom CSS for Chat Bubbles
# -------------------------------
st.markdown(
    """
    <style>
        .chat-container {
            max-width: 800px;
            margin: auto;
        }
        .chat-bubble {
            padding: 10px;
            border-radius: 10px;
            margin: 10px;
            max-width: 85%;
            display: inline-block;
            word-wrap: break-word;
        }
        .user-message {
            background-color: #343166;
            float: right;
            clear: both;
        }
        .bot-message {
            background-color: #343166;
            float: left;
            clear: both;
        }
        .clearfix::after {
            content: "";
            clear: both;
            display: table;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------------
# Sidebar with Chatbot Description
# -------------------------------
st.sidebar.title("GenAI Career Assistant 🚀")
st.sidebar.markdown("""
### Your Personalized AI Career Advisor

**Key Features:**

1. **Learning & Content Creation 🎓:**
   - Tailored learning pathways in GenAI covering essential topics and cutting-edge skills.
   - Assistance in creating engaging tutorials, blogs, and posts based on your interests.

2. **Q&A Support ❓:**
   - Real-time Q&A sessions to help you tackle complex concepts and coding challenges.

3. **Resume Building & Review 💼:**
   - One-on-one resume consultations and expert guidance.
   - Craft personalized, market-relevant resumes optimized for today's job trends.

4. **Interview Preparation 🤖:**
   - Simulated interview sessions featuring both technical and behavioral questions.
   - Experience realistic mock interviews with expert feedback to boost your confidence.

5. **Job Search Assistance 🔍:**
   - Navigate the job market with tailored insights and support.
   - Receive personalized guidance to find the best opportunities in GenAI.

With GenAI Career Assistant, your journey to a successful career in Generative AI becomes organized, personalized, and efficient!
""")

# -------------------------------
# Main Chat Interface
# -------------------------------
st.markdown("<h1 style='text-align: center;'>Chat with Career Assistant 🤖</h1>", unsafe_allow_html=True)

def display_messages():
    for msg in st.session_state.messages:
        sender = msg["sender"]
        text = msg["text"]
        if sender == "user":
            st.markdown(
                f"""
                <div class="chat-container clearfix">
                    <div class="chat-bubble user-message">
                        <strong>🙂 You:</strong> {text}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="chat-container clearfix">
                    <div class="chat-bubble bot-message">
                        <strong>🤖 Bot:</strong> {text}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

display_messages()

# -------------------------------
# User Input & Interaction
# -------------------------------
with st.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_input("Your Message:", key="input", placeholder="Type your query here...")
    submit_button = st.form_submit_button("Send")

if submit_button and user_input.strip():
    # 1) Add user's message to chat history
    st.session_state.messages.append({"sender": "user", "text": user_input})

    # 2) If user says "new topic" or no route chosen yet, do full classification
    if user_input.lower().strip() == "new topic" or not st.session_state.current_route:
        results = run_user_query(user_input)
        st.session_state.current_route = results["category"]  # e.g. "handle_resume_making", "ask_query_bot", ...
        bot_response = results["response"]
    else:
        # 3) We already have an active route, call that route function directly
        route = st.session_state.current_route

        # If you have sub-routes, you can store them too in session_state.
        # For simplicity, we’ll just call the top-level route each time:
        if route == "handle_resume_making":
            next_state = handle_resume_making({"query": user_input, "category": "", "response": ""})
            bot_response = next_state["response"]

        elif route == "handle_learning_resource":
            # This route typically leads to "tutorial_agent" or "ask_query_bot."
            # If you want to store a sub-route in session state, you can do so.
            # But if you want a dynamic approach, call handle_learning_resource again:
            # or just default to ask_query_bot for subsequent queries:
            #   next_state = ask_query_bot({"query": user_input, "category": "", "response": ""})
            #   bot_response = next_state["response"]

            # Actually let's re-classify the sub-route each time:
            sub_category = handle_learning_resource({"query": user_input, "category": "", "response": ""})["category"]
            if "tutorial" in sub_category.lower():
                next_state = tutorial_agent({"query": user_input, "category": "", "response": ""})
            else:
                next_state = ask_query_bot({"query": user_input, "category": "", "response": ""})
            bot_response = next_state["response"]

        elif route == "handle_interview_preparation":
            # Similarly handle sub-route
            sub_category = handle_interview_preparation({"query": user_input, "category": "", "response": ""})["category"]
            if "mock" in sub_category.lower():
                next_state = mock_interview({"query": user_input, "category": "", "response": ""})
            else:
                next_state = interview_topics_questions({"query": user_input, "category": "", "response": ""})
            bot_response = next_state["response"]

        elif route == "job_search":
            next_state = job_search({"query": user_input, "category": "", "response": ""})
            bot_response = next_state["response"]

        else:
            # If route is something else, or empty, re-run classification
            results = run_user_query(user_input)
            st.session_state.current_route = results["category"]
            bot_response = results["response"]

    # 4) Append bot response
    st.session_state.messages.append({"sender": "bot", "text": bot_response})
    st.rerun()

# -------------------------------
# Clear Chat
# -------------------------------
if st.button("Clear Chat"):
    st.session_state.messages = []
    st.session_state.current_route = None
    st.rerun()
