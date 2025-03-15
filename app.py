import streamlit as st
import subprocess
import time
import urllib
import base64
import os
import requests
import streamlit.components.v1 as components


# Must be the first Streamlit command
st.set_page_config(page_title="NexusAI", page_icon="🤖", layout="wide")

#  --- Hide ALL Streamlit warnings in the UI ---
st.markdown("""
<style>
[data-testid="stWarning"] {
    display: none !important;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state flags
if "launched_agents" not in st.session_state:
    st.session_state["launched_agents"] = {}
if "open_url" not in st.session_state:
    st.session_state["open_url"] = None
if "redirect_done" not in st.session_state:
    st.session_state["redirect_done"] = False

# If an agent URL is already set and we've flagged redirect, immediately redirect.
if st.session_state["open_url"] and st.session_state["redirect_done"]:
    redirect_js = f"""
    <script>
      window.open("{st.session_state['open_url']}", "_self");
    </script>
    """
    st.markdown(redirect_js, unsafe_allow_html=True)
    st.stop()

def wait_for_agent(url, timeout=15, interval=1):
    """Polls the given URL until it returns a successful status or timeout is reached."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            r = requests.get(url)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(interval)
    return False

def launch_agent(agent_name, details):
    """Launch the specified agent in a new Streamlit process if not already launched."""
    if not st.session_state["launched_agents"].get(agent_name, False):
        st.session_state["launched_agents"][agent_name] = True
        # Launch the agent in a separate process
        subprocess.Popen(["streamlit", "run", details["script"], "--server.port", str(details["port"])])
        agent_url = f"http://localhost:{details['port']}"
        # Wait until the agent is up (using a polling mechanism)
        if wait_for_agent(agent_url, timeout=15):
            st.session_state["open_url"] = agent_url
            st.session_state["redirect_done"] = True
            st.success(f"{agent_name} is now running at [localhost:{details['port']}]({agent_url})")
        else:
            st.error(f"Timed out waiting for {agent_name} to start. Please check the agent's logs.")
    else:
        st.info(f"{agent_name} is already running at [localhost:{details['port']}](http://localhost:{details['port']}).")


# --- Define AI Agents with Unique Ports, Scripts, Emojis, Image Paths, and Descriptions ---
agents = {
    "Smart Resume Analyzer": {
        "script": "SmartResumeAnalyzer/app.py", 
        "port": 8522,
        "emoji": "📄",
        "img": "images/SmartResumeAnalyzer.webp",
        "description": "An AI-powered tool for analyzing  resumes with job descriptions and providing feedback."
    },

    "24/7 AI Chatbot": {
        "script": "customer_service/app.py", 
        "port": 8511,
        "emoji": "🤖",
        "img": "images/customer-service.jpg",
        "description": "A dedicated chatbot for customer service inquiries."
    },
    "AI Health Assistant": {
        "script": "medical_diagnostics_agent/app.py", 
        "port": 8502,
        "emoji": "💊",
        "img": "images/health-assistant.jpg",
        "description": "Provides health advice and medical information."
    },
    "Virtual Tutor": {
        "script": "EduGPT/src/app.py", 
        "port": 8503,
        "emoji": "📚",
        "img": "images/virtual-tutor.jpg",
        "description": "An intelligent tutor to assist with your learning journey."
    },
    "AI Data Visualization Agent": {
        "script": "Data_Visualization_Agent/ai_data_visualisation_agent.py", 
        "port": 8504,
        "emoji": "📊",
        "img": "images/data-visualization.jpg",
        "description": "Generates insightful data visualizations on demand."
    },
    "Multi-PDFs Chatapp": {
        "script": "Multi-PDFs_ChatApp/chatapp.py", 
        "port": 8505,
        "emoji": "📄",
        "img": "images/multi-PDFs.jpg",
        "description": "Chat with ease while processing multiple PDFs."
    },
    "Career Assistant": {
        "script": "career-assistant-agent/app.py", 
        "port": 8506,
        "emoji": "💼",
        "img": "images/career-assistant.jpg",
        "description": "Get career advice and job search support."
    },
    "Smart Farming Assistant": {
        "script": "KrishiBot/app.py", 
        "port": 8507,
        "emoji": "🌱",
        "img": "images/smart-farming.jpg",
        "description": "Guidance for modern smart farming techniques."
    },
    "AI Travel Agent": {
        "script": "travel-agent/app.py", 
        "port": 8508,
        "emoji": "✈️",
        "img": "images/travel-agent.jpg",
        "description": "Tailored travel planning and recommendations."
    },
    "Image to Speech GenAI Tool": {
        "script": "image-to-speech-Agent/app.py", 
        "port": 8509,
        "emoji": "🖼️",
        "img": "images/image-to-speech.jpg",
        "description": "Converts images into engaging speech descriptions."
    },
    "AI Lead Generation": {
        "script": "lead-generation-agent/ai_lead_generation_agent.py", 
        "port": 8510,
        "emoji": "📈",
        "img": "images/lead-generation.jpg",
        "description": "Automates lead generation for your business growth."
    },
}

@st.cache_data
def embed_images_as_base64(agent_dict):
    for name, info in agent_dict.items():
        ext = os.path.splitext(info["img"])[1].lower()
        mime = "image/jpeg" if ext in [".jpg", ".jpeg"] else "image/png"
        with open(info["img"], "rb") as f:
            encoded = base64.b64encode(f.read()).decode()
        info["img_base64"] = f"data:{mime};base64,{encoded}"
    return agent_dict

agents = embed_images_as_base64(agents)

# Inject stable CSS for styling
st.markdown("""
<style>
[data-testid="stHorizontalBlock"] {
    flex-wrap: wrap;
    justify-content: center;
}
.card {
    width: 350px;
    min-height: 200px;
    margin: 10px;
    background-color: #FFFFFF;
    border-radius: 8px;
    box-shadow: 0 2px 5px rgba(0,0,0,0.3);
    cursor: pointer;
    overflow: hidden;
    text-align: center;
    transition: transform 0.2s;
    display: flex;
    flex-direction: column;
}
.card:hover {
    transform: scale(1.05);
}
.card img {
    width: 100%;
    height: 150px;
    object-fit: contain;
}
.card h5 {
    margin: 0;
    padding: 10px;
    background-color: #3A506B;
    color: #FFFFFF;
}
.card p.description {
    margin: 0;
    padding-bottom: 10px;
    font-size: 0.9rem;
    background-color: #3A506B;
    color: #FFFFFF;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# Page title and description
st.markdown("<h1 style='text-align:center;'>✨NexusAI✨</h1>", unsafe_allow_html=True)
st.markdown("<h5 style='text-align:center;'>- By Ria Choudhari❤️</h5>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Click on any AI agent card to launch and open it automatically.</p>", unsafe_allow_html=True)

# Display cards in rows of 4 columns
columns = st.columns(4)
for idx, (agent_name, details) in enumerate(agents.items()):
    with columns[(idx+1) % 4]:
        st.markdown(f"""
        <a href="?agent={urllib.parse.quote(agent_name)}" style="text-decoration: none;" target="_self">
            <div class="card">
                <img src="{details['img_base64']}" alt="{agent_name}">
                <h5>{agent_name} {details['emoji']}</h5>
                <p class="description">{details['description']}</p>
            </div>
        </a>
        """, unsafe_allow_html=True)

# Handle launching agent via query parameters
query_params = st.experimental_get_query_params()
if "agent" in query_params:
    agent_to_launch = query_params["agent"][0]
    if agent_to_launch in agents:
        launch_agent(agent_to_launch, agents[agent_to_launch])
        # Clear query parameters to avoid re-triggering on rerun
        st.experimental_set_query_params()

# If an agent URL is set (and we haven't already redirected above), do a JavaScript redirect.
if st.session_state["open_url"] and not st.session_state["redirect_done"]:
    st.session_state["redirect_done"] = True
    redirect_js = f"""
    <script>
      window.open("{st.session_state['open_url']}", "_self");
    </script>
    """
    st.markdown(redirect_js, unsafe_allow_html=True)
    st.stop()
