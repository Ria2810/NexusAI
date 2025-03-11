import streamlit as st
from Utils.Agents import (
    Cardiologist, Psychologist, Pulmonologist,
    Dermatologist, Neurologist, Gastroenterologist,
    Endocrinologist, Orthopedist, Nephrologist, Oncologist,
    MultidisciplinaryTeam
)
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

# Set up the page configuration with title and icon
st.set_page_config(page_title="Advanced AI Healthcare Agent", page_icon=":hospital:")

# Sidebar with description, features, and usage instructions
with st.sidebar:
    st.markdown("# Advanced AI Healthcare Agent")
    st.markdown("## Overview")
    st.write(
        "This platform leverages advanced AI-powered healthcare agents to provide a comprehensive analysis of your medical reports. "
        "Upload a medical report (TXT file) and receive detailed insights from multiple specialists including a Cardiologist, Psychologist, "
        "Pulmonologist, Dermatologist, Neurologist, Gastroenterologist, Endocrinologist, Orthopedist, Nephrologist, and Oncologist. "
        "A multidisciplinary team then combines these expert opinions to deliver a final diagnosis."
    )
    st.markdown("## Features")
    st.write("- **Multi-Specialist Analysis:** Individual analysis from various specialized healthcare agents.")
    st.write("- **Concurrent Processing:** Agents run concurrently for faster results.")
    st.write("- **Holistic Diagnosis:** Combines expert opinions for a comprehensive final diagnosis.")
    st.write("- **Downloadable Report:** Save your final diagnosis for future reference.")
    st.markdown("## How to Use")
    st.write("1. Upload a medical report in TXT format.")
    st.write("2. Click **Analyze Report** to process the file.")
    st.write("3. View and download the final diagnosis.")

# Center aligned title with emojis using HTML
st.markdown("<h1 style='text-align: center;'>🩺🤖 Advanced AI Healthcare Agent 🤖🩺</h1>", unsafe_allow_html=True)
st.write("Upload a medical report file (TXT) to receive a comprehensive analysis from multiple healthcare specialists.")

# File uploader
uploaded_file = st.file_uploader("Choose a medical report file", type=["txt"])

if uploaded_file is not None:
    # Decode and display the uploaded medical report in a styled box
    medical_report = uploaded_file.read().decode("utf-8")
    st.markdown(
        "<h3 style='background-color: #3A506B; padding: 10px; border-radius: 5px; color: white;'>📝 Uploaded Medical Report</h3>",
        unsafe_allow_html=True
    )
    st.text_area("", medical_report, height=200)
    
    if st.button("Analyze Report"):
        st.info("Processing the medical report. Please wait...")
        
        # Initialize all specialized agents
        agents = {
            "Cardiologist": Cardiologist(medical_report),
            "Psychologist": Psychologist(medical_report),
            "Pulmonologist": Pulmonologist(medical_report),
            "Dermatologist": Dermatologist(medical_report),
            "Neurologist": Neurologist(medical_report),
            "Gastroenterologist": Gastroenterologist(medical_report),
            "Endocrinologist": Endocrinologist(medical_report),
            "Orthopedist": Orthopedist(medical_report),
            "Nephrologist": Nephrologist(medical_report),
            "Oncologist": Oncologist(medical_report)
        }

        responses = {}
        def get_response(agent_name, agent):
            response = agent.run()
            return agent_name, response

        # Run agents concurrently
        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(get_response, name, agent): name for name, agent in agents.items()}
            for future in as_completed(futures):
                agent_name, response = future.result()
                responses[agent_name] = response

        # Run the multidisciplinary team analysis to combine results
        team_agent = MultidisciplinaryTeam(
            cardiologist_report=responses.get("Cardiologist", ""),
            psychologist_report=responses.get("Psychologist", ""),
            pulmonologist_report=responses.get("Pulmonologist", ""),
            dermatologist_report=responses.get("Dermatologist", ""),
            neurologist_report=responses.get("Neurologist", ""),
            gastroenterologist_report=responses.get("Gastroenterologist", ""),
            endocrinologist_report=responses.get("Endocrinologist", ""),
            orthopedist_report=responses.get("Orthopedist", ""),
            nephrologist_report=responses.get("Nephrologist", ""),
            oncologist_report=responses.get("Oncologist", "")
        )
        final_diagnosis = team_agent.run()

        # Display the final diagnosis in a styled box with emojis
        st.markdown(
            f"<div style='background-color: #3A506B; padding: 15px; border-radius: 5px; margin-top: 20px; color: white;'>"
            f"<strong>🩺 Final Diagnosis:</strong><br>{final_diagnosis}</div>",
            unsafe_allow_html=True
        )

        # Save results to a file and provide download option
        results_path = "results/final_diagnosis.txt"
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        with open(results_path, "w") as file:
            file.write("### Final Diagnosis:\n\n" + final_diagnosis)
        
        with open(results_path, "r") as file:
            result_data = file.read()
        st.download_button("Download Final Diagnosis", result_data, file_name="final_diagnosis.txt")
