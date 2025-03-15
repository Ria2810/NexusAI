import streamlit as st
from dotenv import load_dotenv
import base64
import os
import io
from PIL import Image
import pdf2image
import google.generativeai as genai

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_gemini_response(input_text, pdf_content, prompt):
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content([input_text, pdf_content, prompt])
    return response.text if response else "Error in generating response"

def input_pdf_setup(uploaded_file):
    POPPLER_PATH =r"C:\poppler\Library\bin"
    if uploaded_file is not None:
        # images = pdf2image.convert_from_bytes(uploaded_file.read())
        images = pdf2image.convert_from_bytes(uploaded_file.read(), poppler_path=POPPLER_PATH)
        if not images:
            st.error("Failed to extract images from the PDF.")
            return None
        first_page = images[0]
        img_byte_arr = io.BytesIO()
        first_page.save(img_byte_arr, format='JPEG')
        img_byte_arr = img_byte_arr.getvalue()
        pdf_parts = [{"mime_type": "image/jpeg", "data": base64.b64encode(img_byte_arr).decode()}]
        return pdf_parts[0]  # Returning only the first page as an image input
    else:
        st.warning("No file uploaded.")
        return None

# Streamlit Responsive UI Design
st.set_page_config(page_title="HirEase AI", layout="centered")
st.markdown("""
    <style>
        .main-container {
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
        }
        .stButton button {
            width: 100%;
            font-size: 16px;
            padding: 10px;
            border-radius: 8px;
        }
        .uploaded-file {
            text-align: center;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)

st.title("🤖 HirEase AI ")
st.title("Smart Resume Analyzer")
st.write("Upload your resume and paste the job description to analyze how well your profile matches the job role.")

with st.sidebar:
    st.header("📂 Upload Your Resume")
    uploaded_file = st.file_uploader("Upload your resume (PDF format only)", type=["pdf"])
    if uploaded_file:
        st.success("✅ Resume Uploaded Successfully")
    st.header("📝 Enter Job Description")
    input_text = st.text_area("Paste the job description here", height=150)

col1, col2 = st.columns([1, 1])
with col1:
    submit1 = st.button("📋 Get Resume Details")
with col2:
    submit3 = st.button("📊 Get Resume Match %")

input_prompt1 = """
As an experienced Technical Human Resource Manager, analyze the resume against the job description. 
Provide a detailed evaluation on how well the candidate meets the required qualifications, skills, and experience. 
Highlight key strengths, gaps, and suggestions for improvement in relation to the job role.
"""

input_prompt3 = """
As a highly trained ATS (Applicant Tracking System) scanner, evaluate the resume's alignment with the job description. 
Calculate an accurate match percentage, list missing keywords that affect ranking, and give final thoughts on the suitability of the candidate.
"""

if (submit1 or submit3) and uploaded_file:
    pdf_content = input_pdf_setup(uploaded_file)
    if pdf_content:
        prompt = input_prompt1 if submit1 else input_prompt3
        response = get_gemini_response(input_text, pdf_content, prompt)
        st.subheader("💡 AI Analysis Result:")
        st.write(response)  # Ensures the result is always displayed properly
    else:
        st.error("Could not process the PDF.")
elif (submit1 or submit3) and not uploaded_file:
    st.warning("⚠️ Please upload your resume first.")
