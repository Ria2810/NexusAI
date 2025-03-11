import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
# Configure the Google Generative AI API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=50000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details. If the answer is not in
    the provided context just say, "answer is not available in the context", don't provide a wrong answer.\n\n
    Context:\n {context}?\n
    Question:\n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )
    print(response)
    st.write("Reply: ", response["output_text"])

def main():
    st.set_page_config("Multi PDF Chatbot", page_icon=":scroll:")
    st.header("Multi-PDF's 📚 - Chat Agent 🤖")
    
    # Main input for user questions
    user_question = st.text_input("Ask a Question from the PDF Files uploaded .. ✍️📝")
    if user_question:
        user_input(user_question)

    # Sidebar: Overview, features, and PDF file uploader
    with st.sidebar:
        st.markdown("# Multi-PDF Chat Agent")
        st.title("📁 PDF File's Section")
        pdf_docs = st.file_uploader("Upload your PDF Files & \nClick on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")
                
        st.write("---")
        st.markdown("## Overview")
        st.write(
            "This AI-powered chatbot allows you to upload multiple PDF files and ask detailed questions about their content. "
            "The app extracts text from the PDFs, splits it into manageable chunks, and uses advanced vector embeddings to perform a semantic search. "
            "It then provides detailed answers based on the context found in your documents."
        )
        st.markdown("## Features")
        st.write("- **Multi-PDF Upload:** Process multiple PDF files at once.")
        st.write("- **Semantic Search:** Utilizes Google Generative AI embeddings with FAISS for efficient search.")
        st.write("- **Detailed Q&A:** Provides comprehensive answers based on the document content.")
        st.write("- **Interactive Chat:** A user-friendly interface to ask questions and get real-time responses.")
        st.write("---")

        st.write("AI App created by @ Ria Choudhari")

    # Footer
    st.markdown(
        """
        <div style="position: fixed; bottom: 0; left: 0; width: 100%; background-color: #0E1117; padding: 15px; text-align: center;">
            © <a href="https://github.com/Ria2810" target="_blank">Ria Choudhari</a> | Made with ❤️
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
