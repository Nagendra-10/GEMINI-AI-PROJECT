from dotenv import load_dotenv
from PyPDF2 import PdfReader
import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
import os
import textract
import tempfile

load_dotenv()
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
model = genai.GenerativeModel('gemini-pro')

def process_text(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    VectorDB = Chroma.from_texts(chunks, embeddings)
    return VectorDB

def extract_text_from_file(file, file_type):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.' + file_type) as temp_file:
        temp_file.write(file.getvalue())
        temp_file_path = temp_file.name

    if file_type == 'pdf':
        pdf_reader = PdfReader(temp_file_path)
        text = "".join(page.extract_text() for page in pdf_reader.pages)
    elif file_type == 'docx':
        text = textract.process(temp_file_path, extension='docx').decode('utf-8')

    elif file_type == 'txt':

        with open(temp_file_path, 'r', encoding='utf-8', errors='replace') as file:
            text = file.read()
    else:
        raise ValueError("Unsupported file type")
    return text

def get_gemini_response(question):
    response = model.generate_content(question, stream=True)
    full_response = ""
    for chunk in response:
        full_response += chunk.text
    return full_response

def main():
    st.header("Gemini Application for Multiple Documents ")
    st.sidebar.title("Chat with Documents")
    files = st.sidebar.file_uploader('Upload your Documents', type=['pdf', 'docx','txt'], accept_multiple_files=True)

    if files:
        combined_text = ""
        for file in files:
            file_type = file.name.split('.')[-1]
            text = extract_text_from_file(file, file_type)
            combined_text += text + "\n\n"

        VectorDB = process_text(combined_text)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What is up?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            response = get_gemini_response(prompt)
            message_placeholder.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
