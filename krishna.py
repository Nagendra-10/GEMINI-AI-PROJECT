# import streamlit as st
# from langchain.document_loaders.csv_loader import CSVLoader
# import tempfile
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.vectorstores import Chroma
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from langchain.chains.question_answering import load_qa_chain
# from langchain.prompts import PromptTemplate
# from langchain_google_genai import ChatGoogleGenerativeAI
# import google.generativeai as genai
# import os
# from dotenv import load_dotenv
# import pwd
# load_dotenv()
# #genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
# genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
# def get_model_response(file,query):
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
#     context='\n\n'.join(str(p.page_content)for p in file)
#     data=text_splitter.split_text(context)
#
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#     searcher = Chroma.from_texts(data, embeddings).as_retriever()
#
#     q="which employee has maximum salary?"
#     records=searcher.get_relevant_documents(q)
#
#     prompt_template = """
#         Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
#         provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
#         Context:\n {context}?\n
#         Question: \n{question}\n
#
#         Answer:
#         """
#
#     model = ChatGoogleGenerativeAI(model="gemini-pro",
#                                    temperature=0.9)
#     prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
#     chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
#
#     response= chain(
#         {
#             "input_document":records,
#             "question": query
#         }
#          , return_only_outputs=True)
#
#     return response['output_text']
#
#
# def main():
#     st.title("chat with csv using gemini pro")
#
#     upload_file = st.file_uploader("Upload your CSV file", type=['csv'])
#
#     if upload_file is not None:
#         with tempfile.NamedTemporaryFile(delete=False) as temp_file:
#             temp_file.write(upload_file.getvalue())
#             temp_file_path = temp_file.name
#
#             csv_loader=CSVLoader(file_path=temp_file_path, csv_args={'delimiter': ','})
#
#             data=csv_loader.load()
#
#             user_input= st.text_input("your message")
#             print(user_input)
#
#             if user_input:
#                 get_model_response(data,user_input)
#                 response="response"
#                 st.write(response)




# import streamlit as st
# from langchain.document_loaders.csv_loader import CSVLoader
# import tempfile
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.vectorstores import Chroma
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from langchain.chains.question_answering import load_qa_chain
# from langchain.prompts import PromptTemplate
# from langchain_google_genai import ChatGoogleGenerativeAI
# import google.generativeai as genai
# import os
# from dotenv import load_dotenv
#
# load_dotenv()
# genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
#
# def get_model_response(data, query):
#     responses = []
#     for row in data:
#         context = '\n\n'.join(str(cell) for cell in row)
#
#         text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
#         data_split = text_splitter.split_text(context)
#
#         embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#         searcher = Chroma.from_texts(data_split, embeddings).as_retriever()
#
#         relevant_documents = searcher.get_relevant_documents(query)
#
#         prompt_template = """
#             Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
#             provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
#             Context:\n {context}?\n
#             Question: \n{question}\n
#
#             Answer:
#             """
#
#         model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.9)
#         prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
#         chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
#
#         response = chain({"input_documents": relevant_documents, "question": query}, return_only_outputs=True)
#
#         responses.append(response['output_text'])
#     return responses
#
#
# def main():
#     st.title("Chat with CSV using Gemini Pro")
#
#     upload_file = st.file_uploader("Upload your CSV file", type=['csv'])
#
#     if upload_file is not None:
#         # Save the uploaded file to a temporary location
#         temp_file_path = tempfile.NamedTemporaryFile(delete=False).name
#         with open(temp_file_path, 'wb') as temp_file:
#             temp_file.write(upload_file.getvalue())
#
#         # Initialize the CSVLoader with the file path
#         csv_loader = CSVLoader(file_path=temp_file_path, csv_args={'delimiter': ','})
#         data = csv_loader.load()
#
#         user_input = st.text_input("Your message")
#
#         if user_input:
#             responses = get_model_response(data, user_input)
#             st.write("Responses:")
#             for response in responses:
#                 st.write(response)
#
# if __name__ == "__main__":
#     main()



import streamlit as st
from langchain_community.document_loaders.csv_loader import CSVLoader
import tempfile
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv

try:
    import pwd
except ImportError:
    pwd = None

load_dotenv()


def get_model_response(data, query):
    # Convert each element in the data to a string
    data_strings = []
    for row in data:
        row_strings = [str(cell) for cell in row]
        data_strings.append('\n'.join(row_strings))

    # Combine all row strings into a single context string
    context = '\n\n'.join(data_strings)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    data_split = text_splitter.split_text(context)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    searcher = Chroma.from_texts(data_split, embeddings).as_retriever()

    relevant_documents = searcher.get_relevant_documents(query)

    prompt_template = """
        Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
        provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
        Context:\n {context}?\n
        Question: \n{question}\n

        Answer:
        """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.9)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    response = chain({"input_documents": relevant_documents, "question": query}, return_only_outputs=True)

    return response['output_text']


def main():
    st.title("Chat with CSV using Gemini Pro")

    upload_file = st.file_uploader("Upload your CSV file", type=['csv'])

    if upload_file is not None:
        # Save the uploaded file to a temporary location
        temp_file_path = tempfile.NamedTemporaryFile(delete=False).name
        with open(temp_file_path, 'wb') as temp_file:
            temp_file.write(upload_file.getvalue())

        # Initialize the CSVLoader with the file path
        csv_loader = CSVLoader(file_path=temp_file_path, csv_args={'delimiter': ','})
        data = csv_loader.load()

        user_input = st.text_input("Your message")

        if user_input:
            response = get_model_response(data, user_input)
            st.write("Response:")
            st.write(response)

if __name__ == "__main__":
    main()

