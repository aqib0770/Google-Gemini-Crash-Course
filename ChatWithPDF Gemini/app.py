from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from io import BytesIO

api = os.getenv("GEMINI_API_KEY")
print(api)
genai.configure(api_key=api)


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text

def get_text_chunks(texts):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(texts)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    if not os.path.exists("faiss_index"):
        os.makedirs("faiss_index")
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context,
    make sure to provide all the details,
    if the answer is not in the provided content, just say,
    "answer is not available in the context", don't provide the wrong answers.
    Context:\n{context}?\n
    Question:\n{question}\n

    Answer:

    """
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3, api_key=api)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context","question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain


def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api)
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()

    response = chain(
        {"input_documents":docs, "question":user_question},
        return_only_outputs=True
    )
    print(response)
    st.write("Reply: ", response["output_text"])


def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF using Gemini")

    user_question = st.text_area("Ask a question from the PDF files")

    if user_question:
        user_input(user_question)
    
    with st.sidebar:
        st.title("Menu: ")
        pdf_docs = st.file_uploader("Upload your PDF files and click on submit", type="pdf", accept_multiple_files=True)
        if st.button("Submit and process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")


if __name__ == '__main__':
    main()
