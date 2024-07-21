import streamlit as st
from dotenv import load_dotenv
import pickle
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
import os
from langchain.docstore.document import Document

# Sidebar contents
with st.sidebar:
    st.title('🤗💬 LLM Chat App')
    st.markdown('''
    ## About
    This app is an LLM-powered chatbot built using:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - [OpenAI](https://platform.openai.com/docs/models) LLM model
    ''')
    add_vertical_space(5)
    st.write('Made with ❤️ by Samson Kinyanjui')

load_dotenv()

def extract_text_from_pdf(pdf):
    pdf_reader = PdfReader(pdf)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def create_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
    return text_splitter.split_text(text=text)

def save_faiss_index(vector_store, store_name):
    vector_store.save_local(f"{store_name}_faiss_index")

def load_faiss_index(store_name):
    return FAISS.load_local(f"{store_name}_faiss_index", OpenAIEmbeddings(), allow_dangerous_deserialization=True)

def main():
    st.header("Chat with PDF 💬")

    # Upload PDF
    pdf = st.file_uploader("Upload your PDF", type='pdf')

    if pdf is not None:
        text = extract_text_from_pdf(pdf)
        chunks = create_chunks(text)
        store_name = pdf.name[:-4]

        if os.path.exists(f"{store_name}_faiss_index") and os.path.exists(f"{store_name}_texts.pkl"):
            vector_store = load_faiss_index(store_name)
            with open(f"{store_name}_texts.pkl", "rb") as f:
                docstore = pickle.load(f)
        else:
            embeddings = OpenAIEmbeddings()
            docstore = [Document(page_content=c) for c in chunks]
            vector_store = FAISS.from_documents(docstore, embeddings)
            save_faiss_index(vector_store, store_name)
            with open(f"{store_name}_texts.pkl", "wb") as f:
                pickle.dump(docstore, f)

        # Chat History
        if 'history' not in st.session_state:
            st.session_state.history = []

        # Display chat history
        for entry in st.session_state.history:
            st.write(f"**User**: {entry['user']}")
            st.write(f"**RAG ML Assistant**: {entry['ai']}")

        # User input
        user_input = st.text_input("Ask your question:")

        if st.button("Submit"):
            if user_input:
                # Add user's query to history
                st.session_state.history.append({'user': user_input, 'ai': ''})

                # Query processing
                docs = vector_store.similarity_search(query=user_input, k=3)
                llm = OpenAI()
                chain = load_qa_chain(llm=llm, chain_type="stuff")
                with get_openai_callback() as cb:
                    ai_response = chain.run(input_documents=docs, question=user_input)
                    print(cb)

                # Add AI response to history
                st.session_state.history[-1]['ai'] = ai_response

                # Refresh the chat
                st.experimental_rerun()

if __name__ == '__main__':
    main()
