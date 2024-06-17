import pandas as pd
import streamlit as sl
import pickle
import time
import langchain
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
import os
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings,ChatGoogleGenerativeAI

from dotenv import load_dotenv
load_dotenv()
def test():
    sl.title("News research tool")
    sl.sidebar.title("News article urls")
    urls = []
    for i in range(1):
        url = sl.sidebar.text_input(f"url {i + 1}")
        urls.append(url)

    process_url_clicked = sl.sidebar.button("Proess Urls")
    file_path = "vector_index.pkl"

    main_placeholder = sl.empty()
    llm = ChatGoogleGenerativeAI(model = 'gemini-pro')
    if process_url_clicked:
        loader = UnstructuredURLLoader(urls = urls)
        main_placeholder.text("Data loading...started...✅✅✅")
        data = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            separators = ['\n','\n\n','.',','],
            chunk_size = 1000,
            chunk_overlap = 200
        )
        main_placeholder.text("Text splitter...started...✅✅✅")
        docs = text_splitter.split_documents(data)
        embeddings = GoogleGenerativeAIEmbeddings(model = 'models/embedding-001')
        vector_index = FAISS.from_documents(docs,embeddings)
        main_placeholder.text(vector_index.index.ntotal)
        main_placeholder.text("Embedding...started...✅✅✅")
        with open(file_path,"wb") as f:
            pickle.dump(vector_index,f)
        query = main_placeholder.text("Question... ")
        if query:
            if os.path.exists(file_path):
                with open(file_path,"rb") as f:
                    vector_stores = pickle.load(f)
                    chain = RetrievalQAWithSourcesChain.from_llm(llm = llm,retrieve =vector_stores.as_retriever())
                    result = chain({"question":query},return_only_outputs = True)
                    sl.header("Result")
                    sl.subheader(result["answer"])

def main():
    sl.title("RockyBot: News Research Tool 📈")
    sl.sidebar.title("News Article URLs")

    urls = []
    for i in range(3):
        url = sl.sidebar.text_input(f"URL {i+1}")
        urls.append(url)

    process_url_clicked = sl.sidebar.button("Process URLs")
    file_path = "faiss_store_openai.pkl"

    main_placeholder = sl.empty()
    llm = ChatGoogleGenerativeAI(model = 'gemini-pro')

    if process_url_clicked:
        # load data
        loader = UnstructuredURLLoader(urls=urls)
        main_placeholder.text("Data Loading...Started...✅✅✅")
        data = loader.load()
        # split data
        text_splitter = RecursiveCharacterTextSplitter(
            separators=['\n\n', '\n', '.', ','],
            chunk_size=1000
        )
        main_placeholder.text("Text Splitter...Started...✅✅✅")
        docs = text_splitter.split_documents(data)
        # create embeddings and save it to FAISS index
        embeddings = GoogleGenerativeAIEmbeddings(model = 'models/embedding-001')
        vectorstore_openai = FAISS.from_documents(docs, embeddings)
        main_placeholder.text("Embedding Vector Started Building...✅✅✅")
        time.sleep(2)

        # Save the FAISS index to a pickle file
        with open(file_path, "wb") as f:
            pickle.dump(vectorstore_openai, f)

    query = main_placeholder.text_input("Question: ")
    if query:
        if os.path.exists(file_path):
            with open(file_path, "rb") as f:
                vectorstore = pickle.load(f)
                chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
                result = chain({"question": query}, return_only_outputs=True)
                # result will be a dictionary of this format --> {"answer": "", "sources": [] }
                sl.header("Answer")
                sl.write(result["answer"])

                # Display sources, if available
                sources = result.get("sources", "")
                if sources:
                    sl.subheader("Sources:")
                    sources_list = sources.split("\n")  # Split the sources by newline
                    for source in sources_list:
                        sl.write(source)

if __name__ == '__main__':
    test()