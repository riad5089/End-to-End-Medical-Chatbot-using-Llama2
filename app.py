import streamlit as st
from streamlit_chat import message
from langchain.chains import ConversationalRetrievalChain
# from langchain.document_loaders import PyPDFLoader, DirectoryLoader
# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain_community.llms import CTransformers
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
# from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain.memory import ConversationBufferMemory

DB_FAISS_PATH = 'vectorstore/db_faiss'

#load the pdf
loader= DirectoryLoader("data/",glob='*.pdf',loader_cls=PyPDFLoader)
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)
text_chunks = text_splitter.split_documents(documents)

#create the embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",model_kwargs={'device': 'cpu'})

# vector store
# db = FAISS.from_documents(text_chunks,embeddings)
vector_store = Chroma.from_documents(text_chunks,embeddings)

# create the language model
llm = CTransformers(model="target\model\llama-2-7b-chat.ggmlv3.q4_0.bin",model_type="llama",
                    config={'max_new_tokens':512,'temperature':0.6})

memory = ConversationBufferMemory(memory_key="chat_history",return_messages=True)

chain = ConversationalRetrievalChain.from_llm(llm=llm,chain_type="stuff",retriever=vector_store(search_kwargs={'k': 2}),memory=memory)

st.title("HealthCare ChatBot 🧑🏽‍⚕️")
def conversation_chat(query):
    result=chain({"question":query,"chat_history":st.session['history']})

                 

