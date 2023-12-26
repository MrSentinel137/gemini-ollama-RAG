from langchain.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
from langchain.document_loaders import TextLoader, DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import Ollama
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import streamlit as st
import os

if "history" not in st.session_state:
    st.session_state.history = []

load_dotenv()

model_type= 'ollama'

# Initializing Gemini
if(model_type == "ollama"):
    model = Ollama(
                    model="dolphin-mistral",
                    callback_manager=CallbackManager([StreamingStdOutCallbackHandler])
                )
    
elif(model_type == "gemini"):
    model = ChatGoogleGenerativeAI(
                                model="gemini-pro", 
                                temperature=0.1, 
                                convert_system_message_to_human=True
                            )
    
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# Vector Database
persist_directory =  "./db/Gemini/"
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

if not os.path.exists(persist_directory):
    with st.spinner('🚀 Starting your bot.  This might take some time...'):
        # Data Pre-processing
        pdf_loader = DirectoryLoader("./docs/", glob="./*.pdf", loader_cls=PyPDFLoader)
        text_loader = DirectoryLoader("./docs/", glob="./*.txt", loader_cls=TextLoader)
        
        pdf_documents = pdf_loader.load()
        text_documents = text_loader.load()
        
        splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=0)
        
        pdf_context = "\n\n".join(str(p.page_content) for p in pdf_documents)
        text_context = "\n\n".join(str(p.page_content) for p in text_documents)

        pdfs = splitter.split_text(pdf_context)
        texts = splitter.split_text(text_context)

        data = pdfs + texts

        print("Data Processing Complete")

        vectordb = Chroma.from_texts(data, embeddings, persist_directory=persist_directory)
        vectordb.persist()

        print("Vector DB Creating Complete\n")

        with st.chat_message("Assistant"):
            st.markdown("🫡 Reporting for duty!!")

elif os.path.exists(persist_directory):
    vectordb = Chroma(persist_directory=persist_directory, 
                  embedding_function=embeddings)
    
    print("Vector DB Loaded\n")

# Quering Model
query_chain = RetrievalQA.from_chain_type(
    llm=model,
    retriever=vectordb.as_retriever()
)

for msg in st.session_state.history:
    with st.chat_message(msg['role']):
        st.markdown(msg['content'])


prompt = st.chat_input("Say something")
if prompt:
    st.session_state.history.append({
        'role':'user',
        'content':prompt
    })

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner('💡Thinking'):
        response = query_chain({"query": prompt})

        st.session_state.history.append({
            'role' : 'Assistant',
            'content' : response['result']
        })

        with st.chat_message("Assistant"):
            st.markdown(response['result'])