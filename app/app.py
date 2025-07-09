import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
import tempfile
import streamlit as st
from dotenv import load_dotenv
load_dotenv()

os.environ['OPENAI_API_KEY']=os.getenv("OPENAI_API_KEY")
os.environ['GROQ_API_KEY']=os.getenv("GROQ_API_KEY")

## Langsmith Tracking
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_PROJECT"]=os.getenv("LANGCHAIN_PROJECT")


## Document Uploading and Saving in VectorStoreDB
st.title("Langchain Demo With Chat Model")

# Create a layout with 2 columns
col1, col2 = st.columns([2, 1])  # Wider for input, narrow for model select
with col1:
    uploaded_docs = st.file_uploader("Upload Documents", type=["pdf","txt"], accept_multiple_files=True)

# Right-hand smaller model selector
with col2:
    model_choice = st.selectbox(
        label="",
        options=[
            "🔷 GPT-4o",
            "🔷 GPT-4-turbo",
            "🔷 GPT-4",
            "🟡 Deepseek-R1-Distill-Llama-70b",
            "🟡 Gemma2-9b-It",
            "🟡 Mistral-Saba-24b"
        ],
        index=0
    )

# Map selection to backend model name
model_map = {
    "🔷 GPT-4o": "gpt-4o",
    "🔷 GPT-4-turbo": "gpt-4-turbo",
    "🔷 GPT-4": "gpt-4",
    "🟡 Deepseek-R1-Distill-Llama-70b": "Deepseek-R1-Distill-Llama-70b",
    "🟡 Gemma2-9b-It": "Gemma2-9b-It",
    "🟡 Mistral-Saba-24b": "Mistral-Saba-24b"
}
selected_model = model_map[model_choice]


retriever = None
if uploaded_docs:
    documents=[]
    for file in uploaded_docs:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name
    
        if file.name.endswith(".pdf"):
            loader = PyPDFLoader(tmp_path)
        else:
            loader = TextLoader(tmp_path)

        docs = loader.load()
        documents.extend(docs)
    if documents:
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        chunks = splitter.split_documents(documents)
        
        if chunks:
            embeddings = OpenAIEmbeddings()
            vectorstore = FAISS.from_documents(chunks, embeddings)
            retriever = vectorstore.as_retriever()

## ChatPrompt
prompt = ChatPromptTemplate.from_messages([
    ("system","You are a helpful assistant, use the context to answer the questions."),
    ("user","Context : {context}\n\nQuestion : {question}")
])

## StreamLit - GUI

input_text=st.text_input("What question you have in mind?")

# Define lists of models for each provider
openai_models = ["gpt-4o", "gpt-4-turbo", "gpt-4"]
groq_models = ["Deepseek-R1-Distill-Llama-70b", "Gemma2-9b-It", "Mistral-Saba-24b"]

# Choose LLM based on exact match
if selected_model in openai_models:
    llm = ChatOpenAI(model_name=selected_model)
elif selected_model in groq_models:
    llm = ChatGroq(model=selected_model)
else:
    st.error(f"Unsupported model selected: {model_choice}")
    st.stop()

outputParser = StrOutputParser()
chain = prompt|llm|outputParser

if input_text:
    context = ""
    if retriever:
        retrieved_docs = retriever.invoke(input_text)
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])

    response = chain.invoke({"context": context, "question":input_text})
    st.markdown("### Answer : ")
    st.write(response)

    