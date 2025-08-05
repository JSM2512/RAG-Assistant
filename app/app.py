import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain.memory import ChatMessageHistory
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
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
session_id = st.text_input("Enter Session ID (e.g., user123):", value="default_session")

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

# ## ChatPrompt
# prompt = ChatPromptTemplate.from_messages([
#     ("system","You are a helpful assistant, use the context to answer the questions."),
#     ("user","Context : {context}\n\nQuestion : {question}")
# ])


openai_models = ["gpt-4o", "gpt-4-turbo", "gpt-4"]
groq_models = ["Deepseek-R1-Distill-Llama-70b", "Gemma2-9b-It", "Mistral-Saba-24b"]

if selected_model in openai_models:
    llm = ChatOpenAI(model_name=selected_model)
elif selected_model in groq_models:
    llm = ChatGroq(model=selected_model)
else:
    st.error(f"Unsupported model selected: {model_choice}")
    st.stop()

if "store" not in st.session_state:
    st.session_state.store = {}

def get_session_history(session: str):
    if session not in st.session_state.store:
        st.session_state.store[session] = ChatMessageHistory()
    return st.session_state.store[session]


if retriever:
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question, "
        "rewrite the question so it can be understood without the history. "
        "Do NOT answer the question, just rewrite it if needed."
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

    system_prompt = (
        "You are a helpful assistant. Use the following context to answer the question. "
        "If you don't know, say 'I don't know'.\n\n{context}"
    )

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )

    
user_input = st.text_input("What question do you have?")
if user_input:
    session_history = get_session_history(session_id)

    if retriever:
        # Document-based RAG
        response = conversational_rag_chain.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": session_id}}
        )
        st.markdown("### Answer (From Documents):")
        st.write(response["answer"])
    else:
        # Fallback: Chat History Only
        simple_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are a helpful assistant. Answer based on the chat history. If u have your own answer, seperate what is from context, and what is you know generally"),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        history_only_chain = simple_prompt | llm | StrOutputParser()

        response = history_only_chain.invoke(
            {"input": user_input, "chat_history": session_history.messages}
        )

        # Update session history manually
        session_history.add_user_message(user_input)
        session_history.add_ai_message(response)

        st.markdown("### Answer :")
        st.write(response)