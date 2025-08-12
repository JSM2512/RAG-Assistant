import streamlit as st
import time

from config import *
from llm import get_llm
from utils import load_documents, create_retriever
from prompt_templates import get_contextualize_prompt, get_qa_prompt, get_simple_prompt
from chat_history import get_session_history
from analytics import initialize_analytics, update_analytics, record_feedback, render_dashboard


from langchain_core.output_parsers import StrOutputParser
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables.history import RunnableWithMessageHistory



st.set_page_config(page_title="Langchain Demo", layout="wide")

# Tabs for Chat and Analytics
tab_chat, tab_analytics = st.tabs(["💬 Chat", "📊 Analytics"])

with tab_chat:
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
        documents = load_documents(uploaded_docs)
        if documents:
            retriever = create_retriever(documents)

    llm = get_llm(selected_model)
    initialize_analytics()
    session_history = get_session_history(session_id)

    if retriever:
        contextualize_q_prompt = get_contextualize_prompt()
        history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
        qa_prompt = get_qa_prompt()
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
        start_time = time.time()

        if retriever:

            retrieved_docs = history_aware_retriever.invoke(
                {"chat_history": session_history.messages, "input": user_input}
            )

            answer = question_answer_chain.invoke(
                {"context": retrieved_docs, "chat_history":session_history.messages, "input": user_input}
            )

            end_time = time.time()
            update_analytics(start_time, end_time)

            st.markdown("### Answer (From Documents):")
            st.write(answer)

            ## Sources
            st.markdown("#### Sources Used: ")
            for i, doc in enumerate(retrieved_docs):
                with st.expander(f"Source {i+1}"):
                    meta_lines = []
                    if "source" in doc.metadata:
                        meta_lines.append(f"**File:** {doc.metadata['source']}")
                    if "page" in doc.metadata:
                        meta_lines.append(f"**Page:** {doc.metadata['page']}")
                    if meta_lines:
                        st.markdown("  \n".join(meta_lines))
                    st.write(doc.page_content)

        else:
            # Fallback: Chat History Only
            simple_prompt = get_simple_prompt()
            history_only_chain = simple_prompt | llm | StrOutputParser()

            response = history_only_chain.invoke(
                {"input": user_input, "chat_history": session_history.messages}
            )

            # Update session history manually
            session_history.add_user_message(user_input)
            session_history.add_ai_message(response)
            end_time = time.time()
            update_analytics(start_time, end_time)
            st.markdown("### Answer :")
            st.write(response)
        
        # Feedback for accuracy
        feedback = st.radio(
            "Was this answer correct?",
            ["Yes", "No"],
            key=f"feedback_{st.session_state.analytics['queries']}"
        )
        record_feedback(feedback == "Yes")

with tab_analytics:
    render_dashboard()