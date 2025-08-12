from langchain.memory import ChatMessageHistory
import streamlit as st

def get_session_history(session: str):
    if "store" not in st.session_state:
        st.session_state.store = {}
    if session not in st.session_state.store:
        st.session_state.store[session] = ChatMessageHistory()
    return st.session_state.store[session]