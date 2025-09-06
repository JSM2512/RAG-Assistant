from langchain.memory import ChatMessageHistory
import streamlit as st

def get_session_history(session: str, tab: str = "chat"):
    key = f"{tab}_{session}"
    if "store" not in st.session_state:
        st.session_state.store = {}
    if key not in st.session_state.store:
        st.session_state.store[key] = ChatMessageHistory()
    return st.session_state.store[key]