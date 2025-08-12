import streamlit as st

def initialize_analytics():
    if "analytics" not in st.session_state:
        st.session_state.analytics = {
            "queries": 0,
            "latency": [],
            "accuracy_feedback": [],
        }

def update_analytics(start_time, end_time):
    st.session_state.analytics["queries"] += 1
    st.session_state.analytics["latency"].append(end_time - start_time)

def record_feedback(is_correct: bool):
    st.session_state.analytics["accuracy_feedback"].append(is_correct)

def render_dashboard():
    st.markdown("## Analytics Dashboard")
    total_queries = st.session_state.analytics["queries"]
    avg_latency = (sum(st.session_state.analytics["latency"]) / total_queries) if total_queries else 0
    accuracy = (sum(st.session_state.analytics["accuracy_feedback"]) / len(st.session_state.analytics["accuracy_feedback"])) if st.session_state.analytics["accuracy_feedback"] else 0
    st.metric("Total Queries", total_queries)
    st.metric("Average Latency (seconds)", f"{avg_latency:.2f}")
    st.metric("Accuracy (%)", f"{accuracy*100:.1f}")
    st.line_chart(st.session_state.analytics["latency"], x_label="Query No.", y_label="Latency (s)")