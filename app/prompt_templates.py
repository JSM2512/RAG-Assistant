from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

def get_contextualize_prompt():
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question, "
        "rewrite the question so it can be understood without the history. "
        "Do NOT answer the question, just rewrite it if needed."
    )
    return ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

def get_qa_prompt():
    system_prompt = (
        "You are a helpful assistant. Use the following context to answer the question. "
        "If you don't know, say 'I don't know'."
        "At the end of your answer, list the sources you used from the context.\n\n{context}"
    )
    return ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

def get_simple_prompt():
    return ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Answer based on the chat history. If u have your own answer, separate what is from context, and what is you know generally"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])