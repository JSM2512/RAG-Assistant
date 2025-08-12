from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq

def get_llm(selected_model: str):
    openai_models = ["gpt-4o", "gpt-4-turbo", "gpt-4"]
    groq_models = ["Deepseek-R1-Distill-Llama-70b", "Gemma2-9b-It", "Mistral-Saba-24b"]

    if selected_model in openai_models:
        return ChatOpenAI(model_name=selected_model)
    elif selected_model in groq_models:
        return ChatGroq(model=selected_model)
    else:
        raise ValueError(f"Unsupported model selected: {selected_model}")