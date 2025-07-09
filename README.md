# 💬 LangChain RAG Chat App  

This is an easy-to-use Retrieval-Augmented Generation (RAG) chat app built with **LangChain**, **OpenAI**, **Groq**, **Streamlit**, and **LangSmith** for monitoring.

Features include:  
- 💬 Conversing with powerful AI models.
- 📄 Uploading PDFs or TXT files to provide custom knowledge  
- 🧠 Receiving precise, context-aware answers using document retrieval  
- 📊 Integrated LangSmith tracing for debugging and performance insights  

---

## 🧰 Features

- 🔷 Supports popular OpenAI models: GPT-4, GPT-4-turbo, GPT-4o  
- 🟡 Integrates Groq models like Deepseek LLaMA3, Gemma, and Mistral  
- 📚 Upload multiple PDF and TXT documents for custom context  
- 🔍 Automatic document splitting and embedding with FAISS vector store  
- 🧠 Retrieval-Augmented Generation (RAG) for context-aware, accurate answers  
- ✅ Fully functional whether or not documents are uploaded  
- 📊 Built-in LangSmith integration for monitoring, tracing, and debugging model calls  
- ⚡ Streamlit-powered interactive UI for fast prototyping and usage  


---

## 🚀 Getting Started

1. Clone the repository and navigate into it.
   
   ```
   git clone git clone https://github.com/your-username/langchain-rag-chat.git
   ```

2. Create and activate a Python virtual environment with Python 3.12 or higher.
   
   ```
   conda create -p venvlangchain python=3.12
   conda activate venvlangchain
   ```

3. Install dependencies listed in `requirements.txt`.
   
    ```
    pip install -r requirements.txt
    ```

4. Create a `.env` file in the root directory and add your API keys:

    ```
    OPENAI_API_KEY=your_openai_api_key
    GROQ_API_KEY=your_groq_api_key
    LANGCHAIN_API_KEY=your_langsmith_api_key
    LANGCHAIN_PROJECT=LangchainRAG
    LANGCHAIN_TRACING_V2=true
    ```

> 💡 If you don’t use LangSmith, you can skip the last three lines or set `LANGCHAIN_TRACING_V2=false`.

5. Run the app using Streamlit:

    ```
    streamlit run app.py
    ```

---

## 📦 Requirements

- Python 3.12+
- Access to OpenAI and/or Groq API
- Python packages as per `requirements.txt`

---

## 📁 Project Structure
```
  app/
    ├── app.py                # Main Streamlit application script
    ├──Project.ipynb          # Jupyter notebook for core steps
  requirements.txt      # List of Python dependencies
  .env                  # Environment variables (API keys, **not** committed)
  README.md             # Project documentation and instructions
```
---

## 🤖 Supported Models

| Model Name                    | Model ID              |
|------------------------------|-----------------------|
| GPT-4o                       | `gpt-4o`              |
| GPT-4-turbo                  | `gpt-4-turbo`         |
| GPT-4                        | `gpt-4`               |
| Deepseek-R1-Distill-Llama-70b| `llama3-70b-8192`     |
| Gemma2-9b-It                 | `gemma-7b-it`         |
| Mistral-Saba-24b             | `mixtral-8x7b-32768`  |


---

## 🤝 Contributing

Contributions are welcome!  
Please open issues or submit pull requests for improvements.

---
