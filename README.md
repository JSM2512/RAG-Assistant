# 💬 LangChain RAG Chat App (OpenAI + Groq + Streamlit)

This is a simple Retrieval-Augmented Generation (RAG) chat application built using **LangChain**, **OpenAI**, **Groq**, and **Streamlit**.

It allows you to:
- 💬 Ask general questions to AI models (OpenAI or Groq)
- 📄 Upload PDF or TXT documents
- 🧠 Get context-aware answers using document-based retrieval

---

## 🧰 Features

- 🔷 Supports OpenAI models (GPT-4, GPT-4-turbo, GPT-4o)
- 🟡 Supports Groq models (LLaMA3, Mistral, Gemma)
- 📚 Upload multiple PDFs or TXT files
- 🔍 Automatic document chunking and vector storage using FAISS
- 🤖 Intelligent context-aware responses
- ✅ Works with or without uploaded documents

---

## 🚀 Getting Started

1. Clone the repository and navigate into it.

2. Create and activate a Python virtual environment with Python 3.12 or higher.

3. Install dependencies listed in `requirements.txt`.

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

| Model Name (Label)               | Provider | Model ID                |
|----------------------------------|----------|--------------------------|
| GPT-4o                          | OpenAI   | `gpt-4o`                 |
| GPT-4-turbo                     | OpenAI   | `gpt-4-turbo`            |
| GPT-4                           | OpenAI   | `gpt-4`                  |
| Deepseek-R1-Distill-Llama-70b   | Groq     | `llama3-70b-8192`        |
| Gemma2-9b-It                    | Groq     | `gemma-7b-it`            |
| Mistral-Saba-24b                | Groq     | `mixtral-8x7b-32768`     |

---

## 📜 License

This project is licensed under the MIT License.

---

## 🤝 Contributing

Contributions are welcome!  
Please open issues or submit pull requests for improvements.

---

## 🙋‍♂️ Questions?

Open an issue or discussion in this repository if you need help.
