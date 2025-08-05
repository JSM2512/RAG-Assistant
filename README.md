# 💬 LangChain RAG Chat App with Conversation Memory

This is a production-ready Retrieval-Augmented Generation (RAG) chat app built with **LangChain**, **OpenAI**, **Groq**, **Streamlit**, and **LangSmith** for monitoring. Features intelligent conversation memory for contextual multi-turn discussions.

## ✨ Key Features  
- 💬 **Multi-turn Conversations** - Remembers entire conversation context within current chat session
- 🧠 **Intelligent Memory** - Maintains chat history for contextual responses during active session
- 📄 **Document Upload** - Upload PDFs or TXT files for custom knowledge base
- 🎯 **Context-Aware Answers** - Combines document retrieval with conversation history
- 📊 **LangSmith Monitoring** - Integrated tracing for debugging and performance insights
- 🔄 **Conversation Flow** - Natural multi-turn discussions with memory of previous messages

---

## 🧰 Technical Features

### 🤖 **Multi-Model Support**
- 🔷 OpenAI models: GPT-4, GPT-4-turbo, GPT-4o  
- 🟡 Groq models: Deepseek LLaMA3, Gemma, Mistral

### 💾 **Advanced Memory Management**
- **Conversation Context** - Maintains full chat history during active session
- **Context Retention** - References previous questions and answers for coherent discussions
- **Memory Optimization** - Efficient in-memory storage for current conversation
- **Fresh Start** - Each new chat session starts with clean memory

### 🔍 **Document Processing**
- 📚 Multi-file PDF and TXT document support
- 🔗 Automatic document chunking and embedding
- 🗃️ FAISS vector store for semantic search
- ⚡ Real-time document retrieval integration

### 🛠️ **Production Features**
- ✅ Fully functional with or without uploaded documents
- 📊 Built-in LangSmith integration for monitoring and debugging
- ⚡ Streamlit-powered responsive UI
- 🧹 Clean slate for each new conversation

---

## 🚀 Getting Started

1. **Clone the repository**
   ```bash
   git clone https://github.com/JSM2512/RAG-Assistant.git
   cd RAG-Assistant
   ```

2. **Set up Python environment**
   ```bash
   conda create -p venvlangchain python=3.12
   conda activate venvlangchain
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**
   Create a `.env` file in the root directory:
   ```env
   OPENAI_API_KEY=your_openai_api_key
   GROQ_API_KEY=your_groq_api_key
   LANGCHAIN_API_KEY=your_langsmith_api_key
   LANGCHAIN_PROJECT=LangchainRAG
   LANGCHAIN_TRACING_V2=true
   ```

   > 💡 **Note:** LangSmith tracing is optional. Set `LANGCHAIN_TRACING_V2=false` to disable.

5. **Launch the application**
   ```bash
   streamlit run app.py
   ```

6. **Monitor performance**
   Track conversations on the **[🚀 LangSmith Dashboard](https://smith.langchain.com/o/b30de270-0832-4d48-baa4-c4ce02a836dc/dashboards/10024ed1-1fc3-4b53-b9fe-4f4e6a8cf2a2)**

---

## 🏗️ Architecture Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────────┐
│   Streamlit UI  │────│ Conversation     │────│   Memory Buffer     │
│                 │    │ Memory Manager   │    │   (Current Session) │
└─────────────────┘    └──────────────────┘    └─────────────────────┘
         │                       │                         │
         ▼                       ▼                         ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────────┐
│ Document Upload │────│   RAG Pipeline   │────│   Vector Database   │
└─────────────────┘    └──────────────────┘    └─────────────────────┘
                                │
                                ▼
                    ┌──────────────────────┐
                    │   LLM Integration    │
                    │  (OpenAI/Groq)       │
                    └──────────────────────┘
```

---

## 📦 System Requirements

- **Python:** 3.12+
- **APIs:** OpenAI and/or Groq API access
- **Dependencies:** Listed in `requirements.txt`
- **Memory:** Recommended 4GB+ RAM for document processing

---

## 📁 Project Structure

```
RAG-Assistant/
├── app.py                    # Main Streamlit application
├── Project.ipynb            # Core RAG implementation notebook
├── requirements.txt         # Python dependencies
├── .env                     # API keys and configuration
└── README.md               # Project documentation
```

---

## 🤖 Supported Models

| **Provider** | **Model Name** | **Model ID** |
|-------------|----------------|--------------|
| OpenAI | GPT-4o | `gpt-4o` |
| OpenAI | GPT-4-turbo | `gpt-4-turbo` |
| OpenAI | GPT-4 | `gpt-4` |
| Groq | Deepseek-R1-Distill-Llama-70b | `Deepseek-R1-Distill-Llama-70b` |
| Groq | Gemma2-9b-It | `Gemma2-9b-It` |
| Groq | Mistral-Saba-24b | `Mistral-Saba-24b` |

---

## 🎯 Use Cases

- **Research Assistant:** Upload academic papers and have flowing conversations about content
- **Document Analysis:** Interactive Q&A with memory of previous questions and context  
- **Learning Tool:** Ask follow-up questions that build on previous discussion
- **Knowledge Exploration:** Deep-dive conversations that remember earlier context

---

## 🔧 Memory Management

### Conversation Memory
- **In-Session Context:** Maintains complete conversation history during active chat
- **Follow-up Awareness:** Understands references to previous questions and answers
- **Context Building:** Each response builds on the cumulative conversation
- **Fresh Start:** New conversations begin with clean memory slate

### Performance Features
- Vector embeddings cached for faster retrieval
- Conversation memory optimized for natural dialogue flow
- Memory cleared automatically on new chat sessions

---
