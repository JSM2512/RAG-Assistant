# 📚 Internal Docs Q&A Dashboard — Powered by LangChain RAG

**RAG-Assistant** is a modern web dashboard for secure, AI-powered search and Q&A over your organization’s internal documents. Easily upload PDFs or TXT files, ask natural language questions, and receive context-rich answers with transparent source attribution. Includes analytics for usage, accuracy, and performance monitoring.

---

## ✨ Key Features

- 📄 **Document Upload:** Build a custom knowledge base from internal PDFs and text files.
- 💬 **Natural Language Q&A:** Ask questions and get answers sourced directly from your docs.
- ✈️ **Aviation Data RAG:** Integrate and query 80+ aviation data reports as a specialized vector database for aviation-related Q&A, enabling deep domain insights and analysis.
- 🔄 **Multi-turn Conversations:** Chat with context-aware memory for richer discussions.
- 🔎 **Source Attribution:** Every answer lists the document sources and page numbers.
- 📊 **Analytics Dashboard:** Track query count, response latency, and answer accuracy.
- 🤖 **Multi-Model Support:** Choose from leading OpenAI and Groq models.
- 🛡️ **Session Management:** Persistent chat history by user/session.
- 🛠️ **LangSmith Monitoring:** Integrated tracing for debugging and performance insights.

---

## 🏢 Typical Use Cases

- **Internal Knowledge Base:** Empower staff to search policies, manuals, HR docs, and more.
- **Support Portal:** Fast answers for IT, HR, or helpdesk teams based on company docs.
- **Research & Analysis:** Upload reports, ask complex questions, and get cited answers.
- **Learning Platform:** Deep-dive discussions with memory of previous context.
- **Aviation Data Search:** Analyze and explore aviation datasets for research, compliance, or operational insights.

---

## 🚀 Quick Start

1. **Clone the Repository**
   ```bash
   git clone https://github.com/JSM2512/RAG-Assistant.git
   cd RAG-Assistant
   ```

2. **Set Up Python Environment**
   ```bash
   conda create -p venvlangchain python=3.12
   conda activate venvlangchain
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Add API Keys**
   Create a `.env` file in the repo root:
   ```env
   OPENAI_API_KEY=your_openai_api_key
   GROQ_API_KEY=your_groq_api_key
   LANGCHAIN_API_KEY=your_langsmith_api_key
   LANGCHAIN_PROJECT=LangchainRAG
   LANGCHAIN_TRACING_V2=true
   ```

5. **Launch the Dashboard**
   ```bash
   streamlit run app.py
   ```

---

## 🖥️ Dashboard Overview

- **Chat Tab:** Upload documents, select your preferred model, and ask questions — see answers with sources.
- **Analytics Tab:** Visualize total queries, latency, and answer accuracy over time.
- **Session Selector:** Each user or session gets its own persistent chat history.
- **Aviation Data RAG Tab:** Instantly query 80+ aviation datasets using RAG, suitable for aviation ops, research, and compliance.

---

## 🏗️ Architecture

```
┌───────────────┐   ┌─────────────┐   ┌───────────────┐
│ Streamlit UI  │──▶│ RAG Engine  │──▶│ Vector Store  │
└───────────────┘   └─────────────┘   └───────────────┘
        │                │                 │
        ▼                ▼                 ▼
 ┌─────────────┐   ┌─────────────┐   ┌──────────────┐
 │ User Upload │   │ LLM Models  │   │ Analytics DB │
 └─────────────┘   └─────────────┘   └──────────────┘
        │
        ▼
┌───────────────────────┐
│ Aviation Data VectorDB│
└───────────────────────┘
```

---

## 🤖 Supported Models

| Provider | Model Name                        | Model ID                   |
|----------|-----------------------------------|----------------------------|
| OpenAI   | GPT-4o, GPT-4-turbo, GPT-4        | `gpt-4o`, `gpt-4-turbo`, `gpt-4` |
| Groq     | Deepseek LLaMA3, Gemma, Mistral   | `Deepseek-R1-Distill-Llama-70b`, `Gemma2-9b-It`, `Mistral-Saba-24b` |

---

## ✈️ Aviation Data RAG

- **Integrated 80+ aviation datasets** as a vector database for domain-specific retrieval.
- **Use Cases:** Safety analysis, fleet management, regulatory compliance, route optimization, research.
- **Query Examples:** "Show all incidents for Boeing 737 in 2023", "List ICAO codes for European airports", "Analyze global flight delays trends."
- **Source Attribution:** Answers cite the aviation dataset(s) used.

---

## 📊 Analytics Dashboard

- **Query Tracking:** Count of questions asked per session.
- **Latency Monitoring:** Average and per-query response time (sec).
- **Accuracy Feedback:** Track user feedback on answer correctness.
- **Interactive Charts:** Visualize latency and accuracy trends.

---

## 📁 Project Structure

```
RAG-Assistant/
├── app.py             # Main Streamlit dashboard
├── Project.ipynb      # Core RAG prototype notebook
├── requirements.txt   # Python dependencies
├── .env               # API keys/config
├── aviation_data/     # Aviation datasets (vector db source)
└── README.md          # This documentation
```

---

## 🔒 Security & Privacy

- **Local Processing:** All uploaded docs are processed locally and not sent to third-party servers (except for LLM query).
- **Session Isolation:** Each chat session is isolated for privacy.
- **API Keys:** Store keys securely in `.env` file.
- **Aviation Data:** Aviation datasets processed locally for compliance and privacy.

---

## 🎥 Demo Workflow

1. **Upload internal docs (PDF/TXT)**
2. **Ask questions about those docs or aviation datasets**
3. **Get answers with source citation**
4. **Review analytics for usage and accuracy**

---

## 🧠 How It Works

1. **Document Ingestion:** Upload, chunk, and embed docs into a vector store.
2. **Aviation Data Load:** Aviation datasets embedded as a specialized vector store for domain queries.
3. **RAG Pipeline:** Retrieve relevant chunks using ensemble search (semantic + keyword).
4. **Chat Engine:** Generate contextual answers with selected LLM, citing sources.
5. **Analytics:** Monitor performance and collect feedback for continuous improvement.

---

## 💡 Tips

- Use specific session IDs for different teams/users.
- Upload up-to-date docs for the best answers.
- Review sources for traceability.
- Share the dashboard with your team for internal knowledge sharing.
- Leverage Aviation Data RAG for deep aviation insights and analysis.

---
