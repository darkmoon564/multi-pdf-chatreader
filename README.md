# 📚 Multi-PDF ChatReader

Chat with multiple PDF documents using AI!  
This app enables contextual question-answering over multiple PDFs using state-of-the-art embeddings and an open-source language model.

DEMO APP : https://multi-pdf-chatreader.streamlit.app


---

## 🚀 Features
- Upload and process multiple PDF documents
- Text chunking & semantic embedding (MiniLM via `sentence-transformers`)
- FAISS-based vector store for fast retrieval
- Conversational agent powered by `Flan-T5` (Hugging Face Pipeline)
- Streamlit UI for interactive chat

---

## 🛠️ Tech Stack

| Category | Tools Used |
|----------|------------|
| Embeddings | `sentence-transformers` |
| Vector Store | `FAISS` |
| LLM | `Flan-T5` via HuggingFace |
| UI | `Streamlit` |
| Parsing | `PyPDF2` |
| Orchestration | `LangChain` |

---

## 📦 Installation

```bash
git clone https://github.com/darkmoon564/multi-pdf-chatreader.git
cd multi-pdf-chatreader
pip install -r requirements.txt
