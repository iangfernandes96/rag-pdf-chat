# 🧠 RAG PDF Chat

**RAG PDF Chat** is a local Retrieval-Augmented Generation (RAG) app that lets you upload PDFs and ask natural language questions about them. It runs entirely on your MacBook (including M1/M2), using local embeddings, vector search, and an LLM like Mistral or LLaMA via Ollama.

---

## 🔧 Features

- Upload one or more PDFs
- Chunk and embed text locally
- Store embeddings in a vector database (FAISS or Chroma)
- Ask questions in natural language
- Get answers using a local LLM with source context
- No external APIs required

---

## 🛠️ Tech Stack

| Component        | Tool                        |
|------------------|-----------------------------|
| PDF Parsing      | PyMuPDF or PDFPlumber       |
| Chunking         | LangChain / Custom          |
| Embeddings       | `sentence-transformers`     |
| Vector Store     | FAISS or Chroma             |
| LLM              | Ollama + Mistral / LLaMA 3  |
| Backend          | Python + FastAPI            |
| Frontend         | Streamlit or React          |

---

## 🚀 Getting Started

### 1. Clone the Repo

```bash
git clone https://github.com/your-username/rag-pdf-chat.git
cd rag-pdf-chat
