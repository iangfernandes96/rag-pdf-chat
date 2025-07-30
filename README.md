# 🧠 RAG PDF Chat

**RAG PDF Chat** is a complete Retrieval-Augmented Generation (RAG) application that lets you upload PDFs and ask natural language questions about them. It uses local embeddings, vector search, and Google's Gemini API for LLM inference.

**🎉 PRODUCTION READY** - Fully implemented through Phase 6 with comprehensive testing and UX polish!

---

## 🔧 Features

- **📄 Upload multiple PDFs** with drag-and-drop interface
- **🔍 Smart document chunking** and embedding locally
- **🗃️ Vector database storage** with Qdrant for fast retrieval
- **💬 Natural language chat** interface with message history
- **🤖 LLM inference** via Google Gemini API
- **📚 Source citations** with relevance scores and document references
- **⚡ Background processing** with Arq workers for document uploads
- **🛡️ Comprehensive error handling** with user-friendly messages
- **📊 System monitoring** and health checks
- **🎨 Polished UX** with response quality scoring and user guidance
- **🔄 Real-time job status tracking** with Redis persistence

---

## 🛠️ Tech Stack

| Component        | Tool                        | Status |
|------------------|-----------------------------|--------|
| PDF Parsing      | PDFPlumber                  | ✅ |
| Text Chunking    | Custom with overlap         | ✅ |
| Embeddings       | `sentence-transformers`     | ✅ |
| Vector Store     | Qdrant (AsyncClient)        | ✅ |
| Database         | PostgreSQL (AsyncPG)        | ✅ |
| LLM              | Google Gemini API            | ✅ |
| Backend          | FastAPI + Python            | ✅ |
| Background Jobs  | Arq + Redis                 | ✅ |
| Frontend         | Streamlit                   | ✅ |
| UX Polish        | Enhanced error handling     | ✅ |
| Code Quality     | Black + Ruff               | ✅ |

---

## 🚀 Quick Start

### 1. Environment Setup

First, set up your environment variables:

```bash
# Copy the example environment file
cp config.env.example .env

# Edit the .env file and add your Gemini API key
# Get your API key from: https://makersuite.google.com/app/apikey
```

### 2. One-Command Setup and Run

```bash
# Download and run the complete application
git clone https://github.com/your-username/rag-pdf-chat.git
cd rag-pdf-chat
./start_full_app.sh
```

This script will:
- ✅ Check dependencies (uv, Gemini API key)
- ✅ Install Python packages
- ✅ Run comprehensive tests
- ✅ Start both backend and frontend
- ✅ Open the application in your browser

### 2. Manual Setup (if preferred)

```bash
# Install dependencies
uv sync

# Start individual services
uv run uvicorn backend.app:app --reload  # Backend API
uv run streamlit run frontend/app.py     # Frontend UI

# Or use Docker (includes background workers)
# Make sure your .env file contains your GEMINI_API_KEY
docker compose up

# Start background worker separately (if not using Docker)
python start_worker.py
```

---

## 🎯 Usage

### 1. **Upload Documents**
- Use the sidebar file uploader
- Supports PDF files up to 50MB
- **Background processing** - uploads are queued and processed asynchronously
- **Real-time job tracking** - monitor processing status via job IDs with Redis persistence
- **Job management** - list, filter, and manage background jobs
- Automatic chunking and embedding generation

### 2. **Ask Questions** 
- Type natural language questions in the chat
- Get AI-powered answers with source citations
- Filter by specific documents or search all
- View relevance scores and source snippets

### 3. **Manage Documents**
- View uploaded documents in the sidebar
- See processing statistics (pages, chunks, tokens)
- Delete documents as needed
- Monitor system status and health

---

## 🔄 Background Processing Architecture

The application now uses **Arq workers** for background document processing, providing:

### **Benefits**
- **Non-blocking uploads** - API responds immediately with job ID
- **Scalable processing** - Multiple workers can handle concurrent documents
- **Reliable execution** - Automatic retries and error handling
- **Progress tracking** - Real-time status updates via job endpoints

### **Components**
- **Redis** - Message broker for job queues and job status persistence
- **Arq Workers** - Background task processors
- **Job Tracker** - Redis-based job status management
- **Job Status API** - Real-time status tracking and job management
- **Error Recovery** - Automatic retry with exponential backoff

### **Processing Pipeline**
1. **Upload** → File validation and job creation
2. **Queue** → Document added to processing queue
3. **Parse** → PDF text extraction and chunking
4. **Embed** → Vector embedding generation
5. **Store** → Vector database storage
6. **Complete** → Job status updated and ready for querying

### **API Endpoints**
- `POST /upload` - Queue document for processing
- `GET /jobs/{job_id}` - Check processing status
- `GET /jobs` - List all jobs with optional filtering
- `POST /query` - Query processed documents (unchanged)

### **Job Tracking Features**
- **Real-time Status Updates** - Jobs progress through stages: Upload → Parsing → Chunking → Embedding → Storing → Completed
- **Persistent Storage** - Job information stored in Redis with 7-day TTL
- **Error Tracking** - Failed jobs include detailed error messages
- **Progress Monitoring** - Percentage completion and current stage tracking
- **Job Management** - List, filter, and clean up old jobs

---

## 🧪 Testing & Code Quality

### Code Quality Tools ⭐ **NEW**
The project includes **Black** and **Ruff** for code formatting and linting:

```bash
# 🔧 Format code (run this before committing)
./format.sh

# 🔍 Check code quality without making changes  
./lint.sh

# 🚀 Comprehensive check: format + lint + tests
./check.sh
```

### Manual Quality Commands
```bash
# Format with Black
uv run black backend/ frontend/

# Lint with Ruff
uv run ruff check backend/ frontend/

# Auto-fix linting issues
uv run ruff check backend/ frontend/ --fix
```

### Configuration
- **Black**: Line length 88, Python 3.11+ target
- **Ruff**: pycodestyle, pyflakes, isort, bugbear rules
- **Config**: All settings in `pyproject.toml`

---

## 📋 Implementation Phases

### ✅ Phase 1: Setup & Environment
- ✅ Project structure with `uv` package management
- ✅ Ollama installation and model setup
- ✅ Development environment configuration

### ✅ Phase 2: Document Ingestion  
- ✅ PDF parsing with `pdfplumber`
- ✅ Text chunking with configurable overlap
- ✅ Document models with Pydantic validation

### ✅ Phase 3: Embedding & Vector Store
- ✅ Local embedding generation (`sentence-transformers`)
- ✅ Async Qdrant integration for vector storage
- ✅ Similarity search and retrieval

### ✅ Phase 4: RAG Pipeline & Backend
- ✅ FastAPI server with async endpoints
- ✅ LLM integration with Ollama
- ✅ PostgreSQL for metadata storage
- ✅ Health monitoring and error handling

### ✅ Phase 5: Frontend Interface
- ✅ Streamlit web interface
- ✅ Document upload and management
- ✅ Real-time chat with message history
- ✅ Source citation and system monitoring

### ✅ Phase 6: Testing & UX Polish ⭐ **NEW**
- ✅ Comprehensive integration testing
- ✅ Enhanced error messages and user guidance
- ✅ Response quality scoring and caching
- ✅ Performance optimization and monitoring
- ✅ User feedback collection and analytics

---

## 🎨 UX Enhancements (Phase 6)

### Enhanced User Experience
- **🎯 Smart Error Handling**: User-friendly error messages with helpful suggestions
- **📊 Response Quality Scoring**: Automatic assessment of answer quality
- **💡 User Guidance**: Context-aware tips for better results
- **⚡ Response Caching**: Faster responses for similar queries
- **📈 Usage Analytics**: System performance monitoring and recommendations

### Advanced Features
- **🔍 Enhanced Source Citations**: Relevance descriptions and better formatting
- **📝 Response Formatting**: Automatic paragraph structuring for readability
- **🛡️ Graceful Degradation**: Helpful responses even when information is unavailable
- **⏱️ Performance Monitoring**: Response time tracking and optimization
- **📊 System Health**: Comprehensive service monitoring and diagnostics

---

## 🌐 URLs

Once running, access these URLs:

- **🖥️ Frontend (Streamlit)**: http://localhost:8501
- **🔧 Backend API (FastAPI)**: http://localhost:8000  
- **📚 API Documentation**: http://localhost:8000/docs
- **🤖 Gemini API**: Configured via environment variables

---

## 🔧 Configuration

Key settings can be configured via environment variables:

```bash
# LLM Settings
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-2.0-flash
LLM_TEMPERATURE=0.7

# Vector Database
QDRANT_URL=http://localhost:6333
VECTOR_SIZE=384

# Document Processing  
CHUNK_SIZE=500
CHUNK_OVERLAP=100
MAX_FILE_SIZE_MB=50

# Database
DATABASE_URL=postgresql://user:pass@localhost:5432/rag_pdf_chat
```

---

## 🎉 What's Next?

- **🔐 User Authentication**: Multi-user support with document isolation
- **📱 Mobile Interface**: Responsive design for mobile devices  
- **🌍 Multi-language**: Support for non-English documents
- **📊 Analytics Dashboard**: Usage statistics and insights
- **🔄 Auto-sync**: Watch folders for new documents
- **🎛️ Model Selection**: Choose between different LLMs
- **💾 Export Features**: Save conversations and insights

---

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

