"""
Streamlit frontend for RAG PDF Chat application.
Provides file upload, document management, and chat interface.
"""

import logging
import os
from datetime import datetime
from typing import Any

import httpx
import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
MAX_FILE_SIZE_MB = 50
SUPPORTED_EXTENSIONS = [".pdf"]

# Page configuration
st.set_page_config(
    page_title="🧠 RAG PDF Chat",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better styling
st.markdown(
    """
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        border-bottom: 2px solid #f0f0f0;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0.5rem;
        border-left: 4px solid #4CAF50;
        background-color: #f9f9f9;
    }
    .user-message {
        border-left-color: #2196F3;
        background-color: #e3f2fd;
    }
    .assistant-message {
        border-left-color: #4CAF50;
        background-color: #e8f5e9;
    }
    .source-box {
        background-color: #fff3e0;
        border: 1px solid #ffcc02;
        border-radius: 0.3rem;
        padding: 0.5rem;
        margin: 0.5rem 0;
    }
    .metric-card {
        background-color: #f5f5f5;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""",
    unsafe_allow_html=True,
)


class RAGChatClient:
    """Client for communicating with the RAG PDF Chat API."""

    def __init__(self, backend_url: str):
        self.backend_url = backend_url
        self.client = httpx.Client(timeout=300.0)  # 5 minutes for long operations

    def health_check(self) -> dict[str, Any]:
        """Check backend health status."""
        try:
            response = self.client.get(f"{self.backend_url}/health")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def upload_document(self, file: UploadedFile) -> dict[str, Any]:
        """Upload a PDF document for background processing."""
        try:
            files = {"file": (file.name, file.read(), "application/pdf")}
            response = self.client.post(f"{self.backend_url}/upload", files=files)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_job_status(self, job_id: str) -> dict[str, Any]:
        """Get the status of a background processing job."""
        try:
            response = self.client.get(f"{self.backend_url}/jobs/{job_id}")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"success": False, "error": str(e)}

    def query_documents(
        self,
        query: str,
        document_id: str | None = None,
        limit: int = 5,
        include_context: bool = True,
    ) -> dict[str, Any]:
        """Query documents using RAG."""
        try:
            payload = {
                "query": query,
                "limit": limit,
                "include_context": include_context,
            }
            if document_id:
                payload["document_id"] = document_id

            response = self.client.post(f"{self.backend_url}/query", json=payload)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"success": False, "error": str(e)}

    def list_documents(self) -> dict[str, Any]:
        """List all uploaded documents."""
        try:
            response = self.client.get(f"{self.backend_url}/documents")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"documents": [], "total_count": 0, "error": str(e)}

    def delete_document(self, document_id: str) -> dict[str, Any]:
        """Delete a document."""
        try:
            response = self.client.delete(f"{self.backend_url}/documents/{document_id}")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"success": False, "error": str(e)}


def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "documents" not in st.session_state:
        st.session_state.documents = []

    if "client" not in st.session_state:
        st.session_state.client = RAGChatClient(BACKEND_URL)

    if "backend_healthy" not in st.session_state:
        st.session_state.backend_healthy = False

    if "active_job_id" not in st.session_state:
        st.session_state.active_job_id = None


def check_backend_status():
    """Check and display backend status."""
    health = st.session_state.client.health_check()

    if "error" in health:
        st.session_state.backend_healthy = False
        st.error(f"🔴 Backend connection failed: {health['error']}")
        st.info("Please ensure the FastAPI backend is running on http://localhost:8000")
        return False

    st.session_state.backend_healthy = True
    status = health.get("status", "unknown")

    if status == "healthy":
        st.success("🟢 Backend is healthy and ready")
    else:
        st.warning(f"🟡 Backend status: {status}")

    return True


def display_header():
    """Display the main application header."""
    st.markdown(
        """
        <div class="main-header">
            <h1>🧠 RAG PDF Chat</h1>
            <p>Upload PDFs and ask questions using Retrieval-Augmented Generation</p>
        </div>
    """,
        unsafe_allow_html=True,
    )


def sidebar_document_management():
    """Handle document management in the sidebar."""
    st.sidebar.header("📚 Document Management")

    # Upload section
    st.sidebar.subheader("Upload Document")
    uploaded_file = st.sidebar.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        help=f"Maximum file size: {MAX_FILE_SIZE_MB}MB",
    )

    if uploaded_file is not None:
        if st.sidebar.button("📤 Upload Document", type="primary"):
            if not st.session_state.backend_healthy:
                st.sidebar.error("Backend not available")
                return

            # Validate file
            if uploaded_file.size > MAX_FILE_SIZE_MB * 1024 * 1024:
                st.sidebar.error(f"File too large! Maximum size: {MAX_FILE_SIZE_MB}MB")
                return

            # Upload with progress
            with st.spinner("Uploading and processing document..."):
                result = st.session_state.client.upload_document(uploaded_file)

            if result.get("success", False):
                job_id = result.get("job_id")
                st.session_state.active_job_id = job_id
                st.sidebar.success("✅ Document uploaded successfully!")
                st.sidebar.info(
                    f"Processing started in background (Job ID: {job_id[:8]}...)"
                )

                # Show job status
                if job_id:
                    with st.sidebar.expander("📊 Processing Status", expanded=True):
                        # Refresh button
                        col1, col2 = st.columns([3, 1])
                        with col2:
                            if st.button(
                                "🔄",
                                key="refresh_job_status",
                                help="Refresh job status",
                            ):
                                st.rerun()

                        job_status = st.session_state.client.get_job_status(job_id)
                        if job_status.get("success", False):
                            job_info = job_status.get("job_info", {})
                            status = job_info.get("status", "unknown")
                            stage = job_info.get("stage", "unknown")
                            progress = job_info.get("progress", 0)
                            message = job_info.get("message", "")

                            st.write(f"**Status:** {status}")
                            st.write(f"**Stage:** {stage}")
                            st.write(f"**Progress:** {progress}%")
                            st.write(f"**Message:** {message}")

                            if status == "completed":
                                st.success("✅ Processing completed!")
                                st.session_state.active_job_id = None
                                load_documents()
                                st.rerun()
                            elif status == "failed":
                                st.error(
                                    f"❌ Processing failed: {job_info.get('error', 'Unknown error')}"
                                )
                                st.session_state.active_job_id = None
                            else:
                                st.info("⏳ Processing in progress...")
                                st.info(
                                    "Click the refresh button above to check status"
                                )
                        else:
                            st.error(
                                f"Failed to get job status: {job_status.get('error', 'Unknown error')}"
                            )
            else:
                st.sidebar.error(
                    f"❌ Upload failed: {result.get('error', 'Unknown error')}"
                )

    # Show active job status if exists
    if st.session_state.active_job_id:
        with st.sidebar.expander("📊 Active Job Status", expanded=True):
            # Refresh button
            col1, col2 = st.columns([3, 1])
            with col2:
                if st.button("🔄", key="refresh_active_job", help="Refresh job status"):
                    st.rerun()

            job_status = st.session_state.client.get_job_status(
                st.session_state.active_job_id
            )
            if job_status.get("success", False):
                job_info = job_status.get("job_info", {})
                status = job_info.get("status", "unknown")
                stage = job_info.get("stage", "unknown")
                progress = job_info.get("progress", 0)
                message = job_info.get("message", "")
                filename = job_info.get("filename", "Unknown")

                st.write(f"**File:** {filename}")
                st.write(f"**Status:** {status}")
                st.write(f"**Stage:** {stage}")
                st.write(f"**Progress:** {progress}%")
                st.write(f"**Message:** {message}")

                if status == "completed":
                    st.success("✅ Processing completed!")
                    st.session_state.active_job_id = None
                    load_documents()
                    st.rerun()
                elif status == "failed":
                    st.error(
                        f"❌ Processing failed: {job_info.get('error', 'Unknown error')}"
                    )
                    st.session_state.active_job_id = None
                else:
                    st.info("⏳ Processing in progress...")
                    st.info("Click the refresh button above to check status")
            else:
                st.error(
                    f"Failed to get job status: {job_status.get('error', 'Unknown error')}"
                )

    # Document list section
    st.sidebar.subheader("Uploaded Documents")

    if st.sidebar.button("🔄 Refresh Documents"):
        load_documents()
        st.rerun()

    if not st.session_state.documents:
        st.sidebar.info("No documents uploaded yet")
    else:
        for doc in st.session_state.documents:
            with st.sidebar.expander(f"📄 {doc['original_filename'][:20]}..."):
                st.write(f"**File size:** {doc['file_size'] / 1024:.1f} KB")
                st.write(f"**Pages:** {doc['page_count']}")
                st.write(f"**Chunks:** {doc['chunk_count']}")
                st.write(f"**Uploaded:** {doc['uploaded_at'][:16]}")

                if st.button("🗑️ Delete", key=f"delete_{doc['id']}"):
                    with st.spinner("Deleting document..."):
                        result = st.session_state.client.delete_document(doc["id"])

                    if result.get("success", False):
                        st.success("Document deleted!")
                        load_documents()
                        st.rerun()
                    else:
                        st.error(
                            f"Deletion failed: {result.get('error', 'Unknown error')}"
                        )


def load_documents():
    """Load the list of documents from the backend."""
    if st.session_state.backend_healthy:
        docs_response = st.session_state.client.list_documents()
        st.session_state.documents = docs_response.get("documents", [])


def main_chat_interface():
    """Main chat interface."""
    st.header("💬 Chat with Your Documents")

    if not st.session_state.backend_healthy:
        st.error("Please ensure the backend is running to use the chat interface.")
        return

    if not st.session_state.documents:
        st.info("📚 Upload some PDF documents first to start chatting!")
        return

    # Chat settings
    col1, col2 = st.columns([3, 1])

    with col2:
        st.subheader("🔧 Settings")

        # Document filter
        doc_options = ["All documents"] + [
            doc["original_filename"] for doc in st.session_state.documents
        ]
        selected_doc = st.selectbox("Search in:", doc_options)

        # Search parameters
        num_chunks = st.slider("Max chunks to retrieve:", 1, 10, 5)
        include_sources = st.checkbox("Show sources", value=True)

        # Statistics
        total_docs = len(st.session_state.documents)
        total_chunks = sum(
            doc.get("chunk_count", 0) for doc in st.session_state.documents
        )

        st.markdown(
            f"""
            <div class="metric-card">
                <strong>📊 Statistics</strong><br>
                Documents: {total_docs}<br>
                Total chunks: {total_chunks}<br>
                Messages: {len(st.session_state.messages)}
            </div>
        """,
            unsafe_allow_html=True,
        )

    with col1:
        # Display chat messages
        st.subheader("Chat History")

        chat_container = st.container()
        with chat_container:
            for message in st.session_state.messages:
                display_message(message)

        # Chat input
        if query := st.chat_input("Ask a question about your documents..."):
            # Add user message
            user_message = {
                "role": "user",
                "content": query,
                "timestamp": datetime.now().isoformat(),
            }
            st.session_state.messages.append(user_message)

            # Get document ID filter
            document_id = None
            if selected_doc != "All documents":
                for doc in st.session_state.documents:
                    if doc["original_filename"] == selected_doc:
                        document_id = doc["id"]
                        break

            # Query the backend
            with st.spinner("🤔 Thinking..."):
                response = st.session_state.client.query_documents(
                    query=query,
                    document_id=document_id,
                    limit=num_chunks,
                    include_context=include_sources,
                )

            # Add assistant message
            if response.get("success", False):
                assistant_message = {
                    "role": "assistant",
                    "content": response["answer"],
                    "sources": response.get("sources", []) if include_sources else [],
                    "metadata": {
                        "model_used": response.get("model_used", "unknown"),
                        "response_time": response.get("response_time", 0),
                        "chunks_used": response.get("chunks_used", 0),
                    },
                    "timestamp": datetime.now().isoformat(),
                }
            else:
                assistant_message = {
                    "role": "assistant",
                    "content": f"❌ Sorry, I encountered an error: {response.get('error', 'Unknown error')}",
                    "sources": [],
                    "metadata": {},
                    "timestamp": datetime.now().isoformat(),
                }

            st.session_state.messages.append(assistant_message)
            st.rerun()


def display_message(message: dict[str, Any]):
    """Display a chat message with proper styling."""
    role = message["role"]
    content = message["content"]
    timestamp = message.get("timestamp", "")

    if role == "user":
        st.markdown(
            f"""
            <div class="chat-message user-message">
                <strong>👤 You</strong> <small>({timestamp[:16]})</small><br>
                {content}
            </div>
        """,
            unsafe_allow_html=True,
        )

    elif role == "assistant":
        metadata = message.get("metadata", {})
        sources = message.get("sources", [])

        st.markdown(
            f"""
            <div class="chat-message assistant-message">
                <strong>🤖 Assistant</strong> <small>({timestamp[:16]})</small><br>
                {content}
            </div>
        """,
            unsafe_allow_html=True,
        )

        # Display metadata
        if metadata:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.caption(f"Model: {metadata.get('model_used', 'N/A')}")
            with col2:
                st.caption(f"Response: {metadata.get('response_time', 0):.1f}s")
            with col3:
                st.caption(f"Chunks: {metadata.get('chunks_used', 0)}")

        # Display sources
        if sources:
            with st.expander(f"📚 Sources ({len(sources)} chunks)"):
                for i, source in enumerate(sources, 1):
                    st.markdown(
                        f"""
                        <div class="source-box">
                            <strong>Source {i} (Score: {source['score']:.3f})</strong><br>
                            <small>Document: {source['document_id'][:8]}... | Chunk: {source['chunk_index']}</small><br>
                            {source['content']}
                        </div>
                    """,
                        unsafe_allow_html=True,
                    )


def display_system_status():
    """Display system status and statistics."""
    if not st.session_state.backend_healthy:
        return

    st.header("📊 System Status")

    # Get health information
    health = st.session_state.client.health_check()

    if "services" in health:
        services = health["services"]

        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("🧠 RAG Service")
            rag_service = services.get("rag_service", {})
            if rag_service.get("system_healthy", False):
                st.success("✅ Healthy")

                embedding_service = rag_service.get("embedding_service", {})
                if embedding_service.get("loaded", False):
                    st.info(f"Model: {embedding_service.get('model_name', 'N/A')}")
                    st.info(
                        f"Dimension: {embedding_service.get('embedding_dimension', 'N/A')}"
                    )
            else:
                st.error("❌ Not healthy")

        with col2:
            st.subheader("🤖 LLM Service")
            llm_service = services.get("llm_service", {})
            if llm_service.get("healthy", False):
                st.success("✅ Healthy")
                st.info(f"Model: {llm_service.get('model', 'N/A')}")
                st.info(f"URL: {llm_service.get('url', 'N/A')}")
            else:
                st.error("❌ Not available")

        with col3:
            st.subheader("📊 Database")
            db_service = services.get("database", {})
            if db_service.get("healthy", False):
                st.success("✅ Connected")
                st.info(
                    f"Pool: {db_service.get('pool_size', 0)}/{db_service.get('pool_max_size', 0)}"
                )
            else:
                st.error("❌ Not connected")


def main():
    """Main application function."""
    initialize_session_state()
    display_header()

    # Check backend status
    backend_status = check_backend_status()

    if backend_status:
        # Load documents initially
        if not st.session_state.documents:
            load_documents()

    # Sidebar for document management
    sidebar_document_management()

    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["💬 Chat", "📊 Status", "ℹ️ About"])

    with tab1:
        main_chat_interface()

    with tab2:
        display_system_status()

    with tab3:
        st.header("ℹ️ About RAG PDF Chat")
        st.markdown(
            """
        ### 🧠 What is RAG PDF Chat?

        RAG PDF Chat is a **Retrieval-Augmented Generation** application that allows you to:
        - Upload PDF documents
        - Ask natural language questions about their content
        - Get accurate answers with source citations

        ### 🔧 How it works:

        1. **Document Processing**: PDFs are parsed and split into chunks
        2. **Embedding Generation**: Text chunks are converted to vector embeddings
        3. **Vector Storage**: Embeddings are stored in a vector database (Qdrant)
        4. **Query Processing**: Your questions are embedded and matched with relevant chunks
        5. **Response Generation**: A local LLM (via Ollama) generates answers using the retrieved context

        ### 🛠️ Technology Stack:

        - **Frontend**: Streamlit
        - **Backend**: FastAPI + Python
        - **LLM**: Ollama (Mistral model)
        - **Embeddings**: sentence-transformers (all-MiniLM-L6-v2)
        - **Vector DB**: Qdrant
        - **Database**: PostgreSQL
        - **PDF Processing**: pdfplumber

        ### 🚀 Features:

        - ✅ Local processing (no external APIs)
        - ✅ Multiple document support
        - ✅ Source citation
        - ✅ Real-time chat interface
        - ✅ Document management
        - ✅ System monitoring

        ### 📝 Usage Tips:
        - Ask specific questions for better results
        - Use the document filter to search within specific files
        - Check the sources to verify answers
        - Upload multiple related documents for comprehensive answers
        """
        )


if __name__ == "__main__":
    main()
