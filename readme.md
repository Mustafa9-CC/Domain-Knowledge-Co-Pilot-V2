# 🚀 Domain Knowledge Co-Pilot

Domain Knowledge Co-Pilot Retrieval-Augmented Generation (RAG) based knowledge assistant that enables users to interact with their personal documents using natural language. Instead of manually searching through PDFs and reports, users can upload documents, organize them into knowledge corpora, and receive context-aware answers with source citations.

Built as an IIT Roorkee Capstone Project, Domain Knnowledge Co-Pilot focuses on modular architecture, scalable document processing, and production-inspired RAG workflows.

---

## ✨ Features

- 🔐 JWT Authentication
- 📂 Multi-Corpus Management
- 📄 PDF, DOCX, TXT & Markdown Support
- ⚙️ Asynchronous Background Document Processing
- ✂️ Intelligent Document Chunking
- 🧠 Semantic Embedding Generation
- 🗄️ Chroma Vector Database Integration
- 🔍 Hybrid Retrieval (Vector Search + BM25)
- 🌐 Cross-Corpus Search
- 💬 Context-Aware Conversational Chat
- 📚 Source Citation & Chunk References
- 📜 Conversation History
- 📊 Diagnostics & Health Monitoring
- ⚡ Streaming AI Responses
- 🧪 Benchmark & Performance Utilities

---

# 🏗️ System Architecture

```
                  User
                    │
                    ▼
             Streamlit Frontend
                    │
                    ▼
             FastAPI Backend
                    │
      ┌─────────────┴─────────────┐
      │                           │
 Authentication            Corpus Manager
      │                           │
      └─────────────┬─────────────┘
                    │
             Document Upload
                    │
                    ▼
        Background Processing Worker
                    │
                    ▼
         Text Extraction & Chunking
                    │
                    ▼
          Embedding Generation
                    │
                    ▼
             Chroma Vector Store
                    │
                    ▼
      Hybrid Retrieval (BM25 + Vector)
                    │
                    ▼
             Prompt Construction
                    │
                    ▼
           llamma-3.3-70b-versatile                    │
                    ▼
      Answer with Source Citations
```

---

# 📌 Project Workflow

```
Create Account
      │
      ▼
Login
      │
      ▼
Create Corpus
      │
      ▼
Upload Documents
      │
      ▼
Background Processing
      │
      ▼
Embedding & Indexing
      │
      ▼
Ask Questions
      │
      ▼
Hybrid Retrieval
      │
      ▼
LLM Response
      │
      ▼
Answer + Citations
```

---

# 🛠️ Technology Stack

## Frontend

- Streamlit

## Backend

- FastAPI

## Database

- SQLite
- SQLAlchemy

## Vector Database

- ChromaDB

## AI & NLP

- Google Gemini API
- Sentence Transformers
- BM25 Retrieval

## Authentication

- JWT Authentication

## Document Processing

- PyMuPDF
- python-docx
- Markdown Parser

---

# 📂 Project Structure

```
capstone/
│
├── backend/
│   ├── api/
│   ├── services/
│   ├── models/
│   ├── schemas/
│   ├── workers/
│   ├── retrieval/
│   ├── auth/
│   └── utils/
│
├── frontend/
│
└── README.md
```

---

# 🚀 Getting Started

## Clone Repository

```bash
git clone http.git

cd 
```

---

## Create Virtual Environment

```bash
python -m venv .venv
```

Windows

```bash
.venv\Scripts\activate
```

Linux / macOS

```bash
source .venv/bin/activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Configure Environment Variables

Create a `.env` file.

```env
GEMINI_API_KEY=YOUR_API_KEY
JWT_SECRET_KEY=YOUR_SECRET
DATABASE_URL=sqlite:///graphify.db
CHROMA_DB_PATH=./chroma_db
```

---

## Run Backend

```bash
uvicorn backend.main:app --reload
```

---

## Run Frontend

```bash
streamlit run frontend/app.py
```

---

# 📸 Screenshots

> Add screenshots here after deployment.

- Login
- Dashboard
- Corpus Management
- Document Upload
- Chat Interface
- Source Citations

---

# 🎯 Use Cases

- Research Assistant
- Academic Knowledge Base
- Company Internal Documentation
- Technical Documentation Search
- Legal Document Search
- Personal Knowledge Management

---

# 🔮 Future Enhancements

- OCR Support
- Image-based Retrieval
- Role-Based Access Control
- Cloud Deployment
- Multi-Modal RAG
- Advanced Analytics Dashboard 
- Web Search Across Documents


---

# 📄 License

This project is licensed under the MIT License.

---

# 👨‍💻 Author

Mohd Mustafa

IIT Roorkee Capstone Project

---

## ⭐ Support

If you found this project useful, consider giving it a ⭐ on GitHub.