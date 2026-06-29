import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(dotenv_path=BASE_DIR / ".env")

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_CONTEXT_WINDOW = int(os.getenv("GROQ_CONTEXT_WINDOW", "8192"))
LLM_MAX_OUTPUT_TOKENS = int(os.getenv("LLM_MAX_OUTPUT_TOKENS", "1024"))
PROMPT_SAFETY_MARGIN = int(os.getenv("PROMPT_SAFETY_MARGIN", "256"))

JWT_SECRET = os.getenv("JWT_SECRET", "change-this-to-a-random-secret-in-production")
JWT_ALGORITHM = "HS256"
JWT_EXPIRY_HOURS = 24

DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{BASE_DIR / 'data' / 'app.db'}").replace("\\", "/")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", str(BASE_DIR / "chroma_data"))
UPLOAD_DIR = str(BASE_DIR / "uploads")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "64"))
TOP_K = int(os.getenv("TOP_K", "5"))

DOCUMENT_WORKER_COUNT = int(os.getenv("DOCUMENT_WORKER_COUNT", "2"))
DOCUMENT_JOB_MAX_ATTEMPTS = int(os.getenv("DOCUMENT_JOB_MAX_ATTEMPTS", "3"))
DOCUMENT_JOB_POLL_SECONDS = float(os.getenv("DOCUMENT_JOB_POLL_SECONDS", "0.5"))
STARTUP_WARMUP = os.getenv("STARTUP_WARMUP", "true").lower() in {"1", "true", "yes", "on"}
ENABLE_STREAMING = os.getenv("ENABLE_STREAMING", "false").lower() in {"1", "true", "yes", "on"}
