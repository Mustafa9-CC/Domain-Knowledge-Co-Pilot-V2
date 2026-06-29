import logging

from sqlalchemy import create_engine, event, text, inspect
from sqlalchemy.orm import sessionmaker, declarative_base

from backend.config import DATABASE_URL

logger = logging.getLogger(__name__)

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# ---------------------------------------------------------------------------
# Enable SQLite foreign key enforcement on every connection.
# Without this, ON DELETE CASCADE in the schema is silently ignored.
# ---------------------------------------------------------------------------
@event.listens_for(engine, "connect")
def _set_sqlite_pragma(dbapi_connection, connection_record):
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def _run_migrations(bind):
    """Add columns that are missing from an existing database.

    SQLAlchemy's create_all() only creates *new* tables; it never
    alters existing ones.  This function bridges that gap for simple
    column additions so we don't need Alembic for an MVP.
    """
    inspector = inspect(bind)
    if "documents" in inspector.get_table_names():
        existing = {col["name"] for col in inspector.get_columns("documents")}
        if "process_error" not in existing:
            with bind.connect() as conn:
                conn.execute(text(
                    "ALTER TABLE documents ADD COLUMN process_error TEXT"
                ))
                conn.commit()
            logger.info("Migration: added 'process_error' column to documents table")

    if "chat_sessions" in inspector.get_table_names():
        columns = {col["name"]: col for col in inspector.get_columns("chat_sessions")}
        # Check if corpus_id is NOT NULL (nullable == False)
        if not columns["corpus_id"].get("nullable", True):
            with bind.connect() as conn:
                # Disable FKs for table rebuild
                conn.execute(text("PRAGMA foreign_keys=OFF"))
                conn.execute(text("""
                    CREATE TABLE chat_sessions_new (
                        id INTEGER NOT NULL PRIMARY KEY,
                        user_id INTEGER NOT NULL,
                        corpus_id INTEGER,
                        title VARCHAR NOT NULL,
                        created_at DATETIME,
                        updated_at DATETIME,
                        FOREIGN KEY(user_id) REFERENCES users (id) ON DELETE CASCADE,
                        FOREIGN KEY(corpus_id) REFERENCES corpora (id) ON DELETE CASCADE
                    )
                """))
                conn.execute(text("INSERT INTO chat_sessions_new SELECT * FROM chat_sessions"))
                
                # Copy single-corpus mappings to session_corpora
                if "session_corpora" in inspector.get_table_names():
                    conn.execute(text("INSERT OR IGNORE INTO session_corpora (session_id, corpus_id) SELECT id, corpus_id FROM chat_sessions_new WHERE corpus_id IS NOT NULL"))
                
                conn.execute(text("DROP TABLE chat_sessions"))
                conn.execute(text("ALTER TABLE chat_sessions_new RENAME TO chat_sessions"))
                
                conn.execute(text("CREATE INDEX ix_chat_sessions_id ON chat_sessions (id)"))
                conn.execute(text("CREATE INDEX ix_chat_sessions_user_id ON chat_sessions (user_id)"))
                conn.execute(text("CREATE INDEX ix_chat_sessions_corpus_id ON chat_sessions (corpus_id)"))
                
                conn.commit()
                conn.execute(text("PRAGMA foreign_keys=ON"))
            logger.info("Migration: recreated chat_sessions to make corpus_id nullable and backfilled session_corpora")


def init_db():
    Base.metadata.create_all(bind=engine)
    _run_migrations(engine)
