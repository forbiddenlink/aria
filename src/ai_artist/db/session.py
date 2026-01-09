"""Database session management."""

from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker


def create_db_engine(db_path: Path):
    """Create SQLite engine with optimal settings."""
    db_path.parent.mkdir(parents=True, exist_ok=True)

    engine = create_engine(
        f"sqlite:///{db_path}",
        connect_args={"check_same_thread": False, "timeout": 30},
        pool_pre_ping=True,
    )

    # Enable WAL mode for better concurrency
    with engine.connect() as conn:
        conn.execute(text("PRAGMA journal_mode=WAL"))
        conn.execute(text("PRAGMA synchronous=NORMAL"))
        conn.execute(text("PRAGMA cache_size=-64000"))  # 64MB cache
        conn.commit()

    return engine


def create_session_factory(db_path: Path) -> sessionmaker:
    """Create session factory."""
    engine = create_db_engine(db_path)
    return sessionmaker(bind=engine, autocommit=False, autoflush=False)


@contextmanager
def get_db_session(session_factory: sessionmaker):
    """Context manager for database sessions."""
    session = session_factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
