"""Lineage database — SQLAlchemy models for experiment tracking.

Every graph execution is recorded with its hypothesis, status, metrics,
and the validation level it reached.  This gives the harness a complete
audit trail independent of the LLM conversation history.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from sqlalchemy import (
    JSON,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    Session,
    mapped_column,
    sessionmaker,
)

from uarh.core.config import get_settings


# ═══════════════════════════════════════════════════════════════
#  ORM Models
# ═══════════════════════════════════════════════════════════════


class Base(DeclarativeBase):
    pass


class ExecutionLog(Base):
    """One row per hypothesis execution attempt."""

    __tablename__ = "execution_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    hypothesis_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    hypothesis_title: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False)  # HypothesisStatus value
    validation_level_reached: Mapped[int] = mapped_column(Integer, default=0)
    final_loss: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    metrics: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    error_summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )


class AxiomRecord(Base):
    """Persistent axiom extracted by the Distillation Engine."""

    __tablename__ = "axioms"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    axiom_id: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        unique=True,
        default=lambda: f"axiom-{uuid.uuid4().hex[:8]}",
    )
    text: Mapped[str] = mapped_column(Text, nullable=False)
    source_hypothesis_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )


# ═══════════════════════════════════════════════════════════════
#  Repository
# ═══════════════════════════════════════════════════════════════


class LineageRepository:
    """CRUD operations on the lineage database."""

    def __init__(self) -> None:
        settings = get_settings()
        self._engine = create_engine(settings.sqlite_uri, echo=False, future=True)
        Base.metadata.create_all(self._engine)
        self._session_factory = sessionmaker(bind=self._engine, expire_on_commit=False)

    # ── Execution Logs ─────────────────────────────────────────

    def record_execution(
        self,
        *,
        run_id: str,
        hypothesis_id: str,
        hypothesis_title: str | None = None,
        status: str,
        validation_level_reached: int = 0,
        final_loss: float | None = None,
        metrics: Dict[str, Any] | None = None,
        error_summary: str | None = None,
    ) -> ExecutionLog:
        """Insert a new execution log row."""
        log = ExecutionLog(
            run_id=run_id,
            hypothesis_id=hypothesis_id,
            hypothesis_title=hypothesis_title,
            status=status,
            validation_level_reached=validation_level_reached,
            final_loss=final_loss,
            metrics=metrics,
            error_summary=error_summary,
        )
        with self._session_factory() as session:
            session.add(log)
            session.commit()
            session.refresh(log)
        return log

    def get_recent_executions(self, limit: int = 20) -> list[ExecutionLog]:
        """Return the most recent execution logs."""
        with self._session_factory() as session:
            return (
                session.query(ExecutionLog)
                .order_by(ExecutionLog.created_at.desc())
                .limit(limit)
                .all()
            )

    # ── Axioms ─────────────────────────────────────────────────

    def store_axiom(self, text: str, source_hypothesis_id: str | None = None) -> AxiomRecord:
        """Persist a new axiom."""
        rec = AxiomRecord(text=text, source_hypothesis_id=source_hypothesis_id)
        with self._session_factory() as session:
            session.add(rec)
            session.commit()
            session.refresh(rec)
        return rec

    def load_all_axioms(self) -> list[str]:
        """Return all axiom texts in creation order."""
        with self._session_factory() as session:
            rows = session.query(AxiomRecord).order_by(AxiomRecord.created_at.asc()).all()
            return [r.text for r in rows]
