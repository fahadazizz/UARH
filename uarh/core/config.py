"""Centralised configuration for the UARH harness.

Loads from environment variables (or ``.env`` file) and exposes typed
settings via a Pydantic Settings model so every downstream module has a
single source of truth.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class UARHSettings(BaseSettings):
    """All tuneable knobs for the harness, loaded from environment."""

    # ── LLM Provider Keys ──────────────────────────────────────
    openai_api_key: Optional[str] = Field(default=None, alias="OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = Field(default=None, alias="ANTHROPIC_API_KEY")

    # ── Model Selection ────────────────────────────────────────
    elite_model: str = Field(
        default="openai/gpt-4o",
        alias="UARH_ELITE_MODEL",
        description="High-capability model for PI, Theorist, Architect.",
    )
    fast_model: str = Field(
        default="openai/gpt-4o-mini",
        alias="UARH_FAST_MODEL",
        description="Speed-optimised model for Debug Agent, Data Scientist.",
    )

    # ── Persistence Paths ──────────────────────────────────────
    data_dir: Path = Field(default=Path("./data"), alias="UARH_DATA_DIR")
    chroma_dir: Path = Field(default=Path("./data/chromadb"), alias="UARH_CHROMA_DIR")
    sqlite_uri: str = Field(
        default="sqlite:///./data/lineage.db",
        alias="UARH_SQLITE_URI",
    )
    semantic_graph_path: Path = Field(
        default=Path("./data/semantic_graph.json"),
        alias="UARH_SEMANTIC_GRAPH_PATH",
    )
    axiom_store_path: Path = Field(
        default=Path("./data/axioms.json"),
        alias="UARH_AXIOM_STORE_PATH",
    )

    # ── Governor Safety Limits ─────────────────────────────────
    max_debug_retries: int = Field(default=3, alias="UARH_MAX_DEBUG_RETRIES")
    max_consecutive_failures: int = Field(default=5, alias="UARH_MAX_CONSECUTIVE_FAILURES")
    max_theorist_revisions: int = Field(default=3, alias="UARH_MAX_THEORIST_REVISIONS")

    # ── Sandbox ────────────────────────────────────────────────
    sandbox_timeout_seconds: int = Field(default=300, alias="UARH_SANDBOX_TIMEOUT")
    level2_micro_batch_samples: int = Field(default=100, alias="UARH_L2_SAMPLES")

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
        "populate_by_name": True,
    }

    def ensure_dirs(self) -> None:
        """Create all required data directories on disk."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_dir.mkdir(parents=True, exist_ok=True)


# ── Module-level singleton ─────────────────────────────────────
_settings: Optional[UARHSettings] = None


def get_settings(force_reload: bool = False) -> UARHSettings:
    """Return the global settings singleton, creating it on first call."""
    global _settings
    if _settings is None or force_reload:
        _settings = UARHSettings()
        _settings.ensure_dirs()
    return _settings
