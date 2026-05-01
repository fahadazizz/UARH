"""Distillation engine — extracts and manages axioms from experiment results.

After each completed experiment cycle, the Data Scientist Agent may
produce ``new_axioms_discovered``.  This module persists them in both
the SQL lineage DB and a fast-loading JSON file, and injects them into
the HarnessState for permanent prompt inclusion (Phase 1 strategy).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Optional

from uarh.core.config import get_settings
from uarh.memory.lineage import LineageRepository

logger = logging.getLogger(__name__)


class DistillationEngine:
    """Manages the lifecycle of distilled axioms."""

    def __init__(self, lineage_repo: Optional[LineageRepository] = None) -> None:
        settings = get_settings()
        self._axiom_path: Path = settings.axiom_store_path
        self._lineage = lineage_repo or LineageRepository()

    # ── Load ───────────────────────────────────────────────────

    def load_axioms(self) -> List[str]:
        """Load all axioms from the SQL store (canonical source).

        Falls back to the JSON cache if SQL is empty.
        """
        axioms = self._lineage.load_all_axioms()
        if axioms:
            logger.info("Loaded %d axioms from lineage DB.", len(axioms))
            return axioms

        # Fallback to JSON cache
        if self._axiom_path.exists():
            try:
                with open(self._axiom_path, "r") as f:
                    axioms = json.load(f)
                logger.info("Loaded %d axioms from JSON cache.", len(axioms))
                return axioms
            except (json.JSONDecodeError, Exception) as exc:
                logger.warning("Corrupt axiom cache: %s", exc)

        return []

    # ── Store ──────────────────────────────────────────────────

    def ingest_axioms(
        self,
        new_axioms: List[str],
        source_hypothesis_id: Optional[str] = None,
    ) -> int:
        """Persist new axioms to both SQL and the JSON cache.

        Parameters
        ----------
        new_axioms : list[str]
            Raw axiom text strings from the Data Scientist Agent.
        source_hypothesis_id : str | None
            The hypothesis that produced these axioms.

        Returns
        -------
        int
            Number of axioms actually ingested (deduplicated).
        """
        if not new_axioms:
            return 0

        existing = set(self._lineage.load_all_axioms())
        ingested = 0

        for axiom_text in new_axioms:
            clean = axiom_text.strip()
            if not clean or clean in existing:
                continue
            self._lineage.store_axiom(
                text=clean,
                source_hypothesis_id=source_hypothesis_id,
            )
            existing.add(clean)
            ingested += 1

        if ingested > 0:
            self._flush_json_cache(list(existing))
            logger.info(
                "Ingested %d new axioms (total: %d).",
                ingested,
                len(existing),
            )

        return ingested

    def _flush_json_cache(self, all_axioms: List[str]) -> None:
        """Write the complete axiom list to the JSON cache file."""
        self._axiom_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._axiom_path, "w") as f:
            json.dump(all_axioms, f, indent=2)

    # ── Format for Prompt Injection ────────────────────────────

    @staticmethod
    def format_for_prompt(axioms: List[str]) -> str:
        """Format axioms as a numbered block for system prompt injection.

        Returns
        -------
        str
            Ready-to-inject string, or empty string if no axioms.
        """
        if not axioms:
            return ""
        lines = ["<DISTILLED_AXIOMS>"]
        for i, axiom in enumerate(axioms, 1):
            lines.append(f"  Axiom {i}: {axiom}")
        lines.append("</DISTILLED_AXIOMS>")
        return "\n".join(lines)
