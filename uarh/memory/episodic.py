"""Episodic memory — ChromaDB-backed vector store for past experiments.

Stores embedded hypothesis rationales so the PI Agent can detect
whether a permutation has been tried before and avoid redundant work.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

from uarh.core.config import get_settings

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════
#  Episodic Memory Repository
# ═══════════════════════════════════════════════════════════════


class EpisodicMemory:
    """Thin wrapper around a ChromaDB collection for hypothesis look-up.

    ChromaDB's default embedding function (all-MiniLM-L6-v2) is used for
    lightweight local embedding.  Each document is the hypothesis
    rationale; metadata carries structured experiment data.
    """

    COLLECTION_NAME = "uarh_episodic"

    def __init__(self) -> None:
        settings = get_settings()
        self._client = chromadb.PersistentClient(
            path=str(settings.chroma_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "EpisodicMemory initialised — %d documents in collection.",
            self._collection.count(),
        )

    # ── Write ──────────────────────────────────────────────────

    def store_experiment(
        self,
        *,
        hypothesis_id: str,
        rationale: str,
        status: str,
        final_loss: float | None = None,
        metrics: Dict[str, Any] | None = None,
    ) -> None:
        """Embed and store a completed experiment."""
        metadata: Dict[str, Any] = {"status": status}
        if final_loss is not None:
            metadata["final_loss"] = final_loss
        if metrics:
            # ChromaDB metadata values must be str | int | float | bool
            for k, v in metrics.items():
                if isinstance(v, (str, int, float, bool)):
                    metadata[f"metric_{k}"] = v

        self._collection.upsert(
            ids=[hypothesis_id],
            documents=[rationale],
            metadatas=[metadata],
        )
        logger.info("Stored experiment %s in episodic memory.", hypothesis_id)

    # ── Read ───────────────────────────────────────────────────

    def find_similar(
        self,
        query_text: str,
        n_results: int = 5,
        max_distance: float = 0.35,
    ) -> List[Dict[str, Any]]:
        """Return past experiments whose rationale is semantically close.

        Parameters
        ----------
        query_text : str
            The new hypothesis rationale to compare against.
        n_results : int
            Maximum number of neighbours to return.
        max_distance : float
            Cosine distance threshold — lower means more similar.

        Returns
        -------
        list[dict]
            Each dict contains ``id``, ``document``, ``distance``, and ``metadata``.
        """
        if self._collection.count() == 0:
            return []

        results = self._collection.query(
            query_texts=[query_text],
            n_results=min(n_results, self._collection.count()),
        )

        hits: List[Dict[str, Any]] = []
        ids = results.get("ids", [[]])[0]
        docs = results.get("documents", [[]])[0]
        dists = results.get("distances", [[]])[0]
        metas = results.get("metadatas", [[]])[0]

        for i, doc_id in enumerate(ids):
            dist = dists[i] if i < len(dists) else 1.0
            if dist <= max_distance:
                hits.append(
                    {
                        "id": doc_id,
                        "document": docs[i] if i < len(docs) else "",
                        "distance": dist,
                        "metadata": metas[i] if i < len(metas) else {},
                    }
                )
        return hits

    def count(self) -> int:
        """Number of stored experiments."""
        return self._collection.count()
