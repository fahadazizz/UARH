"""Semantic memory — NetworkX-based knowledge graph for ML concepts.

Instead of a heavyweight graph database (Neo4j), concepts and their
relationships are stored in-process as a NetworkX DiGraph and
serialised to JSON on disk.  The Data Scientist Agent feeds new
entities after each experiment.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx
from networkx.readwrite import json_graph

from uarh.core.config import get_settings

logger = logging.getLogger(__name__)


class SemanticMemory:
    """Persistent in-memory knowledge graph for ML concept traversal."""

    def __init__(self) -> None:
        settings = get_settings()
        self._path: Path = settings.semantic_graph_path
        self._graph: nx.DiGraph = self._load()
        logger.info(
            "SemanticMemory initialised — %d nodes, %d edges.",
            self._graph.number_of_nodes(),
            self._graph.number_of_edges(),
        )

    # ── Persistence ────────────────────────────────────────────

    def _load(self) -> nx.DiGraph:
        """Load from JSON or return an empty graph."""
        if self._path.exists():
            try:
                with open(self._path, "r") as f:
                    data = json.load(f)
                return json_graph.node_link_graph(data, directed=True)
            except (json.JSONDecodeError, Exception) as exc:
                logger.warning("Corrupt semantic graph file, starting fresh: %s", exc)
        return nx.DiGraph()

    def save(self) -> None:
        """Flush the graph to disk as JSON."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = json_graph.node_link_data(self._graph)
        with open(self._path, "w") as f:
            json.dump(data, f, indent=2)
        logger.debug("SemanticMemory saved — %d nodes.", self._graph.number_of_nodes())

    # ── Write ──────────────────────────────────────────────────

    def add_concept(self, concept: str, **attributes: Any) -> None:
        """Add or update a concept node."""
        normalised = concept.strip().lower()
        self._graph.add_node(normalised, label=concept, **attributes)

    def add_relationship(
        self,
        source: str,
        target: str,
        relation: str,
        **attributes: Any,
    ) -> None:
        """Add a directed edge between two concepts.

        Both concepts are created if they don't exist yet.
        """
        src = source.strip().lower()
        tgt = target.strip().lower()
        if src not in self._graph:
            self.add_concept(source)
        if tgt not in self._graph:
            self.add_concept(target)
        self._graph.add_edge(src, tgt, relation=relation, **attributes)

    def add_triplets(self, triplets: List[Tuple[str, str, str]]) -> None:
        """Bulk-add (source, relation, target) triplets and persist."""
        for src, rel, tgt in triplets:
            self.add_relationship(src, tgt, rel)
        self.save()

    # ── Read ───────────────────────────────────────────────────

    def get_neighbours(self, concept: str, depth: int = 1) -> Dict[str, Any]:
        """Return concepts reachable within *depth* hops.

        Returns a dict mapping concept names to their edge data.
        """
        normalised = concept.strip().lower()
        if normalised not in self._graph:
            return {}

        visited: Set[str] = set()
        frontier: Set[str] = {normalised}
        result: Dict[str, Any] = {}

        for _ in range(depth):
            next_frontier: Set[str] = set()
            for node in frontier:
                if node in visited:
                    continue
                visited.add(node)
                for _, neighbour, edge_data in self._graph.edges(node, data=True):
                    result[neighbour] = edge_data
                    if neighbour not in visited:
                        next_frontier.add(neighbour)
            frontier = next_frontier

        return result

    def get_concept_context(self, concept: str) -> str:
        """Return a natural-language description of a concept's neighbourhood.

        Useful for injecting into agent prompts.
        """
        neighbours = self.get_neighbours(concept, depth=2)
        if not neighbours:
            return f"No semantic context found for '{concept}'."
        lines = [f"Semantic context for '{concept}':"]
        for nbr, data in neighbours.items():
            rel = data.get("relation", "related_to")
            label = self._graph.nodes[nbr].get("label", nbr)
            lines.append(f"  - {concept} --[{rel}]--> {label}")
        return "\n".join(lines)

    def list_concepts(self) -> List[str]:
        """Return all concept labels."""
        return [
            self._graph.nodes[n].get("label", n)
            for n in self._graph.nodes()
        ]

    @property
    def stats(self) -> Dict[str, int]:
        return {
            "nodes": self._graph.number_of_nodes(),
            "edges": self._graph.number_of_edges(),
        }
