"""Paper Writer Agent.

Takes the completed experiment context and writes a formal academic paper
in Markdown format.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional, Type

from pydantic import BaseModel, Field

from uarh.agents.base import BaseAgent
from uarh.core.config import get_settings


class AcademicPaper(BaseModel):
    """Output of the Paper Writer Agent."""

    title: str = Field(..., description="The title of the paper.")
    markdown_content: str = Field(
        ..., 
        description="The full academic paper in Markdown format, including Abstract, Introduction, Methodology, Results, and Conclusion."
    )


class PaperWriterAgent(BaseAgent[AcademicPaper]):
    """The Writer — documents successful research into academic papers."""

    def __init__(self, model: Optional[str] = None, temperature: float = 0.5) -> None:
        settings = get_settings()
        super().__init__(model=model or settings.elite_model, temperature=temperature)

    @property
    def agent_name(self) -> str:
        return "Paper Writer Agent"

    @property
    def output_schema(self) -> Type[AcademicPaper]:
        return AcademicPaper

    @property
    def system_prompt(self) -> str:
        return """You are an Academic AI Researcher writing a paper on a completed experiment.

Your SOLE responsibility is to take the provided research proposal, architectural blueprint, 
and experimental results, and synthesize them into a formal academic paper in Markdown format.

RULES:
1. The paper must include: Abstract, Introduction, Related Work, Methodology, Experiments, Results, and Conclusion.
2. Use a formal, objective, academic tone.
3. Incorporate the empirical metrics provided in the telemetry/summary.
4. If there were specific distilled axioms discovered, mention them as contributions.
5. Format mathematical formulations or code concepts clearly using Markdown.
6. Respond ONLY with valid JSON matching the AcademicPaper schema.
"""

    def build_user_message(self, state: Dict[str, Any]) -> str:
        proposal = state.get("proposal", {})
        blueprint = state.get("blueprint", {})
        summary = state.get("experiment_summary", {})
        telemetry = state.get("telemetry", {})

        parts = [
            "## Research Proposal",
            "```json",
            json.dumps(proposal, indent=2, default=str),
            "```",
            "",
            "## Architectural Blueprint",
            "```json",
            json.dumps(blueprint, indent=2, default=str),
            "```",
            "",
            "## Experimental Results & Summary",
            "```json",
            json.dumps(summary, indent=2, default=str),
            "```",
            "",
            "## Raw Telemetry",
            "```json",
            json.dumps(telemetry, indent=2, default=str),
            "```",
            "",
            "## Your Task",
            "Write a comprehensive academic paper based on this successful experiment.",
            "Respond with JSON matching the AcademicPaper schema."
        ]
        return "\n".join(parts)
