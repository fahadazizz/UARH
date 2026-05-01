"""Theorist & Mathematician Agent.

Translates a ``ResearchProposal`` into a formal ``ArchitecturalBlueprint``
with precise specifications — framework-agnostic.  Works for neural nets,
RL algorithms, classical ML pipelines, or any other AI method.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Type

from uarh.agents.base import BaseAgent
from uarh.core.state import ArchitecturalBlueprint


class TheoristAgent(BaseAgent[ArchitecturalBlueprint]):
    """The Theorist — formalises hypotheses into architectural blueprints."""

    @property
    def agent_name(self) -> str:
        return "Theorist Agent"

    @property
    def output_schema(self) -> Type[ArchitecturalBlueprint]:
        return ArchitecturalBlueprint

    @property
    def system_prompt(self) -> str:
        return """You are a Theoretical AI Researcher in an autonomous research lab.

Your SOLE responsibility is to translate a research proposal into a precise
architectural blueprint that a Software Engineer can implement.  You work across
ALL AI domains and frameworks.

RULES:
1. Specify key data structures and their shapes using symbolic notation:
   - For neural nets: [B, S, E], [B, C, H, W], etc.
   - For RL: observation_shape, action_shape, reward structure
   - For classical ML: feature dimensions, target dimensions
2. Your "forward_pass_pseudocode" should describe the core algorithm step-by-step.
3. Your "training_loop_pseudocode" should describe the full training procedure:
   - For supervised learning: data loading → forward → loss → backward → step
   - For RL: env.reset → action → env.step → buffer → update → repeat
   - For GANs: generator step → discriminator step → alternating
   - For diffusion: noise schedule → denoising steps → loss
4. Define the loss/objective function precisely.
5. List every sub-component that must be implemented (modules, policies, critics, etc.).
6. If the proposal seems unsound, note concerns in the "constraints" field.
7. Respond ONLY with valid JSON matching the ArchitecturalBlueprint schema.

Your output will be PROGRAMMATICALLY VALIDATED for completeness and then handed
to a Software Architect who will write executable code.
"""

    def build_user_message(self, state: Dict[str, Any]) -> str:
        proposal = state.get("proposal")
        if proposal is None:
            raise ValueError("TheoristAgent requires a ResearchProposal in state.")

        # If this is a revision, include the previous error
        revision_note = ""
        theorist_revisions = state.get("theorist_revision_count", 0)
        latest_error = state.get("latest_error")
        if theorist_revisions > 0 and latest_error:
            revision_note = (
                f"\n\n## ⚠️ REVISION REQUEST (attempt {theorist_revisions + 1})\n"
                f"Your previous blueprint FAILED validation:\n"
                f"```\n{latest_error}\n```\n"
                f"Fix the identified issues.\n"
            )

        proposal_data = proposal if isinstance(proposal, dict) else proposal.model_dump()

        parts = [
            "## Research Proposal to Formalise",
            "```json",
            json.dumps(proposal_data, indent=2),
            "```",
            revision_note,
            "",
            "## Your Task",
            "Translate this proposal into a precise ArchitecturalBlueprint.",
            "Include BOTH forward_pass_pseudocode AND training_loop_pseudocode.",
            "Respond ONLY with a JSON object matching this schema:",
            "",
            "```json",
            json.dumps(ArchitecturalBlueprint.model_json_schema(), indent=2),
            "```",
        ]
        return "\n".join(parts)
