"""Software Architect & Synthesis Agent.

Implements an ``ArchitecturalBlueprint`` as executable Python code in
whatever framework the proposal specifies.  The generated code MUST be
completely self-contained — including its own data loading, training loop,
and metric reporting.  The harness sandbox does NOT provide any training
infrastructure; it only calls `create_model()` and `run_training()`.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Type

from pydantic import BaseModel

from uarh.agents.base import BaseAgent
from uarh.core.state import ImplementationBlocker, SynthesizedCode


class ArchitectOutput(BaseModel):
    """Union wrapper — the Architect returns either code or a blocker."""

    synthesized_code: SynthesizedCode | None = None
    implementation_blocker: ImplementationBlocker | None = None


class ArchitectAgent(BaseAgent[ArchitectOutput]):
    """The Software Architect — translates blueprints into executable code."""

    @property
    def agent_name(self) -> str:
        return "Architect Agent"

    @property
    def output_schema(self) -> Type[ArchitectOutput]:
        return ArchitectOutput

    @property
    def system_prompt(self) -> str:
        return """You are a Senior AI Software Architect in an autonomous research lab.

Your SOLE responsibility is to implement an ArchitecturalBlueprint as a complete,
self-contained, executable Python module.  You work across ALL frameworks:
PyTorch, JAX/Flax, TensorFlow, scikit-learn, Stable-Baselines3, or pure Python.

THE CODE YOU WRITE MUST BE FULLY SELF-CONTAINED. This means:

1. ALL imports at the top of the file.
2. A `create_model(**kwargs)` factory function that returns the model/agent/pipeline.
3. A `run_training(model, config, **kwargs)` function that:
   a. Loads/creates its own data (from config["dataset_path"], config["environment_name"], or synthetic data).
   b. Runs its own complete training/evaluation loop.
   c. Returns a dict of metrics: {"final_loss": ..., "final_metric": ..., ...}
4. The `run_training` function receives a config dict with:
   - "dataset_path": str or None — path to a data file/directory
   - "environment_name": str or None — RL environment name
   - "device": str — "cpu", "cuda", or "mps"
   - "max_steps": int — maximum training steps
   - "learning_rate": float
5. The module must be runnable as: `model = create_model(); metrics = run_training(model, config)`

CRITICAL RULES:
- The sandbox provides NOTHING — no data loaders, no training loops, no optimizers.
- Your code handles EVERYTHING internally.
- For text data: read the file, tokenize (character-level is fine), create batches.
- For RL: create the environment, run episodes, collect transitions, update.
- For vision: load images, preprocess, create batches.
- Use standard library patterns (no custom CUDA kernels).
- If the blueprint is FUNDAMENTALLY unimplementable, set "implementation_blocker".
- Respond ONLY with valid JSON.
"""

    def build_user_message(self, state: Dict[str, Any]) -> str:
        blueprint = state.get("blueprint")
        if blueprint is None:
            raise ValueError("ArchitectAgent requires an ArchitecturalBlueprint in state.")

        blueprint_data = blueprint if isinstance(blueprint, dict) else blueprint.model_dump()
        proposal = state.get("proposal")
        proposal_data = {}
        if proposal:
            proposal_data = proposal if isinstance(proposal, dict) else proposal.model_dump()

        exp_config = state.get("experiment_config", {})

        parts = [
            "## Architectural Blueprint to Implement",
            "```json",
            json.dumps(blueprint_data, indent=2),
            "```",
            "",
        ]

        if proposal_data:
            parts.extend([
                "## Original Research Proposal (for context)",
                f"**Framework:** {proposal_data.get('framework', 'pytorch')}",
                f"**Domain:** {proposal_data.get('domain', 'unknown')}",
                f"**Dependencies:** {proposal_data.get('dependencies', [])}",
                "```json",
                json.dumps(proposal_data, indent=2),
                "```",
                "",
            ])

        if exp_config:
            parts.extend([
                "## Experiment Configuration (passed to run_training as config dict)",
                "```json",
                json.dumps(exp_config, indent=2),
                "```",
                "",
            ])

        # If previous architect attempt failed
        latest_error = state.get("latest_error")
        if latest_error and state.get("hypothesis_status") == "synthesized":
            parts.extend([
                "## ⚠️ Previous Code Failed Validation",
                f"```\n{latest_error}\n```",
                "Fix the issues. If the blueprint itself is wrong, set implementation_blocker.",
                "",
            ])

        parts.extend([
            "## Your Task",
            "Implement this blueprint as a complete, self-contained, runnable Python module.",
            "The module MUST export `create_model()` and `run_training(model, config)`.",
            "`run_training` must handle its OWN data loading and training loop.",
            "Respond with JSON matching this schema:",
            "",
            "```json",
            json.dumps(ArchitectOutput.model_json_schema(), indent=2),
            "```",
        ])
        return "\n".join(parts)
