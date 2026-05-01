"""Principal Investigator (PI) Agent.

Sets the macro-goal and formulates new hypotheses across ANY AI domain.
Consults episodic memory to avoid redundant experiments, incorporates
distilled axioms, and outputs a strict ``ResearchProposal``.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Type

from uarh.agents.base import BaseAgent
from uarh.core.state import ResearchProposal
from uarh.memory.distillation import DistillationEngine


class PIAgent(BaseAgent[ResearchProposal]):
    """The Principal Investigator — hypothesis generator for any AI domain."""

    @property
    def agent_name(self) -> str:
        return "PI Agent"

    @property
    def output_schema(self) -> Type[ResearchProposal]:
        return ResearchProposal

    @property
    def system_prompt(self) -> str:
        return """You are the Principal Investigator (PI) of an autonomous AI research lab.

Your SOLE responsibility is to formulate ONE new, testable research hypothesis that
could improve the given target metric.  You are a world-class researcher who works
across ALL domains of AI:

DOMAINS YOU COVER:
- Language Modeling (transformers, RNNs, diffusion LMs, state-space models)
- Computer Vision (CNNs, ViTs, diffusion models, segmentation, detection)
- Reinforcement Learning (PPO, SAC, DQN, world models, model-based RL)
- Generative Models (GANs, VAEs, diffusion, flow-based)
- World Models (latent dynamics, dreamer architectures)
- Bio-ML, multi-modal, AutoML, and any other AI paradigm

FRAMEWORKS YOU CAN USE:
- pytorch, jax/flax, tensorflow, sklearn, stable-baselines3, gymnasium, or pure Python
- Choose the BEST framework for the specific hypothesis

RULES:
1. Each hypothesis must be NOVEL — do not repeat past experiments (listed below).
2. Each hypothesis must be SPECIFIC — name concrete architectural changes.
3. Each hypothesis must be TESTABLE — the outcome must be measurable via the target metric.
4. Choose the RIGHT framework for the task (don't default to PyTorch if another is better).
5. List ALL Python dependencies your proposed architecture requires.
6. Specify what kind of data/environment your experiment needs.
7. Respond ONLY with valid JSON matching the ResearchProposal schema.

Your output will be consumed programmatically by the next agent in the pipeline.
"""

    def build_user_message(self, state: Dict[str, Any]) -> str:
        target = state.get("target_metric", "improvement")
        domain = state.get("domain", "general_ai")
        axioms = state.get("axioms", [])
        similar = state.get("similar_past_hypotheses", [])
        exp_config = state.get("experiment_config", {})

        parts = [
            f"## Research Objective",
            f"**Target Metric:** {target}",
            f"**Domain:** {domain}",
            "",
        ]

        # Inject experiment configuration
        if exp_config:
            parts.append("## Experiment Environment")
            if exp_config.get("dataset_path"):
                parts.append(f"**Dataset:** `{exp_config['dataset_path']}`")
            if exp_config.get("environment_name"):
                parts.append(f"**RL Environment:** `{exp_config['environment_name']}`")
            if exp_config.get("max_params"):
                parts.append(f"**Max Parameters:** {exp_config['max_params']:,}")
            if exp_config.get("hardware"):
                parts.append(f"**Hardware:** {exp_config['hardware']}")
            if exp_config.get("extra_context"):
                parts.append(f"**Context:** {exp_config['extra_context']}")
            parts.append("")

        # Inject distilled axioms
        if axioms:
            axiom_block = DistillationEngine.format_for_prompt(axioms)
            parts.append("## Distilled Knowledge (MUST respect these rules)")
            parts.append(axiom_block)
            parts.append("")

        # Inject past experiment warnings
        if similar:
            parts.append("## Past Experiments (DO NOT repeat these)")
            for s in similar:
                parts.append(f"  - {s}")
            parts.append("")

        parts.append("## Your Task")
        parts.append(
            "Formulate a single, novel research hypothesis.  "
            "Choose the best framework and domain for the objective.  "
            "Respond ONLY with a JSON object matching the ResearchProposal schema."
        )
        parts.append("")
        parts.append("```json")
        parts.append(json.dumps(ResearchProposal.model_json_schema(), indent=2))
        parts.append("```")

        return "\n".join(parts)
