"""Data Scientist Agent.

Post-execution analytics persona.  Receives telemetry (metrics dict from
run_training) and produces an ``ExperimentSummary`` with conclusions,
axioms, and recommendations.  Works across all AI domains.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional, Type

from uarh.agents.base import BaseAgent
from uarh.core.config import get_settings
from uarh.core.state import ExperimentSummary


class DataScientistAgent(BaseAgent[ExperimentSummary]):
    """The Data Scientist — evaluates experiment results from any AI domain."""

    def __init__(self, model: Optional[str] = None, temperature: float = 0.3) -> None:
        settings = get_settings()
        super().__init__(model=model or settings.fast_model, temperature=temperature)

    @property
    def agent_name(self) -> str:
        return "Data Scientist Agent"

    @property
    def output_schema(self) -> Type[ExperimentSummary]:
        return ExperimentSummary

    @property
    def system_prompt(self) -> str:
        return """You are a Senior AI Data Scientist in an autonomous research lab.

Your SOLE responsibility is to analyse the metrics from a completed experiment
and produce a rigorous ExperimentSummary.  You work across ALL AI domains.

ANALYSIS RULES:
1. Determine if the hypothesis SUCCEEDED or FAILED based on the target metric.
2. If failed, identify the specific failure mode:
   - "vanishing_gradients": grad norms collapse to near zero.
   - "exploding_gradients": grad norms spike or loss becomes NaN.
   - "overfitting": train metric improves but val metric degrades.
   - "underfitting": neither metric improves meaningfully.
   - "numerical_instability": sporadic NaN/Inf values.
   - "convergence_plateau": metric stops improving after few steps.
   - "environment_error": RL environment interaction failures.
   - "data_error": data loading or preprocessing failures.
   - "import_error": missing dependencies or incompatible versions.
   - "architecture_error": fundamental design flaw in the model/agent.
3. Extract DISTILLED AXIOMS — general rules learned that should guide
   future hypotheses.  These should be domain-aware, e.g.:
   - "For diffusion LMs: linear noise schedule works better than cosine at small scale."
   - "For RL: PPO with clipped objectives converges faster than vanilla PG on continuous action spaces."
4. Provide concrete recommendations for the next experiment cycle.
5. Respond ONLY with valid JSON matching the ExperimentSummary schema.
"""

    def build_user_message(self, state: Dict[str, Any]) -> str:
        proposal = state.get("proposal")
        proposal_data = {}
        if proposal:
            proposal_data = proposal if isinstance(proposal, dict) else proposal.model_dump()

        telemetry = state.get("telemetry", {})
        target_metric = state.get("target_metric", "unknown")
        exp_config = state.get("experiment_config", {})

        parts = [
            "## Experiment Details",
            f"**Target Metric:** {target_metric}",
            f"**Domain:** {proposal_data.get('domain', 'unknown')}",
            f"**Framework:** {proposal_data.get('framework', 'unknown')}",
            "",
        ]

        if proposal_data:
            parts.extend([
                "## Hypothesis Under Test",
                "```json",
                json.dumps(proposal_data, indent=2),
                "```",
                "",
            ])

        parts.extend([
            "## Execution Telemetry (metrics returned by run_training)",
            "```json",
            json.dumps(telemetry, indent=2, default=str),
            "```",
            "",
            "## Your Task",
            "Analyse these results and produce a rigorous ExperimentSummary.",
            "Respond with JSON matching this schema:",
            "",
            "```json",
            json.dumps(ExperimentSummary.model_json_schema(), indent=2),
            "```",
        ])
        return "\n".join(parts)
