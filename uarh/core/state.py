"""Pydantic output schemas and the LangGraph HarnessState definition.

Every LLM agent returns a strict Pydantic model.  The harness graph
operates on ``HarnessState`` — a ``TypedDict`` that flows through every
node, carrying the complete execution context for a single research cycle.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, TypedDict

from pydantic import BaseModel, Field


# ═══════════════════════════════════════════════════════════════
#  Enums
# ═══════════════════════════════════════════════════════════════


class ValidationLevel(int, Enum):
    """Tiered sandbox levels — ordered by cost."""

    STATIC = 0
    SMOKE = 1       # renamed from SHAPE — now it's "can it instantiate?"
    MICRO_TRAIN = 2
    FULL_RUN = 3


class HypothesisStatus(str, Enum):
    """Lifecycle states for a single hypothesis."""

    PROPOSED = "proposed"
    FORMALIZED = "formalized"
    SYNTHESIZED = "synthesized"
    VALIDATING = "validating"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    ABORTED = "aborted"


# ═══════════════════════════════════════════════════════════════
#  Pydantic Agent Output Models
# ═══════════════════════════════════════════════════════════════


class ResearchProposal(BaseModel):
    """Output of the PI Agent — a single, testable research hypothesis."""

    hypothesis_id: str = Field(
        default_factory=lambda: f"hyp-{uuid.uuid4().hex[:8]}",
        description="Unique hypothesis identifier.",
    )
    title: str = Field(
        ...,
        description="One-line summary of the hypothesis.",
    )
    rationale: str = Field(
        ...,
        description="Detailed scientific reasoning explaining *why* this might improve the target metric.",
    )
    domain: str = Field(
        ...,
        description="AI research domain: 'language_modeling', 'reinforcement_learning', 'computer_vision', 'world_models', 'generative_models', 'bio_ml', 'multi_modal', 'other'.",
    )
    framework: str = Field(
        ...,
        description="Primary ML framework to use: 'pytorch', 'jax', 'tensorflow', 'sklearn', 'stable_baselines3', 'pure_python', or others.",
    )
    target_architecture: str = Field(
        ...,
        description="High-level name of the architecture or technique (e.g. 'Diffusion Transformer', 'PPO with LSTM policy', 'Vision Transformer').",
    )
    proposed_changes: List[str] = Field(
        ...,
        description="Bullet-list of concrete modifications or innovations.",
    )
    dependencies: List[str] = Field(
        default_factory=lambda: ["torch"],
        description="Python package names required (e.g. ['torch', 'torchvision'] or ['jax', 'flax'] or ['stable-baselines3', 'gymnasium']).",
    )
    expected_metric_improvement: str = Field(
        ...,
        description="Qualitative or quantitative prediction.",
    )
    data_requirements: str = Field(
        default="",
        description="What kind of data this experiment needs: 'text_file', 'image_directory', 'gym_environment:CartPole-v1', 'synthetic', etc.",
    )


class TensorShapeSpec(BaseModel):
    """A single named tensor/data structure with its expected shape."""

    name: str = Field(..., description="Name, e.g. 'encoder_output' or 'observation'.")
    shape: str = Field(
        ...,
        description="Symbolic shape string, e.g. '[B, S, E]' or '[B, obs_dim]' or '[B, C, H, W]'.",
    )
    dtype: str = Field(default="float32", description="Data-type string.")


class ArchitecturalBlueprint(BaseModel):
    """Output of the Theorist Agent — a formalised architectural blueprint."""

    mathematical_formalization: str = Field(
        ...,
        description="Mathematical or algorithmic description of the architecture/method.",
    )
    tensor_shapes: List[TensorShapeSpec] = Field(
        ...,
        description="Key data tensors/structures with their expected shapes.",
    )
    component_breakdown: List[str] = Field(
        ...,
        description="Ordered list of sub-components/modules that must be implemented.",
    )
    forward_pass_pseudocode: str = Field(
        ...,
        description="Step-by-step pseudocode of the forward pass / main algorithm loop.",
    )
    loss_function: str = Field(
        ...,
        description="Loss/objective function specification.",
    )
    training_loop_pseudocode: str = Field(
        default="",
        description="Pseudocode for the training procedure (important for RL, GAN, diffusion, etc. where loops are non-standard).",
    )
    constraints: List[str] = Field(
        default_factory=list,
        description="Hard constraints (max params, memory budget, specific API requirements, etc.).",
    )


class SynthesizedCode(BaseModel):
    """Output of the Software Architect Agent — executable Python source."""

    source_code: str = Field(
        ...,
        description="Complete, self-contained Python source implementing the blueprint.",
    )
    entrypoint_function: str = Field(
        default="create_model",
        description="Name of the factory function (e.g. 'create_model', 'create_agent').",
    )
    model_class_name: str = Field(
        default="Model",
        description="Name of the primary class defined in source_code.",
    )
    estimated_param_count: Optional[str] = Field(
        default=None,
        description="Rough parameter count estimate (if applicable).",
    )


class ImplementationBlocker(BaseModel):
    """Returned by the Architect when a Blueprint is unimplementable."""

    reason: str = Field(
        ...,
        description="Technical explanation of why the blueprint cannot be realised.",
    )
    suggestions: List[str] = Field(
        ...,
        description="Concrete suggestions for the Theorist to revise the blueprint.",
    )


class DebugPatch(BaseModel):
    """Output of the Debug Agent — a corrected version of failing code."""

    patched_code: str = Field(
        ...,
        description="The complete, corrected Python source.",
    )
    diagnosis: str = Field(
        ...,
        description="Root-cause analysis of the failure.",
    )
    changes_made: List[str] = Field(
        ...,
        description="Bullet list of changes applied.",
    )


class ExperimentSummary(BaseModel):
    """Output of the Data Scientist Agent — post-execution analysis."""

    conclusion: str = Field(
        ...,
        description="Natural-language summary of experimental outcome.",
    )
    successful: bool = Field(
        ...,
        description="Whether the hypothesis improved the target metric.",
    )
    key_metrics: Dict[str, Any] = Field(
        default_factory=dict,
        description="Final metric values (loss, accuracy, reward, perplexity, etc.).",
    )
    failure_mode: Optional[str] = Field(
        default=None,
        description="If failed, the identified failure mode.",
    )
    new_axioms_discovered: List[str] = Field(
        default_factory=list,
        description="Distilled rules inferred from this experiment.",
    )
    recommendations: List[str] = Field(
        default_factory=list,
        description="Suggestions for the next research cycle.",
    )


# ═══════════════════════════════════════════════════════════════
#  Validation Result (returned by sandbox levels)
# ═══════════════════════════════════════════════════════════════


class ValidationResult(BaseModel):
    """Standard result from any sandbox validation level."""

    level: ValidationLevel
    passed: bool
    error_message: Optional[str] = None
    traceback: Optional[str] = None
    metrics: Dict[str, Any] = Field(default_factory=dict)
    duration_seconds: float = 0.0


# ═══════════════════════════════════════════════════════════════
#  Experiment Configuration (passed from CLI into the pipeline)
# ═══════════════════════════════════════════════════════════════


class ExperimentConfig(BaseModel):
    """User-provided context about the research environment."""

    dataset_path: Optional[str] = Field(
        default=None,
        description="Path to a dataset file or directory (e.g. '/data/input.txt', '/data/images/').",
    )
    environment_name: Optional[str] = Field(
        default=None,
        description="RL environment name (e.g. 'CartPole-v1', 'HalfCheetah-v4').",
    )
    max_params: Optional[int] = Field(
        default=None,
        description="Maximum parameter budget (e.g. 10_000_000 for 10M).",
    )
    hardware: str = Field(
        default="cpu",
        description="Available hardware: 'cpu', 'cuda', 'mps'.",
    )
    output_dir: Optional[str] = Field(
        default=None,
        description="Directory to save trained models, visualizations, and artifacts.",
    )
    extra_context: str = Field(
        default="",
        description="Any additional context the user wants the PI to know.",
    )


# ═══════════════════════════════════════════════════════════════
#  LangGraph Harness State (TypedDict for LangGraph compatibility)
# ═══════════════════════════════════════════════════════════════


class HarnessState(TypedDict, total=False):
    """The single state object flowing through the LangGraph DCG.

    All values are plain Python types (dicts, strings, ints) because
    LangGraph merges node outputs via dict update.
    """

    # ── Identity
    run_id: str
    created_at: str

    # ── Research Objective
    target_metric: str
    domain: str

    # ── Experiment Configuration (from user)
    experiment_config: Dict[str, Any]

    # ── Axiom Store (permanent prompt injection)
    axioms: List[str]

    # ── Agent Outputs (populated as graph progresses) — all as dicts
    proposal: Optional[Dict[str, Any]]
    blueprint: Optional[Dict[str, Any]]
    synthesized_code: Optional[Dict[str, Any]]
    implementation_blocker: Optional[Dict[str, Any]]
    debug_patch: Optional[Dict[str, Any]]
    experiment_summary: Optional[Dict[str, Any]]
    paper: Optional[Dict[str, Any]]

    # ── Live Code (mutable — updated by Architect & Debugger)
    code: Optional[str]

    # ── Validation Tracking
    current_level: int
    latest_validation: Optional[Dict[str, Any]]
    latest_error: Optional[str]

    # ── Telemetry (populated by Level 2 run_training output)
    telemetry: Dict[str, Any]

    # ── Governor / Control-Flow Counters
    hypothesis_status: str
    debug_retry_count: int
    theorist_revision_count: int
    consecutive_failure_count: int

    # ── Episodic Memory References
    similar_past_hypotheses: List[str]
