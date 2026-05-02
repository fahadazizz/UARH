"""The Harness Engine — LangGraph Directed Cyclic Graph orchestration.

This is the central nervous system of UARH.  It defines every node
(agent invocations + sandbox executions) and every conditional edge
(including backward edges for cyclic feedback).

Graph Topology
--------------
START → PI → Theorist → MathValidator ──┬── (pass) → Architect ──┬── (code) → L0 → L1 → L2 → Scientist → END
                  ↑                     │                        │
                  └── (fail, ≤ max) ────┘                        │
                  ↑                                              │
                  └────────── (blocker) ─────────────────────────┘

L0/L1/L2 on fail → Debugger → (retry ≤ max) → back to failed level
                             → (retry > max) → ABORT → END
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from langgraph.graph import END, StateGraph

from uarh.agents.architect import ArchitectAgent
from uarh.agents.debug import DebugAgent
from uarh.agents.pi import PIAgent
from uarh.agents.scientist import DataScientistAgent
from uarh.agents.theorist import TheoristAgent
from uarh.agents.writer import PaperWriterAgent
from uarh.core.config import get_settings
from uarh.core.state import (
    HarnessState,
    HypothesisStatus,
    ValidationLevel,
    ValidationResult,
)
from uarh.execution.level0_static import run_level0
from uarh.execution.level1_shape import run_level1
from uarh.execution.level2_train import run_level2
from uarh.memory.distillation import DistillationEngine
from uarh.memory.episodic import EpisodicMemory
from uarh.memory.lineage import LineageRepository
from uarh.memory.semantic import SemanticMemory

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
#  Node functions
# ═══════════════════════════════════════════════════════════════


def node_pi(state: Dict[str, Any]) -> Dict[str, Any]:
    """PI Agent: generate a new ResearchProposal."""
    logger.info("━━━ NODE: Principal Investigator ━━━")

    # Query episodic memory for similar past experiments
    try:
        episodic = EpisodicMemory()
        target = state.get("target_metric", "")
        similar = episodic.find_similar(target, n_results=5)
        past_hypotheses = [h["document"] for h in similar]
    except Exception as exc:
        logger.warning("Episodic memory lookup failed: %s", exc)
        past_hypotheses = []

    state_for_agent = dict(state)
    state_for_agent["similar_past_hypotheses"] = past_hypotheses

    agent = PIAgent()
    proposal = agent.invoke(state_for_agent)

    return {
        "proposal": proposal.model_dump(),
        "hypothesis_status": HypothesisStatus.PROPOSED.value,
        "latest_error": None,
        "debug_retry_count": 0,
        "theorist_revision_count": 0,
    }


def node_theorist(state: Dict[str, Any]) -> Dict[str, Any]:
    """Theorist Agent: formalise a proposal into an ArchitecturalBlueprint."""
    logger.info("━━━ NODE: Theorist ━━━")

    agent = TheoristAgent()
    blueprint = agent.invoke(state)

    return {
        "blueprint": blueprint.model_dump(),
        "hypothesis_status": HypothesisStatus.FORMALIZED.value,
        "latest_error": None,
    }


def node_math_validator(state: Dict[str, Any]) -> Dict[str, Any]:
    """Validate the blueprint for completeness.

    Checks that the Theorist provided all required fields.  True
    programmatic validation happens at Level 1 after code synthesis.
    """
    logger.info("━━━ NODE: Blueprint Validator ━━━")

    blueprint = state.get("blueprint", {})
    tensor_shapes = blueprint.get("tensor_shapes", [])
    components = blueprint.get("component_breakdown", [])
    pseudocode = blueprint.get("forward_pass_pseudocode", "")

    errors = []

    if not tensor_shapes:
        errors.append("No tensor/data shapes defined in blueprint.")
    if not components:
        errors.append("No component breakdown provided.")
    if not pseudocode:
        errors.append("No forward pass / algorithm pseudocode provided.")

    # Validate each tensor shape spec has required fields
    for i, spec in enumerate(tensor_shapes):
        if isinstance(spec, dict):
            if not spec.get("name"):
                errors.append(f"Shape spec #{i} missing 'name'.")
            if not spec.get("shape"):
                errors.append(f"Shape spec #{i} missing 'shape'.")
        else:
            errors.append(f"Shape spec #{i} is not a dict: {spec}")

    if errors:
        error_text = "\n".join(errors)
        logger.warning("Blueprint validation FAILED:\n%s", error_text)
        return {
            "latest_error": error_text,
            "latest_validation": ValidationResult(
                level=ValidationLevel.SMOKE,
                passed=False,
                error_message=error_text,
            ).model_dump(),
        }

    logger.info("Blueprint validation PASSED — blueprint is well-formed.")
    return {
        "latest_error": None,
        "latest_validation": ValidationResult(
            level=ValidationLevel.SMOKE,
            passed=True,
        ).model_dump(),
    }


def node_architect(state: Dict[str, Any]) -> Dict[str, Any]:
    """Software Architect Agent: synthesise code from the blueprint."""
    logger.info("━━━ NODE: Software Architect ━━━")

    agent = ArchitectAgent()
    output = agent.invoke(state)

    # Check if the architect returned a blocker
    if output.implementation_blocker and not output.synthesized_code:
        blocker = output.implementation_blocker
        logger.warning("Architect returned BLOCKER: %s", blocker.reason)
        return {
            "implementation_blocker": blocker.model_dump(),
            "latest_error": blocker.reason,
            "hypothesis_status": HypothesisStatus.VALIDATING.value,
        }

    if output.synthesized_code:
        synth = output.synthesized_code
        logger.info(
            "Code synthesised — entry: %s, class: %s",
            synth.entrypoint_function,
            synth.model_class_name,
        )
        return {
            "synthesized_code": synth.model_dump(),
            "code": synth.source_code,
            "implementation_blocker": None,
            "hypothesis_status": HypothesisStatus.SYNTHESIZED.value,
            "current_level": ValidationLevel.STATIC.value,
            "debug_retry_count": 0,
        }

    raise RuntimeError("Architect returned neither code nor blocker.")


def node_level0(state: Dict[str, Any]) -> Dict[str, Any]:
    """Level 0 sandbox: AST + Ruff static checks."""
    logger.info("━━━ NODE: Level 0 (Static) ━━━")

    code = state.get("code", "")
    result = run_level0(code)

    return {
        "latest_validation": result.model_dump(),
        "current_level": ValidationLevel.STATIC.value,
        "latest_error": result.error_message if not result.passed else None,
    }


def node_level1(state: Dict[str, Any]) -> Dict[str, Any]:
    """Level 1 sandbox: smoke test — import + instantiate."""
    logger.info("━━━ NODE: Level 1 (Smoke Test) ━━━")

    code = state.get("code", "")
    synth = state.get("synthesized_code", {})
    entrypoint = synth.get("entrypoint_function", "create_model") if isinstance(synth, dict) else "create_model"

    # Get dependencies from proposal
    proposal = state.get("proposal", {})
    dependencies = proposal.get("dependencies", []) if isinstance(proposal, dict) else []

    result = run_level1(
        code,
        entrypoint_function=entrypoint,
        dependencies=dependencies,
    )

    update: Dict[str, Any] = {
        "latest_validation": result.model_dump(),
        "current_level": ValidationLevel.SMOKE.value,
        "latest_error": result.error_message if not result.passed else None,
    }
    if result.metrics:
        update["telemetry"] = {**state.get("telemetry", {}), "level1": result.metrics}

    return update


def node_level2(state: Dict[str, Any]) -> Dict[str, Any]:
    """Level 2 sandbox: micro-train — call run_training()."""
    logger.info("━━━ NODE: Level 2 (Micro-Train) ━━━")

    code = state.get("code", "")
    synth = state.get("synthesized_code", {})
    entrypoint = synth.get("entrypoint_function", "create_model") if isinstance(synth, dict) else "create_model"

    # Get dependencies from proposal
    proposal = state.get("proposal", {})
    dependencies = proposal.get("dependencies", []) if isinstance(proposal, dict) else []

    # Pass experiment config through to run_training
    experiment_config = state.get("experiment_config", {})

    result = run_level2(
        code,
        entrypoint_function=entrypoint,
        experiment_config=experiment_config,
        dependencies=dependencies,
    )

    update: Dict[str, Any] = {
        "latest_validation": result.model_dump(),
        "current_level": ValidationLevel.MICRO_TRAIN.value,
        "latest_error": result.error_message if not result.passed else None,
    }
    if result.metrics:
        update["telemetry"] = {**state.get("telemetry", {}), "level2": result.metrics}

    return update


def node_debugger(state: Dict[str, Any]) -> Dict[str, Any]:
    """Debug Agent: patch failing code at any validation level."""
    logger.info("━━━ NODE: Debug Agent (retry %d) ━━━", state.get("debug_retry_count", 0) + 1)

    agent = DebugAgent()
    patch = agent.invoke(state)

    return {
        "debug_patch": patch.model_dump(),
        "code": patch.patched_code,
        "debug_retry_count": state.get("debug_retry_count", 0) + 1,
    }


def node_scientist(state: Dict[str, Any]) -> Dict[str, Any]:
    """Data Scientist Agent: post-training analysis + distillation."""
    logger.info("━━━ NODE: Data Scientist ━━━")

    agent = DataScientistAgent()
    summary = agent.invoke(state)

    # Determine final status
    status = HypothesisStatus.SUCCEEDED if summary.successful else HypothesisStatus.FAILED

    # Persist to episodic memory
    try:
        episodic = EpisodicMemory()
        proposal = state.get("proposal", {})
        rationale = proposal.get("rationale", "") if isinstance(proposal, dict) else ""
        hyp_id = proposal.get("hypothesis_id", state.get("run_id", "unknown")) if isinstance(proposal, dict) else "unknown"
        episodic.store_experiment(
            hypothesis_id=hyp_id,
            rationale=rationale,
            status=status.value,
            final_loss=summary.key_metrics.get("final_loss"),
            metrics=summary.key_metrics,
        )
    except Exception as exc:
        logger.warning("Failed to store in episodic memory: %s", exc)

    # Persist to lineage DB
    try:
        lineage = LineageRepository()
        proposal = state.get("proposal", {})
        hyp_id = proposal.get("hypothesis_id", "unknown") if isinstance(proposal, dict) else "unknown"
        hyp_title = proposal.get("title", "") if isinstance(proposal, dict) else ""
        lineage.record_execution(
            run_id=state.get("run_id", "unknown"),
            hypothesis_id=hyp_id,
            hypothesis_title=hyp_title,
            status=status.value,
            validation_level_reached=state.get("current_level", 0),
            final_loss=summary.key_metrics.get("final_loss"),
            metrics=summary.key_metrics,
        )
    except Exception as exc:
        logger.warning("Failed to record in lineage DB: %s", exc)

    # Run distillation
    try:
        distiller = DistillationEngine()
        proposal = state.get("proposal", {})
        hyp_id = proposal.get("hypothesis_id", None) if isinstance(proposal, dict) else None
        distiller.ingest_axioms(summary.new_axioms_discovered, source_hypothesis_id=hyp_id)
    except Exception as exc:
        logger.warning("Distillation failed: %s", exc)

    # Update semantic memory with learned concepts
    try:
        semantic = SemanticMemory()
        proposal = state.get("proposal", {})
        arch = proposal.get("target_architecture", "") if isinstance(proposal, dict) else ""
        if arch and summary.conclusion:
            semantic.add_concept(arch, status=status.value)
            if summary.failure_mode:
                semantic.add_relationship(arch, summary.failure_mode, "exhibited")
            for axiom in summary.new_axioms_discovered:
                semantic.add_relationship(arch, axiom[:50], "produced_axiom")
            semantic.save()
    except Exception as exc:
        logger.warning("Semantic memory update failed: %s", exc)

    # Update consecutive failure counter
    consec = state.get("consecutive_failure_count", 0)
    if summary.successful:
        consec = 0
    else:
        consec += 1

    return {
        "experiment_summary": summary.model_dump(),
        "hypothesis_status": status.value,
        "consecutive_failure_count": consec,
    }


def node_writer(state: Dict[str, Any]) -> Dict[str, Any]:
    """Paper Writer Agent: document the successful experiment."""
    logger.info("━━━ NODE: Paper Writer ━━━")
    agent = PaperWriterAgent()
    paper = agent.invoke(state)
    return {
        "paper": paper.model_dump(),
    }


def node_abort(state: Dict[str, Any]) -> Dict[str, Any]:
    """Abort node — triggered when retry limits are exceeded."""
    logger.error("━━━ NODE: ABORT ━━━ — hypothesis abandoned.")

    # Record the abort in lineage
    try:
        lineage = LineageRepository()
        proposal = state.get("proposal", {})
        hyp_id = proposal.get("hypothesis_id", "unknown") if isinstance(proposal, dict) else "unknown"
        hyp_title = proposal.get("title", "") if isinstance(proposal, dict) else ""
        lineage.record_execution(
            run_id=state.get("run_id", "unknown"),
            hypothesis_id=hyp_id,
            hypothesis_title=hyp_title,
            status=HypothesisStatus.ABORTED.value,
            validation_level_reached=state.get("current_level", 0),
            error_summary=state.get("latest_error", "Max retries exceeded"),
        )
    except Exception as exc:
        logger.warning("Failed to record abort: %s", exc)

    return {
        "hypothesis_status": HypothesisStatus.ABORTED.value,
    }


# ═══════════════════════════════════════════════════════════════
#  Conditional edge functions
# ═══════════════════════════════════════════════════════════════


def route_after_math_validator(state: Dict[str, Any]) -> str:
    """After MathValidator: pass → architect, fail → theorist (cyclic)."""
    settings = get_settings()
    validation = state.get("latest_validation", {})
    passed = validation.get("passed", False) if isinstance(validation, dict) else False

    if passed:
        return "architect"

    revision_count = state.get("theorist_revision_count", 0)
    if revision_count >= settings.max_theorist_revisions:
        logger.error("Theorist revision limit reached (%d). Aborting.", revision_count)
        return "abort"

    return "theorist_revision"


def route_after_architect(state: Dict[str, Any]) -> str:
    """After Architect: code → level0, blocker → theorist (backward edge)."""
    blocker = state.get("implementation_blocker")
    if blocker:
        settings = get_settings()
        revision_count = state.get("theorist_revision_count", 0)
        if revision_count >= settings.max_theorist_revisions:
            return "abort"
        return "theorist_revision"

    return "level0"


def route_after_level0(state: Dict[str, Any]) -> str:
    """After L0: pass → level1, fail → debugger."""
    validation = state.get("latest_validation", {})
    passed = validation.get("passed", False) if isinstance(validation, dict) else False
    return "level1" if passed else "debugger"


def route_after_level1(state: Dict[str, Any]) -> str:
    """After L1: pass → level2, fail → debugger."""
    validation = state.get("latest_validation", {})
    passed = validation.get("passed", False) if isinstance(validation, dict) else False
    return "level2" if passed else "debugger"


def route_after_level2(state: Dict[str, Any]) -> str:
    """After L2: pass → scientist, fail → debugger."""
    validation = state.get("latest_validation", {})
    passed = validation.get("passed", False) if isinstance(validation, dict) else False
    return "scientist" if passed else "debugger"


def route_after_debugger(state: Dict[str, Any]) -> str:
    """After Debugger: route back to the failed level, or abort."""
    settings = get_settings()
    retry_count = state.get("debug_retry_count", 0)

    if retry_count >= settings.max_debug_retries:
        logger.error("Debug retry limit reached (%d). Aborting.", retry_count)
        return "abort"

    # Route back to the level that failed
    current_level = state.get("current_level", 0)
    level_map = {
        ValidationLevel.STATIC.value: "level0",
        ValidationLevel.SMOKE.value: "level1",
        ValidationLevel.MICRO_TRAIN.value: "level2",
    }
    target = level_map.get(current_level, "level0")
    logger.info("Debugger routing back to %s (retry %d/%d)", target, retry_count, settings.max_debug_retries)
    return target


def route_after_scientist(state: Dict[str, Any]) -> str:
    """After Scientist: if successful, go to writer, else end."""
    status = state.get("hypothesis_status")
    if status == HypothesisStatus.SUCCEEDED.value:
        return "writer"
    return "end"


def theorist_revision_wrapper(state: Dict[str, Any]) -> Dict[str, Any]:
    """Increment the theorist revision counter before re-entering Theorist."""
    new_count = state.get("theorist_revision_count", 0) + 1
    logger.info("Theorist revision #%d", new_count)
    return {
        "theorist_revision_count": new_count,
    }


# ═══════════════════════════════════════════════════════════════
#  Graph Builder
# ═══════════════════════════════════════════════════════════════


def build_graph() -> StateGraph:
    """Construct and compile the UARH LangGraph state machine.

    Returns
    -------
    StateGraph
        The compiled graph, ready to be invoked with an initial state.
    """
    graph = StateGraph(HarnessState)

    # ── Register nodes ─────────────────────────────────────────
    graph.add_node("pi", node_pi)
    graph.add_node("theorist", node_theorist)
    graph.add_node("math_validator", node_math_validator)
    graph.add_node("theorist_revision", theorist_revision_wrapper)
    graph.add_node("architect", node_architect)
    graph.add_node("level0", node_level0)
    graph.add_node("level1", node_level1)
    graph.add_node("level2", node_level2)
    graph.add_node("debugger", node_debugger)
    graph.add_node("scientist", node_scientist)
    graph.add_node("writer", node_writer)
    graph.add_node("abort", node_abort)

    # ── Edges ──────────────────────────────────────────────────

    # START → PI
    graph.set_entry_point("pi")

    # PI → Theorist
    graph.add_edge("pi", "theorist")

    # Theorist → MathValidator
    graph.add_edge("theorist", "math_validator")

    # MathValidator → (conditional)
    graph.add_conditional_edges(
        "math_validator",
        route_after_math_validator,
        {
            "architect": "architect",
            "theorist_revision": "theorist_revision",
            "abort": "abort",
        },
    )

    # Theorist revision → Theorist (cyclic)
    graph.add_edge("theorist_revision", "theorist")

    # Architect → (conditional)
    graph.add_conditional_edges(
        "architect",
        route_after_architect,
        {
            "level0": "level0",
            "theorist_revision": "theorist_revision",
            "abort": "abort",
        },
    )

    # Level 0 → (conditional)
    graph.add_conditional_edges(
        "level0",
        route_after_level0,
        {"level1": "level1", "debugger": "debugger"},
    )

    # Level 1 → (conditional)
    graph.add_conditional_edges(
        "level1",
        route_after_level1,
        {"level2": "level2", "debugger": "debugger"},
    )

    # Level 2 → (conditional)
    graph.add_conditional_edges(
        "level2",
        route_after_level2,
        {"scientist": "scientist", "debugger": "debugger"},
    )

    # Debugger → (conditional: back to failed level or abort)
    graph.add_conditional_edges(
        "debugger",
        route_after_debugger,
        {
            "level0": "level0",
            "level1": "level1",
            "level2": "level2",
            "abort": "abort",
        },
    )

    # Scientist → (conditional: writer or end)
    graph.add_conditional_edges(
        "scientist",
        route_after_scientist,
        {
            "writer": "writer",
            "end": END,
        },
    )

    # Writer → END
    graph.add_edge("writer", END)

    # Abort → END
    graph.add_edge("abort", END)

    return graph.compile()
