"""Debug / QA Agent.

Engaged when ANY sandbox level fails.  Receives the current code and
the failure traceback/telemetry, and produces a ``DebugPatch`` with
corrected code and root-cause analysis.  Works across all frameworks.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional, Type

from uarh.agents.base import BaseAgent
from uarh.core.config import get_settings
from uarh.core.state import DebugPatch, ValidationLevel


class DebugAgent(BaseAgent[DebugPatch]):
    """The Debugger — fixes code failures at any validation level."""

    def __init__(self, model: Optional[str] = None, temperature: float = 0.2) -> None:
        settings = get_settings()
        super().__init__(model=model or settings.fast_model, temperature=temperature)

    @property
    def agent_name(self) -> str:
        return "Debug Agent"

    @property
    def output_schema(self) -> Type[DebugPatch]:
        return DebugPatch

    @property
    def system_prompt(self) -> str:
        return """You are a Senior AI Debugging Engineer in an autonomous research lab.

Your SOLE responsibility is to diagnose and FIX code that failed validation.
You do NOT theorise or propose new architectures — you fix what exists.
You work across ALL frameworks: PyTorch, JAX, TensorFlow, sklearn, SB3, etc.

FAILURE TYPES YOU HANDLE:
- Level 0 (Static): Syntax errors, import failures, AST parse errors.
- Level 1 (Smoke): Module import failures, factory function crashes, model instantiation errors.
- Level 2 (Training): OOM errors, NaN/Inf loss, exploding/vanishing gradients, data loading errors,
  environment errors (RL), training loop crashes, metric reporting failures.

DEBUGGING RULES:
1. Your "patched_code" must be the COMPLETE corrected source — not a diff.
2. For import errors: fix the import or add the missing dependency.
3. For OOM: reduce batch size, add gradient checkpointing, or reduce model size.
4. For NaN loss: add gradient clipping, check for division by zero, fix weight init.
5. For data loading errors: fix file paths, encoding, parsing logic.
6. For RL environment errors: fix observation/action space handling.
7. NEVER change the fundamental architecture — only fix bugs and tune numerics.
8. Ensure `create_model()` and `run_training(model, config)` still work after patching.
9. Provide a clear root-cause "diagnosis".
10. Respond ONLY with valid JSON matching the DebugPatch schema.
"""

    def build_user_message(self, state: Dict[str, Any]) -> str:
        code = state.get("code", "")
        latest_error = state.get("latest_error", "No error details available.")
        current_level = state.get("current_level", 0)
        debug_retry = state.get("debug_retry_count", 0)
        telemetry = state.get("telemetry", {})
        proposal = state.get("proposal", {})

        level_name = ValidationLevel(current_level).name if isinstance(current_level, int) else str(current_level)

        parts = [
            f"## Failure Context",
            f"**Validation Level:** Level {current_level} ({level_name})",
            f"**Debug Attempt:** {debug_retry + 1}",
            f"**Framework:** {proposal.get('framework', 'unknown') if isinstance(proposal, dict) else 'unknown'}",
            "",
            "## Error / Traceback",
            "```",
            str(latest_error),
            "```",
            "",
        ]

        if telemetry:
            parts.extend([
                "## Training Telemetry (if available)",
                "```json",
                json.dumps(telemetry, indent=2, default=str),
                "```",
                "",
            ])

        parts.extend([
            "## Current Code (FAILING)",
            "```python",
            str(code),
            "```",
            "",
            "## Your Task",
            "Diagnose the root cause and provide the COMPLETE corrected code.",
            "The code MUST still export `create_model()` and `run_training(model, config)`.",
            "Respond with JSON matching this schema:",
            "",
            "```json",
            json.dumps(DebugPatch.model_json_schema(), indent=2),
            "```",
        ])
        return "\n".join(parts)
