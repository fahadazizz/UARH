"""Level 2 — Micro-train sandbox.

Calls the generated module's ``run_training(model, config)`` function
and captures the returned metrics dict.  This is completely framework-
agnostic — the generated code handles its own data loading, training
loop, and metric reporting.

The sandbox provides NOTHING except:
1. The experiment config dict (dataset_path, environment_name, device, etc.)
2. A hard timeout.
3. Metrics capture from the returned dict.

ALL training logic lives in the generated code.
"""

from __future__ import annotations

import base64
import json
import logging
from typing import Any, Dict, Optional

from uarh.core.config import get_settings
from uarh.core.state import ValidationLevel, ValidationResult
from uarh.execution.sandbox import SandboxManager

logger = logging.getLogger(__name__)


def run_level2(
    source_code: str,
    *,
    entrypoint_function: str = "create_model",
    experiment_config: Optional[Dict[str, Any]] = None,
    dependencies: list[str] | None = None,
) -> ValidationResult:
    """Execute Level 2 micro-train validation.

    Parameters
    ----------
    source_code : str
        The synthesised Python source.
    entrypoint_function : str
        Factory function name.
    experiment_config : dict | None
        Config dict passed to run_training().
    dependencies : list[str] | None
        Packages to install before running.

    Returns
    -------
    ValidationResult
    """
    import time

    settings = get_settings()
    t0 = time.perf_counter()

    config = experiment_config or {}
    # Set sensible defaults for micro-training
    config.setdefault("max_steps", 50)
    config.setdefault("learning_rate", 1e-3)
    config.setdefault("device", "cpu")

    wrapper_code = _build_training_wrapper(
        source_code=source_code,
        entrypoint_function=entrypoint_function,
        config=config,
    )

    sandbox = SandboxManager(timeout=settings.sandbox_timeout_seconds)
    try:
        # Install dependencies first
        if dependencies:
            sandbox.install_dependencies(dependencies)

        result = sandbox.execute_inline(wrapper_code, filename="micro_train.py")
        elapsed = time.perf_counter() - t0

        if result.success:
            metrics = _parse_training_output(result.stdout)
            diagnosis = _diagnose_training(metrics)

            if diagnosis is not None:
                logger.warning("Level 2 FAILED — training anomaly: %s", diagnosis)
                return ValidationResult(
                    level=ValidationLevel.MICRO_TRAIN,
                    passed=False,
                    error_message=diagnosis,
                    metrics=metrics,
                    duration_seconds=elapsed,
                )

            logger.info("Level 2 PASSED — training completed successfully (%.2fs)", elapsed)
            return ValidationResult(
                level=ValidationLevel.MICRO_TRAIN,
                passed=True,
                metrics=metrics,
                duration_seconds=elapsed,
            )
        else:
            error_msg = result.stderr or result.error_summary or "Training subprocess failed"
            if "OutOfMemoryError" in error_msg or "CUDA out of memory" in error_msg:
                error_msg = f"OOM_ERROR: {error_msg}"
            logger.warning("Level 2 FAILED:\n%s", error_msg)
            return ValidationResult(
                level=ValidationLevel.MICRO_TRAIN,
                passed=False,
                error_message=error_msg,
                traceback=result.stderr,
                duration_seconds=elapsed,
            )
    finally:
        sandbox.cleanup()


def _build_training_wrapper(
    *,
    source_code: str,
    entrypoint_function: str,
    config: Dict[str, Any],
) -> str:
    """Generate a self-contained training script.

    This wrapper is framework-agnostic.  It:
    1. Writes the generated code to a temp file.
    2. Imports it.
    3. Calls create_model().
    4. Calls run_training(model, config).
    5. Captures and prints the returned metrics.

    ALL training logic is in the generated code — not here.
    """
    encoded_source = base64.b64encode(source_code.encode()).decode()
    config_json = json.dumps(config)

    wrapper = (
        '"""Auto-generated Level 2 micro-train wrapper."""\n'
        "import sys, os, json, importlib.util, tempfile, base64\n"
        "\n"
        "# ── Decode and write experiment module ──────────────────\n"
        f'_encoded = "{encoded_source}"\n'
        "_source = base64.b64decode(_encoded).decode()\n"
        '_tmp_dir = tempfile.mkdtemp(prefix="uarh_l2_")\n'
        '_mod_path = os.path.join(_tmp_dir, "experiment.py")\n'
        'with open(_mod_path, "w") as _f:\n'
        "    _f.write(_source)\n"
        "\n"
        "# ── Import the module ───────────────────────────────────\n"
        "try:\n"
        '    spec = importlib.util.spec_from_file_location("experiment", _mod_path)\n'
        "    experiment = importlib.util.module_from_spec(spec)\n"
        '    sys.modules["experiment"] = experiment\n'
        "    spec.loader.exec_module(experiment)\n"
        "except Exception as e:\n"
        '    print(f"IMPORT_ERROR: {type(e).__name__}: {e}", file=sys.stderr)\n'
        "    sys.exit(1)\n"
        "\n"
        "# ── Create model/agent ──────────────────────────────────\n"
        "try:\n"
        f'    factory = getattr(experiment, "{entrypoint_function}", None)\n'
        "    if factory is None:\n"
        f'        print("FACTORY_ERROR: No function named \'{entrypoint_function}\' in module", file=sys.stderr)\n'
        "        sys.exit(1)\n"
        "    model = factory()\n"
        "except Exception as e:\n"
        '    print(f"MODEL_INIT_ERROR: {type(e).__name__}: {e}", file=sys.stderr)\n'
        "    sys.exit(1)\n"
        "\n"
        "# ── Load experiment config ──────────────────────────────\n"
        f"config = json.loads('{config_json}')\n"
        "\n"
        "# ── Call run_training ────────────────────────────────────\n"
        'run_training_fn = getattr(experiment, "run_training", None)\n'
        "if run_training_fn is None:\n"
        '    print("MISSING_FUNCTION: No run_training function found in module", file=sys.stderr)\n'
        "    sys.exit(1)\n"
        "\n"
        "try:\n"
        "    metrics = run_training_fn(model, config)\n"
        "except Exception as e:\n"
        "    import traceback\n"
        '    print(f"TRAINING_ERROR: {type(e).__name__}: {e}", file=sys.stderr)\n'
        "    traceback.print_exc(file=sys.stderr)\n"
        "    sys.exit(1)\n"
        "\n"
        "# ── Report results ──────────────────────────────────────\n"
        "if not isinstance(metrics, dict):\n"
        '    print(f"RETURN_ERROR: run_training returned {type(metrics).__name__}, expected dict", file=sys.stderr)\n'
        "    sys.exit(1)\n"
        "\n"
        'metrics["status"] = "PASS"\n'
        'print("TRAIN_REPORT:" + json.dumps(metrics, default=str))\n'
    )
    return wrapper


def _parse_training_output(stdout: str) -> Dict[str, Any]:
    """Extract TRAIN_REPORT JSON from stdout."""
    for line in stdout.strip().splitlines():
        if line.startswith("TRAIN_REPORT:"):
            try:
                return json.loads(line[len("TRAIN_REPORT:"):])
            except json.JSONDecodeError:
                pass
    return {}


def _diagnose_training(metrics: Dict[str, Any]) -> Optional[str]:
    """Detect degenerate training patterns from returned metrics.

    Returns None if training looks healthy, otherwise a diagnostic string.
    This is intentionally loose — different domains return different metrics.
    """
    if not metrics:
        return "No metrics returned by run_training — function may not have completed."

    if metrics.get("status") != "PASS":
        return f"run_training returned non-PASS status: {metrics.get('status')}"

    # Check for NaN in any numeric metric
    import math
    for key, val in metrics.items():
        if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
            return f"NaN/Inf detected in metric '{key}' = {val}"

    # Check loss history if present (common across domains)
    loss_history = metrics.get("loss_history", [])
    if isinstance(loss_history, list) and len(loss_history) >= 20:
        valid_losses = [v for v in loss_history if isinstance(v, (int, float)) and not math.isnan(v)]
        if valid_losses:
            quarter = max(1, len(valid_losses) // 4)
            first_q = sum(valid_losses[:quarter]) / quarter
            last_q = sum(valid_losses[-quarter:]) / quarter
            if last_q >= first_q * 0.995:
                return (
                    f"Loss did not decrease: first quarter avg={first_q:.4f}, "
                    f"last quarter avg={last_q:.4f}."
                )

    # Check gradient norms if present
    grad_norms = metrics.get("grad_norms", [])
    if isinstance(grad_norms, list) and grad_norms:
        valid_norms = [v for v in grad_norms if isinstance(v, (int, float))]
        if valid_norms:
            avg_norm = sum(valid_norms) / len(valid_norms)
            if avg_norm < 1e-7:
                return f"Vanishing gradients: avg grad norm = {avg_norm:.2e}"
            if avg_norm > 1e6:
                return f"Exploding gradients: avg grad norm = {avg_norm:.2e}"

    return None
