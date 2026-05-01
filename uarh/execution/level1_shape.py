"""Level 1 — Smoke-test sandbox.

Imports the generated module and calls ``create_model()`` to verify the
code can instantiate without crashing.  This is framework-agnostic — it
works for PyTorch models, JAX modules, sklearn pipelines, RL agents, or
any other Python object.

No hardcoded shapes, no hardcoded frameworks.  If ``create_model()``
runs without error, Level 1 passes.
"""

from __future__ import annotations

import base64
import logging
from typing import Any, Dict

from uarh.core.state import ValidationLevel, ValidationResult
from uarh.execution.sandbox import SandboxManager

logger = logging.getLogger(__name__)


def run_level1(
    source_code: str,
    *,
    entrypoint_function: str = "create_model",
    dependencies: list[str] | None = None,
) -> ValidationResult:
    """Execute Level 1 smoke-test validation.

    Steps:
    1. Install any declared dependencies.
    2. Import the generated module.
    3. Call the factory function (create_model / create_agent).
    4. Report success/failure.

    Parameters
    ----------
    source_code : str
        The synthesised Python source to validate.
    entrypoint_function : str
        Name of the factory function.
    dependencies : list[str] | None
        Python packages to pip install before running.

    Returns
    -------
    ValidationResult
    """
    import time

    t0 = time.perf_counter()

    wrapper_code = _build_smoke_wrapper(
        source_code=source_code,
        entrypoint_function=entrypoint_function,
    )

    sandbox = SandboxManager(timeout=120)
    try:
        # Install dependencies first
        if dependencies:
            sandbox.install_dependencies(dependencies)

        result = sandbox.execute_inline(wrapper_code, filename="smoke_test.py", timeout=120)
        elapsed = time.perf_counter() - t0

        if result.success:
            logger.info("Level 1 PASSED — module instantiates OK (%.2fs)", elapsed)
            metrics = _parse_output(result.stdout)
            return ValidationResult(
                level=ValidationLevel.SMOKE,
                passed=True,
                metrics=metrics,
                duration_seconds=elapsed,
            )
        else:
            error_msg = result.stderr or result.error_summary or "Unknown error"
            logger.warning("Level 1 FAILED:\n%s", error_msg)
            return ValidationResult(
                level=ValidationLevel.SMOKE,
                passed=False,
                error_message=error_msg,
                traceback=result.stderr,
                duration_seconds=elapsed,
            )
    finally:
        sandbox.cleanup()


def _build_smoke_wrapper(
    *,
    source_code: str,
    entrypoint_function: str,
) -> str:
    """Generate a self-contained Python script for smoke testing.

    This wrapper is completely framework-agnostic.  It just:
    1. Writes the generated code to a temp file.
    2. Imports it.
    3. Calls the factory function.
    4. Reports what was created.
    """
    encoded_source = base64.b64encode(source_code.encode()).decode()

    wrapper = (
        '"""Auto-generated Level 1 smoke-test wrapper."""\n'
        "import sys, os, json, importlib.util, tempfile, base64\n"
        "\n"
        "# ── Step 1: Decode and write the experiment module ──────\n"
        f'_encoded = "{encoded_source}"\n'
        "_source = base64.b64decode(_encoded).decode()\n"
        '_tmp_dir = tempfile.mkdtemp(prefix="uarh_l1_")\n'
        '_mod_path = os.path.join(_tmp_dir, "experiment.py")\n'
        'with open(_mod_path, "w") as _f:\n'
        "    _f.write(_source)\n"
        "\n"
        "# ── Step 2: Import the module ───────────────────────────\n"
        "try:\n"
        '    spec = importlib.util.spec_from_file_location("experiment", _mod_path)\n'
        "    experiment = importlib.util.module_from_spec(spec)\n"
        '    sys.modules["experiment"] = experiment\n'
        "    spec.loader.exec_module(experiment)\n"
        "except Exception as e:\n"
        '    print(f"IMPORT_ERROR: {type(e).__name__}: {e}", file=sys.stderr)\n'
        "    sys.exit(1)\n"
        "\n"
        "# ── Step 3: Call factory function ────────────────────────\n"
        "try:\n"
        f'    factory = getattr(experiment, "{entrypoint_function}", None)\n'
        "    if factory is None:\n"
        f'        print("FACTORY_ERROR: No function named \'{entrypoint_function}\' found in module", file=sys.stderr)\n'
        "        sys.exit(1)\n"
        "    model = factory()\n"
        "except Exception as e:\n"
        '    print(f"INSTANTIATION_ERROR: {type(e).__name__}: {e}", file=sys.stderr)\n'
        "    sys.exit(1)\n"
        "\n"
        "# ── Step 4: Report success ──────────────────────────────\n"
        "report = {\n"
        '    "status": "PASS",\n'
        '    "type": type(model).__name__,\n'
        '    "module": type(model).__module__,\n'
        "}\n"
        "\n"
        "# Try to get param count (works for PyTorch, may not for others)\n"
        "try:\n"
        "    param_count = sum(p.numel() for p in model.parameters())\n"
        '    report["param_count"] = param_count\n'
        "except (AttributeError, TypeError):\n"
        "    pass\n"
        "\n"
        "# Try to get string representation\n"
        "try:\n"
        '    report["repr"] = repr(model)[:500]\n'
        "except Exception:\n"
        "    pass\n"
        "\n"
        'print("SMOKE_REPORT:" + json.dumps(report, default=str))\n'
    )
    return wrapper


def _parse_output(stdout: str) -> Dict[str, Any]:
    """Extract structured metrics from the wrapper's stdout."""
    import json as _json

    for line in stdout.strip().splitlines():
        if line.startswith("SMOKE_REPORT:"):
            try:
                return _json.loads(line[len("SMOKE_REPORT:"):])
            except _json.JSONDecodeError:
                pass
    return {}
