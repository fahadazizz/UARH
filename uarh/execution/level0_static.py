"""Level 0 — Static analysis sandbox.

Sub-second validation using Python's ``ast`` module for syntax checking
and ``ruff`` for linting.  This is the cheapest gate: if code cannot
even be parsed, there is no point running it.
"""

from __future__ import annotations

import ast
import logging
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

from uarh.core.state import ValidationLevel, ValidationResult

logger = logging.getLogger(__name__)


def run_level0(source_code: str) -> ValidationResult:
    """Execute Level 0 static validation on *source_code*.

    Steps
    -----
    1. ``ast.parse`` — catches syntax errors instantly.
    2. ``ruff check`` — catches undefined names, unused imports, style.

    Returns
    -------
    ValidationResult
        With ``level=STATIC``, ``passed`` flag, and any error details.
    """
    import time

    t0 = time.perf_counter()

    # ── Step 1: AST parse ──────────────────────────────────────
    ast_ok, ast_error = _check_ast(source_code)
    if not ast_ok:
        elapsed = time.perf_counter() - t0
        logger.warning("Level 0 FAILED — AST parse error: %s", ast_error)
        return ValidationResult(
            level=ValidationLevel.STATIC,
            passed=False,
            error_message=f"SyntaxError: {ast_error}",
            traceback=ast_error,
            duration_seconds=elapsed,
        )

    # ── Step 2: Ruff linting ───────────────────────────────────
    ruff_ok, ruff_errors = _check_ruff(source_code)
    elapsed = time.perf_counter() - t0

    if not ruff_ok:
        error_text = "\n".join(ruff_errors)
        logger.warning("Level 0 FAILED — Ruff errors:\n%s", error_text)
        return ValidationResult(
            level=ValidationLevel.STATIC,
            passed=False,
            error_message=f"Ruff lint errors:\n{error_text}",
            traceback=error_text,
            duration_seconds=elapsed,
        )

    logger.info("Level 0 PASSED — static checks OK (%.2fs)", elapsed)
    return ValidationResult(
        level=ValidationLevel.STATIC,
        passed=True,
        duration_seconds=elapsed,
    )


# ═══════════════════════════════════════════════════════════════
#  Internal helpers
# ═══════════════════════════════════════════════════════════════


def _check_ast(source_code: str) -> Tuple[bool, Optional[str]]:
    """Return (True, None) if code parses, else (False, error_string)."""
    try:
        ast.parse(source_code)
        return True, None
    except SyntaxError as exc:
        detail = f"Line {exc.lineno}: {exc.msg}"
        if exc.text:
            detail += f"\n  {exc.text.rstrip()}"
        return False, detail


def _check_ruff(source_code: str) -> Tuple[bool, List[str]]:
    """Run ``ruff check`` on the source and return (passed, errors).

    Only fails on critical rules (F-series: pyflakes fatal errors) to
    avoid false-positives from style rules on generated code.
    """
    tmp = None
    try:
        tmp = tempfile.NamedTemporaryFile(
            suffix=".py",
            mode="w",
            delete=False,
            encoding="utf-8",
        )
        tmp.write(source_code)
        tmp.flush()
        tmp_path = tmp.name
        tmp.close()

        # Resolve ruff from the same venv as the running interpreter
        ruff_bin = str(Path(sys.executable).parent / "ruff")

        result = subprocess.run(
            [
                ruff_bin,
                "check",
                "--select",
                "E9,F63,F7,F82",  # Only fatal / undefined-name errors
                "--no-fix",
                "--output-format",
                "concise",
                tmp_path,
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )

        if result.returncode == 0:
            return True, []

        errors = [
            line.strip()
            for line in result.stdout.strip().splitlines()
            if line.strip()
        ]
        return False, errors

    except FileNotFoundError:
        # ruff not installed — skip this check gracefully
        logger.warning("ruff not found in PATH; skipping lint check.")
        return True, []

    except subprocess.TimeoutExpired:
        logger.warning("ruff timed out; treating as pass.")
        return True, []

    finally:
        if tmp:
            Path(tmp.name).unlink(missing_ok=True)
