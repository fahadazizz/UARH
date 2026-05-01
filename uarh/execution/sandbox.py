"""Sandbox manager — isolated subprocess execution for generated code.

All LLM-generated code is written to a temporary directory and executed
in a child process with hard timeouts.  This module provides the shared
infrastructure used by every validation level.

Now includes dynamic dependency installation so that the harness can
work with ANY framework the PI Agent chooses.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from uarh.core.config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class SandboxResult:
    """Outcome of a sandboxed execution."""

    success: bool
    stdout: str = ""
    stderr: str = ""
    return_code: int = -1
    metrics: Dict[str, Any] = field(default_factory=dict)
    error_summary: Optional[str] = None
    duration_seconds: float = 0.0


class SandboxManager:
    """Write code to a temp dir and run it in a subprocess with a timeout.

    The sandbox creates an isolated working directory for each run so
    that file-system side-effects from generated code cannot leak into
    the host project.
    """

    def __init__(self, timeout: Optional[int] = None) -> None:
        settings = get_settings()
        self._timeout = timeout or settings.sandbox_timeout_seconds
        self._work_dir: Optional[Path] = None

    # ── Public API ─────────────────────────────────────────────

    def install_dependencies(self, packages: List[str]) -> None:
        """Install Python packages into the active venv before execution.

        This is what makes the harness truly universal — if the PI Agent
        declares dependencies: ["jax", "flax"], they get installed here.
        Packages already installed are skipped (--upgrade is NOT used to
        avoid breaking existing installs).
        """
        if not packages:
            return

        # Filter out packages likely already available
        packages_to_install = []
        for pkg in packages:
            # Normalise package name for import check
            import_name = pkg.replace("-", "_").split("[")[0].split(">=")[0].split("==")[0]
            try:
                __import__(import_name)
                logger.debug("Package '%s' already available, skipping.", pkg)
            except ImportError:
                packages_to_install.append(pkg)

        if not packages_to_install:
            logger.info("All declared dependencies already installed.")
            return

        logger.info("Installing dependencies: %s", packages_to_install)
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "--quiet", *packages_to_install],
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.returncode != 0:
                logger.warning(
                    "pip install failed for %s:\n%s",
                    packages_to_install,
                    result.stderr,
                )
            else:
                logger.info("Dependencies installed successfully.")
        except subprocess.TimeoutExpired:
            logger.warning("pip install timed out for %s", packages_to_install)
        except Exception as exc:
            logger.warning("Failed to install dependencies: %s", exc)

    def write_module(self, source_code: str, filename: str = "experiment.py") -> Path:
        """Write *source_code* into a fresh temp directory and return its path."""
        self._work_dir = Path(tempfile.mkdtemp(prefix="uarh_sandbox_"))
        filepath = self._work_dir / filename
        filepath.write_text(source_code, encoding="utf-8")
        logger.debug("Sandbox module written to %s", filepath)
        return filepath

    def execute_script(
        self,
        script_path: Path,
        *,
        timeout: Optional[int] = None,
        extra_env: Optional[Dict[str, str]] = None,
    ) -> SandboxResult:
        """Run a Python script in a subprocess with a hard timeout.

        Parameters
        ----------
        script_path : Path
            Absolute path to the ``.py`` file.
        timeout : int | None
            Override the default timeout (seconds).
        extra_env : dict | None
            Additional environment variables for the child process.

        Returns
        -------
        SandboxResult
        """
        import time

        effective_timeout = timeout or self._timeout

        env = os.environ.copy()
        if extra_env:
            env.update(extra_env)

        t0 = time.perf_counter()
        try:
            proc = subprocess.run(
                [sys.executable, str(script_path)],
                capture_output=True,
                text=True,
                timeout=effective_timeout,
                cwd=str(script_path.parent),
                env=env,
            )
            elapsed = time.perf_counter() - t0

            success = proc.returncode == 0
            error_summary = None
            if not success:
                stderr_lines = proc.stderr.strip().splitlines()
                error_summary = stderr_lines[-1] if stderr_lines else "Unknown error"

            return SandboxResult(
                success=success,
                stdout=proc.stdout,
                stderr=proc.stderr,
                return_code=proc.returncode,
                error_summary=error_summary,
                duration_seconds=elapsed,
            )

        except subprocess.TimeoutExpired:
            elapsed = time.perf_counter() - t0
            logger.warning("Sandbox timed out after %ds", effective_timeout)
            return SandboxResult(
                success=False,
                stderr=f"TIMEOUT: Execution exceeded {effective_timeout}s limit.",
                error_summary=f"Execution timed out after {effective_timeout}s.",
                duration_seconds=elapsed,
            )

        except Exception as exc:
            elapsed = time.perf_counter() - t0
            logger.error("Sandbox execution error: %s", exc)
            return SandboxResult(
                success=False,
                stderr=str(exc),
                error_summary=str(exc),
                duration_seconds=elapsed,
            )

    def execute_inline(
        self,
        source_code: str,
        *,
        filename: str = "experiment.py",
        timeout: Optional[int] = None,
    ) -> SandboxResult:
        """Convenience: write + execute in one call."""
        path = self.write_module(source_code, filename=filename)
        return self.execute_script(path, timeout=timeout)

    def cleanup(self) -> None:
        """Remove the temporary working directory."""
        if self._work_dir and self._work_dir.exists():
            import shutil

            shutil.rmtree(self._work_dir, ignore_errors=True)
            logger.debug("Sandbox cleaned up: %s", self._work_dir)
            self._work_dir = None
