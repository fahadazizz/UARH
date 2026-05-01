"""UARH CLI — entrypoint for launching autonomous research cycles.

Usage
-----
    # Autonomous research on a text dataset
    python -m uarh.main run --target "validation perplexity" --dataset /data/input.txt

    # RL research
    python -m uarh.main run --target "episode reward" --domain reinforcement_learning --env CartPole-v1

    # Vision research
    python -m uarh.main run --target "classification accuracy" --domain computer_vision --dataset /data/images/

    # Multiple cycles
    python -m uarh.main run --target "validation perplexity" --cycles 5

    # Status check
    python -m uarh.main status
"""

from __future__ import annotations

import logging
import sys
import uuid
from datetime import datetime, timezone
from typing import Optional

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table

from uarh.core.config import get_settings
from uarh.core.graph import build_graph
from uarh.core.state import HypothesisStatus
from uarh.memory.distillation import DistillationEngine
from uarh.memory.lineage import LineageRepository

# ── Logging ────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True, markup=True)],
)
logger = logging.getLogger("uarh")

# ── CLI ────────────────────────────────────────────────────────
app = typer.Typer(
    name="uarh",
    help="Universal Autonomous Research Harness — autonomous AI research agent.",
    add_completion=False,
)
console = Console()


@app.command()
def run(
    target: str = typer.Option(
        ...,
        "--target", "-t",
        help="The north-star metric to optimise (e.g. 'validation perplexity', 'episode reward', 'classification accuracy').",
    ),
    domain: str = typer.Option(
        "general_ai",
        "--domain", "-d",
        help="Research domain: general_ai, language_modeling, reinforcement_learning, computer_vision, world_models, generative_models, bio_ml.",
    ),
    dataset: Optional[str] = typer.Option(
        None,
        "--dataset",
        help="Path to dataset file or directory.",
    ),
    env: Optional[str] = typer.Option(
        None,
        "--env",
        help="RL environment name (e.g. 'CartPole-v1', 'HalfCheetah-v4').",
    ),
    max_params: Optional[int] = typer.Option(
        None,
        "--max-params",
        help="Maximum parameter budget (e.g. 10000000 for 10M).",
    ),
    hardware: str = typer.Option(
        "cpu",
        "--hardware",
        help="Available hardware: cpu, cuda, mps.",
    ),
    context: str = typer.Option(
        "",
        "--context",
        help="Extra context for the PI agent (e.g. 'Use diffusion architecture instead of autoregressive').",
    ),
    cycles: int = typer.Option(
        1,
        "--cycles", "-n",
        help="Number of hypothesis cycles to run.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose", "-v",
        help="Enable DEBUG-level logging.",
    ),
) -> None:
    """Launch an autonomous research cycle."""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    settings = get_settings()

    # Build experiment config from CLI args
    experiment_config = {
        "dataset_path": dataset,
        "environment_name": env,
        "max_params": max_params,
        "hardware": hardware,
        "extra_context": context,
        "max_steps": 50,  # micro-train default
        "learning_rate": 1e-3,
        "device": hardware,
    }

    console.print(
        Panel.fit(
            f"[bold cyan]Universal Autonomous Research Harness[/bold cyan]\n"
            f"Target: [yellow]{target}[/yellow]  |  Domain: [yellow]{domain}[/yellow]\n"
            f"Dataset: [dim]{dataset or 'none'}[/dim]  |  Env: [dim]{env or 'none'}[/dim]  |  Params: [dim]{max_params or 'unlimited'}[/dim]\n"
            f"Hardware: [dim]{hardware}[/dim]  |  Cycles: [yellow]{cycles}[/yellow]\n"
            f"Elite: [dim]{settings.elite_model}[/dim]  |  Fast: [dim]{settings.fast_model}[/dim]",
            title="UARH v0.2.0",
            border_style="bright_blue",
        )
    )

    # Load persisted axioms
    distiller = DistillationEngine()
    axioms = distiller.load_axioms()
    if axioms:
        console.print(f"[dim]Loaded {len(axioms)} distilled axiom(s) from previous runs.[/dim]")

    # Build the graph
    graph = build_graph()

    # Governor: track consecutive failures across cycles
    consecutive_failures = 0

    for cycle_num in range(1, cycles + 1):
        console.print(f"\n[bold]═══ Cycle {cycle_num}/{cycles} ═══[/bold]")

        # Check governor: too many consecutive failures
        if consecutive_failures >= settings.max_consecutive_failures:
            console.print(
                f"[bold red]GOVERNOR HALT:[/bold red] "
                f"{consecutive_failures} consecutive failures. Stopping.",
            )
            break

        run_id = f"run-{uuid.uuid4().hex[:12]}"
        
        # Setup output directory for this run
        from pathlib import Path
        output_dir = Path("workspace/runs") / run_id
        output_dir.mkdir(parents=True, exist_ok=True)
        experiment_config["output_dir"] = str(output_dir.absolute())

        # Build initial state as a plain dict (NOT Pydantic model_dump)
        initial_state = {
            "run_id": run_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "target_metric": target,
            "domain": domain,
            "experiment_config": experiment_config,
            "axioms": axioms,
            "hypothesis_status": HypothesisStatus.PROPOSED.value,
            "debug_retry_count": 0,
            "theorist_revision_count": 0,
            "consecutive_failure_count": consecutive_failures,
            "telemetry": {},
            "similar_past_hypotheses": [],
        }

        try:
            # Execute the graph
            final_state = graph.invoke(initial_state)

            # Analyse result
            status = final_state.get("hypothesis_status", "unknown")
            proposal = final_state.get("proposal", {})
            hyp_title = proposal.get("title", "Unknown") if isinstance(proposal, dict) else "Unknown"
            hyp_id = proposal.get("hypothesis_id", "?") if isinstance(proposal, dict) else "?"

            if status == HypothesisStatus.SUCCEEDED.value:
                consecutive_failures = 0
                console.print(
                    f"[bold green]✓ SUCCEEDED[/bold green] — "
                    f"[{hyp_id}] {hyp_title}"
                )
                summary = final_state.get("experiment_summary", {})
                if isinstance(summary, dict):
                    console.print(f"  Conclusion: {summary.get('conclusion', 'N/A')}")
                    metrics = summary.get("key_metrics", {})
                    if metrics:
                        console.print(f"  Metrics: {metrics}")

                # Save artifacts
                code = final_state.get("code", "")
                if code:
                    (output_dir / "experiment.py").write_text(code, encoding="utf-8")
                
                paper_dict = final_state.get("paper", {})
                if paper_dict and "markdown_content" in paper_dict:
                    (output_dir / "paper.md").write_text(paper_dict["markdown_content"], encoding="utf-8")
                
                console.print(f"  [bold blue]Artifacts saved to:[/bold blue] {output_dir}")

            elif status == HypothesisStatus.FAILED.value:
                consecutive_failures += 1
                console.print(
                    f"[bold yellow]✗ FAILED[/bold yellow] — "
                    f"[{hyp_id}] {hyp_title}"
                )
                summary = final_state.get("experiment_summary", {})
                if isinstance(summary, dict):
                    console.print(f"  Failure mode: {summary.get('failure_mode', 'Unknown')}")

            elif status == HypothesisStatus.ABORTED.value:
                consecutive_failures += 1
                console.print(
                    f"[bold red]⊘ ABORTED[/bold red] — "
                    f"[{hyp_id}] {hyp_title}"
                )
                console.print(f"  Error: {final_state.get('latest_error', 'Unknown')}")

            else:
                consecutive_failures += 1
                console.print(f"[dim]Status: {status}[/dim]")

            # Reload axioms for next cycle (may have new ones from distillation)
            axioms = distiller.load_axioms()

        except KeyboardInterrupt:
            console.print("\n[bold red]Interrupted by user.[/bold red]")
            sys.exit(130)
        except Exception as exc:
            consecutive_failures += 1
            logger.exception("Unhandled error in cycle %d", cycle_num)
            console.print(f"[bold red]ERROR:[/bold red] {exc}")

    console.print(f"\n[bold]═══ UARH Complete ({cycles} cycle(s)) ═══[/bold]")


@app.command()
def status() -> None:
    """Show recent execution history and system status."""
    console.print(Panel.fit("[bold cyan]UARH Status[/bold cyan]", border_style="bright_blue"))

    # Recent executions
    try:
        lineage = LineageRepository()
        logs = lineage.get_recent_executions(limit=10)

        if not logs:
            console.print("[dim]No executions recorded yet.[/dim]")
        else:
            table = Table(title="Recent Executions")
            table.add_column("Run ID", style="dim", max_width=16)
            table.add_column("Hypothesis", max_width=40)
            table.add_column("Status", justify="center")
            table.add_column("Level", justify="center")
            table.add_column("Loss", justify="right")
            table.add_column("Time", style="dim")

            for log in logs:
                status_style = {
                    "succeeded": "[green]✓[/green]",
                    "failed": "[yellow]✗[/yellow]",
                    "aborted": "[red]⊘[/red]",
                }.get(log.status, log.status)

                table.add_row(
                    log.run_id[:16],
                    log.hypothesis_title or log.hypothesis_id,
                    status_style,
                    str(log.validation_level_reached),
                    f"{log.final_loss:.4f}" if log.final_loss is not None else "—",
                    str(log.created_at)[:19] if log.created_at else "—",
                )
            console.print(table)
    except Exception as exc:
        console.print(f"[red]Error reading lineage: {exc}[/red]")

    # Axiom count
    try:
        distiller = DistillationEngine()
        axioms = distiller.load_axioms()
        console.print(f"\n[bold]Distilled Axioms:[/bold] {len(axioms)}")
        for i, ax in enumerate(axioms[-5:], max(1, len(axioms) - 4)):
            console.print(f"  [dim]Axiom {i}:[/dim] {ax}")
    except Exception:
        pass


def cli() -> None:
    """Package entrypoint."""
    app()


if __name__ == "__main__":
    cli()
