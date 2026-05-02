import yaml
from pathlib import Path
import os

@app.command()
def init(
    experiment_dir: str = typer.Argument(
        ".",
        help="Directory to initialize the experiment configuration in.",
    )
) -> None:
    """Initialize a new experiment with config.yaml and program.md templates."""
    out_dir = Path(experiment_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    config_path = out_dir / "config.yaml"
    if not config_path.exists():
        config_data = {
            "system": {
                "elite_model": "ollama/qwen3.5:397b-cloud",
                "fast_model": "ollama/qwen3.5:397b-cloud"
            },
            "experiment": {
                "domain": "language_modeling",
                "dataset_path": "dataset/input.txt",
                "environment_name": "",
                "max_params": 10000000,
                "hardware": "cpu",
                "cycles": 1,
                "max_steps": 50,
                "learning_rate": 0.001
            }
        }
        with open(config_path, "w") as f:
            yaml.dump(config_data, f, sort_keys=False)
            
    program_path = out_dir / "program.md"
    if not program_path.exists():
        with open(program_path, "w") as f:
            f.write("# Experiment Target\n\nDesign a 10M Parameter Diffusion Language Model that...\n\n## Extra Context\n\nEnsure that you use Pre-LN and weight tying.\n")
            
    console.print(f"[bold green]Initialized experiment configuration in {out_dir.absolute()}[/bold green]")
    console.print("Edit [bold cyan]program.md[/bold cyan] and [bold cyan]config.yaml[/bold cyan], then run: [bold]python -m uarh.main launch[/bold]")


@app.command()
def launch(
    experiment_dir: str = typer.Argument(
        ".",
        help="Directory containing config.yaml and program.md.",
    ),
    tui_mode: bool = typer.Option(
        False,
        "--tui",
        help="Launch in interactive TUI mode.",
    )
) -> None:
    """Launch an experiment reading configuration from config.yaml and program.md."""
    import yaml
    
    out_dir = Path(experiment_dir)
    config_path = out_dir / "config.yaml"
    program_path = out_dir / "program.md"
    
    if not config_path.exists() or not program_path.exists():
        console.print(f"[bold red]Error:[/bold red] Could not find config.yaml and program.md in {out_dir.absolute()}")
        console.print("Run [bold cyan]python -m uarh.main init[/bold cyan] to generate them.")
        raise typer.Exit(1)
        
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    with open(program_path, "r") as f:
        target = f.read().strip()
        
    # Override environment variables for system models so get_settings() picks them up
    system_cfg = config.get("system", {})
    if "elite_model" in system_cfg:
        os.environ["UARH_ELITE_MODEL"] = system_cfg["elite_model"]
    if "fast_model" in system_cfg:
        os.environ["UARH_FAST_MODEL"] = system_cfg["fast_model"]
        
    exp_cfg = config.get("experiment", {})
    
    if tui_mode:
        from uarh.tui_app import UarhApp
        tui_app = UarhApp(
            target=target,
            domain=exp_cfg.get("domain", "general_ai"),
            dataset=exp_cfg.get("dataset_path"),
            env=exp_cfg.get("environment_name"),
            max_params=exp_cfg.get("max_params"),
            hardware=exp_cfg.get("hardware", "cpu"),
            context="",  # Context is folded into program.md
            cycles=exp_cfg.get("cycles", 1),
        )
        tui_app.run()
    else:
        run(
            target=target,
            domain=exp_cfg.get("domain", "general_ai"),
            dataset=exp_cfg.get("dataset_path"),
            env=exp_cfg.get("environment_name"),
            max_params=exp_cfg.get("max_params"),
            hardware=exp_cfg.get("hardware", "cpu"),
            context="",
            cycles=exp_cfg.get("cycles", 1),
            verbose=False
        )


@app.command()
def status() -> None:
