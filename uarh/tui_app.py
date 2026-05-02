"""Textual UI Application for UARH."""

import asyncio
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path

from textual import work
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Header, Footer, Log, Static, Tree
from textual_plotext import PlotextPlot

from uarh.core.config import get_settings
from uarh.core.graph import build_graph
from uarh.core.state import HypothesisStatus
from uarh.memory.distillation import DistillationEngine


class UarhApp(App):
    """The Universal Autonomous Research Harness Interactive TUI."""
    
    CSS = """
    #main-container {
        layout: horizontal;
    }
    #sidebar {
        width: 30;
        border-right: solid green;
    }
    #content {
        width: 1fr;
    }
    #logs {
        height: 2fr;
        border-bottom: solid gray;
    }
    #viz {
        height: 1fr;
    }
    .panel-title {
        text-align: center;
        text-style: bold;
        background: $boost;
        padding: 1;
    }
    """

    def __init__(self, target: str, domain: str, dataset: str, env: str, max_params: int, hardware: str, context: str, cycles: int):
        super().__init__()
        self.target = target
        self.domain = domain
        self.dataset = dataset
        self.env = env
        self.max_params = max_params
        self.hardware = hardware
        self.context = context
        self.cycles = cycles

        self.graph = build_graph()
        self.settings = get_settings()

    def compose(self) -> ComposeResult:
        yield Header()
        with Container(id="main-container"):
            with Vertical(id="sidebar"):
                yield Static("Pipeline Status", classes="panel-title")
                self.pipeline_tree = Tree("Nodes")
                yield self.pipeline_tree
            with Vertical(id="content"):
                self.log_widget = Log(id="logs", highlight=True)
                yield self.log_widget
                self.viz_widget = PlotextPlot(id="viz")
                yield self.viz_widget
        yield Footer()

    def on_mount(self):
        self.title = f"UARH — {self.domain} | {self.target}"
        
        # Setup logging to go to Textual Log widget
        class TextualHandler(logging.Handler):
            def __init__(self, log_widget):
                super().__init__()
                self.log_widget = log_widget
            def emit(self, record):
                msg = self.format(record)
                
                # Add text color based on level
                if record.levelno >= logging.ERROR:
                    msg = f"[bold red]{msg}[/bold red]"
                elif record.levelno >= logging.WARNING:
                    msg = f"[bold yellow]{msg}[/bold yellow]"
                else:
                    msg = f"[blue]{msg}[/blue]"

                try:
                    self.log_widget.app.call_from_thread(self.log_widget.write_line, msg)
                except Exception:
                    pass

        # Adjust the root logger
        root_logger = logging.getLogger()
        for h in list(root_logger.handlers):
            root_logger.removeHandler(h)
        root_logger.setLevel(logging.INFO)
        th = TextualHandler(self.log_widget)
        th.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s", "%H:%M:%S"))
        root_logger.addHandler(th)

        # Initialize the pipeline tree
        self.pipeline_tree.root.expand()
        self.nodes = {
            "pi": self.pipeline_tree.root.add("Principal Investigator"),
            "theorist": self.pipeline_tree.root.add("Theorist"),
            "validator": self.pipeline_tree.root.add("Blueprint Validator"),
            "architect": self.pipeline_tree.root.add("Architect"),
            "level0": self.pipeline_tree.root.add("Level 0 (Static)"),
            "level1": self.pipeline_tree.root.add("Level 1 (Smoke)"),
            "level2": self.pipeline_tree.root.add("Level 2 (Train)"),
            "scientist": self.pipeline_tree.root.add("Data Scientist"),
            "writer": self.pipeline_tree.root.add("Paper Writer"),
            "debugger": self.pipeline_tree.root.add("Debugger"),
        }

        # Setup initial plot
        plt = self.viz_widget.plt
        plt.title("Training Loss Curve")
        plt.xlabel("Step")
        plt.ylabel("Loss")
        self.viz_widget.refresh()

        # Start research
        self.run_research()

    @work(thread=True)
    def run_research(self):
        distiller = DistillationEngine()
        axioms = distiller.load_axioms()
        
        for cycle_num in range(1, self.cycles + 1):
            logging.info(f"=== Cycle {cycle_num}/{self.cycles} ===")
            
            run_id = f"run-{uuid.uuid4().hex[:12]}"
            output_dir = Path("workspace/runs") / run_id
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Add run-specific log handler
            file_handler = logging.FileHandler(output_dir / "run_log.txt")
            file_handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s - %(name)s - %(message)s"))
            logging.getLogger().addHandler(file_handler)
            
            experiment_config = {
                "dataset_path": self.dataset,
                "environment_name": self.env,
                "max_params": self.max_params,
                "hardware": self.hardware,
                "extra_context": self.context,
                "max_steps": 50,
                "learning_rate": 1e-3,
                "device": self.hardware,
                "output_dir": str(output_dir.absolute()),
            }
            
            initial_state = {
                "run_id": run_id,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "target_metric": self.target,
                "domain": self.domain,
                "experiment_config": experiment_config,
                "axioms": axioms,
                "hypothesis_status": HypothesisStatus.PROPOSED.value,
                "debug_retry_count": 0,
                "theorist_revision_count": 0,
                "consecutive_failure_count": 0,
                "telemetry": {},
                "similar_past_hypotheses": [],
            }
            
            # Stream the graph execution to update the UI
            try:
                final_state = initial_state
                # reset tree
                self.app.call_from_thread(self._reset_tree)
                
                for step_dict in self.graph.stream(initial_state, stream_mode="updates"):
                    # step_dict format: {"node_name": state_update}
                    for node_name, state_update in step_dict.items():
                        self.app.call_from_thread(self._update_tree, node_name)
                        final_state.update(state_update)
                        
                        # Update visualization if telemetry is present
                        if "telemetry" in state_update and state_update["telemetry"]:
                            self.app.call_from_thread(self._update_plot, state_update["telemetry"])
                
                status = final_state.get("hypothesis_status")
                if status == HypothesisStatus.SUCCEEDED.value:
                    logging.info(f"✓ SUCCEEDED — artifacts saved to {output_dir}")
                    
                    code = final_state.get("code", "")
                    if code:
                        (output_dir / "experiment.py").write_text(code, encoding="utf-8")
                    paper_dict = final_state.get("paper", {})
                    if paper_dict and "markdown_content" in paper_dict:
                        (output_dir / "paper.md").write_text(paper_dict["markdown_content"], encoding="utf-8")

                elif status == HypothesisStatus.FAILED.value:
                    logging.error("✗ FAILED")
                elif status == HypothesisStatus.ABORTED.value:
                    logging.error("⊘ ABORTED")
                
            except Exception as e:
                logging.error(f"Error in cycle: {e}")
                
            axioms = distiller.load_axioms()
        
        logging.info("Research Complete.")

    def _reset_tree(self):
        for node in self.nodes.values():
            # Keep only the base label without any emojis
            base_label = str(node.label).replace(" ✅", "").replace(" 🔄", "").replace(" ❌", "")
            node.label = base_label

    def _update_tree(self, active_node_name):
        # We received an update FROM active_node_name.
        if active_node_name in self.nodes:
            node = self.nodes[active_node_name]
            base_label = str(node.label).replace(" ✅", "").replace(" 🔄", "").replace(" ❌", "")
            node.label = f"{base_label} ✅"
            
    def _update_plot(self, telemetry):
        plt = self.viz_widget.plt
        plt.clear_data()
        plt.clear_figure()
        
        losses = []
        if "level2" in telemetry and isinstance(telemetry["level2"], dict):
            # Try to grab loss_curve if it exists, otherwise just plot the final loss to show *something*
            if "loss_curve" in telemetry["level2"]:
                losses = telemetry["level2"]["loss_curve"]
            elif "final_loss" in telemetry["level2"]:
                losses = [telemetry["level2"]["final_loss"]]
                
        if losses:
            plt.plot(losses, marker="dot", color="blue")
            plt.title("Training Loss Curve")
            plt.xlabel("Step")
            plt.ylabel("Loss")
        else:
            plt.title("No plot data yet")
            
        self.viz_widget.refresh()
