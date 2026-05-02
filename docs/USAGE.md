# USAGE & CONFIGURATION: The Expert Guide

UARH v1.1 implements a **Research Manifest** system. This decouples the "What" (Research Goal) from the "How" (System Config).

---

## 1. The `program.md` (The Research Manifest)
The `program.md` is the primary communication channel with the Principal Investigator. While the system can work with a single sentence, high-tier research requires structure.

### Recommended Structure:
1.  **Primary Objective**: State the clear north-star goal.
2.  **Architecture Constraints**: "Must use RMSNorm," "No Dropout," etc.
3.  **Algorithmic Strategy**: Describe the optimization flow.
4.  **Evaluation Criteria**: Tell the Scientist what metrics matter most.

**Expert Tip**: Think of `program.md` as the "Introduction" and "Methodology" section of a research paper. The clearer this is, the less the PI has to "guess" your intent.

---

## 2. The `config.yaml` (The System Engine)
The `config.yaml` controls the orchestration and hardware mapping.

### Schema Breakdown:

#### `experiment` (Necessary)
- **`domain`**: The research category. This changes the PI's "Internal Encyclopedia." 
    - *Options*: `language_modeling`, `reinforcement_learning`, `computer_vision`, `generative_models`, `bio_ml`, `world_models`.
- **`hardware`**: Where the code runs. 
    - *Options*: `cpu`, `cuda` (NVIDIA), `mps` (Apple).

#### `hyperparameters` (Important)
- **`cycles`**: How many times the system should "Dream" and "Test." 
- **`max_steps`**: The duration of the Level 3 Micro-train. 
- **`learning_rate`**: The global optimizer setting.
- **`max_params`**: A strict guardrail. The Architect will attempt to keep the model size below this number (e.g., `30000000` for 30M params).

#### `system` (Advanced)
- **`elite_model`**: The LLM for PI, Theorist, and Debugger tasks (e.g., `ollama/qwen2.5:32b`).
- **`fast_model`**: The LLM for the Scientist and trivial logic (e.g., `ollama/llama3:8b`).
- **`sandbox_timeout`**: Hard kill-switch for runaway experiments.

---

## 3. CLI Command Suite

### Initialization
```bash
python -m uarh.main init <folder_name>
```
*Creates a fresh workspace with the documented templates.*

### The Launch Command
The `launch` command is the entrypoint for folder-based research.
```bash
# Standard Launch
python -m uarh.main launch <folder_name>

# TUI Launch (Recommended for visibility)
python -m uarh.main launch <folder_name> --tui
```

### The Run Command (Legacy/Quick-Fire)
Use this for one-off tests without creating a folder.
```bash
python -m uarh.main run --target "Accuracy" --cycles 1 --max-params 1000000
```

### The Status Command
Check the "Lineage" and see which hypotheses have been distilled into Axioms.
```bash
python -m uarh.main status
```

---

## 4. Understanding Output
When an experiment runs, UARH creates a unique `runs/run-xxx/` folder. This is your **Research Artifact**.
- **`model.py`**: The definitive architecture. You can copy-paste this directly into your own projects.
- **`run_log.txt`**: The forensic trail of the training run.
- **`hypothesis.json`**: The original "Why" behind the architecture.
