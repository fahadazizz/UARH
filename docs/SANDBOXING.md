# SANDBOXING: Security, Isolation & Safety

The Universal Autonomous Research Harness (UARH) is designed to execute **arbitrary, agent-synthesized Python code**. To do this safely and effectively, UARH utilizes a multi-layered isolation strategy managed by the `SandboxManager`.

---

## 1. The Isolation Philosophy
Agent-generated code is inherently unpredictable. It may contain:
- Resource-heavy infinite loops.
- Inefficient memory allocations (OOM risks).
- Unintended filesystem interactions.
- Dependency conflicts.

UARH treats every experiment as a **disposable guest process**.

---

## 2. Dynamic Module Isolation
UARH avoids installing the research code as a package. Instead, it uses **Python's `importlib` and `sys.path` injection**.

### The Flow:
1.  **Code Synthesis**: The Architect writes `model.py` to a unique, ephemeral directory (`workspace/runs/run-xxx/`).
2.  **Path Hijacking**: The SandboxManager temporarily injects this unique directory into `sys.path`.
3.  **Encapsulation**: The model is instantiated within a `try-except-finally` block. After the run, the directory is purged from `sys.path`, ensuring that Cycle 2 cannot accidentally import code from Cycle 1.

---

## 3. Resource & Execution Guardrails

### Execution Timeouts
In AI training, it is easy to accidentally write a loop that never terminates. 
- **The Switch**: `sandbox_timeout` in `config.yaml`.
- **The Mechanism**: UARH runs the training logic in a **separate thread/subprocess**. If the timer expires, the Governor sends a `SIGKILL` or raises a `TimeoutError`, instantly halting hardware usage.

### Device Management
The Sandbox Manager enforces hardware constraints. If `config.yaml` specifies `hardware: cpu`, the Sandbox Manager intercepts device-selection logic to ensure the experiment doesn't attempt to seize an unavailable GPU, which would crash the cycle.

---

## 4. Intelligent Dependency Resolution
AI Agents often assume certain libraries are present (e.g., `scipy`, `einops`, `timm`).

### The Resolution Loop:
1.  **Detection**: The system attempts to import the code.
2.  **Capture**: If an `ImportError` occurs, the Sandbox Manager parses the missing package name.
3.  **Auto-Installation**: It attempts a non-interactive `.venv/bin/pip install <package>`. 
4.  **Logging**: All installations are logged in `run_log.txt` so the user knows which packages were added to the environment during the research cycle.

---

## 5. Data & Artifact Capture
The sandbox is a "One-Way Mirror." Code inside the sandbox can see the dataset, but it cannot permanently modify the UARH codebase. 

- **Stdout Redirection**: All print statements, progress bars (Tqdm), and logger outputs are piped into the UARH `run_log.txt`.
- **Metric Scraping**: The Scientist agent monitors the output stream to extract numerical data without requiring the Architect to explicitly "save" a CSV or JSON file. This makes the code generation much simpler and less prone to "File Not Found" errors.
