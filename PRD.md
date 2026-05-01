# Product Requirements Document: Universal Autonomous Research Harness (UARH)

## 1. Product Overview & Objective
The Universal Autonomous Research Harness (UARH) is an orchestration framework designed to transition Large Language Models (LLMs) from passive text generators into localized, autonomous scientific agents capable of end-to-end research across **any domain of AI**: supervised learning, reinforcement learning, world models, generative models, bio-ML, vision, NLP, multi-modal systems, and beyond.

This system moves away from monolithic scripts and single-agent loops ("prompt-in, code-out") to a **Directed Cyclic Graph (DCG) multi-agent state machine**. The harness is **framework-agnostic** (PyTorch, JAX, TensorFlow, scikit-learn, Stable-Baselines3, or pure Python) and **domain-agnostic** — the PI Agent dynamically selects the architecture, framework, and execution strategy based on the research objective.

---

## 2. Core Architectural Pillars

### A. The State Graph (The Harness Engine)
- **Concept:** The Harness operates as a State Machine (via LangGraph). LLMs do not execute control flow; the Harness dictates which agent acts based on the current localized state node.
- **Implementation Rules:**
  - Enforce strict typing of state objects passing between nodes via TypedDict schemas.
  - Make state transitions deterministic.
  - Implement idempotent graph nodes so that an interrupted workflow can safely resume.

### B. Context & Memory Isolation (The "Clean Room" Principle)
- **Concept:** Agents must suffer zero "context poisoning." They are spun up ephemerally and given exactly the tokens they need for their explicit task.
- **Implementation Rules:**
  - Do NOT share conversational `messages` arrays between agents.
  - Pass structured artifacts (JSON/AST trees/blueprints), not unstructured chat history.

### C. Tiered Scientific Sandbox (Fast Failing)
- **Concept:** Every hypothesis requires execution, but research is computationally heavy. The sandbox enforces rapid, tiered failure modes.
- **Implementation Rules:**
  - **Level 0 (Static):** Sub-second Python `ast` syntax checks and `ruff` linting.
  - **Level 1 (Smoke Run):** Import the generated module and call its `create_model()` / `create_agent()` factory. Verify it instantiates without crashes. For neural nets, validate shapes via meta tensors. For RL, verify env-agent compatibility. For sklearn, verify pipeline construction.
  - **Level 2 (Micro-Train):** Call the module's `run_training()` function on minimal data/episodes. Verify metrics are returned, loss decreases (if applicable), no crashes.
  - **Level 3 (Full Run):** Reserved for distributed/cluster execution (future).

### D. Domain Agnosticism (The Universal Principle)
- **Concept:** The system MUST NOT be hardcoded to any framework (PyTorch) or domain (language modeling). The PI Agent dynamically determines framework, dependencies, data format, and evaluation criteria.
- **Implementation Rules:**
  - The `ResearchProposal` defines the framework and dependencies.
  - The sandbox installs declared dependencies before execution.
  - Sandboxes delegate ALL training/evaluation to the generated code's `run_training()` function — no hardcoded training loops in the harness itself.
  - The generated code is a complete, self-contained experiment — the harness only orchestrates, validates structure, and captures telemetry.

---

## 3. System Components: The Multi-Agent Swarm

### 3.1. Principal Investigator (PI) Agent
- **Role:** Sets the macro-goal, formulates new hypotheses for ANY AI domain.
- **Inputs:** Previous axioms, target objective, experiment configuration (dataset paths, env names), domain context.
- **Outputs:** Strict `ResearchProposal` with framework choice, dependencies, and domain-appropriate evaluation criteria.

### 3.2. Theorist & Mathematician Agent
- **Role:** Translates the `ResearchProposal` into formal architecture specification.
- **Inputs:** `ResearchProposal`.
- **Outputs:** `ArchitecturalBlueprint` (architecture logic, dimensionality rules, algorithm pseudocode) — framework-agnostic.

### 3.3. Software Architect & Synthesis Agent
- **Role:** Implements the `ArchitecturalBlueprint` as executable code in the chosen framework.
- **Inputs:** `ArchitecturalBlueprint`, proposal's framework/dependency context.
- **Outputs:** Complete, self-contained Python module with `create_model()` and `run_training()`.
- **Critical Rule:** The code MUST contain its own data loading, training loop, and metric reporting. The sandbox does NOT provide any of these.

### 3.4. Debug / QA Agent
- **Role:** Engaged solely if sandboxes fail. Does not theorise.
- **Inputs:** Python Traceback, failing code.
- **Outputs:** Patched complete code.

### 3.5. Data Scientist Agent
- **Role:** Post-execution analytics.
- **Inputs:** Telemetry metrics from `run_training()` output.
- **Outputs:** `ExperimentSummary` (conclusions, axioms, failure modes).

---

## 4. The Memory Subsystem

1. **Episodic Memory (Vector DB):** Stores experimental runs for deduplication.
2. **Semantic Memory (Knowledge Graph):** Maps concepts across ALL AI domains.
3. **Distillation Engine (Axiom Builder):** Compresses experiments into permanent rules injected into PI Agent's prompt.

---

## 5. Execution Flow

### Phase 1: Ideation
1. Harness receives target metric + experiment config (dataset path, env name, hardware).
2. PI Agent reads axioms + episodic memory, formulates a `ResearchProposal`.

### Phase 2: Formalization
1. Theorist translates proposal into `ArchitecturalBlueprint`.
2. Blueprint is validated for completeness (not empty fields).

### Phase 3: Code Synthesis
1. Architect synthesises a **complete, self-contained Python module** in the chosen framework.
2. Module MUST export: `create_model(**kwargs)` and `run_training(model, config)`.
3. `run_training()` handles its own data loading, training loop, and returns a metrics dict.

### Phase 4: Tiered Validation
1. **Level 0:** AST + Ruff static checks.
2. **Level 1:** Import module, call `create_model()`, verify no crashes.
3. **Level 2:** Call `run_training()` with the experiment config, capture returned metrics.
4. On failure: Debug Agent patches → retry (max 3).

### Phase 5: Analysis & Distillation
1. Data Scientist analyses metrics from `run_training()` output.
2. Episodic + Semantic memory updated.
3. Axioms distilled if threshold met.

---

## 6. Technology Stack

- **Orchestration:** `langgraph`
- **LLM Abstraction:** `litellm`
- **Data Validation:** `pydantic`
- **Episodic Memory:** `chromadb`
- **Semantic Memory:** `networkx` (local, JSON-serialised)
- **Lineage DB:** `SQLAlchemy` over SQLite
- **Static Analysis:** `ruff`, `ast`
- **Sandbox:** Subprocess-based with dynamic `pip install`

---

## 7. Universality & Safety

### 7.1. Domain Agnosticism
- The system works for: Language Models, Vision Models, RL Agents, World Models, GANs, Diffusion Models, Bio-ML, AutoML, and any other AI paradigm.
- The PI Agent chooses the framework and the Architect writes framework-appropriate code.
- The sandbox installs whatever dependencies the proposal declares.

### 7.2. Governor (Budget & Safety)
- Debug retries capped at 3 per hypothesis.
- Consecutive failure halt at 5.
- Theorist revisions capped at 3.
- Sandbox hard timeout (configurable, default 300s).
