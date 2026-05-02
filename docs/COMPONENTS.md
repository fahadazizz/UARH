# AGENT COMPONENTS: The Cognitive Engine

UARH utilizes a **Heterogeneous Agent Swarm**. This means each agent is not just a general-purpose LLM, but is specialized through "Prompt Injection" and "Role-Specific Constraints" to think like a specific member of a high-tier research lab.

---

## 1. The Principal Investigator (PI)
**The Theoretical Visionary**

The PI is the highest-level agent in the UARH ecosystem. Its primary goal is to maintain the "Research Direction." 

- **Cognitive Profile**: Strategic, highly creative, and mathematically rigorous.
- **Input**: User's `program.md`, `config.yaml`, and the `axiom_repository`.
- **Reasoning Process**:
    1. It analyzes the user's primary objective.
    2. It cross-references current goals with past failures (Axioms).
    3. It formulates a **Strategic Hypothesis** (e.g., "Standard self-attention is O(N^2); we should implement a Performer-style linear attention to handle the long sequences in the dataset").
- **Constraint**: The PI is forbidden from writing implementation code. It must speak only in high-level scientific terms.

---

## 2. The Theorist
**The Mathematical Bridge**

The Theorist acts as the bridge between "Vision" (PI) and "Engineering" (Architect). It translates abstract ideas into technical requirements.

- **Cognitive Profile**: Methodical, formal, and precise.
- **Responsibilities**:
    - **Hyperparameter Estimation**: Suggesting appropriate learning rates based on the domain.
    - **Mathematical Formalization**: Defining the exact loss function (e.g., "Contrastive InfoNCE loss" instead of just "Contrastive loss").
    - **Invariant Enforcement**: Specifying that certain layers must remain frozen or that specific normalization must be used.

---

## 3. The Architect
**The Senior Engineer**

The Architect is the "Implementer." It is responsible for the actual synthesis of Python code.

- **Cognitive Profile**: Pragmatic, clean-code focused, and an expert in deep learning frameworks (PyTorch, JAX).
- **Core Output**: A self-contained Python module that includes:
    1.  The model class (e.g., `nn.Module`).
    2.  A robust `run_training(model, config)` entrypoint.
- **Production Standards**: The Architect is prompted to use **Modular Design**. It avoids monolithic scripts and prioritizes code that can be easily inspected and debugged.

---

## 4. The Scientist
**The Objective Truth-Seeker**

The Scientist is the only agent that interacts with the "Real World" (the sandbox). It is the judge of success.

- **Cognitive Profile**: Skeptical, data-driven, and analytical.
- **Process**:
    1. It monitors the stdout of the sandbox.
    2. It parses the loss values and validation metrics in real-time.
    3. It performs **Outcome Classification**:
        - **SUCCESS**: Significant improvement in the target metric.
        - **LOGICAL FAILURE**: Code ran but the model did not learn.
        - **RUNTIME ERROR**: The code crashed.
- **Critical Role**: The Scientist decides when an experiment is "Distillable" (worthy of becoming an Axiom).

---

## 5. The Debugger
**The Forensic Investigator**

The Debugger is only summoned when the state graph detects a failure. 

- **Cognitive Profile**: Detail-oriented, logical, and highly technical.
- **The Forensic Loop**:
    1. It receives the **Crash Context** (The exact line of code that failed + the full stack trace).
    2. It analyzes the trace (e.g., "Dimension mismatch at layer 3: expected 512, got 256").
    3. It creates a **Fix Directive**. This is not a fix itself, but a set of instructions for the Architect to prevent a "Fix-Break" cycle.
