# Universal Autonomous Research Harness (UARH)

[![Version](https://img.shields.io/badge/version-v1.1-blue.svg)](https://github.com/fahadazizz/UARH)
[![Domain](https://img.shields.io/badge/domain-Any_AI-orange.svg)](https://github.com/fahadazizz/UARH)

The Universal Autonomous Research Harness (UARH) is a production-grade, domain-agnostic research framework designed to bridge the gap between human intuition and machine-speed experimentation. It is not merely a code generator; it is a full-lifecycle autonomous scientific agent capable of traversing the entire research stack—from high-level theoretical hypothesis generation to the low-level management of hardware-bound training kernels.

## 🌟 Research Philosophy
Traditional AI research is bottle-necked by the "Human-in-the-loop" latency: writing boilerplate training code, debugging tensor shape mismatches, and manually tracking metrics. UARH inverts this paradigm. By offloading the mechanical and engineering burdens to a graph of specialized agents, the human researcher moves to a "Governor" role, defining the objective function and constraints while the machine explores the search space of architectures and algorithms.

### Why UARH?
- **Domain Agnosticism**: Unlike task-specific tools, UARH treats RL, NLP, GANs, and World Models as first-class citizens by synthesizing the execution logic on-the-fly.
- **Cognitive Specialization**: It utilizes a multi-agent swarm where reasoning (PI), engineering (Architect), and analysis (Scientist) are isolated to prevent cognitive bleed and hallucination.
- **Fail-Safe Iteration**: Built-in 3-level validation ensures that even the most "hallucinatory" architectural ideas are caught and debugged before they consume significant compute.

## 🚀 Getting Started (Comprehensive)

### Prerequisites
- **Python 3.10+**: Required for modern typing and async orchestration.
- **Ollama / LiteLLM Backend**: Access to a reasoning-capable LLM (e.g., Qwen 2.5, Llama 3, or Claude-3.5 via API).
- **GPU (Optional but Recommended)**: Support for `cuda` (NVIDIA) or `mps` (Apple Silicon).

### Installation & Environment Setup
```bash
# Clone the repository
git clone https://github.com/fahadazizz/UARH.git
cd UARH

# Setup a clean virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install core and research dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### The "First Light" Experiment
UARH uses the **`program.md`** and **`config.yaml`** paradigm to ensure reproducibility.

1.  **Initialize**: `python -m uarh.main init my_first_study`
2.  **Define**: Open `my_first_study/program.md` and define a goal (e.g., "A Transformer with Linear Attention").
3.  **Configure**: Edit `my_first_study/config.yaml` to set your hardware and models.
4.  **Launch**: `python -m uarh.main launch my_first_study --tui`

## 📂 Documentation Deep-Dive
To truly understand the internal mechanics of UARH, please refer to the following comprehensive technical manuals:

- 🏗️ **[Architecture & Flow](docs/ARCHITECTURE.md)**: Deep dive into the State-Graph, tiered validation levels, and the Governor logic.
- 🤖 **[Agent Components](docs/COMPONENTS.md)**: Breakdown of the PI, Theorist, Architect, Scientist, and Debugger reasoning patterns.
- ⚙️ **[Usage & Configuration](docs/USAGE.md)**: Comprehensive guide to the YAML schema and the Markdown-based prompting interface.
- 🛡️ **[Sandboxing & Security](docs/SANDBOXING.md)**: How UARH executes arbitrary code safely, handles timeouts, and manages dependencies.

## 🗺️ Project Roadmap
- [x] v1.0: Core Multi-Agent Graph Implementation.
- [x] v1.1: Standardized Config/Program paradigm and TUI integration.
- [ ] v1.2: Multi-Node training support and distributed orchestration.
- [ ] v1.3: Native integration with Weights & Biases / MLFlow for artifact tracking.

## 🤝 Contribution
UARH is an open-research platform. We welcome contributions in the form of new agent prompts, better sandbox isolation strategies, and support for additional research domains.

## ⚖️ License
Distributed under the MIT License. See `LICENSE` for more information.
