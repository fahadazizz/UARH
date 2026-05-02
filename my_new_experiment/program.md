# Autonomous Research Specification

## 1. Primary Objective
<!-- Specify the core target of this experiment (e.g., "Design a Character-Level GAN", "Implement a PPO RL agent for CartPole", "Train a Diffusion Language Model"). -->
[INSERT PRIMARY OBJECTIVE HERE]

## 2. Architecture Demands & Constraints
<!-- Detail the structural requirements for the Neural Network, Pipeline, or Mathematical model. -->
- [e.g., Must use Pre-Layer Normalization]
- [e.g., Weight tying across embedding and unembedding]

## 3. Training & Algorithmic Strategy
<!-- Detail the exact optimization loops, loss functions, or heuristics the Architect should implement inside `run_training(model, config)`. -->
- [e.g., Use alternating optimization for Generator/Discriminator]
- [e.g., Track both L1 loss and adversarial binary cross-entropy]

## 4. Evaluation Metrics
<!-- Define the metrics that the Data Scientist should extract from the training output. -->
- [e.g., generator_loss]
- [e.g., discriminator_loss]

## 5. Additional Context
<!-- Any literature references, ablation study specifics, or known issues the Principal Investigator should avoid. -->
- [e.g., Avoid standard autoregressive loss, prioritize parallel decoding]
