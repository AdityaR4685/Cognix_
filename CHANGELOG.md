# Changelog

All notable changes to this project will be documented in this file.

## [0.1.0] - Unreleased

### Added
- Initial distribution packaging for `cognix-uncertainty-framework` on PyPI/TestPyPI
- Domain-agnostic generic decision pipeline (`DecisionEngine`) with dependency-injected strategies
- Standardized agent interfaces (`AgentInterface`, `StandardAgent`, `AgentRegistry`)
- Core uncertainty estimation methods (Monte Carlo Dropout, Deep Ensembles, epistemic/aleatoric decomposition)
- Multi-agent belief and evidence fusion (Bayesian belief updates, epistemic weighted fusion)
- Inductive conformal prediction (`ConformalPredictor`) and risk assessment/escalation (`EscalationEngine`)
- Fine-grained optional dependency extras (`[graph]`, `[vision]`, `[explainability]`, `[carla]`, `[llm]`, `[marl]`, `[dashboard]`, `[dev]`, `[all]`)
- Decoupled runtime architecture allowing clean installation and execution without requiring PyTorch or simulator runtimes

## [0.1.0.dev0] - 2026-09-07

### Added
- Initial project skeleton
- Basic utilities and configurations
- Configuration schema and default configs
