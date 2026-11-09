---
name: op-dev
description: Scaffold a new Asim NPU operator and its typed actual-instruction reconstruction adapter. Use when implementing an algorithm or adding a new simulator operator entry.
---

# New Asim operator

Follow [op-flow](../op-flow/SKILL.md) before code. It is the shared current workflow; legacy UOBS/reference-inverse registration alone is insufficient.

Add an isolated Model/Operation under `src/inverse/<variant>/`, a mode in `src/main.cc`, and an example model config. `src/CMakeLists.txt` uses CONFIGURE_DEPENDS recursive source discovery. Produce actual `Instruction` views with `src/semantic/SourceViews.h`, not a second artificial FormulaLogger execution plan.

The evaluator owns `orchestrator/codex_operators.json` and trusted parameter/reference modules. Register scope, accepted profiles, iteration limits, full algorithm parameters and independent reference. See `DOCS/research/ADDING_OPERATOR.md` for the adapter protocol and current limits. Old `orchestrator/operator_registry.json` remains the historical Claude route; do not repurpose its old entries or scores for a new candidate.

Full input/output tensors and all constants/buffers must be accounted for. MMSE direct solve and explicit inverse are different tasks. Run the op-flow audit, real Asim prefilter, common reconstruction, same-algorithm precision reference, task and applicable communication gates before claiming success. Keep score null until promotion is complete.
