---
name: op-flow
description: Develop or optimize an Asim operator with a frozen task, isolated build, actual-instruction reconstruction and independent numerical acceptance. Use for new operator implementations and instruction/layout optimizations.
---

# Asim operator development

This is the current operator workflow for Claude Code and Codex. In Codex, read and execute these steps; do not claim an unavailable slash command ran. Historical `orchestrator/pipeline.json`/UOBS scripts remain available for old experiments, but their optional FormulaLogger/DAG check cannot certify a new candidate.

## Before implementation

Read `AGENTS.md`, `CLAUDE.md`, `DOCS/OPERATOR_DEVELOPMENT_STANDARD_V3.md`, `DOCS/research/ADDING_OPERATOR.md`, and the applicable frozen algorithm experiment. Create `DOCS/operators/<operator>.md` before code: file inventory, formula derivation, SPAD/ACCUM layout, formula→instruction/view mapping, trace metadata, and verification scope.

Freeze the actual task and algorithm separately. For MMSE, input H/Y/λ and output xhat; deriving an explicit inverse is optional. Record initialization, preconditioner, real/complex block definition, recurrence, K, spectral bounds, all storage/temporary precision and rounding points. Calibration selects K/parameters before fresh holdout. Never tune on holdout and call its rerun independent acceptance.

Use an independent worktree/build/log directory. On the current 2-vCPU host, build and simulation are serial (`-j1`); check memory/disk and bound runs. Preserve historical results, user files and old operators. Candidate scope excludes evaluator/reference/threshold changes; prepare and review trusted integration changes separately.

## Generate executable semantics and costs together

At each actual C++ `Instruction` constructor, attach typed source/destination views using `src/semantic/SourceViews.h`; include region base, offsets, shape, transpose/conjugate and explicit scalar broadcast. Header binds Model tensors, regions/capacities, task, algorithm and numeric policy. Export comes from the real Operation, not an independent handwritten formula stream.

Use the shared `engine`/`source_views`/`binding`/runtime checker. Add a new trusted adapter and independent algorithm reference, not a solver-name branch in the replay engine. Unknown operations, absent inputs or unsupported precision fail closed.

Non-obvious constraints:

- Every read must have a real preceding write/LOAD. `ADD(RegI,RegI)` is 2λI, not COPY/zero/identity. Explicit scalar 0/1 synthesis or declared common inputs are allowed and charged.
- Scalar work is one element. Batch/row reductions need real GEMM, Vector or scalar sequences with corresponding costs. Do not hide a matrix solve/reduction in FormulaLogger.
- Raw Vector has no SUB in the accepted profile: use explicit negation MUL + ADD, or genuine per-element SCALAR_SUB. Freeze intermediate writebacks and check signed zero.
- Mathematical equivalence may change FP16 bits: reassociation, diagonal GEMM→MUL, copy/negation lowering, and output packing require independent precision checks and adversarial zero tests.
- Allocate the complete physical region before partial writes where Asim allocates only the first destination span. All slices/strides/output batches must fit the declared tensor. A logical view alone does not expand an existing allocation.
- `MOVIN` uses full DRAM addresses with `base_addr=0`; charge input/output DMA, preprocessing, preconditioning, coefficients, initialization, state copies/compensation and output recovery. A direct solve cannot claim inverse-task speedup.
- Declare actual barriers and check actual read/write events. Do not impose identical layouts/instruction counts on different algorithms; enforce the same task, primitive semantics, precision and cost boundaries.
- FormulaLogger is a diagnostic index, not executable correctness evidence. Do not “fix” only the log to make a candidate pass.

## Build, audit and evaluate

1. Build `Simulator` in the isolated build, retain configure/build exit status and logs.
2. Have an independent audit sub-agent compare derivation, C++ and frozen reference. Resolve findings and retain the audit. This explicit operator review is authorized by this skill; avoid concurrent builds/simulations.
3. Freeze source/binary/library/config/task/algorithm/evaluator versions. Real Asim must finish and pass instruction/event/binding checks and the declared performance prefilter before candidate reconstruction. Failed or noncompetitive runs are numerics NOT_RUN with score=null. Correctness baseline creation is explicit, not a claim of optimization.
4. Reconstruct the actual operations with the common engine; compare complete output against a separately implemented same-algorithm precision model, then FP64 task reference. Numerical acceptance is mandatory. Unsupported, mismatch and algorithm-quality failure are distinct outcomes. Never clip/replace nonfinite values or invent semantics.
5. Run applicable Rayleigh/CDL-B inputs, task precision gates and uncoded QPSK tests. State exact channel/geometry/SNR/shape/seed coverage. An inverse-error pass does not imply inverse-then-apply MMSE passes. FEC and hardware timing are not prerequisites for the current software task.
6. Verify negative cases, reproducibility and immutable artifact hashes. Report total cycles and detailed pipeline statistics only for accepted comparable task identities. Utilization diagnoses useful work; it does not reward redundant GEMM. Do not score before the complete applicable promotion gates.

## Archive and report

Use a new `results/<campaign>/<version>/` and small tracked `experiments/<campaign>/<version>/` summaries/manifests. Preserve failures/retries and original frozen inputs. Record Asim, event-check, reference and replay wall time separately; pure model thinking time is unobservable (`null`). Document software-model assumptions versus board-level evidence.

Update the operator document and current research index/decisions. Paper snapshots go under the fixed `DOCS/research/paper_archive/README.md` index, never overwrite old versions. Publish a new pinned runtime when its dependencies change; never edit an old runtime's hashes to claim continuity. Commit logical changes without automatic push.
