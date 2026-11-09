---
name: audit-operator
description: Independently audit an Asim operator against its frozen task, algorithm, actual instructions and numerical evidence. Use for op-flow review and when instruction correctness or claimed speedups are disputed.
---

# Independent Asim operator audit

Read [op-flow](../op-flow/SKILL.md), `AGENTS.md`, the operator design, frozen algorithm/precision contract and C++ generation code. In Codex, perform the review with an available sub-agent/tool; never claim a slash command ran when unavailable. Review is evidence, not a substitute for dynamic gates.

Establish the target first: inverse, intermediate preconditioned variable, or MMSE output. Check derivation from H/Y/λ, initialization, block definition, iteration feedback, preconditioner and output recovery. A Python helper or chapter is a source to audit, not automatically an authority; compare its definitions independently. Direct MMSE is not required to materialize an inverse.

Inspect actual Instruction constructors and exported views together:

- Opcode/arity, base, offsets, shape/stride, transpose/conjugate and explicit broadcast agree with the formula and raw workload fields. Unknown/ambiguous semantics block acceptance.
- Full regions fit capacities and Model tensors include every batch. Real writes precede all reads; first partial writes must not underallocate physical storage. Verify constants: ADD(Reg,Reg) is 2λI; unity is 1, not 0.
- Real feedback reaches the next iteration; source/destination aliasing and barriers agree with actual observed read/write order. State copies/compensation and initialization are charged.
- Each SCALAR operates on one element. Reductions, triangular solves and matrix updates have real instruction work. No single fixed-latency SCALAR stands for a vector or hidden solve.
- FP16 lowering preserves declared rounding and signed zero; reassociation, diagonal GEMM→MUL and packing need independent bit checks. Arithmetic/storage assumptions do not imply hardware certification.
- Iterative coefficients satisfy the intended stability assumptions. Changing damping/momentum changes the recurrence: recheck α/β together instead of reusing the original Chebyshev interval as a guarantee. Calibration success is not a universal proof.

Do not demand identical layouts, barriers or instruction counts from different algorithms. Demand the same external task, precision and full timing boundary for a performance comparison. Preprocessing, preconditioner/coefficients, initialization, recovery and DMA are included. Reject suspicious speedups caused by skipped work.

Dynamic evidence must distinguish:

1. Real Asim and actual event/binding/performance checks; failed/skipped candidates have numerics NOT_RUN.
2. Common-engine replay of actual operations versus an independent same-algorithm precision reference.
3. Task accuracy versus high-precision MMSE/inverse target and applicable QPSK gates.

FormulaLogger/DAG self-consistency, matching operation counts, a report's PASS label, or low average BER alone is insufficient. Inverse error passing does not prove inverse-then-apply MMSE passes. Preserve quality failures even when replay is bit-exact. Verify complete case/batch coverage, frozen source/reference/toolchain closure, fresh holdout separation and immutable artifact hashes.

Report findings with path/line, severity, evidence, affected scope and remaining gates. Use STATIC_REVIEW_PASSED only for static review; it is not full operator acceptance. Record observable review wall time; pure model thinking time is null. Keep historical failures and final score null when promotion is incomplete.
