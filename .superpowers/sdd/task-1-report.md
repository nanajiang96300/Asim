# Task 1 Report: Add Mandatory Operator Pipeline Rule to CLAUDE.md

## What was changed
Appended a new "## Operator Development Pipeline (MANDATORY)" section to `/home/nanajiang/Asim/CLAUDE.md` after the "Code conventions" section. The block requires all operator development/optimization work to go through the `/op-flow` pipeline and enforces 5 stages (design doc, v3.0 standard, CMake build, /audit-operator review, benchmark archiving).

## Verification output
```
## Operator Development Pipeline (MANDATORY)

When developing a NEW operator or optimizing an EXISTING operator, you MUST invoke `/op-flow <operator_name> <action>` before writing any code. The pipeline enforces:

1. Design document with formula derivation exists before code
2. Code follows v3.0 standard (DOCS/OPERATOR_DEVELOPMENT_STANDARD_V3.md)
3. CMake build passes
4. /audit-operator sub-agent review passes (formula↔code consistency)
5. Benchmark results archived to results/<operator>/

**NEVER skip the pipeline.** If you write operator code without running /op-flow first, stop, delete the code, and run /op-flow.
```

## Commit hash
`014a89a7f3566524c6dc1446991f4a117e9dd93a`
