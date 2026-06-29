# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Asim is a multi-core NPU cycle-level simulator (forked from ONNXim). It explicitly models Core (Cube/Vector/MTE), SRAM/SPAD/ACCUM, NoC (Simple/Booksim2), and DRAM (Simple/Ramulator1/2), producing cycle counts, utilization metrics, and instruction-level traces.

## Build

```bash
cmake -S . -B build
cmake --build build --target Simulator -j$(nproc)
```

CMake options:
- `USE_RAMULATOR` (default ON) — include Ramulator2 DRAM model
- `BRI_ENABLE_WEIGHTED_UPDATE` (default OFF) — experimental weighted residual in BlockRichardsonOp
- `BRI_ENABLE_PRECOND_ELEM_ADDR` (default OFF) — experimental element-addressed preconditioner

The primary active build directory is `build_asim/`. Binary lands at `build_asim/bin/Simulator`.

## Run Simulator

```bash
export ONNXIM_TRACE_CSV=results/trace.csv           # optional: emit trace CSV
export ONNXIM_FORMULA_JSON=/tmp/formula.json        # optional: emit formula metadata

./build_asim/bin/Simulator \
  --config configs/ascend_910b_quiet.json \
  --models_list example/<test>.json \
  --mode <mode> \
  --log_level info
```

## Test

```bash
cmake --build build --target Simulator_test -j$(nproc)
./Simulator_test  # GTest binary, output in repo root
```

GTest is fetched as an external project. Tests live in `tests/` and link against `Simulator_lib`.

## Architecture

### Execution model

`Simulator` owns a main loop that cycles cores, interconnect, and DRAM. Models are registered into the simulator, each producing Tiles (work units) dispatched by a Scheduler onto Cores.

**Flow**: `main.cc` parses mode → creates a `Model` subclass → model creates `Operation` objects → operations generate `Tile`s with `Instruction` lists → `Simulator::run_simulator()` executes cycle-by-cycle.

### Key classes

- **`Operation`** (`src/operations/Operation.h`) — abstract base for all compute nodes. Subclasses override `initialize_tiles()` and `initialize_instructions()`. Instructions are the core output: they spell out `MOVIN`/`MOVOUT` (data movement), `GEMM_PRELOAD` (Cube matmul), and vector ops (`ADD`, `MUL`, `MAC`, `DIV`, `EXP`, etc.) with SPAD addresses.
- **`Model`** (`src/Model.h`) — represents a workload graph. Creates `Operation` nodes connected by `Tensor` edges.
- **`Instruction`** (`src/Common.h:76`) — the fundamental unit of work, containing `opcode`, source/dest addresses, tile dimensions, and timing. Full `Opcode` enum at `src/Common.h:49`.
- **`SimulationConfig`** (`src/SimulationConfig.h`) — hardware parameters: core count/size, Cube dimensions, vector latencies, DRAM/NoC type, frequency, etc. Loaded from JSON config files.
- **`Core` / `Dram` / `Interconnect`** — cycle-accurate hardware models executed each tick.
- **`FormulaLogger`** (`src/FormulaLogger.h`) — emits algorithm metadata per instruction step for UOBS black-box scoring. Must be called in every operator's `initialize_instructions()`.
- **`TraceLogger`** (`src/TraceLogger.h`) — writes CSV trace when `ONNXIM_TRACE_CSV` is set.
- **`OperationFactory`** (`src/operations/OperationFactory.cc`) — maps ONNX op_type strings to C++ classes.

### Addressing

SPAD uses base addresses (`SPAD_BASE = 0x10000000`, `ACCUM_SPAD_BASE = 0x20000000`). DRAM addresses are linear offsets. `Operation::make_address()` maps logical indices to linear offsets; `get_operand_addr()` resolves operand IDs.

### Ascend Cube model

When `ascend_cube_model.enabled = true` in config, GEMM instructions are tiled into `cube_m × cube_n × cube_k` sub-blocks (typically 16×16×16). The cycle model is in `SystolicWS::get_inst_compute_cycles()`.

### Adding new operators (Path B — C++ model)

1. Create `Operation` subclass in `src/operations/` — implement `initialize_tiles()` and `initialize_instructions()`
2. Create `Model` subclass in `src/models/` — creates and connects the operation
3. Add mode branch in `src/main.cc`
4. Create test JSON in `example/`
5. Register in `orchestrator/operator_registry.json` (for `/eval-patch` and `/opt-round`)
6. Call `FormulaLogger::set_algorithm()` + `FormulaLogger::emit_step()` for each math step
7. Add source files to `src/CMakeLists.txt` (may already be covered by `GLOB_RECURSE`, but explicit listing is safer)

See `DOCS/OPERATOR_DEVELOPMENT_STANDARD.md` for the full v2.0 standard with UOBS integration.

## Project-Specific Skills

Three skills are configured in `.claude/skills/`:
- **`/eval-patch`** — black-box evaluate an operator patch (Score/Cycle/Cube%/Error)
- **`/op-dev`** — scaffold a new operator following the v2.0 standard
- **`/opt-round`** — automated multi-round AI-driven operator optimization

## Dependencies (Conan + Git Submodules)

Conan packages: `boost/1.79.0`, `robin-hood-hashing/3.11.5`, `spdlog/1.11.0`, `nlohmann_json/3.11.2`.

Git submodules: `booksim`, `protobuf`, `onnx`, `ramulator_custom`, `ramulator2`.

## Code conventions

- C++20 (`CMAKE_CXX_STANDARD 20`)
- `_GLIBCXX_USE_CXX11_ABI=0` (legacy ABI)
- ASan enabled in Debug builds (`-fsanitize=address`)
- Operator attributes are passed as `std::map<std::string, std::string>` — always strings, parse in `parse_attributes()`
- Tiles use `std::unique_ptr` throughout; `std::deque<std::unique_ptr<Tile>>` is the tile container
- Header-only lib `robin_hood` provides the hash map; `nlohmann/json` for all JSON needs
- `spdlog` for logging — levels: trace, debug, info
- Use `spdlog::info("msg")` style; prefer lambda helpers for repeated SPAD address patterns in `initialize_instructions()`
