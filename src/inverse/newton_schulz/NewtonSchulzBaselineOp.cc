#include <algorithm>
#include "NewtonSchulzBaselineOp.h"
#include "Model.h"
#include "FormulaLogger.h"
#include <set>

NewtonSchulzBaselineOp::NewtonSchulzBaselineOp(
    SimulationConfig config, Model* model, const std::string& name,
    std::map<std::string, std::string>& attributes, uint32_t target_core)
    : Operation(config, model, name, attributes, target_core) {
  _optype = "NewtonSchulzBaselineOp";
  parse_attributes();
}

void NewtonSchulzBaselineOp::parse_attributes() {
  auto it = _attributes.find("batch_size");
  if (it != _attributes.end()) try { _batch_size = stoul(it->second); } catch(...) {}
  it = _attributes.find("iterations");
  if (it != _attributes.end()) try { _iterations = std::max(1u, (uint32_t)stoul(it->second)); } catch(...) {}
  if (_matrix_shape.empty()) _matrix_shape = {32, 32};
}

void NewtonSchulzBaselineOp::initialize_tiles(MappingTable&) {
  for (uint32_t b = 0; b < _batch_size; ++b) {
    auto tile = std::make_unique<Tile>(Tile{
        .status = Tile::Status::INITIALIZED, .optype = _optype,
        .layer_id = _id, .batch = b, .core_id = static_cast<int>(b % _config.num_cores)});
    initialize_instructions(tile.get(), Mapping{});
    if (!tile->instructions.empty()) _tiles.push_back(std::move(tile));
  }
}

void NewtonSchulzBaselineOp::initialize_instructions(Tile* tile, Mapping) {
  const uint32_t N = _matrix_shape[0];  // N×N matrix
  const uint32_t U = _matrix_shape[1];
  (void)U;
  const uint32_t K = _iterations;
  const addr_type size_nn = static_cast<addr_type>(N) * N * _config.precision;
  const uint32_t epa = std::max(1u, _config.dram_req_size / _config.precision);

  const addr_type dA = get_operand_addr(_INPUT_OPERAND + 0);
  const addr_type dX0= get_operand_addr(_INPUT_OPERAND + 1);
  const addr_type dC = get_operand_addr(_INPUT_OPERAND + 2);
  const addr_type dOut=get_operand_addr(_OUTPUT_OPERAND + 0) + static_cast<addr_type>(tile->batch) * size_nn;

  const addr_type aA  = SPAD_BASE;
  const addr_type aX  = aA  + size_nn;
  const addr_type aC  = aX  + size_nn;
  const addr_type aT  = aC  + size_nn;
  const addr_type aR  = aT  + size_nn;
  const addr_type aAinv= ACCUM_SPAD_BASE;

  const std::vector<uint32_t> sNN{N, N};
  FormulaLogger::instance().set_algorithm("newton_schulz_v3", 0, K, N);

  auto movin = [&](addr_type dram, addr_type spad, uint32_t rows, uint32_t cols,
                   const std::vector<uint32_t>& shape, uint32_t op_id) {
    std::set<addr_type> addrs;
    for (uint32_t r = 0; r < rows; ++r)
      for (uint32_t c = 0; c < cols; c += epa)
        addrs.insert(dram + make_address({r, std::min(c, cols-1)}, shape));
    tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
        .opcode = Opcode::MOVIN, .dest_addr = spad,
        .size = static_cast<uint32_t>(addrs.size()),
        .src_addrs = std::vector<addr_type>(addrs.begin(), addrs.end()),
        .operand_id = op_id, .base_addr = 0,
        .tile_m = rows, .tile_k = cols, .my_tile = tile}));
  };

  auto barrier = [&](const std::string& id, uint32_t type) {
    tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
        .opcode = Opcode::PIPE_BARRIER, .id = id,
        .my_tile = tile, .is_barrier = true, .barrier_type = type}));
  };

  // Phase 1: MOVIN
  // Load input matrices per-batch (A, X_init, C=2I are all N×N per batch)
  addr_type bOff = static_cast<addr_type>(tile->batch) * size_nn;
  movin(dA  + bOff, aA, N, N, sNN, _INPUT_OPERAND + 0);
  movin(dX0 + bOff, aX, N, N, sNN, _INPUT_OPERAND + 1);
  movin(dC,         aC, N, N, sNN, _INPUT_OPERAND + 2);
  barrier("NS_LOAD", 1);

  // Phase 2: Newton-Schulz iterations
  // X_{k+1} = X_k @ (2I - A @ X_k)
  for (uint32_t k = 0; k < K; ++k) {
    // T_k = A @ X_k
    tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
        .opcode = Opcode::GEMM_PRELOAD,
        .id = "NS_T_" + std::to_string(k),
        .dest_addr = aT, .compute_size = N,
        .src_addrs = {aA, aX},
        .tile_m = N, .tile_k = N, .tile_n = N, .my_tile = tile}));
    FormulaLogger::instance().emit_step("NS_GEMM_T_" + std::to_string(k), "GEMM",
        {std::string("A"), std::string(k==0?"X":"X_"+std::to_string(k-1))}, "T_" + std::to_string(k), {{N,N},{N,N}}, {N,N}, tile->batch,
        "NS_T_" + std::to_string(k));

    barrier("NS_T2R_" + std::to_string(k), 2);

    // R_k = 2I - T_k  (C=2I loaded, subtract T_k)
    tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
        .opcode = Opcode::ADD,
        .id = "NS_R_" + std::to_string(k),
        .dest_addr = aR, .compute_size = N * N,
        .src_addrs = {aC, aT},
        .tile_m = N, .tile_k = N, .tile_n = N, .my_tile = tile}));
    FormulaLogger::instance().emit_step("NS_RESIDUAL_" + std::to_string(k), "MATRIX_SUB",
        {"2I", "T_" + std::to_string(k)}, "R_" + std::to_string(k), {{N,N},{N,N}}, {N,N}, tile->batch,
        "NS_R_" + std::to_string(k));

    barrier("NS_R2X_" + std::to_string(k), 3);

    // X_{k+1} = X_k @ R_k  (write to aX, overwriting old X)
    tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
        .opcode = Opcode::GEMM,
        .id = "NS_X_" + std::to_string(k),
        .dest_addr = aX, .compute_size = N,
        .src_addrs = {aX, aR},
        .tile_m = N, .tile_k = N, .tile_n = N, .my_tile = tile}));
    FormulaLogger::instance().emit_step("NS_UPDATE_" + std::to_string(k), "GEMM",
        {std::string(k==0?"X":"X_"+std::to_string(k-1)), "R_" + std::to_string(k)}, "X_" + std::to_string(k), {{N,N},{N,N}}, {N,N}, tile->batch,
        "NS_X_" + std::to_string(k));

    if (k + 1 < K) barrier("NS_ITER_" + std::to_string(k), 4);
  }

  // Emit final BWD assembly: Ainv = X_{K-1} @ X_{K-1} (X is symmetric, no transpose needed)
  FormulaLogger::instance().emit_step("NS_BWD_ASSEMBLE", "GEMM",
      {"X_" + std::to_string(K-1), "X_" + std::to_string(K-1)}, "Ainv", {{N,N},{N,N}}, {N,N}, tile->batch, "NS_FINAL");

  // Phase 3: Final GEMM to ACCUM and MOVOUT
  tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
      .opcode = Opcode::GEMM_PRELOAD,
      .id = "NS_FINAL",
      .dest_addr = aAinv, .compute_size = N,
      .src_addrs = {aX, aX},
      .tile_m = N, .tile_k = N, .tile_n = N, .my_tile = tile}));
  barrier("NS_PRE_MOVOUT", 6);

  std::set<addr_type> outs;
  for (uint32_t r = 0; r < N; ++r)
    for (uint32_t c = 0; c < N; c += epa)
      outs.insert(dOut + make_address({r, std::min(c, N-1)}, sNN));
  tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
      .opcode = Opcode::MOVOUT, .id = "NS_STORE",
      .dest_addr = aAinv, .size = static_cast<uint32_t>(outs.size()),
      .src_addrs = std::vector<addr_type>(outs.begin(), outs.end()),
      .operand_id = _OUTPUT_OPERAND,
      .tile_m = N, .tile_k = N, .tile_n = N,
      .src_from_accum = true, .last_inst = true, .my_tile = tile}));
}
