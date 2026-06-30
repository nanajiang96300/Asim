#include "LDLNoBlockMergeOp.h"
#include "Model.h"
#include "FormulaLogger.h"
#include <set>

LDLNoBlockMergeOp::LDLNoBlockMergeOp(
    SimulationConfig config, Model* model, const std::string& name,
    std::map<std::string, std::string>& attributes, uint32_t target_core)
    : Operation(config, model, name, attributes, target_core) {
  _optype = "LDLNoBlockMergeOp";
  parse_attributes();
  infer_shapes_from_model();
}

void LDLNoBlockMergeOp::parse_attributes() {
  auto it_batch = _attributes.find("batch_size");
  if (it_batch != _attributes.end()) {
    try { _batch_size = static_cast<uint32_t>(std::stoul(it_batch->second)); }
    catch (...) { _batch_size = 96; }
  }
}

void LDLNoBlockMergeOp::infer_shapes_from_model() {
  if (!_matrix_shape.empty() || !_model) return;
  if (_inputs.empty()) return;
  Tensor* h_tensor = _model->get_tensor(_inputs[0]);
  if (!h_tensor) return;
  std::vector<uint32_t> dims = h_tensor->get_dims();
  if (dims.size() == 3) { _batch_size = dims[0]; _matrix_shape = {dims[1], dims[2]}; }
  else { _matrix_shape = dims; }
}

void LDLNoBlockMergeOp::initialize_tiles(MappingTable&) {
  for (uint32_t b = 0; b < _batch_size; ++b) {
    auto tile = std::make_unique<Tile>(Tile{
        .status = Tile::Status::INITIALIZED, .optype = _optype,
        .layer_id = _id, .batch = b, .core_id = static_cast<int>(b % _config.num_cores)});
    initialize_instructions(tile.get(), Mapping{});
    if (!tile->instructions.empty()) _tiles.push_back(std::move(tile));
  }
}

void LDLNoBlockMergeOp::initialize_instructions(Tile* tile, Mapping) {
  const uint32_t M = _matrix_shape[0];
  const uint32_t U = _matrix_shape[1];
  const addr_type size_mu = static_cast<addr_type>(M) * U * _config.precision;
  const addr_type size_uu = static_cast<addr_type>(U) * U * _config.precision;
  const uint32_t epa = std::max(1u, _config.dram_req_size / _config.precision);

  const addr_type dH   = get_operand_addr(_INPUT_OPERAND + 0) + static_cast<addr_type>(tile->batch) * size_mu;
  const addr_type dReg = get_operand_addr(_INPUT_OPERAND + 1);
  const addr_type dOut = get_operand_addr(_OUTPUT_OPERAND + 0) + static_cast<addr_type>(tile->batch) * size_uu;

  // SPAD: same layout as Cholesky + extra D/Dinv regions
  const addr_type aH   = SPAD_BASE;
  const addr_type aReg = aH   + size_mu;
  const addr_type aG   = aReg + size_uu;
  const addr_type aA   = aG   + size_uu;
  const addr_type aL   = aA   + size_uu;
  const addr_type aD   = aL   + size_uu;
  const addr_type aDinv= aD   + size_uu;
  const addr_type aTmp = aDinv+ size_uu;
  const addr_type aY   = aTmp + size_uu;
  const addr_type aAinv= ACCUM_SPAD_BASE;

  const std::vector<uint32_t> sMU{M, U}, sUU{U, U};

  FormulaLogger::instance().set_algorithm("ldl_noblock_merge", 1, 0, U);

  auto movin = [&](addr_type dram, addr_type spad, uint32_t rows, uint32_t cols,
                   const std::vector<uint32_t>& shape, uint32_t op_id) {
    std::set<addr_type> addrs;
    for (uint32_t r = 0; r < rows; ++r)
      for (uint32_t c = 0; c < cols; c += epa)
        addrs.insert(dram + make_address({r, std::min(c, cols - 1)}, shape));
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

  // ===== Phase 1: MOVIN =====
  movin(dH, aH, M, U, sMU, _INPUT_OPERAND + 0);
  movin(dReg, aReg, U, U, sUU, _INPUT_OPERAND + 1);
  barrier("LDL_NB_LOAD", 1);

  // ===== Phase 2: GRAM + REG =====
  tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
      .opcode = Opcode::GEMM_PRELOAD, .id = "LDL_NB_GRAM",
      .dest_addr = aG, .compute_size = U, .src_addrs = {aH, aH},
      .tile_m = U, .tile_k = M, .tile_n = U, .my_tile = tile}));
  FormulaLogger::instance().emit_step("LDL_NB_GRAM", "GEMM",
      {"H", "H^H"}, "G", {{M, U}, {U, M}}, {U, U}, tile->batch, "LDL_NB_GRAM");

  tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
      .opcode = Opcode::ADD, .id = "LDL_NB_REG",
      .dest_addr = aA, .compute_size = U * U, .src_addrs = {aG, aReg},
      .tile_m = U, .tile_k = U, .tile_n = U, .my_tile = tile}));
  FormulaLogger::instance().emit_step("LDL_NB_REG", "DIAG_ADD",
      {"G", "lambda*I"}, "A", {{U, U}, {U, U}}, {U, U}, tile->batch, "LDL_NB_REG");
  barrier("LDL_NB_REG2DECOMP", 3);

  // Initialize D, Dinv, Tmp, L, Y regions (ensure SPAD allocation)
  // Use ADD with zero-like init: copy aReg then use SCALAR_SUB to zero-out
  tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
      .opcode = Opcode::ADD, .id = "LDL_NB_INIT_L",
      .dest_addr = aL, .compute_size = U * U,
      .src_addrs = {aReg, aReg}, .tile_m = U, .tile_k = U, .tile_n = U, .my_tile = tile}));
  tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
      .opcode = Opcode::ADD, .id = "LDL_NB_INIT_Y",
      .dest_addr = aY, .compute_size = U * U,
      .src_addrs = {aReg, aReg}, .tile_m = U, .tile_k = U, .tile_n = U, .my_tile = tile}));
  tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
      .opcode = Opcode::ADD, .id = "LDL_NB_INIT_D",
      .dest_addr = aD, .compute_size = U,
      .src_addrs = {aReg, aReg}, .tile_m = 1, .tile_k = U, .tile_n = 1, .my_tile = tile}));
  tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
      .opcode = Opcode::ADD, .id = "LDL_NB_INIT_DINV",
      .dest_addr = aDinv, .compute_size = U,
      .src_addrs = {aReg, aReg}, .tile_m = 1, .tile_k = U, .tile_n = 1, .my_tile = tile}));
  tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
      .opcode = Opcode::ADD, .id = "LDL_NB_INIT_TMP",
      .dest_addr = aTmp, .compute_size = U,
      .src_addrs = {aReg, aReg}, .tile_m = 1, .tile_k = U, .tile_n = 1, .my_tile = tile}));

  // ===== Phase 3: LDL Decomposition (column-by-column) =====
  // For each column j:
  //   D[j] = A[j,j] - sum_{k<j} D[k] * |L[j,k]|^2
  //   D_inv[j] = 1 / D[j]
  //   For i > j: L[i,j] = (A[i,j] - sum_{k<j} L[i,k] * D[k] * conj(L[j,k])) * D_inv[j]
  for (uint32_t j = 0; j < U; ++j) {
    // D_UPDATE: accumulate Schur complement
    // For each k<j: D[j] -= D[k] * |L[j,k]|^2
    // OPT1: merge j×3 MUL/SUB → 3 merged ops (compute_size=j)
    if (j > 0) {
      tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
          .opcode = Opcode::SCALAR_MUL,
          .id = "LDL_NB_DU_SQ_" + std::to_string(j),
          .dest_addr = aTmp, .compute_size = j,
          .src_addrs = {aL, aL}, .tile_m = 1, .tile_k = j, .tile_n = 1, .my_tile = tile}));
      tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
          .opcode = Opcode::SCALAR_MUL,
          .id = "LDL_NB_DU_DMUL_" + std::to_string(j),
          .dest_addr = aTmp, .compute_size = j,
          .src_addrs = {aTmp, aD}, .tile_m = 1, .tile_k = j, .tile_n = 1, .my_tile = tile}));
      tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
          .opcode = Opcode::SCALAR_SUB,
          .id = "LDL_NB_DU_SUB_" + std::to_string(j),
          .dest_addr = aA, .compute_size = j,
          .src_addrs = {aA, aTmp}, .tile_m = 1, .tile_k = j, .tile_n = 1, .my_tile = tile}));
    // D_inv[j] = 1 / A[j,j]  (A[j,j] now holds D[j] after Schur complement)
    // Step 1: unity = Reg/Reg = 1 (synthesize identity constant)
    tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
        .opcode = Opcode::SCALAR_DIV,
        .id = "LDL_NB_UNITY_" + std::to_string(j),
        .dest_addr = aTmp, .compute_size = 1,
        .src_addrs = {aReg, aReg}, .tile_m = 1, .tile_k = 1, .tile_n = 1, .my_tile = tile}));
    // Step 2: D_inv[j] = 1 / A[j,j]
    tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
        .opcode = Opcode::SCALAR_DIV,
        .id = "LDL_NB_DINV_" + std::to_string(j),
        .dest_addr = aDinv, .compute_size = 1,
        .src_addrs = {aTmp, aA}, .tile_m = 1, .tile_k = 1, .tile_n = 1, .my_tile = tile}));
    // Store D[j] = A[j,j] (copy from aA to aD, using unity*D identity)
    tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
        .opcode = Opcode::SCALAR_MUL,
        .id = "LDL_NB_DSTORE_" + std::to_string(j),
        .dest_addr = aD, .compute_size = 1,
        .src_addrs = {aA, aTmp}, .tile_m = 1, .tile_k = 1, .tile_n = 1, .my_tile = tile}));
    FormulaLogger::instance().emit_step("LDL_NB_DUPDATE_" + std::to_string(j), "DIAG_INV",
        {"A"}, "D_inv", {{U, U}}, {1, 1}, tile->batch, "LDL_NB_DINV_" + std::to_string(j));

    // L_UPDATE for rows i > j
    for (uint32_t i = j + 1; i < U; ++i) {
      // OPT1: merge j×3 MUL/SUB per (i,j) → 3 merged ops
      if (j > 0) {
        tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
            .opcode = Opcode::SCALAR_MUL,
            .id = "LDL_NB_LU_LD_" + std::to_string(i) + "_" + std::to_string(j),
            .dest_addr = aTmp, .compute_size = j,
            .src_addrs = {aL, aD}, .tile_m = 1, .tile_k = j, .tile_n = 1, .my_tile = tile}));
        tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
            .opcode = Opcode::SCALAR_MUL,
            .id = "LDL_NB_LU_CONJ_" + std::to_string(i) + "_" + std::to_string(j),
            .dest_addr = aTmp, .compute_size = j,
            .src_addrs = {aTmp, aL}, .tile_m = 1, .tile_k = j, .tile_n = 1, .my_tile = tile}));
        tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
            .opcode = Opcode::SCALAR_SUB,
            .id = "LDL_NB_LU_SUB_" + std::to_string(i) + "_" + std::to_string(j),
            .dest_addr = aA, .compute_size = j,
            .src_addrs = {aA, aTmp}, .tile_m = 1, .tile_k = 1, .tile_n = 1, .my_tile = tile}));
      }
      // L[i,j] = (A[i,j] - sum) * D_inv[j]
      tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
          .opcode = Opcode::SCALAR_MUL,
          .id = "LDL_NB_LAPPLY_" + std::to_string(i) + "_" + std::to_string(j),
          .dest_addr = aL, .compute_size = 1,
          .src_addrs = {aA, aDinv}, .tile_m = 1, .tile_k = 1, .tile_n = 1, .my_tile = tile}));
      FormulaLogger::instance().emit_step(
          "LDL_NB_LUPDATE_" + std::to_string(i) + "_" + std::to_string(j),
          "TRSM", {"A", "D_inv"}, "L_ij", {{U, U}, {1, 1}}, {1, 1}, tile->batch,
          "LDL_NB_LAPPLY_" + std::to_string(i) + "_" + std::to_string(j));
    }
    barrier("LDL_NB_COL_" + std::to_string(j), 4);
  }

  // ===== Phase 4: Forward Solve Z = L^{-1} =====
  // L is unit lower triangular: Z[c,c] = 1, Z[i,c] = -sum_{k=c}^{i-1} L[i,k] * Z[k,c]
  for (uint32_t c = 0; c < U; ++c) {
    // Z[c,c] = 1 (unit diagonal, write 1 = Reg/Reg)
    tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
        .opcode = Opcode::SCALAR_DIV,
        .id = "LDL_NB_FWD_DIAG_" + std::to_string(c),
        .dest_addr = aY, .compute_size = 1,
        .src_addrs = {aReg, aReg}, .tile_m = 1, .tile_k = 1, .tile_n = 1, .my_tile = tile}));

    for (uint32_t i = c + 1; i < U; ++i) {
      // Accumulate sum_{k=c}^{i-1} L[i,k] * Z[k,c]
      for (uint32_t k = c; k < i; ++k) {
      // OPT1: merge (i-c) MUL → 1 MUL
        tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
            .opcode = Opcode::SCALAR_MUL,
            .id = "LDL_NB_FWD_MUL_" + std::to_string(i) + "_" + std::to_string(c) + "_" + std::to_string(k),
            .dest_addr = aTmp, .compute_size = 1,
            .src_addrs = {aL, aY}, .tile_m = 1, .tile_k = 1, .tile_n = 1, .my_tile = tile}));
      }
      // Z[i,c] = -sum (negate the accumulated result)
      tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
          .opcode = Opcode::SCALAR_SUB,
          .id = "LDL_NB_FWD_NEG_" + std::to_string(i) + "_" + std::to_string(c),
          .dest_addr = aY, .compute_size = 1,
          .src_addrs = {aReg, aTmp}, .tile_m = 1, .tile_k = 1, .tile_n = 1, .my_tile = tile}));
    }
    barrier("LDL_NB_FB_" + std::to_string(c), 5);
  }

  // ===== Phase 5: Weight by sqrt(D_inv) =====
  // Y = sqrt(D_inv) * Z  (scale each column by sqrt(D_inv[c]))
  for (uint32_t c = 0; c < U; ++c) {
    // sqrt_Dinv = sqrt(Dinv[c])
    tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
        .opcode = Opcode::SCALAR_SQRT,
        .id = "LDL_NB_SQRT_DINV_" + std::to_string(c),
        .dest_addr = aTmp, .compute_size = 1,
        .src_addrs = {aDinv}, .tile_m = 1, .tile_k = 1, .tile_n = 1, .my_tile = tile}));
    // Scale column c: Y[:,c] *= sqrt(Dinv[c])
    tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
        .opcode = Opcode::SCALAR_MUL,
        .id = "LDL_NB_SCALE_" + std::to_string(c),
        .dest_addr = aY, .compute_size = U,
        .src_addrs = {aY, aTmp}, .tile_m = U, .tile_k = 1, .tile_n = 1, .my_tile = tile}));
  }

  // ===== Phase 6: Backward Assembly Ainv = Y^H @ Y =====
  tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
      .opcode = Opcode::GEMM_PRELOAD, .id = "LDL_NB_BWD",
      .dest_addr = aAinv, .compute_size = U, .src_addrs = {aY, aY},
      .tile_m = U, .tile_k = U, .tile_n = U, .my_tile = tile}));
  FormulaLogger::instance().emit_step("LDL_NB_BWD_ASSEMBLE", "GEMM",
      {"Y^H", "Y"}, "Ainv", {{U, U}, {U, U}}, {U, U}, tile->batch, "LDL_NB_BWD");
  barrier("LDL_NB_PRE_MOVOUT", 6);

  // ===== Phase 7: MOVOUT =====
  std::set<addr_type> outs;
  for (uint32_t r = 0; r < U; ++r)
    for (uint32_t c = 0; c < U; c += epa)
      outs.insert(dOut + make_address({r, std::min(c, U - 1)}, sUU));
  tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
      .opcode = Opcode::MOVOUT, .id = "LDL_NB_STORE",
      .dest_addr = aAinv, .size = static_cast<uint32_t>(outs.size()),
      .src_addrs = std::vector<addr_type>(outs.begin(), outs.end()),
      .operand_id = _OUTPUT_OPERAND, .tile_m = U, .tile_k = U, .tile_n = U,
      .src_from_accum = true, .last_inst = true, .my_tile = tile}));
}
