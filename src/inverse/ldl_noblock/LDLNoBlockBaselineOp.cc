#include "LDLNoBlockBaselineOp.h"
#include "Model.h"
#include "FormulaLogger.h"
#include <set>
#include "semantic/SourceViews.h"

LDLNoBlockBaselineOp::LDLNoBlockBaselineOp(
    SimulationConfig config, Model* model, const std::string& name,
    std::map<std::string, std::string>& attributes, uint32_t target_core)
    : Operation(config, model, name, attributes, target_core) {
  _optype = "LDLNoBlockBaselineOp";
  parse_attributes();
  infer_shapes_from_model();
}

void LDLNoBlockBaselineOp::parse_attributes() {
  auto it_batch = _attributes.find("batch_size");
  if (it_batch != _attributes.end()) {
    try { _batch_size = static_cast<uint32_t>(std::stoul(it_batch->second)); }
    catch (...) { _batch_size = 96; }
  }
}

void LDLNoBlockBaselineOp::infer_shapes_from_model() {
  if (!_matrix_shape.empty() || !_model) return;
  if (_inputs.empty()) return;
  Tensor* h_tensor = _model->get_tensor(_inputs[0]);
  if (!h_tensor) return;
  std::vector<uint32_t> dims = h_tensor->get_dims();
  if (dims.size() == 3) { _batch_size = dims[0]; _matrix_shape = {dims[1], dims[2]}; }
  else { _matrix_shape = dims; }
}

void LDLNoBlockBaselineOp::initialize_tiles(MappingTable&) {
  for (uint32_t b = 0; b < _batch_size; ++b) {
    auto tile = std::make_unique<Tile>(Tile{
        .status = Tile::Status::INITIALIZED, .optype = _optype,
        .layer_id = _id, .batch = b, .core_id = static_cast<int>(b % _config.num_cores)});
    initialize_instructions(tile.get(), Mapping{});
    if (!tile->instructions.empty()) _tiles.push_back(std::move(tile));
  }
}

void LDLNoBlockBaselineOp::initialize_instructions(Tile* tile, Mapping) {
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

  record_source_regions(json::array({region(aH,M,U,"aH"), region(aReg,U,U,"aReg"), region(aG,U,U,"aG"), region(aA,U,U,"aA"), region(aL,U,U,"aL"), region(aD,U,U,"aD"), region(aDinv,U,U,"aDinv"), region(aTmp,U,U,"aTmp"), region(aY,U,U,"aY"), region(aAinv,U,U,"aAinv")}), M, U);

  FormulaLogger::instance().set_algorithm("ldl_noblock_v2", 1, 0, U);

  auto movin = [&](addr_type dram, addr_type spad, uint32_t rows, uint32_t cols,
                   const std::vector<uint32_t>& shape, uint32_t op_id) {
    std::set<addr_type> addrs;
    for (uint32_t r = 0; r < rows; ++r)
      for (uint32_t c = 0; c < cols; c += epa)
        addrs.insert(dram + make_address({r, std::min(c, cols - 1)}, shape));
    tile->instructions.push_back(source_instruction(Instruction{
        .opcode = Opcode::MOVIN, .dest_addr = spad,
        .size = static_cast<uint32_t>(addrs.size()),
        .src_addrs = std::vector<addr_type>(addrs.begin(), addrs.end()),
        .operand_id = op_id, .base_addr = 0,
        .tile_m = rows, .tile_k = cols, .my_tile = tile}, json::array({view(dram,0,0,rows,cols,false,false)}), view(spad,0,0,rows,cols,false,false), "src/inverse/ldl_noblock/LDLNoBlockBaselineOp.cc:MOVIN"));
  };

  auto barrier = [&](const std::string& id, uint32_t type) {
    tile->instructions.push_back(source_instruction(Instruction{
        .opcode = Opcode::PIPE_BARRIER, .id = id,
        .my_tile = tile, .is_barrier = true, .barrier_type = type}, json::array({}), nullptr, "src/inverse/ldl_noblock/LDLNoBlockBaselineOp.cc:PIPE_BARRIER"));
  };

  // ===== Phase 1: MOVIN =====
  movin(dH, aH, M, U, sMU, _INPUT_OPERAND + 0);
  movin(dReg, aReg, U, U, sUU, _INPUT_OPERAND + 1);
  barrier("LDL_NB_LOAD", 1);

  // ===== Phase 2: GRAM + REG =====
  tile->instructions.push_back(source_instruction(Instruction{
      .opcode = Opcode::GEMM_PRELOAD, .id = "LDL_NB_GRAM",
      .dest_addr = aG, .compute_size = U, .src_addrs = {aH, aH},
      .tile_m = U, .tile_k = M, .tile_n = U, .my_tile = tile}, json::array({view(aH,0,0,M,U,true,true), view(aH,0,0,M,U,false,false)}), view(aG,0,0,U,U,false,false), "src/inverse/ldl_noblock/LDLNoBlockBaselineOp.cc:LDL_NB_GRAM"));
  FormulaLogger::instance().emit_step("LDL_NB_GRAM", "GEMM",
      {"H^H", "H"}, "G", {{M, U}, {U, M}}, {U, U}, tile->batch, "LDL_NB_GRAM");

  tile->instructions.push_back(source_instruction(Instruction{
      .opcode = Opcode::ADD, .id = "LDL_NB_REG",
      .dest_addr = aA, .compute_size = U * U, .src_addrs = {aG, aReg},
      .tile_m = U, .tile_k = U, .tile_n = U, .my_tile = tile}, json::array({view(aG,0,0,U,U,false,false), view(aReg,0,0,U,U,false,false)}), view(aA,0,0,U,U,false,false), "src/inverse/ldl_noblock/LDLNoBlockBaselineOp.cc:LDL_NB_REG"));
  FormulaLogger::instance().emit_step("LDL_NB_REG", "DIAG_ADD",
      {"G", "lambda*I"}, "A", {{U, U}, {U, U}}, {U, U}, tile->batch, "LDL_NB_REG");
  barrier("LDL_NB_REG2DECOMP", 3);

  // Initialize D, Dinv, Tmp, L, Y regions (ensure SPAD allocation)
  // Use ADD with zero-like init: copy aReg then use SCALAR_SUB to zero-out
  tile->instructions.push_back(source_instruction(Instruction{
      .opcode = Opcode::ADD, .id = "LDL_NB_INIT_L",
      .dest_addr = aL, .compute_size = U * U,
      .src_addrs = {aReg, aReg}, .tile_m = U, .tile_k = U, .tile_n = U, .my_tile = tile}, json::array({view(aReg,0,0,U,U,false,false), view(aReg,0,0,U,U,false,false)}), view(aL,0,0,U,U,false,false), "src/inverse/ldl_noblock/LDLNoBlockBaselineOp.cc:LDL_NB_INIT_L"));
  tile->instructions.push_back(source_instruction(Instruction{
      .opcode = Opcode::ADD, .id = "LDL_NB_INIT_Y",
      .dest_addr = aY, .compute_size = U * U,
      .src_addrs = {aReg, aReg}, .tile_m = U, .tile_k = U, .tile_n = U, .my_tile = tile}, json::array({view(aReg,0,0,U,U,false,false), view(aReg,0,0,U,U,false,false)}), view(aY,0,0,U,U,false,false), "src/inverse/ldl_noblock/LDLNoBlockBaselineOp.cc:LDL_NB_INIT_Y"));
  tile->instructions.push_back(source_instruction(Instruction{
      .opcode = Opcode::ADD, .id = "LDL_NB_INIT_D",
      .dest_addr = aD, .compute_size = U,
      .src_addrs = {aReg, aReg}, .tile_m = 1, .tile_k = U, .tile_n = 1, .my_tile = tile}, json::array({view(aReg,0,0,1,U,false,false), view(aReg,0,0,1,U,false,false)}), view(aD,0,0,1,U,false,false), "src/inverse/ldl_noblock/LDLNoBlockBaselineOp.cc:LDL_NB_INIT_D"));
  tile->instructions.push_back(source_instruction(Instruction{
      .opcode = Opcode::ADD, .id = "LDL_NB_INIT_DINV",
      .dest_addr = aDinv, .compute_size = U,
      .src_addrs = {aReg, aReg}, .tile_m = 1, .tile_k = U, .tile_n = 1, .my_tile = tile}, json::array({view(aReg,0,0,1,U,false,false), view(aReg,0,0,1,U,false,false)}), view(aDinv,0,0,1,U,false,false), "src/inverse/ldl_noblock/LDLNoBlockBaselineOp.cc:LDL_NB_INIT_DINV"));
  tile->instructions.push_back(source_instruction(Instruction{
      .opcode = Opcode::ADD, .id = "LDL_NB_INIT_TMP",
      .dest_addr = aTmp, .compute_size = U,
      .src_addrs = {aReg, aReg}, .tile_m = 1, .tile_k = U, .tile_n = 1, .my_tile = tile}, json::array({view(aReg,0,0,1,U,false,false), view(aReg,0,0,1,U,false,false)}), view(aTmp,0,0,1,U,false,false), "src/inverse/ldl_noblock/LDLNoBlockBaselineOp.cc:LDL_NB_INIT_TMP"));

  // ===== Phase 3: LDL Decomposition (column-by-column) =====
  // For each column j:
  //   D[j] = A[j,j] - sum_{k<j} D[k] * |L[j,k]|^2
  //   D_inv[j] = 1 / D[j]
  //   For i > j: L[i,j] = (A[i,j] - sum_{k<j} L[i,k] * D[k] * conj(L[j,k])) * D_inv[j]
  for (uint32_t j = 0; j < U; ++j) {
    // D_UPDATE: accumulate Schur complement
    // For each k<j: D[j] -= D[k] * |L[j,k]|^2
    for (uint32_t k = 0; k < j; ++k) {
      // Step 1: |L[j,k]|^2 = L[j,k] * conj(L[j,k])
      tile->instructions.push_back(source_instruction(Instruction{
          .opcode = Opcode::SCALAR_MUL,
          .id = "LDL_NB_DU_SQ_" + std::to_string(j) + "_" + std::to_string(k),
          .dest_addr = aTmp, .compute_size = 1,
          .src_addrs = {aL, aL}, .tile_m = 1, .tile_k = 1, .tile_n = 1, .my_tile = tile}, json::array({view(aL,j,k,1,1,false,false), view(aL,j,k,1,1,false,true)}), view(aTmp,0,0,1,1,false,false), "src/inverse/ldl_noblock/LDLNoBlockBaselineOp.cc:LDL_NB_DU_SQ_"));
      // Step 2: D[k] * |L[j,k]|^2
      tile->instructions.push_back(source_instruction(Instruction{
          .opcode = Opcode::SCALAR_MUL,
          .id = "LDL_NB_DU_DMUL_" + std::to_string(j) + "_" + std::to_string(k),
          .dest_addr = aTmp, .compute_size = 1,
          .src_addrs = {aTmp, aD}, .tile_m = 1, .tile_k = 1, .tile_n = 1, .my_tile = tile}, json::array({view(aTmp,0,0,1,1,false,false), view(aD,0,k,1,1,false,false)}), view(aTmp,0,0,1,1,false,false), "src/inverse/ldl_noblock/LDLNoBlockBaselineOp.cc:LDL_NB_DU_DMUL_"));
      // Step 3: A[j,j] -= D[k] * |L[j,k]|^2  (subtract from accumulator)
      tile->instructions.push_back(source_instruction(Instruction{
          .opcode = Opcode::SCALAR_SUB,
          .id = "LDL_NB_DU_SUB_" + std::to_string(j) + "_" + std::to_string(k),
          .dest_addr = aA, .compute_size = 1,
          .src_addrs = {aA, aTmp}, .tile_m = 1, .tile_k = 1, .tile_n = 1, .my_tile = tile}, json::array({view(aA,j,j,1,1,false,false), view(aTmp,0,0,1,1,false,false)}), view(aA,j,j,1,1,false,false), "src/inverse/ldl_noblock/LDLNoBlockBaselineOp.cc:LDL_NB_DU_SUB_"));
    }
    // D_inv[j] = 1 / A[j,j]  (A[j,j] now holds D[j] after Schur complement)
    // Step 1: unity = Reg/Reg = 1 (synthesize identity constant)
    tile->instructions.push_back(source_instruction(Instruction{
        .opcode = Opcode::SCALAR_DIV,
        .id = "LDL_NB_UNITY_" + std::to_string(j),
        .dest_addr = aTmp, .compute_size = 1,
        .src_addrs = {aReg, aReg}, .tile_m = 1, .tile_k = 1, .tile_n = 1, .my_tile = tile}, json::array({view(aReg,j,j,1,1,false,false), view(aReg,j,j,1,1,false,false)}), view(aTmp,0,0,1,1,false,false), "src/inverse/ldl_noblock/LDLNoBlockBaselineOp.cc:LDL_NB_UNITY_"));
    // Step 2: D_inv[j] = 1 / A[j,j]
    tile->instructions.push_back(source_instruction(Instruction{
        .opcode = Opcode::SCALAR_DIV,
        .id = "LDL_NB_DINV_" + std::to_string(j),
        .dest_addr = aDinv, .compute_size = 1,
        .src_addrs = {aTmp, aA}, .tile_m = 1, .tile_k = 1, .tile_n = 1, .my_tile = tile}, json::array({view(aTmp,0,0,1,1,false,false), view(aA,j,j,1,1,false,false)}), view(aDinv,0,j,1,1,false,false), "src/inverse/ldl_noblock/LDLNoBlockBaselineOp.cc:LDL_NB_DINV_"));
    // Store D[j] = A[j,j] (copy from aA to aD, using unity*D identity)
    tile->instructions.push_back(source_instruction(Instruction{
        .opcode = Opcode::SCALAR_MUL,
        .id = "LDL_NB_DSTORE_" + std::to_string(j),
        .dest_addr = aD, .compute_size = 1,
        .src_addrs = {aA, aTmp}, .tile_m = 1, .tile_k = 1, .tile_n = 1, .my_tile = tile}, json::array({view(aA,j,j,1,1,false,false), view(aTmp,0,0,1,1,false,false)}), view(aD,0,j,1,1,false,false), "src/inverse/ldl_noblock/LDLNoBlockBaselineOp.cc:LDL_NB_DSTORE_"));
    // LDL_DECOMPOSE step emitted once after decomposition loop (see below)
    // Removed per-column emit_step: prim_ldl_decompose runs on full A,
    // per-column calls were redundant (M5 fix).

    // L_UPDATE for rows i > j
    for (uint32_t i = j + 1; i < U; ++i) {
      // Accumulate sum_{k<j} L[i,k] * D[k] * conj(L[j,k])
      for (uint32_t k = 0; k < j; ++k) {
        // Step 1: L[i,k] * D[k]
        tile->instructions.push_back(source_instruction(Instruction{
            .opcode = Opcode::SCALAR_MUL,
            .id = "LDL_NB_LU_LD_" + std::to_string(i) + "_" + std::to_string(j) + "_" + std::to_string(k),
            .dest_addr = aTmp, .compute_size = 1,
            .src_addrs = {aL, aD}, .tile_m = 1, .tile_k = 1, .tile_n = 1, .my_tile = tile}, json::array({view(aL,i,k,1,1,false,false), view(aD,0,k,1,1,false,false)}), view(aTmp,0,0,1,1,false,false), "src/inverse/ldl_noblock/LDLNoBlockBaselineOp.cc:LDL_NB_LU_LD_"));
        // Step 2: (L[i,k] * D[k]) * conj(L[j,k])
        tile->instructions.push_back(source_instruction(Instruction{
            .opcode = Opcode::SCALAR_MUL,
            .id = "LDL_NB_LU_CONJ_" + std::to_string(i) + "_" + std::to_string(j) + "_" + std::to_string(k),
            .dest_addr = aTmp, .compute_size = 1,
            .src_addrs = {aTmp, aL}, .tile_m = 1, .tile_k = 1, .tile_n = 1, .my_tile = tile}, json::array({view(aTmp,0,0,1,1,false,false), view(aL,j,k,1,1,false,true)}), view(aTmp,0,0,1,1,false,false), "src/inverse/ldl_noblock/LDLNoBlockBaselineOp.cc:LDL_NB_LU_CONJ_"));
        // Step 3: A[i,j] -= (L[i,k] * D[k] * conj(L[j,k]))
        tile->instructions.push_back(source_instruction(Instruction{
            .opcode = Opcode::SCALAR_SUB,
            .id = "LDL_NB_LU_SUB_" + std::to_string(i) + "_" + std::to_string(j) + "_" + std::to_string(k),
            .dest_addr = aA, .compute_size = 1,
            .src_addrs = {aA, aTmp}, .tile_m = 1, .tile_k = 1, .tile_n = 1, .my_tile = tile}, json::array({view(aA,i,j,1,1,false,false), view(aTmp,0,0,1,1,false,false)}), view(aA,i,j,1,1,false,false), "src/inverse/ldl_noblock/LDLNoBlockBaselineOp.cc:LDL_NB_LU_SUB_"));
      }
      // L[i,j] = (A[i,j] - sum) * D_inv[j]
      tile->instructions.push_back(source_instruction(Instruction{
          .opcode = Opcode::SCALAR_MUL,
          .id = "LDL_NB_LAPPLY_" + std::to_string(i) + "_" + std::to_string(j),
          .dest_addr = aL, .compute_size = 1,
          .src_addrs = {aA, aDinv}, .tile_m = 1, .tile_k = 1, .tile_n = 1, .my_tile = tile}, json::array({view(aA,i,j,1,1,false,false), view(aDinv,0,j,1,1,false,false)}), view(aL,i,j,1,1,false,false), "src/inverse/ldl_noblock/LDLNoBlockBaselineOp.cc:LDL_NB_LAPPLY_"));
      // LUPDATE TRSM: removed dead DAG code — output "L" was never consumed by any step (M6 fix).
      // The DAG chain is: GRAM→REG→LDL_DECOMPOSE→BWD_ASSEMBLE, where LDL_DECOMPOSE
      // internally covers forward solve + sqrt(Dinv) scaling (H2/H3 known limitation).
    }
    barrier("LDL_NB_COL_" + std::to_string(j), 4);
  }

  // Single LDL_DECOMPOSE emit_step after decomposition loop (M5 fix).
  // prim_ldl_decompose internally covers: L·D·L^H decomposition +
  // forward solve Z=L^{-1} + sqrt(Dinv) column scaling → Y.
  // Phases 4 (forward solve) and 5 (sqrt scaling) below are hardware
  // phases whose semantics are covered by this single DAG step (H2 known limitation).
  FormulaLogger::instance().emit_step("LDL_NB_DECOMPOSE", "LDL_DECOMPOSE",
      {"A"}, "Y", {{U, U}}, {U, U}, tile->batch, "LDL_NB_COL_0");

  // ===== Phase 4: Forward Solve Z = L^{-1} =====
  // L is unit lower triangular: Z[c,c] = 1, Z[i,c] = -sum_{k=c}^{i-1} L[i,k] * Z[k,c]
  for (uint32_t c = 0; c < U; ++c) {
    // Z[c,c] = 1 (unit diagonal, write 1 = Reg/Reg)
    tile->instructions.push_back(source_instruction(Instruction{
        .opcode = Opcode::SCALAR_DIV,
        .id = "LDL_NB_FWD_DIAG_" + std::to_string(c),
        .dest_addr = aY, .compute_size = 1,
        .src_addrs = {aReg, aReg}, .tile_m = 1, .tile_k = 1, .tile_n = 1, .my_tile = tile}, json::array({view(aReg,c,c,1,1,false,false), view(aReg,c,c,1,1,false,false)}), view(aY,c,c,1,1,false,false), "src/inverse/ldl_noblock/LDLNoBlockBaselineOp.cc:LDL_NB_FWD_DIAG_"));

    for (uint32_t i = c + 1; i < U; ++i) {
      // Accumulate sum_{k=c}^{i-1} L[i,k] * Z[k,c]
      for (uint32_t k = c; k < i; ++k) {
        tile->instructions.push_back(source_instruction(Instruction{
            .opcode = Opcode::SCALAR_MUL,
            .id = "LDL_NB_FWD_MUL_" + std::to_string(i) + "_" + std::to_string(c) + "_" + std::to_string(k),
            .dest_addr = aTmp, .compute_size = 1,
            .src_addrs = {aL, aY}, .tile_m = 1, .tile_k = 1, .tile_n = 1, .my_tile = tile}, json::array({view(aL,i,k,1,1,false,false), view(aY,k,c,1,1,false,false)}), view(aTmp,0,0,1,1,false,false), "src/inverse/ldl_noblock/LDLNoBlockBaselineOp.cc:LDL_NB_FWD_MUL_"));
      }
      // Z[i,c] = -sum (negate the accumulated result)
      tile->instructions.push_back(ambiguous_instruction(Instruction{
          .opcode = Opcode::SCALAR_SUB,
          .id = "LDL_NB_FWD_NEG_" + std::to_string(i) + "_" + std::to_string(c),
          .dest_addr = aY, .compute_size = 1,
          .src_addrs = {aReg, aTmp}, .tile_m = 1, .tile_k = 1, .tile_n = 1, .my_tile = tile}, "zero-source row/column is not uniquely specified by legacy base address; diagonal lambda and off-diagonal zero differ", "src/inverse/ldl_noblock/LDLNoBlockBaselineOp.cc:LDL_NB_FWD_NEG_"));
    }
    barrier("LDL_NB_FB_" + std::to_string(c), 5);
  }

  // ===== Phase 5: Weight by sqrt(D_inv) =====
  // Y = sqrt(D_inv) * Z  (scale each column by sqrt(D_inv[c]))
  for (uint32_t c = 0; c < U; ++c) {
    // sqrt_Dinv = sqrt(Dinv[c])
    tile->instructions.push_back(source_instruction(Instruction{
        .opcode = Opcode::SCALAR_SQRT,
        .id = "LDL_NB_SQRT_DINV_" + std::to_string(c),
        .dest_addr = aTmp, .compute_size = 1,
        .src_addrs = {aDinv}, .tile_m = 1, .tile_k = 1, .tile_n = 1, .my_tile = tile}, json::array({view(aDinv,0,c,1,1,false,false)}), view(aTmp,0,0,1,1,false,false), "src/inverse/ldl_noblock/LDLNoBlockBaselineOp.cc:LDL_NB_SQRT_DINV_"));
    // Scale column c: Y[:,c] *= sqrt(Dinv[c])
    tile->instructions.push_back(source_instruction(Instruction{
        .opcode = Opcode::SCALAR_MUL,
        .id = "LDL_NB_SCALE_" + std::to_string(c),
        .dest_addr = aY, .compute_size = U,
        .src_addrs = {aY, aTmp}, .tile_m = U, .tile_k = 1, .tile_n = 1, .my_tile = tile}, json::array({view(aY,0,c,U,1,false,false), view(aTmp,0,0,1,1,false,false)}), view(aY,0,c,U,1,false,false), "src/inverse/ldl_noblock/LDLNoBlockBaselineOp.cc:LDL_NB_SCALE_"));
    tile->instructions.back()->numeric_semantics["broadcast"] = "scalar_rhs";
  }

  // ===== Phase 6: Backward Assembly Ainv = Y^H @ Y =====
  // FWD solve complete: Z = L^{-1}
  
  tile->instructions.push_back(source_instruction(Instruction{
      .opcode = Opcode::GEMM_PRELOAD, .id = "LDL_NB_BWD",
      .dest_addr = aAinv, .compute_size = U, .src_addrs = {aY, aY},
      .tile_m = U, .tile_k = U, .tile_n = U, .my_tile = tile}, json::array({view(aY,0,0,U,U,true,true), view(aY,0,0,U,U,false,false)}), view(aAinv,0,0,U,U,false,false), "src/inverse/ldl_noblock/LDLNoBlockBaselineOp.cc:LDL_NB_BWD"));
  FormulaLogger::instance().emit_step("LDL_NB_BWD_ASSEMBLE", "GEMM",
      {"Y^H", "Y"}, "Ainv", {{U, U}, {U, U}}, {U, U}, tile->batch, "LDL_NB_BWD");
  barrier("LDL_NB_PRE_MOVOUT", 6);

  // ===== Phase 7: MOVOUT =====
  std::set<addr_type> outs;
  for (uint32_t r = 0; r < U; ++r)
    for (uint32_t c = 0; c < U; c += epa)
      outs.insert(dOut + make_address({r, std::min(c, U - 1)}, sUU));
  tile->instructions.push_back(source_instruction(Instruction{
      .opcode = Opcode::MOVOUT, .id = "LDL_NB_STORE",
      .dest_addr = aAinv, .size = static_cast<uint32_t>(outs.size()),
      .src_addrs = std::vector<addr_type>(outs.begin(), outs.end()),
      .operand_id = _OUTPUT_OPERAND, .tile_m = U, .tile_k = U, .tile_n = U,
      .src_from_accum = true, .last_inst = true, .my_tile = tile}, json::array({view(aAinv,0,0,U,U,false,false)}), view(dOut,0,0,U,U,false,false), "src/inverse/ldl_noblock/LDLNoBlockBaselineOp.cc:LDL_NB_STORE"));
}
