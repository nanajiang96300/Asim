#include "CholeskyNoBlockMergeOp.h"
#include "Model.h"
#include "FormulaLogger.h"
#include <set>

CholeskyNoBlockMergeOp::CholeskyNoBlockMergeOp(
    SimulationConfig config, Model* model, const std::string& name,
    std::map<std::string, std::string>& attributes, uint32_t target_core)
    : Operation(config, model, name, attributes, target_core) {
  _optype = "CholeskyNoBlockMergeOp";
  parse_attributes();
  infer_shapes_from_model();
}

void CholeskyNoBlockMergeOp::parse_attributes() {
  auto it_batch = _attributes.find("batch_size");
  if (it_batch != _attributes.end()) {
    try { _batch_size = static_cast<uint32_t>(std::stoul(it_batch->second)); }
    catch (...) { _batch_size = 96; }
  }
}

void CholeskyNoBlockMergeOp::infer_shapes_from_model() {
  if (!_matrix_shape.empty() || !_model) return;
  if (_inputs.empty()) return;
  Tensor* h_tensor = _model->get_tensor(_inputs[0]);
  if (!h_tensor) return;
  std::vector<uint32_t> dims = h_tensor->get_dims();
  if (dims.size() == 3) { _batch_size = dims[0]; _matrix_shape = {dims[1], dims[2]}; }
  else { _matrix_shape = dims; }
}

void CholeskyNoBlockMergeOp::initialize_tiles(MappingTable&) {
  for (uint32_t b = 0; b < _batch_size; ++b) {
    auto tile = std::make_unique<Tile>(Tile{
        .status = Tile::Status::INITIALIZED, .optype = _optype,
        .layer_id = _id, .batch = b, .core_id = static_cast<int>(b % _config.num_cores)});
    initialize_instructions(tile.get(), Mapping{});
    if (!tile->instructions.empty()) _tiles.push_back(std::move(tile));
  }
}

void CholeskyNoBlockMergeOp::initialize_instructions(Tile* tile, Mapping) {
  const uint32_t M = _matrix_shape[0];
  const uint32_t U = _matrix_shape[1];
  const addr_type size_mu = static_cast<addr_type>(M) * U * _config.precision;
  const addr_type size_uu = static_cast<addr_type>(U) * U * _config.precision;
  const uint32_t epa = std::max(1u, _config.dram_req_size / _config.precision);

  const addr_type dH   = get_operand_addr(_INPUT_OPERAND + 0) + static_cast<addr_type>(tile->batch) * size_mu;
  const addr_type dReg = get_operand_addr(_INPUT_OPERAND + 1);
  const addr_type dOut = get_operand_addr(_OUTPUT_OPERAND + 0) + static_cast<addr_type>(tile->batch) * size_uu;

  const addr_type aH   = SPAD_BASE;
  const addr_type aReg = aH   + size_mu;
  const addr_type aG   = aReg + size_uu;
  const addr_type aA   = aG   + size_uu;
  const addr_type aL   = aA   + size_uu;
  const addr_type aInv = aL   + size_uu;
  const addr_type aY   = aInv + size_uu;
  const addr_type aTmp = aY   + size_uu;
  const addr_type aAinv= ACCUM_SPAD_BASE;

  const std::vector<uint32_t> sMU{M, U}, sUU{U, U};
  FormulaLogger::instance().set_algorithm("cholesky_noblock_merge", 1, 0, U);

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

  // Phase 1: MOVIN
  movin(dH, aH, M, U, sMU, _INPUT_OPERAND + 0);
  movin(dReg, aReg, U, U, sUU, _INPUT_OPERAND + 1);
  barrier("CHOL_NB_LOAD", 1);

  // Phase 2: GRAM + REG
  tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
      .opcode = Opcode::GEMM_PRELOAD, .id = "CHOL_NB_GRAM",
      .dest_addr = aG, .compute_size = U, .src_addrs = {aH, aH},
      .tile_m = U, .tile_k = M, .tile_n = U, .my_tile = tile}));
  FormulaLogger::instance().emit_step("CHOL_NB_GRAM", "GEMM",
      {"H", "H^H"}, "G", {{M, U}, {U, M}}, {U, U}, tile->batch, "CHOL_NB_GRAM");

  tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
      .opcode = Opcode::ADD, .id = "CHOL_NB_REG",
      .dest_addr = aA, .compute_size = U * U, .src_addrs = {aG, aReg},
      .tile_m = U, .tile_k = U, .tile_n = U, .my_tile = tile}));
  FormulaLogger::instance().emit_step("CHOL_NB_REG", "DIAG_ADD",
      {"G", "lambda*I"}, "A", {{U, U}, {U, U}}, {U, U}, tile->batch, "CHOL_NB_REG");
  barrier("CHOL_NB_REG2DECOMP", 3);

  // Init L, Y, Tmp regions
  tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
      .opcode = Opcode::ADD, .id = "CHOL_NB_INIT_L",
      .dest_addr = aL, .compute_size = U * U,
      .src_addrs = {aReg, aReg}, .tile_m = U, .tile_k = U, .tile_n = U, .my_tile = tile}));
  tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
      .opcode = Opcode::ADD, .id = "CHOL_NB_INIT_Y",
      .dest_addr = aY, .compute_size = U * U,
      .src_addrs = {aReg, aReg}, .tile_m = U, .tile_k = U, .tile_n = U, .my_tile = tile}));
  tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
      .opcode = Opcode::ADD, .id = "CHOL_NB_INIT_TMP",
      .dest_addr = aTmp, .compute_size = U,
      .src_addrs = {aReg, aReg}, .tile_m = 1, .tile_k = U, .tile_n = 1, .my_tile = tile}));

  // Phase 3: Cholesky Decomposition (column-by-column)
  for (uint32_t j = 0; j < U; ++j) {
    // POTRF: L[j,j]^2 = A[j,j] - sum_{k<j} |L[j,k]|^2
    // OPT1: merge j individual MUL(1)+SUB(1) → 1 MUL(j)+SUB(j)
    if (j > 0) {
      tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
          .opcode = Opcode::SCALAR_MUL,
          .id = "CHOL_NB_POTRF_SQ_" + std::to_string(j),
          .dest_addr = aTmp, .compute_size = j,
          .src_addrs = {aL, aL}, .tile_m = 1, .tile_k = j, .tile_n = 1, .my_tile = tile}));
      tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
          .opcode = Opcode::SCALAR_SUB,
          .id = "CHOL_NB_POTRF_SUB_" + std::to_string(j),
          .dest_addr = aA, .compute_size = j,
          .src_addrs = {aA, aTmp}, .tile_m = 1, .tile_k = j, .tile_n = 1, .my_tile = tile}));
    }
    // L[j,j] = sqrt(A[j,j])
    tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
        .opcode = Opcode::SCALAR_SQRT,
        .id = "CHOL_NB_POTRF_SQRT_" + std::to_string(j),
        .dest_addr = aL, .compute_size = 1,
        .src_addrs = {aA}, .tile_m = 1, .tile_k = 1, .tile_n = 1, .my_tile = tile}));
    FormulaLogger::instance().emit_step("CHOL_NB_POTRF_" + std::to_string(j), "CHOLESKY",
        {"A"}, "L_diag", {{U, U}}, {1, 1}, tile->batch,
        "CHOL_NB_POTRF_SQRT_" + std::to_string(j));

    // TRSM: L[i,j] = (A[i,j] - sum_{k<j} L[i,k]*conj(L[j,k])) / L[j,j]
    // OPT1: merge j MUL+SUB per (i,j) → 1 MUL(j)+SUB(j) per (i,j)
    for (uint32_t i = j + 1; i < U; ++i) {
      if (j > 0) {
        tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
            .opcode = Opcode::SCALAR_MUL,
            .id = "CHOL_NB_TRSM_MUL_" + std::to_string(i) + "_" + std::to_string(j),
            .dest_addr = aTmp, .compute_size = j,
            .src_addrs = {aL, aL}, .tile_m = 1, .tile_k = j, .tile_n = 1, .my_tile = tile}));
        tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
            .opcode = Opcode::SCALAR_SUB,
            .id = "CHOL_NB_TRSM_SUB_" + std::to_string(i) + "_" + std::to_string(j),
            .dest_addr = aA, .compute_size = j,
            .src_addrs = {aA, aTmp}, .tile_m = 1, .tile_k = j, .tile_n = 1, .my_tile = tile}));
      }
      // L[i,j] = A[i,j] / L[j,j]
      tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
          .opcode = Opcode::SCALAR_DIV,
          .id = "CHOL_NB_TRSM_DIV_" + std::to_string(i) + "_" + std::to_string(j),
          .dest_addr = aL, .compute_size = 1,
          .src_addrs = {aA, aL}, .tile_m = 1, .tile_k = 1, .tile_n = 1, .my_tile = tile}));
      FormulaLogger::instance().emit_step("CHOL_NB_TRSM_" + std::to_string(i) + "_" + std::to_string(j),
          "TRSM", {"A", "L"}, "L_ij", {{U, U}, {U, U}}, {1, 1}, tile->batch,
          "CHOL_NB_TRSM_DIV_" + std::to_string(i) + "_" + std::to_string(j));
    }
    barrier("CHOL_NB_COL_" + std::to_string(j), 4);
  }

  // Phase 4: Forward Solve Y = L^{-1}
  for (uint32_t c = 0; c < U; ++c) {
    // Y[c,c] = 1 / L[c,c] (use Reg as identity numerator, Reg/Reg=1 then scale)
    tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
        .opcode = Opcode::SCALAR_DIV,
        .id = "CHOL_NB_FWD_DIAG_" + std::to_string(c),
        .dest_addr = aInv, .compute_size = 1,
        .src_addrs = {aReg, aReg}, .tile_m = 1, .tile_k = 1, .tile_n = 1, .my_tile = tile}));
    // Y[c,c] = 1/L[c,c]
    tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
        .opcode = Opcode::SCALAR_DIV,
        .id = "CHOL_NB_FWD_DIAG2_" + std::to_string(c),
        .dest_addr = aY, .compute_size = 1,
        .src_addrs = {aInv, aL}, .tile_m = 1, .tile_k = 1, .tile_n = 1, .my_tile = tile}));

    for (uint32_t i = c + 1; i < U; ++i) {
      // OPT1: merge (i-c) MUL → 1 MUL(compute_size=i-c)
      uint32_t len = i - c;
      if (len > 0) {
        tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
            .opcode = Opcode::SCALAR_MUL,
            .id = "CHOL_NB_FWD_MUL_" + std::to_string(i) + "_" + std::to_string(c),
            .dest_addr = aTmp, .compute_size = len,
            .src_addrs = {aL, aY}, .tile_m = 1, .tile_k = len, .tile_n = 1, .my_tile = tile}));
      }
      // Y[i,c] = -sum / L[i,i]
      tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
          .opcode = Opcode::SCALAR_SUB,
          .id = "CHOL_NB_FWD_NEG_" + std::to_string(i) + "_" + std::to_string(c),
          .dest_addr = aTmp, .compute_size = 1,
          .src_addrs = {aReg, aTmp}, .tile_m = 1, .tile_k = 1, .tile_n = 1, .my_tile = tile}));
      tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
          .opcode = Opcode::SCALAR_DIV,
          .id = "CHOL_NB_FWD_DIV_" + std::to_string(i) + "_" + std::to_string(c),
          .dest_addr = aY, .compute_size = 1,
          .src_addrs = {aTmp, aL}, .tile_m = 1, .tile_k = 1, .tile_n = 1, .my_tile = tile}));
    }
    barrier("CHOL_NB_FB_" + std::to_string(c), 5);
  }

  // Phase 5: Backward Assembly Ainv = Y^H @ Y
  tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
      .opcode = Opcode::GEMM_PRELOAD, .id = "CHOL_NB_BWD_GEMM",
      .dest_addr = aAinv, .compute_size = U, .src_addrs = {aY, aY},
      .tile_m = U, .tile_k = U, .tile_n = U, .my_tile = tile}));
  FormulaLogger::instance().emit_step("CHOL_NB_BWD_ASSEMBLE", "GEMM",
      {"Y^H", "Y"}, "Ainv", {{U, U}, {U, U}}, {U, U}, tile->batch, "CHOL_NB_BWD_GEMM");
  barrier("CHOL_NB_PRE_MOVOUT", 6);

  // Phase 6: MOVOUT
  std::set<addr_type> outs;
  for (uint32_t r = 0; r < U; ++r)
    for (uint32_t c = 0; c < U; c += epa)
      outs.insert(dOut + make_address({r, std::min(c, U - 1)}, sUU));
  tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
      .opcode = Opcode::MOVOUT, .id = "CHOL_NB_STORE",
      .dest_addr = aAinv, .size = static_cast<uint32_t>(outs.size()),
      .src_addrs = std::vector<addr_type>(outs.begin(), outs.end()),
      .operand_id = _OUTPUT_OPERAND, .tile_m = U, .tile_k = U, .tile_n = U,
      .src_from_accum = true, .last_inst = true, .my_tile = tile}));
}
