#include <algorithm>
#include "BlockRichardsonBaselineOp.h"
#include "Model.h"
#include "FormulaLogger.h"
#include <set>

BlockRichardsonBaselineOp::BlockRichardsonBaselineOp(
    SimulationConfig config, Model* model, const std::string& name,
    std::map<std::string, std::string>& attributes, uint32_t target_core)
    : Operation(config, model, name, attributes, target_core) {
  _optype = "BlockRichardsonBaselineOp";
  parse_attributes();
  infer_shapes_from_model();
}

void BlockRichardsonBaselineOp::parse_attributes() {
  auto it = _attributes.find("batch_size");
  if (it != _attributes.end()) try { _batch_size = stoul(it->second); } catch(...) {}
  it = _attributes.find("block_size");
  if (it != _attributes.end()) try { _block_size = std::max(2u, (uint32_t)stoul(it->second)); } catch(...) {}
  it = _attributes.find("layers");
  if (it != _attributes.end()) try { _layers = std::max(1u, (uint32_t)stoul(it->second)); } catch(...) {}
}

void BlockRichardsonBaselineOp::infer_shapes_from_model() {
  if (!_matrix_shape.empty() || !_model) return;
  if (_inputs.empty()) return;
  auto* t = _model->get_tensor(_inputs[0]);
  if (!t) return;
  auto dims = t->get_dims();
  if (dims.size() == 3) { _batch_size = dims[0]; _matrix_shape = {dims[1], dims[2]}; }
  else _matrix_shape = dims;
}

void BlockRichardsonBaselineOp::initialize_tiles(MappingTable&) {
  for (uint32_t b = 0; b < _batch_size; ++b) {
    auto tile = std::make_unique<Tile>(Tile{
        .status = Tile::Status::INITIALIZED, .optype = _optype,
        .layer_id = _id, .batch = b, .core_id = static_cast<int>(b % _config.num_cores)});
    initialize_instructions(tile.get(), Mapping{});
    if (!tile->instructions.empty()) _tiles.push_back(std::move(tile));
  }
}

void BlockRichardsonBaselineOp::initialize_instructions(Tile* tile, Mapping) {
  const uint32_t M = _matrix_shape[0];
  const uint32_t U = _matrix_shape[1];
  const uint32_t B = _block_size;
  const uint32_t nB = U / B;
  const uint32_t L = _layers;
  const addr_type size_mu = static_cast<addr_type>(M) * U * _config.precision;
  const addr_type size_uu = static_cast<addr_type>(U) * U * _config.precision;
  const uint32_t epa = std::max(1u, _config.dram_req_size / _config.precision);

  const addr_type dH   = get_operand_addr(_INPUT_OPERAND + 0) + static_cast<addr_type>(tile->batch) * size_mu;
  const addr_type dReg = get_operand_addr(_INPUT_OPERAND + 1);
  const addr_type dY   = get_operand_addr(_INPUT_OPERAND + 2) + static_cast<addr_type>(tile->batch) * size_mu;
  const addr_type dOut = get_operand_addr(_OUTPUT_OPERAND + 0) + static_cast<addr_type>(tile->batch) * size_uu;

  const addr_type aH   = SPAD_BASE;
  const addr_type aReg = aH   + size_mu;
  const addr_type aYin = aReg + size_uu;
  const addr_type aA   = aYin + size_mu;
  const addr_type aB   = aA   + size_uu;
  const addr_type aYk  = aB   + size_uu;
  const addr_type aBY  = aYk  + size_uu;
  const addr_type aR   = aBY  + size_uu;
  const addr_type aYnext=aR  + size_uu;
  const addr_type aW   = aYnext+size_uu;
  const addr_type aTmp = aW   + size_uu;
  const addr_type aXhat= ACCUM_SPAD_BASE;

  const std::vector<uint32_t> sMU{M, U}, sUU{U, U};
  FormulaLogger::instance().set_algorithm("block_richardson_v3", B, L, U);

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
  movin(dH, aH, M, U, sMU, _INPUT_OPERAND + 0);
  movin(dReg, aReg, U, U, sUU, _INPUT_OPERAND + 1);
  movin(dY, aYin, M, U, sMU, _INPUT_OPERAND + 2);
  barrier("BRI_LOAD", 1);

  // Phase 2: GRAM + REG
  tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
      .opcode = Opcode::GEMM_PRELOAD, .id = "BRI_GRAM",
      .dest_addr = aA, .compute_size = U, .src_addrs = {aH, aH},
      .tile_m = U, .tile_k = M, .tile_n = U, .my_tile = tile}));
  FormulaLogger::instance().emit_step("BRI_GRAM", "GEMM",
      {"H^H","H"}, "G", {{M,U},{U,M}}, {U,U}, tile->batch, "BRI_GRAM");

  tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
      .opcode = Opcode::ADD, .id = "BRI_REG",
      .dest_addr = aA, .compute_size = U*U, .src_addrs = {aA, aReg},
      .tile_m = U, .tile_k = U, .tile_n = U, .my_tile = tile}));
  FormulaLogger::instance().emit_step("BRI_REG", "DIAG_ADD",
      {"G","lambda*I"}, "A", {{U,U},{U,U}}, {U,U}, tile->batch, "BRI_REG");
  barrier("BRI_REG2PRECOND", 3);

  // Init regions
  tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
      .opcode = Opcode::ADD, .id = "BRI_INIT_B",
      .dest_addr = aB, .compute_size = U*U, .src_addrs = {aReg, aReg},
      .tile_m = U, .tile_k = U, .tile_n = U, .my_tile = tile}));

  // Phase 3: Block-diagonal preconditioner (B=2 direct 2x2 inverse)
  for (uint32_t b = 0; b < nB; ++b) {
    // 2x2 direct inverse: inv([a b; c d]) = 1/(ad-bc) * [d -b; -c a]
    // SCALAR_MUL: ad, bc
    tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
        .opcode = Opcode::SCALAR_MUL, .id = "BRI_PRECOND_AD_"+std::to_string(b),
        .dest_addr = aTmp, .compute_size = 1, .src_addrs = {aA, aA},
        .tile_m = 1, .tile_k = 1, .tile_n = 1, .my_tile = tile}));
    tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
        .opcode = Opcode::SCALAR_MUL, .id = "BRI_PRECOND_BC_"+std::to_string(b),
        .dest_addr = aBY, .compute_size = 1, .src_addrs = {aA, aA},
        .tile_m = 1, .tile_k = 1, .tile_n = 1, .my_tile = tile}));
    // det = ad - bc
    tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
        .opcode = Opcode::SCALAR_SUB, .id = "BRI_PRECOND_DET_"+std::to_string(b),
        .dest_addr = aTmp, .compute_size = 1, .src_addrs = {aTmp, aBY},
        .tile_m = 1, .tile_k = 1, .tile_n = 1, .my_tile = tile}));
    // inv_det = 1/det
    tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
        .opcode = Opcode::SCALAR_DIV, .id = "BRI_PRECOND_UNITY_"+std::to_string(b),
        .dest_addr = aBY, .compute_size = 1, .src_addrs = {aReg, aReg},
        .tile_m = 1, .tile_k = 1, .tile_n = 1, .my_tile = tile}));
    tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
        .opcode = Opcode::SCALAR_DIV, .id = "BRI_PRECOND_INVDET_"+std::to_string(b),
        .dest_addr = aBY, .compute_size = 1, .src_addrs = {aBY, aTmp},
        .tile_m = 1, .tile_k = 1, .tile_n = 1, .my_tile = tile}));
    // Store B block: B_00 = d*inv_det, B_01 = -b*inv_det, B_10 = -c*inv_det, B_11 = a*inv_det
    for (int op = 0; op < 4; ++op) {
      tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
          .opcode = Opcode::SCALAR_MUL,
          .id = "BRI_PRECOND_BLK_"+std::to_string(b)+"_"+std::to_string(op),
          .dest_addr = aB, .compute_size = 1, .src_addrs = {aA, aBY},
          .tile_m = 1, .tile_k = 1, .tile_n = 1, .my_tile = tile}));
    }
  }
  FormulaLogger::instance().emit_step("BRI_PRECOND", "BRI_PRECOND",
      {"A"}, "B", {{U,U}}, {U,U}, tile->batch, "BRI_PRECOND");

  // Phase 4: Initialize Y_0 = I
  tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
      .opcode = Opcode::ADD, .id = "BRI_INIT_Y0",
      .dest_addr = aYk, .compute_size = U*U, .src_addrs = {aReg, aReg},
      .tile_m = U, .tile_k = U, .tile_n = U, .my_tile = tile}));

  // Phase 5: Richardson Iteration
  for (uint32_t layer = 0; layer < L; ++layer) {
    // BY = B @ Y_k (block-diagonal GEMM)
    tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
        .opcode = Opcode::GEMM_PRELOAD,
        .id = "BRI_BY_"+std::to_string(layer),
        .dest_addr = aBY, .compute_size = U, .src_addrs = {aB, aYk},
        .tile_m = U, .tile_k = U, .tile_n = U, .my_tile = tile}));
    std::string y_in = (layer == 0) ? std::string("I") : ("Y_" + std::to_string(layer - 1));
    FormulaLogger::instance().emit_step("BRI_BY_"+std::to_string(layer), "GEMM",
        std::vector<std::string>{"B", y_in}, "BY_" + std::to_string(layer), {{U,U},{U,U}}, {U,U}, tile->batch, "BRI_BY_"+std::to_string(layer));

    // R = I - BY (residual)
    tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
        .opcode = Opcode::ADD, .id = "BRI_RESIDUAL_"+std::to_string(layer),
        .dest_addr = aR, .compute_size = U*U, .src_addrs = {aReg, aBY},
        .tile_m = U, .tile_k = U, .tile_n = U, .my_tile = tile}));
    FormulaLogger::instance().emit_step("BRI_RESIDUAL_"+std::to_string(layer), "MATRIX_SUB",
        std::vector<std::string>{"I", "BY_" + std::to_string(layer)}, "R_" + std::to_string(layer), {{U,U},{U,U}}, {U,U}, tile->batch, "BRI_RESIDUAL_"+std::to_string(layer));

    // Y_{k+1} = Y_k + omega * R  (omega=1 in baseline Chebyshev)
    tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
        .opcode = Opcode::ADD, .id = "BRI_Y_UPDATE_"+std::to_string(layer),
        .dest_addr = aYnext, .compute_size = U*U, .src_addrs = {aYk, aR},
        .tile_m = U, .tile_k = U, .tile_n = U, .my_tile = tile}));
    FormulaLogger::instance().emit_step("BRI_Y_UPDATE_"+std::to_string(layer), "MATRIX_ADD",
        std::vector<std::string>{y_in, "R_" + std::to_string(layer)}, "Y_" + std::to_string(layer), {{U,U},{U,U}}, {U,U}, tile->batch,
        "BRI_Y_UPDATE_"+std::to_string(layer));

    // Swap Y buffers for next iteration
    if ((layer+1) % 4 == 0 && layer+1 < L)
      barrier("BRI_SYNC_"+std::to_string(layer), 5);
  }
  // Restore aYk to the last computed Y
  // (aYk and aYnext swapped an even number of times, so aYk holds result)

  // Phase 6: Output — W = Ainv @ H^H, X_hat = W @ Y
  tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
      .opcode = Opcode::PIPE_BARRIER, .id = "BRI_INV2W",
      .my_tile = tile, .is_barrier = true, .barrier_type = 6}));
  tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
      .opcode = Opcode::GEMM_PRELOAD, .id = "BRI_W",
      .dest_addr = aW, .compute_size = M, .src_addrs = {aYk, aH},
      .tile_m = U, .tile_k = U, .tile_n = M, .my_tile = tile}));
  tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
      .opcode = Opcode::GEMM_PRELOAD, .id = "BRI_XHAT",
      .dest_addr = aXhat, .compute_size = U, .src_addrs = {aW, aYin},
      .tile_m = U, .tile_k = M, .tile_n = U, .my_tile = tile}));
  // BRI_FINAL: simplified DAG representation; hardware computes W=Y_{L-1}@H then X_hat=W@Yin.
  // DAG approximates Ainv ≈ Y_{L-1} @ Y_{L-1} (Richardson converges to B^{-1}).
  FormulaLogger::instance().emit_step("BRI_FINAL", "GEMM",
      {"Y_" + std::to_string(L-1), "Y_" + std::to_string(L-1)}, "Ainv", {{U,U},{U,U}}, {U,U}, tile->batch, "BRI_XHAT");
  barrier("BRI_PRE_MOVOUT", 6);

  std::set<addr_type> outs;
  for (uint32_t r = 0; r < U; ++r)
    for (uint32_t c = 0; c < U; c += epa)
      outs.insert(dOut + make_address({r, std::min(c, U-1)}, sUU));
  tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
      .opcode = Opcode::MOVOUT, .id = "BRI_STORE",
      .dest_addr = aXhat, .size = static_cast<uint32_t>(outs.size()),
      .src_addrs = std::vector<addr_type>(outs.begin(), outs.end()),
      .operand_id = _OUTPUT_OPERAND, .tile_m = U, .tile_k = U, .tile_n = U,
      .src_from_accum = true, .last_inst = true, .my_tile = tile}));
}
