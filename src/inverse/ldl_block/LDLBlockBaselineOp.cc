#include <algorithm>
#include "LDLBlockBaselineOp.h"
#include "Model.h"
#include "FormulaLogger.h"
#include <set>

LDLBlockBaselineOp::LDLBlockBaselineOp(
    SimulationConfig config, Model* model, const std::string& name,
    std::map<std::string, std::string>& attributes, uint32_t target_core)
    : Operation(config, model, name, attributes, target_core) {
  _optype = "LDLBlockBaselineOp";
  parse_attributes();
  infer_shapes_from_model();
}

void LDLBlockBaselineOp::parse_attributes() {
  auto it = _attributes.find("batch_size");
  if (it != _attributes.end()) try { _batch_size = stoul(it->second); } catch(...) {}
  it = _attributes.find("block_size");
  if (it != _attributes.end()) try { _block_size = std::max(2u, (uint32_t)stoul(it->second)); } catch(...) {}
}

void LDLBlockBaselineOp::infer_shapes_from_model() {
  if (!_matrix_shape.empty() || !_model) return;
  if (_inputs.empty()) return;
  auto* t = _model->get_tensor(_inputs[0]);
  if (!t) return;
  auto dims = t->get_dims();
  if (dims.size() == 3) { _batch_size = dims[0]; _matrix_shape = {dims[1], dims[2]}; }
  else _matrix_shape = dims;
}

void LDLBlockBaselineOp::initialize_tiles(MappingTable&) {
  for (uint32_t b = 0; b < _batch_size; ++b) {
    auto tile = std::make_unique<Tile>(Tile{
        .status = Tile::Status::INITIALIZED, .optype = _optype,
        .layer_id = _id, .batch = b, .core_id = static_cast<int>(b % _config.num_cores)});
    initialize_instructions(tile.get(), Mapping{});
    if (!tile->instructions.empty()) _tiles.push_back(std::move(tile));
  }
}

void LDLBlockBaselineOp::initialize_instructions(Tile* tile, Mapping) {
  const uint32_t M = _matrix_shape[0];
  const uint32_t U = _matrix_shape[1];
  const uint32_t B = _block_size;
  const uint32_t nB = U / B;
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
  const addr_type aD   = aL   + size_uu;
  const addr_type aDinv= aD   + size_uu;
  const addr_type aTmp = aDinv+ size_uu;
  const addr_type aY   = aTmp + size_uu;
  const addr_type aAinv= ACCUM_SPAD_BASE;

  const std::vector<uint32_t> sMU{M, U}, sUU{U, U};
  FormulaLogger::instance().set_algorithm("ldl_block_v3", B, 0, U);

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

  // Phase 1-2: MOVIN + GRAM + REG
  movin(dH, aH, M, U, sMU, _INPUT_OPERAND + 0);
  movin(dReg, aReg, U, U, sUU, _INPUT_OPERAND + 1);
  barrier("LDL_BLK_LOAD", 1);

  tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
      .opcode = Opcode::GEMM_PRELOAD, .id = "LDL_BLK_GRAM",
      .dest_addr = aG, .compute_size = U, .src_addrs = {aH, aH},
      .tile_m = U, .tile_k = M, .tile_n = U, .my_tile = tile}));
  FormulaLogger::instance().emit_step("LDL_BLK_GRAM", "GEMM",
      {"H^H","H"}, "G", {{M,U},{U,M}}, {U,U}, tile->batch, "LDL_BLK_GRAM");

  tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
      .opcode = Opcode::ADD, .id = "LDL_BLK_REG",
      .dest_addr = aA, .compute_size = U*U, .src_addrs = {aG, aReg},
      .tile_m = U, .tile_k = U, .tile_n = U, .my_tile = tile}));
  FormulaLogger::instance().emit_step("LDL_BLK_REG", "DIAG_ADD",
      {"G","lambda*I"}, "A", {{U,U},{U,U}}, {U,U}, tile->batch, "LDL_BLK_REG");
  barrier("LDL_BLK_REG2DECOMP", 3);

  // Init regions
  tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
      .opcode = Opcode::ADD, .id = "LDL_BLK_INIT_L",
      .dest_addr = aL, .compute_size = U*U, .src_addrs = {aReg, aReg},
      .tile_m = U, .tile_k = U, .tile_n = U, .my_tile = tile}));
  tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
      .opcode = Opcode::ADD, .id = "LDL_BLK_INIT_D",
      .dest_addr = aD, .compute_size = U, .src_addrs = {aReg, aReg},
      .tile_m = 1, .tile_k = U, .tile_n = 1, .my_tile = tile}));
  tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
      .opcode = Opcode::ADD, .id = "LDL_BLK_INIT_DINV",
      .dest_addr = aDinv, .compute_size = U, .src_addrs = {aReg, aReg},
      .tile_m = 1, .tile_k = U, .tile_n = 1, .my_tile = tile}));
  tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
      .opcode = Opcode::ADD, .id = "LDL_BLK_INIT_TMP",
      .dest_addr = aTmp, .compute_size = U, .src_addrs = {aReg, aReg},
      .tile_m = 1, .tile_k = U, .tile_n = 1, .my_tile = tile}));
  tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
      .opcode = Opcode::ADD, .id = "LDL_BLK_INIT_Y",
      .dest_addr = aY, .compute_size = U*U, .src_addrs = {aReg, aReg},
      .tile_m = U, .tile_k = U, .tile_n = U, .my_tile = tile}));

  // Phase 3: Block LDL Decomposition
  for (uint32_t j = 0; j < nB; ++j) {
    // D_UPDATE: Schur complement for diagonal block
    for (uint32_t k = 0; k < j; ++k) {
      tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
          .opcode = Opcode::GEMM,
          .id = "LDL_BLK_DUPD_GEMM_" + std::to_string(j) + "_" + std::to_string(k),
          .dest_addr = aTmp, .compute_size = B,
          .src_addrs = {aL, aD}, .tile_m = B, .tile_k = B, .tile_n = B, .my_tile = tile}));
    }
    // B×B LDL on diagonal block
    for (uint32_t jj = 0; jj < B; ++jj) {
      for (uint32_t kk = 0; kk < jj; ++kk) {
        tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
            .opcode = Opcode::SCALAR_MUL, .id = "LDL_BLK_DU_SQ_"+std::to_string(j)+"_"+std::to_string(jj)+"_"+std::to_string(kk),
            .dest_addr = aTmp, .compute_size = 1, .src_addrs = {aA, aA},
            .tile_m = 1, .tile_k = 1, .tile_n = 1, .my_tile = tile}));
        tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
            .opcode = Opcode::SCALAR_MUL, .id = "LDL_BLK_DU_DMUL_"+std::to_string(j)+"_"+std::to_string(jj)+"_"+std::to_string(kk),
            .dest_addr = aTmp, .compute_size = 1, .src_addrs = {aTmp, aD},
            .tile_m = 1, .tile_k = 1, .tile_n = 1, .my_tile = tile}));
        tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
            .opcode = Opcode::SCALAR_SUB, .id = "LDL_BLK_DU_SUB_"+std::to_string(j)+"_"+std::to_string(jj)+"_"+std::to_string(kk),
            .dest_addr = aA, .compute_size = 1, .src_addrs = {aA, aTmp},
            .tile_m = 1, .tile_k = 1, .tile_n = 1, .my_tile = tile}));
      }
      tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
          .opcode = Opcode::SCALAR_DIV, .id = "LDL_BLK_UNITY_"+std::to_string(j)+"_"+std::to_string(jj),
          .dest_addr = aTmp, .compute_size = 1, .src_addrs = {aReg, aReg},
          .tile_m = 1, .tile_k = 1, .tile_n = 1, .my_tile = tile}));
      tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
          .opcode = Opcode::SCALAR_DIV, .id = "LDL_BLK_DINV_"+std::to_string(j)+"_"+std::to_string(jj),
          .dest_addr = aDinv, .compute_size = 1, .src_addrs = {aTmp, aA},
          .tile_m = 1, .tile_k = 1, .tile_n = 1, .my_tile = tile}));
      tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
          .opcode = Opcode::SCALAR_MUL, .id = "LDL_BLK_DSTORE_"+std::to_string(j)+"_"+std::to_string(jj),
          .dest_addr = aD, .compute_size = 1, .src_addrs = {aA, aTmp},
          .tile_m = 1, .tile_k = 1, .tile_n = 1, .my_tile = tile}));
    }
    // DUPDATE: FormulaLogger emits single LDL_DECOMPOSE after loop (see below)

    // L_UPDATE for off-diagonal blocks
    for (uint32_t i = j + 1; i < nB; ++i) {
      for (uint32_t k = 0; k < j; ++k) {
        tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
            .opcode = Opcode::GEMM,
            .id = "LDL_BLK_LUPD_GEMM_" + std::to_string(i)+"_"+std::to_string(j)+"_"+std::to_string(k),
            .dest_addr = aTmp, .compute_size = B,
            .src_addrs = {aL, aD}, .tile_m = B, .tile_k = B, .tile_n = B, .my_tile = tile}));
      }
      for (uint32_t ii = 0; ii < B; ++ii) {
        tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
            .opcode = Opcode::SCALAR_MUL,
            .id = "LDL_BLK_LAPPLY_"+std::to_string(i)+"_"+std::to_string(j)+"_"+std::to_string(ii),
            .dest_addr = aL, .compute_size = 1, .src_addrs = {aA, aDinv},
            .tile_m = 1, .tile_k = 1, .tile_n = 1, .my_tile = tile}));
      }
      // LUPDATE: FormulaLogger emits single LDL_DECOMPOSE after loop (see below)
    }
    barrier("LDL_BLK_COL_"+std::to_string(j), 4);
  }

  // Emit single high-level LDL_DECOMPOSE for DAG verification
  // (prim_ldl_decompose computes L,D,Dinv + forward solve + sqrt(Dinv) scaling internally)
  FormulaLogger::instance().emit_step("LDL_BLK_DECOMPOSE", "LDL_DECOMPOSE",
      {"A"}, "Y", {{U,U}}, {U,U}, tile->batch, "LDL_BLK_COL_" + std::to_string(nB-1));

  // Phase 4: Forward Solve Z = L^{-1} (unit triangular)
  // Hardware forward solve + Phase 6 sqrt(Dinv) scaling are covered by the
  // LDL_DECOMPOSE DAG primitive which internally handles: L·D·L^H → Z = L^{-1} → Y = Z*sqrt(Dinv).
  // No separate emit_step needed for these phases (H3 known limitation).
  for (uint32_t c = 0; c < nB; ++c) {
    for (uint32_t ii = 0; ii < B; ++ii) {
      tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
          .opcode = Opcode::SCALAR_DIV, .id = "LDL_BLK_FWD_DIAG_"+std::to_string(c),
          .dest_addr = aY, .compute_size = 1, .src_addrs = {aReg, aReg},
          .tile_m = 1, .tile_k = 1, .tile_n = 1, .my_tile = tile}));
    }
    for (uint32_t i = c + 1; i < nB; ++i) {
      tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
          .opcode = Opcode::GEMM, .id = "LDL_BLK_FWD_GEMM_"+std::to_string(i)+"_"+std::to_string(c),
          .dest_addr = aTmp, .compute_size = B, .src_addrs = {aL, aY},
          .tile_m = B, .tile_k = B, .tile_n = B, .my_tile = tile}));
      for (uint32_t ii = 0; ii < B; ++ii) {
        tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
            .opcode = Opcode::SCALAR_SUB, .id = "LDL_BLK_FWD_NEG_"+std::to_string(i)+"_"+std::to_string(c),
            .dest_addr = aY, .compute_size = 1, .src_addrs = {aReg, aTmp},
            .tile_m = 1, .tile_k = 1, .tile_n = 1, .my_tile = tile}));
      }
    }
    barrier("LDL_BLK_FB_"+std::to_string(c), 5);
  }

  // Phase 5: Forward Solve Complete — Y = L^{-1}

  // Phase 6: sqrt(Dinv) weighting + BWD GEMM
  for (uint32_t c = 0; c < nB; ++c)
    for (uint32_t ii = 0; ii < B; ++ii) {
      tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
          .opcode = Opcode::SCALAR_SQRT, .id = "LDL_BLK_SQRT_"+std::to_string(c),
          .dest_addr = aTmp, .compute_size = 1, .src_addrs = {aDinv},
          .tile_m = 1, .tile_k = 1, .tile_n = 1, .my_tile = tile}));
      tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
          .opcode = Opcode::SCALAR_MUL, .id = "LDL_BLK_SCALE_"+std::to_string(c),
          .dest_addr = aY, .compute_size = B, .src_addrs = {aY, aTmp},
          .tile_m = B, .tile_k = 1, .tile_n = 1, .my_tile = tile}));
    }

  // Phase 6: sqrt(Dinv) weighting applied in hardware;
  // DAG already has sqrt-scaled Y from LDL_DECOMPOSE, so no separate SQRT_SCALE emit_step.

  tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
      .opcode = Opcode::GEMM_PRELOAD, .id = "LDL_BLK_BWD",
      .dest_addr = aAinv, .compute_size = U, .src_addrs = {aY, aY},
      .tile_m = U, .tile_k = U, .tile_n = U, .my_tile = tile}));
  FormulaLogger::instance().emit_step("LDL_BLK_BWD_ASSEMBLE", "GEMM",
      {"Y^H","Y"}, "Ainv", {{U,U},{U,U}}, {U,U}, tile->batch, "LDL_BLK_BWD");
  barrier("LDL_BLK_PRE_MOVOUT", 6);

  std::set<addr_type> outs;
  for (uint32_t r = 0; r < U; ++r)
    for (uint32_t c = 0; c < U; c += epa)
      outs.insert(dOut + make_address({r, std::min(c, U-1)}, sUU));
  tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
      .opcode = Opcode::MOVOUT, .id = "LDL_BLK_STORE",
      .dest_addr = aAinv, .size = static_cast<uint32_t>(outs.size()),
      .src_addrs = std::vector<addr_type>(outs.begin(), outs.end()),
      .operand_id = _OUTPUT_OPERAND, .tile_m = U, .tile_k = U, .tile_n = U,
      .src_from_accum = true, .last_inst = true, .my_tile = tile}));
}
