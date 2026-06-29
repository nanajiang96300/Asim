#include "LDLDecompNoBlockAlignedOp.h"
#include "Model.h"
#include "FormulaLogger.h"
#include <algorithm>

LDLDecompNoBlockAlignedOp::LDLDecompNoBlockAlignedOp(
    SimulationConfig config, Model* model, const std::string& name,
    std::map<std::string, std::string>& attributes, uint32_t target_core)
    : Operation(config, model, name, attributes, target_core) {
  _optype = "LDLDecompNoBlockAlignedOp";
  parse_attributes();
  infer_shapes_from_model();
}

LDLDecompNoBlockAlignedOp::LDLDecompNoBlockAlignedOp(
    SimulationConfig config, MappingTable& mapping_table,
    const std::vector<uint32_t>& matrix_shape, uint32_t target_core)
    : Operation(config, mapping_table, target_core) {
  (void)mapping_table;
  _optype = "LDLDecompNoBlockAlignedOp";
  _matrix_shape = matrix_shape;
  parse_attributes();
}

void LDLDecompNoBlockAlignedOp::parse_attributes() {
  auto it_batch = _attributes.find("batch_size");
  if (it_batch != _attributes.end()) {
    try { _batch_size = static_cast<uint32_t>(std::stoul(it_batch->second)); }
    catch (...) { _batch_size = 96; }
  }
  auto it_ll = _attributes.find("use_left_looking");
  if (it_ll != _attributes.end()) {
    _use_left_looking = (it_ll->second == "1" || it_ll->second == "true");
  }
}

void LDLDecompNoBlockAlignedOp::infer_shapes_from_model() {
  if (!_matrix_shape.empty() || !_model) return;
  if (_inputs.empty()) return;
  Tensor* h_tensor = _model->get_tensor(_inputs[0]);
  if (!h_tensor) return;
  std::vector<uint32_t> dims = h_tensor->get_dims();
  if (dims.size() == 3) { _batch_size = dims[0]; _matrix_shape = {dims[1], dims[2]}; }
  else { _matrix_shape = dims; }
}

void LDLDecompNoBlockAlignedOp::initialize_tiles(MappingTable&) {
  std::vector<int> core_load(_config.num_cores, 0);
  for (uint32_t b = 0; b < _batch_size; ++b) {
    uint32_t assigned_core = b % _config.num_cores;
    auto tile = std::make_unique<Tile>(Tile{
        .status = Tile::Status::INITIALIZED, .optype = _optype,
        .layer_id = _id, .fused_op_id = 0, .batch = b,
        .Q = 1, .P = 1, .M = 0, .C = 0, .S = 1, .R = 1,
        .stat = {}, .instructions = {}, .accum = false, .skip = false,
        .spad_id = 0, .accum_spad_id = 0,
        .core_id = static_cast<int>(assigned_core), .inst_finished = false});
    initialize_instructions(tile.get(), Mapping{});
    if (!tile->instructions.empty()) { _tiles.push_back(std::move(tile)); core_load[assigned_core]++; }
  }
}

void LDLDecompNoBlockAlignedOp::initialize_instructions(Tile* tile, Mapping) {
  if (_matrix_shape.size() < 2) return;
  const uint32_t M = _matrix_shape[_matrix_shape.size() - 2];
  const uint32_t U = _matrix_shape[_matrix_shape.size() - 1];
  const uint32_t n_scalars = U;  // element-wise, same as Cholesky NoBlock

  FormulaLogger::instance().set_algorithm("ldl_noblock_aligned", 1, 0, U);

  addr_type size_mu = static_cast<addr_type>(M) * U * _config.precision;
  addr_type size_uu = static_cast<addr_type>(U) * U * _config.precision;
  addr_type batch_offset_mu = static_cast<addr_type>(tile->batch) * size_mu;
  addr_type batch_offset_uu = static_cast<addr_type>(tile->batch) * size_uu;

  addr_type h_base = get_operand_addr(_INPUT_OPERAND + 0) + batch_offset_mu;
  addr_type reg_base = get_operand_addr(_INPUT_OPERAND + 1);
  addr_type out_base = get_operand_addr(_OUTPUT_OPERAND + 0) + batch_offset_uu;

  addr_type addr_H = SPAD_BASE, addr_Reg = addr_H + size_mu;
  addr_type addr_G = addr_Reg + size_uu, addr_A = addr_G + size_uu;
  addr_type addr_L = addr_A + size_uu, addr_D = addr_L + size_uu;
  addr_type addr_tmp = addr_D + size_uu, addr_inv = addr_tmp + size_uu;
  addr_type addr_Y = addr_inv + size_uu, addr_Ainv = ACCUM_SPAD_BASE;

  int elems_per_access = _config.dram_req_size / _config.precision;
  if (elems_per_access <= 0) elems_per_access = 1;
  std::vector<uint32_t> shape_mu{M, U}, shape_uu{U, U};

  auto emit_movin = [&](addr_type dram_base, addr_type spad_dest,
                        uint32_t rows, uint32_t cols,
                        const std::vector<uint32_t>& shape, uint32_t operand_id) {
    std::set<addr_type> addrs;
    for (uint32_t r = 0; r < rows; ++r)
      for (uint32_t c = 0; c < cols; c += static_cast<uint32_t>(elems_per_access)) {
        uint32_t col = std::min(c, cols - 1);
        addrs.insert(dram_base + make_address({r, col}, shape));
      }
    if (!addrs.empty())
      tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
          .opcode = Opcode::MOVIN, .dest_addr = spad_dest,
          .size = static_cast<uint32_t>(addrs.size()),
          .src_addrs = std::vector<addr_type>(addrs.begin(), addrs.end()),
          .operand_id = operand_id, .base_addr = dram_base,
          .tile_m = rows, .tile_k = cols, .my_tile = tile}));
  };

  emit_movin(h_base, addr_H, M, U, shape_mu, _INPUT_OPERAND + 0);
  emit_movin(reg_base, addr_Reg, U, U, shape_uu, _INPUT_OPERAND + 1);

  tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
      .opcode = Opcode::PIPE_BARRIER, .id = "LDL_NA_BARRIER_LOAD2GRAM",
      .my_tile = tile, .is_barrier = true, .barrier_type = 1}));

  // Gram: G = H^H @ H
  FormulaLogger::instance().emit_step("LDL_NA_GRAM", "GEMM", {"H", "H^H"}, "G",
      {{M, U}, {U, M}}, {U, U}, tile->batch, "LDL_NA_GRAM");
  tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
      .opcode = Opcode::GEMM_PRELOAD, .id = "LDL_NA_GRAM", .dest_addr = addr_G,
      .compute_size = U, .src_addrs = std::vector<addr_type>{addr_H, addr_H},
      .tile_m = U, .tile_k = M, .tile_n = U, .src_from_accum = false, .my_tile = tile}));

  // Reg: A = G + lambda*I
  tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
      .opcode = Opcode::ADD, .id = "LDL_NA_REG", .dest_addr = addr_A,
      .compute_size = U * U, .src_addrs = std::vector<addr_type>{addr_G, addr_Reg},
      .tile_m = U, .tile_k = U, .tile_n = U, .src_from_accum = false, .my_tile = tile}));
  FormulaLogger::instance().emit_step("LDL_NA_REG", "DIAG_ADD", {"G", "lambda*I"}, "A",
      {{U, U}, {U, U}}, {U, U}, tile->batch, "LDL_NA_REG");

  tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
      .opcode = Opcode::PIPE_BARRIER, .my_tile = tile, .is_barrier = true, .barrier_type = 3}));

  const std::string prefix = "LDL_NA_";

  // ===== RIGHT-LOOKING LDL: per-element SCALAR ops, explicit RK_UPDATE =====
  for (uint32_t j = 0; j < n_scalars; ++j) {
    // --- D_UPDATE: D_jj = A_jj - Σ_{k<j} L_jk·D_kk·L_jk^H ---
    // Per-k SCALAR_MUL (same pattern as Cholesky POTRF, no compute_size merging!)
    for (uint32_t k = 0; k < j; ++k) {
      tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
          .opcode = Opcode::SCALAR_MUL,
          .id = prefix + "D_UPDATE_" + std::to_string(j) + "_" + std::to_string(k),
          .dest_addr = addr_tmp, .compute_size = 1,
          .src_addrs = std::vector<addr_type>{addr_L, addr_D},
          .tile_m = 1, .tile_k = 1, .tile_n = 1, .src_from_accum = false, .my_tile = tile}));
      // Extra D_kk multiply (LDL has one more multiply per term vs Cholesky)
      tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
          .opcode = Opcode::SCALAR_MUL,
          .id = prefix + "D_UPDATE_DK_" + std::to_string(j) + "_" + std::to_string(k),
          .dest_addr = addr_tmp, .compute_size = 1,
          .src_addrs = std::vector<addr_type>{addr_tmp, addr_tmp},
          .tile_m = 1, .tile_k = 1, .tile_n = 1, .src_from_accum = false, .my_tile = tile}));
    }
    // D_jj = A_jj - sum (用 SCALAR_ADD 累减, 先初始化为 A_jj)
    // Simplified: just use SCALAR_DIV for D_jj^{-1}
    tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
        .opcode = Opcode::SCALAR_DIV,
        .id = prefix + "D_DIAG_INV_" + std::to_string(j),
        .dest_addr = addr_D, .compute_size = 1,
        .src_addrs = std::vector<addr_type>{addr_Reg, addr_A},  // 1/A_jj
        .tile_m = 1, .tile_k = 1, .tile_n = 1, .src_from_accum = false, .my_tile = tile}));

    FormulaLogger::instance().emit_step(
        prefix + "D_UPDATE_" + std::to_string(j), "DIAG_INV",
        {"A"}, "D_inv", {{U, U}}, {1, 1}, tile->batch, prefix + "D_UPDATE");

    // --- L_UPDATE: L_ij = (A_ij - Σ_{k<j} L_ik·D_kk·L_jk^H) / D_jj ---
    // Per-(i,k) SCALAR_MUL (same pattern as Cholesky TRSM)
    for (uint32_t i = j + 1; i < n_scalars; ++i) {
      for (uint32_t k = 0; k < j; ++k) {
        tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
            .opcode = Opcode::SCALAR_MUL,
            .id = prefix + "L_UPDATE_" + std::to_string(i) + "_" + std::to_string(j) + "_" + std::to_string(k),
            .dest_addr = addr_tmp, .compute_size = 1,
            .src_addrs = std::vector<addr_type>{addr_L, addr_D},
            .tile_m = 1, .tile_k = 1, .tile_n = 1, .src_from_accum = false, .my_tile = tile}));
        // Extra D_kk multiply
        tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
            .opcode = Opcode::SCALAR_MUL,
            .id = prefix + "L_UPDATE_DK_" + std::to_string(i) + "_" + std::to_string(j) + "_" + std::to_string(k),
            .dest_addr = addr_tmp, .compute_size = 1,
            .src_addrs = std::vector<addr_type>{addr_tmp, addr_tmp},
            .tile_m = 1, .tile_k = 1, .tile_n = 1, .src_from_accum = false, .my_tile = tile}));
      }
      // L_ij = (A_ij - sum) / D_jj
      tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
          .opcode = Opcode::SCALAR_DIV,
          .id = prefix + "L_DIV_" + std::to_string(i) + "_" + std::to_string(j),
          .dest_addr = addr_L, .compute_size = 1,
          .src_addrs = std::vector<addr_type>{addr_A, addr_D},
          .tile_m = 1, .tile_k = 1, .tile_n = 1, .src_from_accum = false, .my_tile = tile}));
    }

    // --- RK_UPDATE (right-looking): skipped in left-looking mode ---
    if (!_use_left_looking) {
    for (uint32_t i = j + 1; i < n_scalars; ++i) {
      for (uint32_t k = i; k < n_scalars; ++k) {
        tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
            .opcode = Opcode::SCALAR_MUL,
            .id = prefix + "RK_UPDATE_" + std::to_string(i) + "_" + std::to_string(k) + "_" + std::to_string(j),
            .dest_addr = addr_A, .compute_size = 1,
            .src_addrs = std::vector<addr_type>{addr_L, addr_D},
            .tile_m = 1, .tile_k = 1, .tile_n = 1, .src_from_accum = false, .my_tile = tile}));
        tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
            .opcode = Opcode::SCALAR_MUL,
            .id = prefix + "RK_UPDATE_DJ_" + std::to_string(i) + "_" + std::to_string(k) + "_" + std::to_string(j),
            .dest_addr = addr_A, .compute_size = 1,
            .src_addrs = std::vector<addr_type>{addr_A, addr_A},
            .tile_m = 1, .tile_k = 1, .tile_n = 1, .src_from_accum = false, .my_tile = tile}));
      }
    }
    }  // end if (!_use_left_looking)

    tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
        .opcode = Opcode::PIPE_BARRIER, .id = prefix + "BARRIER_STEP_" + std::to_string(j),
        .my_tile = tile, .is_barrier = true, .barrier_type = 4}));
  }

  // FWD solve + BWD assemble (simplified — same GEMM as Cholesky NoBlock)
  for (uint32_t c = 0; c < n_scalars; ++c) {
    tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
        .opcode = Opcode::SCALAR_DIV, .id = prefix + "FWD_DIAG_INV_" + std::to_string(c),
        .dest_addr = addr_Y, .compute_size = 1,
        .src_addrs = std::vector<addr_type>{addr_Reg, addr_D},
        .tile_m = 1, .tile_k = 1, .tile_n = 1, .src_from_accum = false, .my_tile = tile}));
    for (uint32_t i = c + 1; i < n_scalars; ++i) {
      tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
          .opcode = Opcode::SCALAR_MUL,
          .id = prefix + "FWD_OFF_" + std::to_string(i) + "_" + std::to_string(c),
          .dest_addr = addr_Y, .compute_size = 1,
          .src_addrs = std::vector<addr_type>{addr_L, addr_Y},
          .tile_m = 1, .tile_k = 1, .tile_n = 1, .src_from_accum = false, .my_tile = tile}));
    }
  }

  // BWD: geometric-series assemble via BWD MAC (same as Cholesky BWD)
  tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
      .opcode = Opcode::GEMM_PRELOAD, .id = prefix + "BWD_MAC_FULL",
      .dest_addr = addr_Ainv, .compute_size = U,
      .src_addrs = std::vector<addr_type>{addr_Y, addr_Y},
      .tile_m = U, .tile_k = U, .tile_n = U, .src_from_accum = false, .my_tile = tile}));

  FormulaLogger::instance().emit_step(
      prefix + "BWD_ASSEMBLE", "GEMM", {"Y", "Y^H"}, "A_inv",
      {{U, U}, {U, U}}, {U, U}, tile->batch, prefix + "BWD");

  // Store output
  std::set<addr_type> output_addrs;
  for (uint32_t r = 0; r < U; ++r)
    for (uint32_t c = 0; c < U; c += static_cast<uint32_t>(elems_per_access)) {
      uint32_t col = std::min(c, U - 1);
      output_addrs.insert(out_base + make_address({r, col}, shape_uu));
    }
  tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
      .opcode = Opcode::MOVOUT, .id = prefix + "STORE",
      .dest_addr = addr_Ainv, .size = static_cast<uint32_t>(output_addrs.size()),
      .src_addrs = std::vector<addr_type>(output_addrs.begin(), output_addrs.end()),
      .operand_id = _OUTPUT_OPERAND, .tile_m = U, .tile_k = U, .tile_n = 0,
      .src_from_accum = true, .last_inst = true, .my_tile = tile}));

  if (tile->instructions.empty())
    spdlog::error("LDLDecompNoBlockAlignedOp: No instructions for Batch {} Core {}",
                  tile->batch, tile->core_id);
}
