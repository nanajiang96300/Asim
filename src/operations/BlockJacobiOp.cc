#include "BlockJacobiOp.h"

#include "../Model.h"

#include <algorithm>
#include <set>

BlockJacobiOp::BlockJacobiOp(SimulationConfig config,
                             Model* model,
                             const std::string& name,
                             std::map<std::string, std::string>& attributes,
                             uint32_t target_core)
    : Operation(config, model, name, attributes, target_core) {
  _optype = "BlockJacobiOp";
  parse_attributes();
  infer_shapes_from_model();
}

BlockJacobiOp::BlockJacobiOp(SimulationConfig config,
                             MappingTable& mapping_table,
                             const std::vector<uint32_t>& matrix_shape,
                             uint32_t target_core)
    : Operation(config, mapping_table, target_core) {
  (void)mapping_table;
  _optype = "BlockJacobiOp";
  _matrix_shape = matrix_shape;
  parse_attributes();
}

void BlockJacobiOp::parse_attributes() {
  auto it_layers = _attributes.find("layers");
  if (it_layers != _attributes.end()) {
    try {
      _layers = std::max(1u, static_cast<uint32_t>(std::stoul(it_layers->second)));
    } catch (...) {
      _layers = 16;
    }
  }

  auto it_block = _attributes.find("block_size");
  if (it_block != _attributes.end()) {
    try {
      _block_size = std::max(1u, static_cast<uint32_t>(std::stoul(it_block->second)));
    } catch (...) {
      _block_size = 2;
    }
  }

  auto it_group = _attributes.find("group_sync");
  if (it_group != _attributes.end()) {
    try {
      _group_sync = std::max(1u, static_cast<uint32_t>(std::stoul(it_group->second)));
    } catch (...) {
      _group_sync = 2;
    }
  }

  auto it_batch = _attributes.find("batch_size");
  if (it_batch != _attributes.end()) {
    try {
      _batch_size = static_cast<uint32_t>(std::stoul(it_batch->second));
    } catch (...) {
      _batch_size = 96;
    }
  }

  auto it_adapt = _attributes.find("adaptive_bounds");
  if (it_adapt != _attributes.end()) {
    const std::string v = it_adapt->second;
    _adaptive_bounds = (v == "1" || v == "true" || v == "TRUE" || v == "on" || v == "ON");
  }
}

void BlockJacobiOp::infer_shapes_from_model() {
  if (!_matrix_shape.empty() || !_model) return;
  if (_inputs.empty()) return;

  Tensor* h_tensor = _model->get_tensor(_inputs[0]);
  if (!h_tensor) return;

  std::vector<uint32_t> dims = h_tensor->get_dims();
  if (dims.size() == 3) {
    _batch_size = dims[0];
    _matrix_shape = {dims[1], dims[2]};
  } else {
    _matrix_shape = dims;
  }
}

void BlockJacobiOp::initialize_tiles(MappingTable& /*mapping_table*/) {
  if (_config.num_cores == 0) {
    spdlog::error("BlockJacobiOp: Invalid core count 0!");
    return;
  }

  std::vector<int> core_load(_config.num_cores, 0);

  for (uint32_t b = 0; b < _batch_size; ++b) {
    uint32_t assigned_core = b % _config.num_cores;
    auto tile = std::make_unique<Tile>(Tile{
        .status = Tile::Status::INITIALIZED,
        .optype = _optype,
        .layer_id = _id,
        .fused_op_id = 0,
        .batch = b,
        .Q = 1,
        .P = 1,
        .M = 0,
        .C = 0,
        .S = 1,
        .R = 1,
        .stat = {},
        .instructions = {},
        .accum = false,
        .skip = false,
        .spad_id = 0,
        .accum_spad_id = 0,
        .core_id = static_cast<int>(assigned_core),
        .inst_finished = false});

    initialize_instructions(tile.get(), Mapping{});
    if (!tile->instructions.empty()) {
      _tiles.push_back(std::move(tile));
      core_load[assigned_core]++;
    }
  }

  spdlog::info("BlockJacobiOp '{}': Dispatched {} batches across {} cores.",
               _name, _batch_size, _config.num_cores);
  std::string load_msg = "  > Load Distribution:";
  for (uint32_t core = 0; core < _config.num_cores; ++core) {
    load_msg += " Core" + std::to_string(core) + ": " + std::to_string(core_load[core]);
  }
  spdlog::info("{}", load_msg);
}

void BlockJacobiOp::initialize_instructions(Tile* tile, Mapping /*mapping*/) {
  if (_matrix_shape.size() < 2) {
    spdlog::error("BlockJacobiOp: matrix shape not set for layer {}", _name);
    return;
  }

  const uint32_t M = _matrix_shape[_matrix_shape.size() - 2];
  const uint32_t U = _matrix_shape[_matrix_shape.size() - 1];

  addr_type size_mu = static_cast<addr_type>(M) * U * _config.precision;
  addr_type size_uu = static_cast<addr_type>(U) * U * _config.precision;

  addr_type batch_offset_mu = static_cast<addr_type>(tile->batch) * size_mu;
  addr_type batch_offset_uu = static_cast<addr_type>(tile->batch) * size_uu;

  addr_type h_base = get_operand_addr(_INPUT_OPERAND + 0) + batch_offset_mu;
  addr_type reg_base = get_operand_addr(_INPUT_OPERAND + 1);
  addr_type y_base = get_operand_addr(_INPUT_OPERAND + 2) + batch_offset_mu;
  addr_type out_base = get_operand_addr(_OUTPUT_OPERAND + 0) + batch_offset_uu;

  addr_type addr_H = SPAD_BASE;
  addr_type addr_Y = addr_H + size_mu;
  addr_type addr_Reg = addr_Y + size_mu;
  addr_type addr_A = addr_Reg + size_uu;
  addr_type addr_B = addr_A + size_uu;
  addr_type addr_Yk = addr_B + size_uu;
  addr_type addr_BY = addr_Yk + size_uu;
  addr_type addr_R = addr_BY + size_uu;
  addr_type addr_Ynext = addr_R + size_uu;
  addr_type addr_W = addr_Ynext + size_uu;
  addr_type addr_det = addr_W + size_mu;
  addr_type addr_det_inv = addr_det + _config.precision;
  addr_type addr_tmp_mul = addr_det_inv + _config.precision;
  addr_type addr_Xhat = ACCUM_SPAD_BASE;

  int elems_per_access = _config.dram_req_size / _config.precision;
  if (elems_per_access <= 0) elems_per_access = 1;

  std::vector<uint32_t> shape_mu{M, U};
  std::vector<uint32_t> shape_uu{U, U};

  auto emit_movin = [&](addr_type dram_base,
                        addr_type spad_dest,
                        uint32_t rows,
                        uint32_t cols,
                        const std::vector<uint32_t>& shape,
                        uint32_t operand_id) {
    std::set<addr_type> addrs;
    for (uint32_t r = 0; r < rows; ++r) {
      for (uint32_t c = 0; c < cols; c += static_cast<uint32_t>(elems_per_access)) {
        uint32_t col = std::min(c, cols - 1);
        std::vector<uint32_t> index = {r, col};
        addr_type off = make_address(index, shape);
        addrs.insert(dram_base + off);
      }
    }
    if (!addrs.empty()) {
      tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
          .opcode = Opcode::MOVIN,
          .dest_addr = spad_dest,
          .size = static_cast<uint32_t>(addrs.size()),
          .src_addrs = std::vector<addr_type>(addrs.begin(), addrs.end()),
          .operand_id = operand_id,
          .base_addr = dram_base,
          .tile_m = rows,
          .tile_k = cols,
          .tile_n = 0,
          .my_tile = tile}));
    }
  };

  emit_movin(h_base, addr_H, M, U, shape_mu, _INPUT_OPERAND + 0);
  emit_movin(reg_base, addr_Reg, U, U, shape_uu, _INPUT_OPERAND + 1);
  emit_movin(y_base, addr_Y, M, U, shape_mu, _INPUT_OPERAND + 2);

  tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
      .opcode = Opcode::PIPE_BARRIER,
      .id = "BJ_BARRIER_LOAD2GRAM",
      .my_tile = tile,
      .is_barrier = true,
      .barrier_type = 1}));

  tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
      .opcode = Opcode::GEMM_PRELOAD,
      .id = "BJ_GRAM",
      .dest_addr = addr_A,
      .compute_size = U,
      .src_addrs = std::vector<addr_type>{addr_H, addr_H},
      .tile_m = U,
      .tile_k = M,
      .tile_n = U,
      .src_from_accum = false,
      .my_tile = tile}));

  tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
      .opcode = Opcode::ADD,
      .id = "BJ_REG",
      .dest_addr = addr_A,
      .compute_size = U * U,
      .src_addrs = std::vector<addr_type>{addr_A, addr_Reg},
      .tile_m = U,
      .tile_k = U,
      .tile_n = U,
      .src_from_accum = false,
      .my_tile = tile}));

  auto elem_addr = [&](addr_type base, uint32_t /*r*/, uint32_t /*c*/) {
    return base;
  };

  const uint32_t blk = std::max(1u, _block_size);
  const bool use_block2_inverse = (blk == 2) && (U % 2 == 0);
  if (use_block2_inverse) {
    tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
        .opcode = Opcode::ADD,
        .id = "BJ_PRECOND_INIT_FULL",
        .dest_addr = addr_B,
        .compute_size = U * U,
        .src_addrs = std::vector<addr_type>{addr_Reg, addr_Reg},
        .tile_m = U,
        .tile_k = U,
        .tile_n = U,
        .src_from_accum = false,
        .my_tile = tile}));

    const uint32_t n_blocks = U / 2;
    for (uint32_t b = 0; b < n_blocks; ++b) {
      const uint32_t i = 2 * b;
      const uint32_t j = i + 1;

      const addr_type a00 = elem_addr(addr_A, i, i);
      const addr_type a01 = elem_addr(addr_A, i, j);
      const addr_type a10 = elem_addr(addr_A, j, i);
      const addr_type a11 = elem_addr(addr_A, j, j);

      const addr_type b00 = elem_addr(addr_B, i, i);
      const addr_type b01 = elem_addr(addr_B, i, j);
      const addr_type b10 = elem_addr(addr_B, j, i);
      const addr_type b11 = elem_addr(addr_B, j, j);

      tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
          .opcode = Opcode::SCALAR_MUL,
          .id = "BJ_PRECOND_B2_DET_MUL0_" + std::to_string(b),
          .dest_addr = addr_tmp_mul,
          .compute_size = 1,
          .src_addrs = std::vector<addr_type>{a00, a11},
          .tile_m = 1,
          .tile_k = 1,
          .tile_n = 1,
          .src_from_accum = false,
          .my_tile = tile}));

      tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
          .opcode = Opcode::SCALAR_MUL,
          .id = "BJ_PRECOND_B2_DET_MUL1_" + std::to_string(b),
          .dest_addr = addr_det,
          .compute_size = 1,
          .src_addrs = std::vector<addr_type>{a01, a10},
          .tile_m = 1,
          .tile_k = 1,
          .tile_n = 1,
          .src_from_accum = false,
          .my_tile = tile}));

      tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
          .opcode = Opcode::SCALAR_ADD,
          .id = "BJ_PRECOND_B2_DET_SUB_" + std::to_string(b),
          .dest_addr = addr_det,
          .compute_size = 1,
          .src_addrs = std::vector<addr_type>{addr_tmp_mul, addr_det},
          .tile_m = 1,
          .tile_k = 1,
          .tile_n = 1,
          .src_from_accum = false,
          .my_tile = tile}));

      tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
          .opcode = Opcode::SCALAR_DIV,
          .id = "BJ_PRECOND_B2_DET_INV_" + std::to_string(b),
          .dest_addr = addr_det_inv,
          .compute_size = 1,
          .src_addrs = std::vector<addr_type>{addr_Reg, addr_det},
          .tile_m = 1,
          .tile_k = 1,
          .tile_n = 1,
          .src_from_accum = false,
          .my_tile = tile}));

      tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
          .opcode = Opcode::SCALAR_MUL,
          .id = "BJ_PRECOND_B2_B00_" + std::to_string(b),
          .dest_addr = b00,
          .compute_size = 1,
          .src_addrs = std::vector<addr_type>{a11, addr_det_inv},
          .tile_m = 1,
          .tile_k = 1,
          .tile_n = 1,
          .src_from_accum = false,
          .my_tile = tile}));

      tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
          .opcode = Opcode::SCALAR_MUL,
          .id = "BJ_PRECOND_B2_B01_NEG_" + std::to_string(b),
          .dest_addr = b01,
          .compute_size = 1,
          .src_addrs = std::vector<addr_type>{a01, addr_det_inv},
          .tile_m = 1,
          .tile_k = 1,
          .tile_n = 1,
          .src_from_accum = false,
          .my_tile = tile}));

      tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
          .opcode = Opcode::SCALAR_MUL,
          .id = "BJ_PRECOND_B2_B10_NEG_" + std::to_string(b),
          .dest_addr = b10,
          .compute_size = 1,
          .src_addrs = std::vector<addr_type>{a10, addr_det_inv},
          .tile_m = 1,
          .tile_k = 1,
          .tile_n = 1,
          .src_from_accum = false,
          .my_tile = tile}));

      tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
          .opcode = Opcode::SCALAR_MUL,
          .id = "BJ_PRECOND_B2_B11_" + std::to_string(b),
          .dest_addr = b11,
          .compute_size = 1,
          .src_addrs = std::vector<addr_type>{a00, addr_det_inv},
          .tile_m = 1,
          .tile_k = 1,
          .tile_n = 1,
          .src_from_accum = false,
          .my_tile = tile}));
    }
  } else {
    tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
        .opcode = Opcode::ADD,
        .id = "BJ_PRECOND_BLOCK",
        .dest_addr = addr_B,
        .compute_size = U * U,
        .src_addrs = std::vector<addr_type>{addr_A, addr_Reg},
        .tile_m = U,
        .tile_k = U,
        .tile_n = U,
        .src_from_accum = false,
        .my_tile = tile}));
  }

  tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
      .opcode = Opcode::ADD,
      .id = "BJ_INIT_Y0",
      .dest_addr = addr_Yk,
      .compute_size = U * U,
      .src_addrs = std::vector<addr_type>{addr_Reg, addr_Reg},
      .tile_m = U,
      .tile_k = U,
      .tile_n = U,
      .src_from_accum = false,
      .my_tile = tile}));

  if (_adaptive_bounds) {
    tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
      .opcode = Opcode::SCALAR_ADD,
      .id = "BJ_ADAPTIVE_BOUNDS",
      .dest_addr = addr_B,
      .compute_size = 1,
      .src_addrs = std::vector<addr_type>{addr_B, addr_Reg},
      .tile_m = 1,
      .tile_k = 1,
      .tile_n = 1,
      .src_from_accum = false,
      .my_tile = tile}));
  }

  addr_type y_curr = addr_Yk;

  for (uint32_t layer = 0; layer < _layers; ++layer) {
    tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
        .opcode = Opcode::GEMM_PRELOAD,
        .id = "BJ_BY_" + std::to_string(layer),
        .dest_addr = addr_BY,
        .compute_size = U,
        .src_addrs = std::vector<addr_type>{addr_B, y_curr},
        .tile_m = U,
        .tile_k = U,
        .tile_n = U,
        .src_from_accum = false,
        .my_tile = tile}));

    tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
        .opcode = Opcode::ADD,
        .id = "BJ_RESIDUAL_" + std::to_string(layer),
        .dest_addr = addr_R,
        .compute_size = U * U,
        .src_addrs = std::vector<addr_type>{addr_Reg, addr_BY},
        .tile_m = U,
        .tile_k = U,
        .tile_n = U,
        .src_from_accum = false,
        .my_tile = tile}));

    tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
        .opcode = Opcode::ADD,
        .id = "BJ_Y_UPDATE_" + std::to_string(layer),
        .dest_addr = addr_Ynext,
        .compute_size = U * U,
        .src_addrs = std::vector<addr_type>{y_curr, addr_R},
        .tile_m = U,
        .tile_k = U,
        .tile_n = U,
        .src_from_accum = false,
        .my_tile = tile}));

    y_curr = addr_Ynext;

    if (((layer + 1) % _group_sync) == 0 && (layer + 1) < _layers) {
      tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
          .opcode = Opcode::PIPE_BARRIER,
          .id = "BJ_GROUP_SYNC_" + std::to_string(layer),
          .my_tile = tile,
          .is_barrier = true,
          .barrier_type = 5}));
    }
  }

  tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
      .opcode = Opcode::PIPE_BARRIER,
      .id = "BJ_BARRIER_INV2W",
      .my_tile = tile,
      .is_barrier = true,
      .barrier_type = 6}));

  tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
      .opcode = Opcode::GEMM_PRELOAD,
      .id = "BJ_W",
      .dest_addr = addr_W,
      .compute_size = M,
      .src_addrs = std::vector<addr_type>{y_curr, addr_H},
      .tile_m = U,
      .tile_k = U,
      .tile_n = M,
      .src_from_accum = false,
      .my_tile = tile}));

  tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
      .opcode = Opcode::GEMM_PRELOAD,
      .id = "BJ_XHAT",
      .dest_addr = addr_Xhat,
      .compute_size = U,
      .src_addrs = std::vector<addr_type>{addr_W, addr_Y},
      .tile_m = U,
      .tile_k = M,
      .tile_n = U,
      .src_from_accum = false,
      .my_tile = tile}));

  std::set<addr_type> output_addrs;
  for (uint32_t r = 0; r < U; ++r) {
    for (uint32_t c = 0; c < U; c += static_cast<uint32_t>(elems_per_access)) {
      uint32_t col = std::min(c, U - 1);
      std::vector<uint32_t> index = {r, col};
      output_addrs.insert(out_base + make_address(index, shape_uu));
    }
  }

  tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
      .opcode = Opcode::MOVOUT,
      .id = "BJ_STORE_XHAT",
      .dest_addr = addr_Xhat,
      .size = static_cast<uint32_t>(output_addrs.size()),
      .src_addrs = std::vector<addr_type>(output_addrs.begin(), output_addrs.end()),
      .operand_id = _OUTPUT_OPERAND,
      .tile_m = U,
      .tile_k = U,
      .tile_n = 0,
      .src_from_accum = true,
      .last_inst = true,
      .my_tile = tile}));

}
