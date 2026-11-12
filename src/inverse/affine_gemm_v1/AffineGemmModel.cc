#include "AffineGemmModel.h"

#include "operations/Operation.h"
#include "semantic/SourceViews.h"
#include "FormulaLogger.h"

#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

class AffineGemmOperation : public Operation {
 public:
  AffineGemmOperation(SimulationConfig c,
                      Model* m,
                      uint32_t mm,
                      uint32_t kk,
                      uint32_t nn,
                      uint32_t batches,
                      json task,
                      json policy,
                      std::map<std::string, std::string>& attrs)
      : Operation(c, m, "AffineGemmV1", attrs, 0),
        M_(mm),
        K_(kk),
        N_(nn),
        B_(batches),
        task_(task),
        policy_(policy) {
    _optype = "AffineGemmV1";
  }

  void initialize_tiles(MappingTable&) override;

 private:
  uint32_t M_;
  uint32_t K_;
  uint32_t N_;
  uint32_t B_;
  json task_;
  json policy_;

  void initialize_instructions(Tile* tile, Mapping) override;
};

void AffineGemmOperation::initialize_tiles(MappingTable&) {
  if (_config.precision != 2 ||
      _config.dram_req_size != 64 ||
      _config.num_cores != 24 ||
      B_ == 0 || B_ > 384 ||
      (B_ % 24) != 0 ||
      M_ == 0 || K_ == 0 || N_ == 0 ||
      (M_ % 16) != 0 || (K_ % 16) != 0 || (N_ % 16) != 0 ||
      M_ > 64 || K_ > 64 || N_ > 64) {
    throw std::runtime_error(
        "AffineGemmV1 requires precision=2 (FP16 storage), "
        "dram_req_size=64, num_cores=24, batch multiple of 24, "
        "and m/k/n in {16,32,48,64}");
  }

  for (uint32_t b = 0; b < B_; ++b) {
    auto tile = std::make_unique<Tile>(
        Tile{.status = Tile::Status::INITIALIZED,
             .optype = _optype,
             .layer_id = _id,
             .batch = b,
             .core_id = static_cast<int>(b % 24)});
    initialize_instructions(tile.get(), Mapping{});
    _tiles.push_back(std::move(tile));
  }
}

void AffineGemmOperation::initialize_instructions(Tile* tile, Mapping) {
  const uint32_t P = _config.precision;
  addr_type next = SPAD_BASE;
  json regions = json::array();

  auto alloc = [&](const std::string& name, uint32_t rows, uint32_t cols) {
    addr_type base = next;
    next += ceil_div(uint64_t(rows) * cols * P, 64) * 64;
    regions.push_back(region(base, rows, cols, name));
    return base;
  };

  const addr_type a_addr = alloc("A_batch", M_, K_);
  const addr_type b_addr = alloc("B_batch", K_, N_);
  const addr_type c_addr = alloc("C_batch", M_, N_);
  const addr_type gemm_addr = alloc("GemmTmpFp16", M_, N_);
  const addr_type z_addr = alloc("Z_batch", M_, N_);
  addr_type sync = next;
  next += 32768;
  regions.push_back(region(sync, 1, 32768 / P, "sync"));

  record_source_regions(regions, M_, K_);

  _numeric_header["schema"] = "source-views-program-v1";
  _numeric_header["profile"] = "modeled_fp16_fp32_v1";
  if (!policy_.is_null()) {
    _numeric_header["precision_policy"] = policy_;
  }
  _numeric_header["task_contract"] = task_;
  _numeric_header["algorithm"] = {
      {"id", "affine_gemm_v1"},
      {"formula", "Z=half(half(A@B)+C)"},
      {"gemm_order", "k_ascending_multiply_then_add"}};

  FormulaLogger::instance().set_algorithm("affine_gemm_v1", 0, K_, N_);

  uint32_t serial = 0;
  const std::string prefix =
      "batch" + std::to_string(tile->batch) + ".";

  auto barrier = [&](addr_type written) {
    Instruction raw{
        .opcode = Opcode::PIPE_BARRIER,
        .id = prefix + "sync" + std::to_string(serial++),
        .dest_addr = sync,
        .size = 1,
        .src_addrs = {written},
        .src_from_accum = true,
        .my_tile = tile,
        .is_barrier = true};
    tile->instructions.push_back(
        source_instruction(std::move(raw), json::array(), nullptr,
                           "affine_gemm_v1:sync"));
    sync += 64;
  };

  auto load = [&](uint32_t slot, addr_type local, uint32_t rows,
                  uint32_t cols) {
    const addr_type dram =
        get_operand_addr(_INPUT_OPERAND + slot) +
        uint64_t(tile->batch) * rows * cols * P;
    const uint32_t lines = static_cast<uint32_t>(
        ceil_div(uint64_t(rows) * cols * P, 64));
    Instruction raw{
        .opcode = Opcode::MOVIN,
        .id = prefix + "load" + std::to_string(slot),
        .dest_addr = local,
        .size = lines,
        .operand_id = _INPUT_OPERAND + slot,
        .base_addr = 0,
        .tile_m = rows,
        .tile_k = cols,
        .my_tile = tile};
    for (uint32_t j = 0; j < raw.size; ++j) {
      raw.src_addrs.push_back(dram + uint64_t(j) * 64);
    }
    const std::string site =
        "affine_gemm_v1:load" + std::to_string(slot);
    tile->instructions.push_back(
        source_instruction(std::move(raw),
                           json::array({view(dram, 0, 0, rows, cols)}),
                           view(local, 0, 0, rows, cols), site));
    barrier(local);
  };

  load(0, a_addr, M_, K_);
  load(1, b_addr, K_, N_);
  load(2, c_addr, M_, N_);

  const uint32_t m_n_lines = static_cast<uint32_t>(
      ceil_div(uint64_t(M_) * N_ * P, 64));

  Instruction gemm{
      .opcode = Opcode::GEMM_PRELOAD,
      .id = prefix + "gemm",
      .dest_addr = gemm_addr,
      .size = m_n_lines,
      .compute_size = N_,
      .tile_m = M_,
      .tile_k = K_,
      .tile_n = N_,
      .src_from_accum = true,
      .my_tile = tile};
  gemm.src_addrs = {a_addr, b_addr};
  const json gemm_src = json::array(
      {view(a_addr, 0, 0, M_, K_), view(b_addr, 0, 0, K_, N_)});
  const json gemm_dst = view(gemm_addr, 0, 0, M_, N_);
  tile->instructions.push_back(
      source_instruction(std::move(gemm), gemm_src, gemm_dst,
                         "affine_gemm_v1:gemm"));
  {
    const std::vector<std::string> names{
        std::to_string(a_addr), std::to_string(b_addr)};
    const std::vector<std::vector<uint32_t>> shapes{{M_, K_}, {K_, N_}};
    const std::vector<uint32_t> out_shape{M_, N_};
    FormulaLogger::instance().emit_step(prefix + "gemm", "GEMM", names,
                                        std::to_string(gemm_addr), shapes,
                                        out_shape, tile->batch,
                                        prefix + "gemm");
  }
  barrier(gemm_addr);

  Instruction add_raw{
      .opcode = Opcode::ADD,
      .id = prefix + "add",
      .dest_addr = z_addr,
      .size = m_n_lines,
      .compute_size = static_cast<uint32_t>(M_ * N_ * 4),
      .tile_m = M_,
      .tile_k = N_,
      .tile_n = N_,
      .src_from_accum = true,
      .my_tile = tile};
  add_raw.src_addrs = {gemm_addr, b_addr};
  const json add_src = json::array(
      {view(gemm_addr, 0, 0, M_, N_), view(b_addr, 0, 0, M_, N_)});
  const json add_dst = view(z_addr, 0, 0, M_, N_);
  tile->instructions.push_back(
      source_instruction(std::move(add_raw), add_src, add_dst,
                         "affine_gemm_v1:add"));
  {
    const std::vector<std::string> names{
        std::to_string(gemm_addr), std::to_string(c_addr)};
    const std::vector<std::vector<uint32_t>> shapes{{M_, N_}, {M_, N_}};
    const std::vector<uint32_t> out_shape{M_, N_};
    FormulaLogger::instance().emit_step(prefix + "add", "ELEMENTWISE", names,
                                        std::to_string(z_addr), shapes,
                                        out_shape, tile->batch,
                                        prefix + "add");
  }
  barrier(z_addr);

  const addr_type dram_out =
      get_operand_addr(_OUTPUT_OPERAND) +
      uint64_t(tile->batch) * M_ * N_ * P;
  Instruction store{
      .opcode = Opcode::MOVOUT,
      .id = prefix + "store",
      .dest_addr = z_addr,
      .size = m_n_lines,
      .operand_id = _OUTPUT_OPERAND,
      .base_addr = 0,
      .tile_m = M_,
      .tile_k = N_,
      .src_from_accum = true,
      .last_inst = true,
      .my_tile = tile};
  for (uint32_t j = 0; j < store.size; ++j) {
    store.src_addrs.push_back(dram_out + uint64_t(j) * 64);
  }
  tile->instructions.push_back(
      source_instruction(std::move(store),
                         json::array({view(z_addr, 0, 0, M_, N_)}),
                         view(dram_out, 0, 0, M_, N_),
                         "affine_gemm_v1:store"));
}

}  // namespace

void AffineGemmModel::initialize_model(
    std::vector<std::unique_ptr<Tensor>>&) {
  const uint32_t M = _model_config.value("matrix_m", 32u);
  const uint32_t K = _model_config.value("matrix_k", 32u);
  const uint32_t N = _model_config.value("matrix_n", 32u);
  const uint32_t B = _model_config.value("batch_size", 96u);

  std::map<std::string, std::string> attrs;
  auto op = std::make_unique<AffineGemmOperation>(
      _config, this, M, K, N, B,
      _model_config.at("task_contract"),
      _model_config.value("precision_policy", json(nullptr)), attrs);

  std::vector<std::pair<std::string, std::vector<uint32_t>>> entries = {
      {"A", {B, M, K}},
      {"B", {B, K, N}},
      {"C", {B, M, N}},
      {"Z", {B, M, N}}};
  for (auto& entry : entries) {
    const bool input = entry.first != "Z";
    auto tensor = std::make_unique<Tensor>(
        get_root_node_id(), name_gen(get_name(), entry.first), entry.second,
        _config.precision, input);
    const uint32_t id = tensor->get_id();
    if (input) {
      tensor->set_produced();
    }
    add_tensor(std::move(tensor));
    if (input) {
      op->add_input(id);
    } else {
      op->add_output(id);
    }
  }

  op->initialize_tiles(_mapping_table);
  _executable_layer.push_back(op.get());
  _operation_map[op->get_id()] = std::move(op);
}
