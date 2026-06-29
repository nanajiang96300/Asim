#include "BlockRichardsonModel.h"

#include <chrono>

void BlockRichardsonModel::initialize_weight(
    std::vector<std::unique_ptr<Tensor>>& /*weight_table*/) {
}

void BlockRichardsonModel::initialize_model(
    std::vector<std::unique_ptr<Tensor>>& weight_table) {
  (void)weight_table;
  auto start = std::chrono::high_resolution_clock::now();

  uint32_t M = 64;
  uint32_t K = 16;
  uint32_t batch_size = 96;

  if (_model_config.contains("matrix_m")) {
    M = static_cast<uint32_t>(_model_config["matrix_m"]);
  }
  if (_model_config.contains("matrix_k")) {
    K = static_cast<uint32_t>(_model_config["matrix_k"]);
  }
  if (_model_config.contains("batch_size")) {
    batch_size = static_cast<uint32_t>(_model_config["batch_size"]);
  }

  const std::vector<uint32_t> shape_MK{batch_size, M, K};
  const std::vector<uint32_t> shape_KK{batch_size, K, K};
  const std::vector<uint32_t> shape_reg{K, K};
  const std::vector<uint32_t> matrix_shape{M, K};

  uint32_t root_id = get_root_node_id();

  auto make_tensor = [&](const std::string& base,
                         const std::vector<uint32_t>& shape,
                         bool produced) -> uint32_t {
    auto t = std::make_unique<Tensor>(
        root_id,
        name_gen(get_name(), base),
        const_cast<std::vector<uint32_t>&>(shape),
        _config.precision,
        produced);
    if (produced) {
      t->set_produced();
    }
    uint32_t id = t->get_id();
    add_tensor(std::move(t));
    return id;
  };

  std::map<std::string, std::string> attrs;
  attrs["batch_size"] = std::to_string(batch_size);
  attrs["layers"] = "8";
  attrs["block_size"] = "2";
  attrs["group_sync"] = "2";
  attrs["adaptive_bounds"] = "1";
  attrs["iter_weight"] = "1";
  attrs["fused_by_gemm"] = "0";
  attrs["by_preload_period"] = "4";
  attrs["fuse_residual_update"] = "0";
  attrs["by_kernel_fuse_factor"] = "1";

  if (_model_config.contains("attributes")) {
    if (_model_config["attributes"].contains("layers")) {
      attrs["layers"] = _model_config["attributes"]["layers"].get<std::string>();
    }
    if (_model_config["attributes"].contains("block_size")) {
      attrs["block_size"] = _model_config["attributes"]["block_size"].get<std::string>();
    }
    if (_model_config["attributes"].contains("group_sync")) {
      attrs["group_sync"] = _model_config["attributes"]["group_sync"].get<std::string>();
    }
    if (_model_config["attributes"].contains("adaptive_bounds")) {
      attrs["adaptive_bounds"] = _model_config["attributes"]["adaptive_bounds"].get<std::string>();
    }
    if (_model_config["attributes"].contains("iter_weight")) {
      attrs["iter_weight"] = _model_config["attributes"]["iter_weight"].get<std::string>();
    }
    if (_model_config["attributes"].contains("fused_by_gemm")) {
      attrs["fused_by_gemm"] = _model_config["attributes"]["fused_by_gemm"].get<std::string>();
    }
    if (_model_config["attributes"].contains("by_preload_period")) {
      attrs["by_preload_period"] = _model_config["attributes"]["by_preload_period"].get<std::string>();
    }
    if (_model_config["attributes"].contains("fuse_residual_update")) {
      attrs["fuse_residual_update"] = _model_config["attributes"]["fuse_residual_update"].get<std::string>();
    }
    if (_model_config["attributes"].contains("by_kernel_fuse_factor")) {
      attrs["by_kernel_fuse_factor"] = _model_config["attributes"]["by_kernel_fuse_factor"].get<std::string>();
    }
    if (_model_config["attributes"].contains("precond_solver")) {
      attrs["precond_solver"] = _model_config["attributes"]["precond_solver"].get<std::string>();
    }
    if (_model_config["attributes"].contains("omega_relaxed")) {
      attrs["omega_relaxed"] = _model_config["attributes"]["omega_relaxed"].get<std::string>();
    }
  }

  uint32_t h_id = make_tensor("H", shape_MK, true);
  uint32_t reg_id = make_tensor("RegI", shape_reg, true);
  uint32_t y_id = make_tensor("Y", shape_MK, true);
  uint32_t out_id = make_tensor("X_hat", shape_KK, false);

  std::string op_name = name_gen(get_name(), "BlockRichardsonOp");
  auto op = std::make_unique<BlockRichardsonOp>(_config, this, op_name, attrs,
                                            /*target_core=*/0);
  op->set_matrix_shape(matrix_shape);

  op->add_input(h_id);
  op->add_input(reg_id);
  op->add_input(y_id);
  op->add_output(out_id);

  op->initialize_tiles(_mapping_table);

  uint32_t op_id = op->get_id();
  _operation_map[op_id] = std::move(op);

  _executable_layer.clear();
  for (auto& [key, val] : _operation_map) {
    if (val->check_executable()) {
      _executable_layer.push_back(val.get());
    }
  }

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> duration = end - start;
  spdlog::info("BlockRichardsonModel initialization time: {:2f} seconds", duration.count());
}
