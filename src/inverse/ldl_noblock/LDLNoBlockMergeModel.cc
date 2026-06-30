#include "LDLNoBlockMergeModel.h"
#include "LDLNoBlockMergeOp.h"

void LDLNoBlockMergeModel::initialize_model(
    std::vector<std::unique_ptr<Tensor>>& weight_table) {
  (void)weight_table;
  uint32_t M = 64, U = 16, batch_size = 96;
  if (_model_config.contains("matrix_m"))
    M = static_cast<uint32_t>(_model_config["matrix_m"]);
  if (_model_config.contains("matrix_k"))
    U = static_cast<uint32_t>(_model_config["matrix_k"]);
  if (_model_config.contains("batch_size"))
    batch_size = static_cast<uint32_t>(_model_config["batch_size"]);

  const std::vector<uint32_t> shape_MU{batch_size, M, U};
  const std::vector<uint32_t> shape_UU{U, U};
  const std::vector<uint32_t> matrix_shape{M, U};

  uint32_t root_id = get_root_node_id();
  auto make_tensor = [&](const std::string& base, const std::vector<uint32_t>& shape,
                         bool produced) -> uint32_t {
    auto t = std::make_unique<Tensor>(root_id, name_gen(get_name(), base),
        const_cast<std::vector<uint32_t>&>(shape), _config.precision, produced);
    if (produced) t->set_produced();
    uint32_t id = t->get_id();
    add_tensor(std::move(t));
    return id;
  };

  uint32_t h_id = make_tensor("H", shape_MU, true);
  uint32_t reg_id = make_tensor("RegI", shape_UU, true);
  uint32_t out_id = make_tensor("Ainv", shape_MU, false);

  std::map<std::string, std::string> attrs{{"batch_size", std::to_string(batch_size)}};
  auto op = std::make_unique<LDLNoBlockMergeOp>(_config, this,
      name_gen(get_name(), "LDLNoBlockMergeOp"), attrs, 0);
  op->set_matrix_shape(matrix_shape);
  op->add_input(h_id);
  op->add_input(reg_id);
  op->add_output(out_id);
  op->initialize_tiles(_mapping_table);

  _operation_map[op->get_id()] = std::move(op);
  _executable_layer.clear();
  for (auto& [key, val] : _operation_map)
    if (val->check_executable()) _executable_layer.push_back(val.get());
}
