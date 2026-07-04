#include "NewtonSchulzModel.h"
#include "NewtonSchulzBaselineOp.h"

void NewtonSchulzModel::initialize_model(std::vector<std::unique_ptr<Tensor>>& wt) {
  (void)wt;
  uint32_t M = 64, U = 16, bs = 96;
  if (_model_config.contains("matrix_m")) M = _model_config["matrix_m"];
  if (_model_config.contains("matrix_k")) U = _model_config["matrix_k"];
  if (_model_config.contains("batch_size")) bs = _model_config["batch_size"];

  const std::vector<uint32_t> sMU{bs, M, U}, sUU{U, U}, ms{M, U};
  uint32_t root_id = get_root_node_id();
  auto make_t = [&](const std::string& base, const std::vector<uint32_t>& shape, bool prod) -> uint32_t {
    auto t = std::make_unique<Tensor>(root_id, name_gen(get_name(), base),
        const_cast<std::vector<uint32_t>&>(shape), _config.precision, prod);
    if (prod) t->set_produced();
    uint32_t id = t->get_id(); add_tensor(std::move(t)); return id;
  };

  uint32_t h_id = make_t("H", sMU, true);
  uint32_t r_id = make_t("RegI", sUU, true);
  uint32_t x_id = make_t("X_init", sUU, true);
  uint32_t c_id = make_t("C", sUU, true);
  uint32_t o_id = make_t("Ainv", sUU, false);

  std::map<std::string, std::string> attrs{{"batch_size", std::to_string(bs)}};
  auto op = std::make_unique<NewtonSchulzBaselineOp>(_config, this,
      name_gen(get_name(), "NewtonSchulzBaselineOp"), attrs, 0);
  op->set_matrix_shape(ms);
  op->add_input(h_id); op->add_input(r_id);
  op->add_input(x_id); op->add_input(c_id);
  op->add_output(o_id);
  op->initialize_tiles(_mapping_table);
  _operation_map[op->get_id()] = std::move(op);
  _executable_layer.clear();
  for (auto& [k,v] : _operation_map)
    if (v->check_executable()) _executable_layer.push_back(v.get());
}
void NewtonSchulzModel::initialize_weight(std::vector<std::unique_ptr<Tensor>>&) {}
