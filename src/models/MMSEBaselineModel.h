#pragma once

#include "../Model.h"
#include "../operations/MMSEBaselineOp.h"

class MMSEBaselineModel : public Model {
 public:
  MMSEBaselineModel(json model_config, SimulationConfig config, std::string name)
      : Model(model_config, config, name) {}

  void initialize_model(
      std::vector<std::unique_ptr<Tensor>>& weight_table) override;
  void initialize_weight(
      std::vector<std::unique_ptr<Tensor>>& weight_table) override;
};
