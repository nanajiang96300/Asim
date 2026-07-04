#pragma once
#include "Model.h"
class LDLDecompNoBlockAlignedModel : public Model {
 public:
  LDLDecompNoBlockAlignedModel(const nlohmann::json& model_config,
                                SimulationConfig config,
                                const std::string& model_name)
    : Model(model_config, config, model_name) {}
  void initialize_weight(std::vector<std::unique_ptr<Tensor>>&) override;
  void initialize_model(std::vector<std::unique_ptr<Tensor>>& weight_table) override;
};
