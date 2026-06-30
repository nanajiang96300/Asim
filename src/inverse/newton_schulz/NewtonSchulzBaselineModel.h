#pragma once
#include "Model.h"
class NewtonSchulzBaselineModel : public Model {
 public:
  NewtonSchulzBaselineModel(json mc, SimulationConfig c, const std::string& n)
      : Model(mc, c, n) {}
  void initialize_model(std::vector<std::unique_ptr<Tensor>>& wt) override;
  void initialize_weight(std::vector<std::unique_ptr<Tensor>>&) override {}
};
