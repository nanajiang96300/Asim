#pragma once
#include "Model.h"
class ScalarCostProbeModel : public Model {
 public:
  ScalarCostProbeModel(json mc,SimulationConfig c,const std::string& n):Model(mc,c,n){}
  void initialize_model(std::vector<std::unique_ptr<Tensor>>&) override;
  void initialize_weight(std::vector<std::unique_ptr<Tensor>>&) override {}
};
