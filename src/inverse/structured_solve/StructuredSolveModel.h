#pragma once
#include "Model.h"
class StructuredSolveModel : public Model {
  std::string variant_;
 public:
  StructuredSolveModel(json mc,SimulationConfig c,const std::string& n,std::string v):Model(mc,c,n),variant_(v){}
  void initialize_model(std::vector<std::unique_ptr<Tensor>>&) override;
  void initialize_weight(std::vector<std::unique_ptr<Tensor>>&) override {}
};
