#pragma once
#include "../Model.h"
class NNGraphModel: public Model {
 public:
 NNGraphModel(json mc,SimulationConfig c,const std::string& name):Model(mc,c,name){}
 void initialize_model(std::vector<std::unique_ptr<Tensor>>&) override;
 void initialize_weight(std::vector<std::unique_ptr<Tensor>>&) override {}
};
