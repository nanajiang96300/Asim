#pragma once

#include <memory>
#include <string>
#include <vector>

#include "Model.h"

class AffineGemmModel : public Model {
 public:
  AffineGemmModel(json mc, SimulationConfig c, const std::string& n)
      : Model(mc, c, n) {}

  void initialize_model(std::vector<std::unique_ptr<Tensor>>&) override;
  void initialize_weight(std::vector<std::unique_ptr<Tensor>>&) override {}
};
