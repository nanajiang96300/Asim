#pragma once
#include "operations/Operation.h"

/// Newton-Schulz iterative matrix inversion — baseline v3.
/// X_{k+1} = X_k @ (2I - A @ X_k), quadratic convergence.
/// Pure GEMM + Vector ADD, no SCALAR operations needed.
class NewtonSchulzBaselineOp : public Operation {
 public:
  NewtonSchulzBaselineOp(SimulationConfig config, Model* model,
                         const std::string& name,
                         std::map<std::string, std::string>& attributes,
                         uint32_t target_core = 0);
  void initialize_tiles(MappingTable& mapping_table) override;
  void set_matrix_shape(const std::vector<uint32_t>& s) { _matrix_shape = s; }
 private:
  void initialize_instructions(Tile* tile, Mapping mapping) override;
  void parse_attributes();
  std::vector<uint32_t> _matrix_shape;
  uint32_t _batch_size{96};
  uint32_t _iterations{16};  // K=16: ΔSER=0.000 on Rayleigh, 0.021 on CDL-B (verified 2026-07-08)
};
