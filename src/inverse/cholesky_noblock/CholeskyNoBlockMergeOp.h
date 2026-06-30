#pragma once
#include "operations/Operation.h"

/// Cholesky NoBlock — SCALAR merge optimization (Opt1).
/// Merges inner-loop SCALAR_MUL ops: j×MUL(1) → 1×MUL(compute_size=j).
/// Reduces Scalar Pipeline serialization bottleneck.
class CholeskyNoBlockMergeOp : public Operation {
 public:
  CholeskyNoBlockMergeOp(SimulationConfig c, Model* m, const std::string& n,
                         std::map<std::string, std::string>& a, uint32_t tc = 0);
  void initialize_tiles(MappingTable& mt) override;
  void set_matrix_shape(const std::vector<uint32_t>& s) { _matrix_shape = s; }
 private:
  void initialize_instructions(Tile* t, Mapping m) override;
  void parse_attributes();
  void infer_shapes_from_model();
  std::vector<uint32_t> _matrix_shape;
  uint32_t _batch_size{96};
};
