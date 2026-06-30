#pragma once

#include "operations/Operation.h"

/// LDL NoBlock matrix inversion — unified baseline v2.
///
/// Pure column-by-column LDL: A = L·D·L^H → A^{-1} = L^{-H}·D^{-1}·L^{-1}.
/// No SQRT (key difference from Cholesky). D is real diagonal.
/// Every scalar operation is compute_size=1.
/// FormulaLogger covers ALL phases (GRAM, REG, D_UPDATE, L_UPDATE, BWD).
class LDLNoBlockBaselineOp : public Operation {
 public:
  LDLNoBlockBaselineOp(SimulationConfig config, Model* model,
                       const std::string& name,
                       std::map<std::string, std::string>& attributes,
                       uint32_t target_core = 0);

  void initialize_tiles(MappingTable& mapping_table) override;

  void set_matrix_shape(const std::vector<uint32_t>& shape) { _matrix_shape = shape; }

 private:
  void initialize_instructions(Tile* tile, Mapping mapping) override;
  void parse_attributes();
  void infer_shapes_from_model();

  std::vector<uint32_t> _matrix_shape;
  uint32_t _batch_size{96};
};
