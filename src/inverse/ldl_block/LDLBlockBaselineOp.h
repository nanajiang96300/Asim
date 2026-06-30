#pragma once
#include "operations/Operation.h"

/// LDL Block matrix inversion — baseline v3.
/// Block LDL: A = L·D·L^H with B×B blocks. No SQRT.
class LDLBlockBaselineOp : public Operation {
 public:
  LDLBlockBaselineOp(SimulationConfig config, Model* model,
                     const std::string& name,
                     std::map<std::string, std::string>& attributes,
                     uint32_t target_core = 0);
  void initialize_tiles(MappingTable& mapping_table) override;
  void set_matrix_shape(const std::vector<uint32_t>& s) { _matrix_shape = s; }
 private:
  void initialize_instructions(Tile* tile, Mapping mapping) override;
  void parse_attributes();
  void infer_shapes_from_model();
  std::vector<uint32_t> _matrix_shape;
  uint32_t _batch_size{96};
  uint32_t _block_size{2};
};
