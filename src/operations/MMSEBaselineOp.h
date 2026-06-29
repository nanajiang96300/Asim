#pragma once

#include "Operation.h"

class MMSEBaselineOp : public Operation {
 public:
  MMSEBaselineOp(SimulationConfig config,
                 Model* model,
                 const std::string& name,
                 std::map<std::string, std::string>& attributes,
                 uint32_t target_core = 0);

  MMSEBaselineOp(SimulationConfig config,
                 MappingTable& mapping_table,
                 const std::vector<uint32_t>& matrix_shape,
                 uint32_t target_core = 0);

  void initialize_tiles(MappingTable& mapping_table) override;

  void set_matrix_shape(const std::vector<uint32_t>& shape) { _matrix_shape = shape; }

 protected:
  void initialize_instructions(Tile* tile, Mapping mapping) override;

 private:
  void parse_attributes();
  void infer_shapes_from_model();

  std::vector<uint32_t> _matrix_shape;
  uint32_t _batch_size{96};
  bool _strict_iso_lowering{true};
};
