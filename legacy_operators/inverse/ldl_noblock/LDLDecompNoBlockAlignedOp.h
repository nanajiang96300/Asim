#pragma once
#include "operations/Operation.h"

// LDL NoBlock with RIGHT-LOOKING mode — fully aligned with Cholesky NoBlock.
// Every SCALAR_MUL is emitted per-element (no compute_size merging).
// Explicit RK_UPDATE loop (right-looking).
// This is the FAIR baseline for Cholesky vs LDL comparison.
class LDLDecompNoBlockAlignedOp : public Operation {
 public:
  LDLDecompNoBlockAlignedOp(SimulationConfig config, Model* model,
                            const std::string& name,
                            std::map<std::string, std::string>& attributes,
                            uint32_t target_core = 0);
  LDLDecompNoBlockAlignedOp(SimulationConfig config, MappingTable& mapping_table,
                            const std::vector<uint32_t>& matrix_shape,
                            uint32_t target_core = 0);

  void initialize_tiles(MappingTable& mapping_table) override;
  void set_matrix_shape(const std::vector<uint32_t>& shape) { _matrix_shape = shape; }
  void set_batch_size(uint32_t batch) { _batch_size = batch; }

 protected:
  void initialize_instructions(Tile* tile, Mapping mapping) override;

 private:
  void parse_attributes();
  void infer_shapes_from_model();
  std::vector<uint32_t> _matrix_shape;
  uint32_t _batch_size{96};
  bool _use_left_looking{false};
};
