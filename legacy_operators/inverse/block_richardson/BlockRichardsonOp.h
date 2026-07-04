#pragma once

#include "operations/Operation.h"

class BlockRichardsonOp : public Operation {
 public:
  BlockRichardsonOp(SimulationConfig config,
                Model* model,
                const std::string& name,
                std::map<std::string, std::string>& attributes,
                uint32_t target_core = 0);

  BlockRichardsonOp(SimulationConfig config,
                MappingTable& mapping_table,
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
  uint32_t _layers{16};
  uint32_t _block_size{2};
  uint32_t _group_sync{2};
  bool _adaptive_bounds{false};
  bool _iter_weight{true};
  bool _omega_relaxed{true};
  bool _fused_by_gemm{false};
  uint32_t _by_preload_period{4};
  bool _fuse_residual_update{false};
  std::string _precond_solver{"cholesky"};
  uint32_t _by_kernel_fuse_factor{1};
};
