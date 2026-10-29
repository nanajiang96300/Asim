#include "InstructionExport.h"
#include <fstream>
#include <cstdlib>
#include <stdexcept>
void export_instructions(Operation& op) {
  const char* path = std::getenv("ASIM_INSTRUCTION_EXPORT");
  if (!path) return;
  json record{{"format", "actual-instructions-v1"}, {"operation", op.get_name()},
              {"header", op.semantic_header()}, {"instructions", json::array()}};
  for (auto& tile : op.get_tiles()) {
    for (auto& ptr : tile->instructions) {
      const auto& i = *ptr;
      record["instructions"].push_back({{"id", i.id}, {"opcode", static_cast<int>(i.opcode)},
          {"dest_addr", i.dest_addr}, {"src_addrs", i.src_addrs}, {"size", i.size},
          {"compute_size", i.compute_size}, {"base_addr", i.base_addr},
          {"m", i.tile_m}, {"k", i.tile_k}, {"n", i.tile_n}, {"src_from_accum", i.src_from_accum},
          {"barrier", i.is_barrier}, {"semantic", i.numeric_semantics}, {"batch", tile->batch}});
    }
  }
  std::ofstream stream(path, std::ios::app);
  if (!stream) throw std::runtime_error("instruction export unavailable");
  stream << record.dump() << '\n';
  if (!stream) throw std::runtime_error("instruction export failed");
}
