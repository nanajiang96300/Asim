#pragma once

#include <cstdint>
#include <mutex>
#include <string>
#include <vector>
#include <limits>

// Lightweight global trace logger for pipeline visualization.
// Usage:
//   TraceLogger::log_event("CubeCore", "MatMul_Tile0", start_cycle, end_cycle);
//   TraceLogger::instance().dump_to_csv("trace.csv");
class TraceLogger {
 public:
  struct TraceEvent {
    std::string name;       // e.g., "MatMul_Tile0", "Load_GM_to_UB"
    std::string unit;       // e.g., "CubeCore", "VectorCore", "MTE2_Load", "MTE3_Store", "MTE1_Mov"
    std::uint64_t start_cycle;
    std::uint64_t end_cycle;
    std::uint64_t request_id=std::numeric_limits<std::uint64_t>::max();
    std::string op_id;
    std::uint32_t fragment=0;
  };

  // Get singleton instance.
  static TraceLogger& instance();

  // Thread-safe, non-throwing logging API.
  static void log_event(const std::string& unit,
                        const std::string& name,
                        std::uint64_t start_cycle,
                        std::uint64_t end_cycle,
                        std::uint64_t request_id=std::numeric_limits<std::uint64_t>::max(),
                        const std::string& op_id="",std::uint32_t fragment=0);

  // Dump all events to a CSV file with header:
  //   name,unit,start_cycle,end_cycle
  void dump_to_csv(const std::string& filename);

  // Access to the in-memory event list (for tests or tooling).
  const std::vector<TraceEvent>& events() const { return _events; }

 private:
  TraceLogger() = default;
  TraceLogger(const TraceLogger&) = delete;
  TraceLogger& operator=(const TraceLogger&) = delete;

  void add_event(const std::string& unit,
                 const std::string& name,
                 std::uint64_t start_cycle,
                 std::uint64_t end_cycle,std::uint64_t request_id,
                 const std::string& op_id,std::uint32_t fragment);

  std::mutex _mutex;
  std::vector<TraceEvent> _events;
};
