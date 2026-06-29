#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>

// ============================================================================
// FormulaStep — describes one mathematical operation in the operator's chain
// ============================================================================
struct FormulaStep {
    /// Unique step identifier within the tile/batch (e.g. "GRAM", "POTRF_0")
    std::string step_id;
    /// Operation type from the UOBS primitive set
    /// (GEMM, DIAG_ADD, CHOLESKY, TRSM, DIAG_INV, MATRIX_INV_2x2,
    ///  MATRIX_SUB, MATRIX_ADD, SCALE, BLOCK_DIAG, PACK_2x2_TO_16x16)
    std::string op_type;
    /// Human-readable names of input tensors (e.g. {"H", "H^H"})
    std::vector<std::string> input_names;
    /// Name of the output tensor (e.g. "G")
    std::string output_name;
    /// Shapes of input tensors: input_shapes[i] = {rows, cols} for input i
    std::vector<std::vector<uint32_t>> input_shapes;
    /// Shape of output tensor: {rows, cols}
    std::vector<uint32_t> output_shape;
    /// Batch index (0-based)
    uint32_t batch;
    /// Instruction ID prefix used to link formula steps back to trace
    /// instructions (e.g. "CHOL_GRAM" links to all instructions whose .id
    /// starts with "CHOL_GRAM")
    std::string relation_id;
};

// ============================================================================
// FormulaLogger — singleton that records the mathematical semantics of every
// operator step.  Operators call emit_step() during initialize_instructions().
// At simulation end, dump_to_json() writes the full formula chain.
//
// Operators SHOULD call set_algorithm() at the start of initialize_instructions()
// to declare their algorithm identity and key parameters.  This enables the UOBS
// scorer to reliably identify the algorithm without fragile pattern matching.
// ============================================================================
struct FormulaMetadata {
    std::string algorithm;     // e.g. "cholesky_block", "block_richardson"
    int block_size = 0;        // block size (0 = not applicable)
    int layers = 0;            // iteration layers (0 = direct method)
    int matrix_dim = 0;        // matrix dimension (nt)
};

class FormulaLogger {
 public:
    static FormulaLogger& instance();

    /// Set algorithm metadata — call ONCE at the beginning of
    /// initialize_instructions().  This is the authoritative declaration
    /// of what algorithm this operator implements.
    void set_algorithm(const std::string& name, int block_size = 0,
                       int layers = 0, int matrix_dim = 0);

    /// Record a formula step.
    void emit_step(const std::string& step_id,
                   const std::string& op_type,
                   const std::vector<std::string>& input_names,
                   const std::string& output_name,
                   const std::vector<std::vector<uint32_t>>& input_shapes,
                   const std::vector<uint32_t>& output_shape,
                   uint32_t batch,
                   const std::string& relation_id);

    /// Serialise all recorded steps to a JSON file (with metadata).
    void dump_to_json(const std::string& filepath);

    /// Clear all recorded steps and metadata.
    void clear();

    /// Number of recorded steps.
    size_t size() const { return _steps.size(); }

 private:
    FormulaLogger() = default;
    FormulaLogger(const FormulaLogger&) = delete;
    FormulaLogger& operator=(const FormulaLogger&) = delete;

    FormulaMetadata _meta;
    bool _has_meta = false;
    std::vector<FormulaStep> _steps;
};
