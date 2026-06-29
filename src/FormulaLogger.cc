#include "FormulaLogger.h"

#include <fstream>
#include <spdlog/spdlog.h>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

FormulaLogger& FormulaLogger::instance() {
    static FormulaLogger logger;
    return logger;
}

void FormulaLogger::set_algorithm(const std::string& name, int block_size,
                                   int layers, int matrix_dim) {
    _meta.algorithm   = name;
    _meta.block_size  = block_size;
    _meta.layers      = layers;
    _meta.matrix_dim  = matrix_dim;
    _has_meta         = true;
}

void FormulaLogger::emit_step(const std::string& step_id,
                              const std::string& op_type,
                              const std::vector<std::string>& input_names,
                              const std::string& output_name,
                              const std::vector<std::vector<uint32_t>>& input_shapes,
                              const std::vector<uint32_t>& output_shape,
                              uint32_t batch,
                              const std::string& relation_id) {
    FormulaStep step;
    step.step_id     = step_id;
    step.op_type     = op_type;
    step.input_names = input_names;
    step.output_name = output_name;
    step.input_shapes = input_shapes;
    step.output_shape = output_shape;
    step.batch       = batch;
    step.relation_id = relation_id;
    _steps.push_back(std::move(step));
}

void FormulaLogger::dump_to_json(const std::string& filepath) {
    json out;
    if (_has_meta) {
        out["_metadata"] = {
            {"algorithm",  _meta.algorithm},
            {"block_size", _meta.block_size},
            {"layers",     _meta.layers},
            {"matrix_dim", _meta.matrix_dim}
        };
    }

    json steps_arr = json::array();
    for (const auto& s : _steps) {
        json shapes_in = json::array();
        for (const auto& sh : s.input_shapes) {
            shapes_in.push_back(sh);
        }
        steps_arr.push_back({
            {"step_id",     s.step_id},
            {"op_type",     s.op_type},
            {"input_names", s.input_names},
            {"output_name", s.output_name},
            {"input_shapes", shapes_in},
            {"output_shape", s.output_shape},
            {"batch",       s.batch},
            {"relation_id", s.relation_id}
        });
    }
    out["steps"] = steps_arr;

    std::ofstream ofs(filepath);
    if (!ofs.is_open()) {
        spdlog::error("FormulaLogger: cannot open {} for writing.", filepath);
        return;
    }
    ofs << out.dump(2) << std::endl;
    ofs.close();
    spdlog::info("FormulaLogger: dumped {} formula steps to {}",
                 _steps.size(), filepath);
}

void FormulaLogger::clear() {
    _steps.clear();
    _meta = FormulaMetadata{};
    _has_meta = false;
}
