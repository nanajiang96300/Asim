#pragma once
#include "../../semantic/SourceViews.h"
namespace nn_units {
// Frozen software-model contracts; raw cost fields and executable views agree.
std::unique_ptr<Instruction> softmax(Instruction raw, json source, json destination);
std::unique_ptr<Instruction> gelu(Instruction raw, json source, json destination);
}
