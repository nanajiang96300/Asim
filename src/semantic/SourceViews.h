#pragma once
#include "../Common.h"
inline json view(addr_type base,int row,int col,int rows=1,int cols=1,bool transpose=false,bool conjugate=false) {
  return {{"base",base},{"row",row},{"col",col},{"rows",rows},{"cols",cols},{"transpose",transpose},{"conjugate",conjugate}};
}
inline json region(addr_type base,int rows,int cols,const std::string& name) {
  return {{"base",base},{"rows",rows},{"cols",cols},{"name",name}};
}
inline std::unique_ptr<Instruction> source_instruction(Instruction i,json src,json dst,const std::string& site) {
  i.numeric_semantics={{"schema","source-views-v1"},{"src",src},{"dst",dst},{"site",site},
                       {"opcode",static_cast<int>(i.opcode)}};
  return std::make_unique<Instruction>(std::move(i));
}
inline std::unique_ptr<Instruction> ambiguous_instruction(Instruction i,const std::string& reason,const std::string& site) {
  i.numeric_semantics={{"schema","source-views-v1"},{"opcode",static_cast<int>(i.opcode)},
                       {"invalid_reason",reason},{"site",site}};
  return std::make_unique<Instruction>(std::move(i));
}
