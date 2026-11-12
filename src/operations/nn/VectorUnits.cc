#include "VectorUnits.h"
#include <stdexcept>
namespace nn_units {
static std::unique_ptr<Instruction> unary(Instruction raw,json src,json dst,bool soft) {
  const uint32_t m=dst.at("rows"), n=dst.at("cols");
  if(!m || !n || src.at("rows")!=m || src.at("cols")!=n || src.at("transpose").get<bool>() || src.at("conjugate").get<bool>())
    throw std::runtime_error("nn-vector-v1 requires matching full unary views");
  raw.opcode=soft?Opcode::SOFTMAX:Opcode::GELU;
  raw.tile_m=m;raw.tile_k=n;raw.tile_n=n;
  raw.size=ceil_div(uint64_t(m)*n*2,64);
  raw.compute_size=soft?n*4:m*n*4;
  raw.src_addrs={src.at("base").get<addr_type>()};raw.dest_addr=dst.at("base").get<addr_type>();
  auto result=source_instruction(std::move(raw),json::array({src}),dst,soft?"nn.softmax":"nn.gelu");
  result->numeric_semantics["kernel_contract"]="nn-vector-v1";
  if(soft)result->numeric_semantics["axis"]=-1;
  else result->numeric_semantics["formula"]="gelu_tanh_f32_v1";
  return result;
}
std::unique_ptr<Instruction> softmax(Instruction raw,json src,json dst){return unary(std::move(raw),src,dst,true);}
std::unique_ptr<Instruction> gelu(Instruction raw,json src,json dst){return unary(std::move(raw),src,dst,false);}
}
