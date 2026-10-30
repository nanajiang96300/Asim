#pragma once
#include "../Common.h"
#include <cstdlib>
#include <fstream>
#include <mutex>
#include <stdexcept>
namespace rw_observer {
inline bool enabled(){static const bool on=std::getenv("ASIM_RW_EXPORT")!=nullptr;return on;}
inline void emit(json e,uint32_t core,cycle_type cycle){
  if(!enabled())return;
  static std::mutex mutex;std::lock_guard<std::mutex> lock(mutex);
  static std::ofstream stream(std::getenv("ASIM_RW_EXPORT"));
  static uint64_t order=0;
  if(!stream)throw std::runtime_error("RW observer export unavailable");
  if(order==0)stream<<json({{"kind","header"},{"schema","core-rw-v4"},{"producer","asim_core_observer_v4"},{"clock","homogeneous_core_cycle_v1"},{"compute_reads","conservative_issue_to_complete_v1"},{"dma_reads","request_snapshot_v1"},{"dma_writes","fragment_response_commit_v1"},{"dma_identity","trace_request_op_fragment_v1"},{"compute_identity","trace_issue_commit_cycle_v1"},{"dma_descriptor","per_fragment_address_v1"}}).dump()<<'\n';
  e["order"]=order++;e["core"]=core;e["cycle"]=cycle;stream<<e.dump()<<'\n';
  if(!stream)throw std::runtime_error("RW observer export failed");
}
inline json descriptor(const Instruction& i){
  return {{"id",i.id},{"opcode",int(i.opcode)},{"dest_addr",i.dest_addr},{"src_addrs",i.src_addrs},{"size",i.size},{"compute_size",i.compute_size},{"m",i.tile_m},{"k",i.tile_k},{"n",i.tile_n},{"spad_buffer",i.spad_id},{"accum_buffer",i.accum_spad_id},{"base_addr",i.base_addr},{"src_from_accum",i.src_from_accum},{"operand_id",i.operand_id}};
}
inline void compute(const char* kind,const Instruction& i,uint32_t core,cycle_type cycle){
  if(!enabled())return;json e=descriptor(i);e["kind"]=kind;
  if(std::string(kind)=="compute_read_begin")e["semantic"]=i.numeric_semantics;
  emit(e,core,cycle);
}
}
