#include "ProgramModel.h"
#include <stdexcept>

namespace {
class ProgramOperation : public Operation {
  json program_;
  uint32_t batch_size_;
 public:
  ProgramOperation(SimulationConfig c, Model* model, json program, uint32_t batches, std::map<std::string,std::string>& attrs)
      : Operation(c,model,"SemanticProgram",attrs,0), program_(program), batch_size_(batches) { _optype="SemanticProgram"; }
  json semantic_header() const override { auto h=program_; h.erase("ops"); return h; }
  json semantic_execution() const override { return {{"batch_size",batch_size_},{"num_cores",_config.num_cores},{"placement","simple_round_robin_v1"}}; }
  void initialize_tiles(MappingTable&) override {
    if (_config.precision != 8 || program_.at("schema") != "structure-program-v1")
      throw std::runtime_error("generic adapter requires explicit precision=8 structure profile");
    if (!batch_size_ || batch_size_>384 || !_config.num_cores) throw std::runtime_error("invalid batch/core budget");
    uint64_t dram_stride=0;
    for(auto& [name,r]:program_.at("regions").items()) if(r.at("space")=="DRAM") {
      auto shape=r.at("shape");
      uint64_t bytes=shape.at(0).get<uint64_t>()*shape.at(1).get<uint64_t>()*8;
      dram_stride+=ceil_div(bytes,_config.dram_req_size)*_config.dram_req_size;
    }
    for(uint32_t batch=0;batch<batch_size_;++batch) {
    auto tile=std::make_unique<Tile>(Tile{.status=Tile::Status::INITIALIZED,.optype=_optype,.layer_id=_id,.batch=batch,.core_id=static_cast<int>(batch%_config.num_cores)});
    const std::string prefix=batch_size_==1?"":"batch"+std::to_string(batch)+".";
    if (program_.at("regions").size()>32 || program_.at("ops").empty() || program_.at("ops").size()>256)
      throw std::runtime_error("prototype program budget exceeded");
    std::map<std::string,addr_type> addresses;
    std::map<std::string,uint64_t> totals;
    addr_type dram=0x10000+batch*dram_stride,spad=SPAD_BASE,acc=ACCUM_SPAD_BASE;
    auto count=[&](const std::string& name) -> uint32_t {
      auto shape=program_.at("regions").at(name).at("shape");
      uint64_t bytes=shape.at(0).get<uint64_t>()*shape.at(1).get<uint64_t>()*8;
      if (!bytes || bytes>1024*1024) throw std::runtime_error("invalid region size");
      return ceil_div(bytes,_config.dram_req_size);
    };
    for (auto& [name,r] : program_.at("regions").items()) {
      auto space=r.at("space").get<std::string>();
      addr_type* next=space=="DRAM"?&dram:space=="SPAD"?&spad:&acc;
      auto bytes=count(name)*_config.dram_req_size;
      totals[space]+=bytes;
      if(totals[space]>65536) throw std::runtime_error("prototype space budget exceeded");
      addresses[name]=*next; *next+=bytes;
    }
    uint32_t barrier=0;
    for (auto& op:program_.at("ops")) {
      auto code=op.at("opcode").get<std::string>(), dst=op.at("dst").get<std::string>();
      auto shape=program_.at("regions").at(dst).at("shape");
      auto inst=std::make_unique<Instruction>(Instruction{});
      inst->id=prefix+op.at("id").get<std::string>(); inst->dest_addr=addresses.at(dst); inst->size=count(dst);
      inst->tile_m=shape.at(0);inst->tile_n=shape.at(1);inst->tile_k=shape.at(1);
      inst->my_tile=tile.get(); inst->base_addr=0; inst->numeric_semantics=op;
      auto src=op.at("src").get<std::vector<std::string>>();
      if (code=="LOAD" || code=="STORE") {
        auto dram_name=code=="LOAD"?src.at(0):dst;
        auto local_name=code=="LOAD"?dst:src.at(0);
        inst->opcode=code=="LOAD"?Opcode::MOVIN:Opcode::MOVOUT;
        inst->dest_addr=addresses.at(local_name);inst->size=count(local_name);
        for(uint32_t j=0;j<count(dram_name);++j) inst->src_addrs.push_back(addresses.at(dram_name)+j*_config.dram_req_size);
        inst->src_from_accum=addresses.at(local_name)>=ACCUM_SPAD_BASE;
      } else {
        if(code=="GEMM") {
          inst->opcode=Opcode::GEMM_PRELOAD;
          inst->tile_k=program_.at("regions").at(src.at(0)).at("shape").at(1);
          inst->compute_size=inst->tile_n;
        } else if(code=="ADD" || code=="SUB" || code=="MUL") {
          inst->opcode=code=="MUL"?Opcode::MUL:Opcode::ADD;
          inst->compute_size=inst->tile_m*inst->tile_n*8;
        } else throw std::runtime_error("unsupported generic Asim opcode");
        for(auto& s:src) inst->src_addrs.push_back(addresses.at(s));
        inst->src_from_accum=true;
      }
      auto written=inst->dest_addr;
      tile->instructions.push_back(std::move(inst));
      // Explicit functional scheduling: wait for previous write before next operation.
      if(code!="STORE") {
        auto sync=std::make_unique<Instruction>(Instruction{});
        sync->opcode=Opcode::PIPE_BARRIER;sync->id=prefix+"semantic_barrier_"+std::to_string(barrier++);
        sync->dest_addr=spad;spad+=_config.dram_req_size;sync->size=1;
        sync->src_addrs={written};sync->src_from_accum=true;sync->is_barrier=true;sync->my_tile=tile.get();
        tile->instructions.push_back(std::move(sync));
      }
    }
    if(tile->instructions.empty()) throw std::runtime_error("empty program");
    _tiles.push_back(std::move(tile));
    }
  }
};
}
void ProgramModel::initialize_model(std::vector<std::unique_ptr<Tensor>>&) {
  std::map<std::string,std::string> attrs;
  auto op=std::make_unique<ProgramOperation>(_config,this,_model_config.at("program"),_model_config.value("batch_size",1u),attrs);
  op->initialize_tiles(_mapping_table);
  _executable_layer.push_back(op.get());_operation_map[op->get_id()]=std::move(op);
}
