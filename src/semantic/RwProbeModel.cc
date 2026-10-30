#include "RwProbeModel.h"
#include "../operations/Operation.h"
#include "SourceViews.h"
#include "../FormulaLogger.h"
namespace {
class RwProbeOp : public Operation {
  std::string scenario_;
 public:
  RwProbeOp(SimulationConfig c,Model* m,std::map<std::string,std::string>& attrs,std::string scenario):Operation(c,m,"RWProbe",attrs,0),scenario_(scenario){_optype="RWProbe";}
  void initialize_tiles(MappingTable&) override {
    if(_config.precision!=8 || _config.dram_req_size!=64 || _config.num_cores!=1)throw std::runtime_error("probe requires core1/precision8/request64");
    if(scenario_!="serialized" && scenario_!="independent" && scenario_!="missing_barrier" && scenario_!="overwrite_input")throw std::runtime_error("unknown RW probe");
    auto tile=std::make_unique<Tile>(Tile{.status=Tile::Status::INITIALIZED,.optype=_optype,.layer_id=_id,.batch=0,.core_id=0});
    initialize_instructions(tile.get(),Mapping{});_tiles.push_back(std::move(tile));
  }
 private:
  void initialize_instructions(Tile* tile,Mapping) override {
    const addr_type a=SPAD_BASE,b=a+2048,c=ACCUM_SPAD_BASE,d=c+2048,sync=b+2048;
    record_source_regions(json::array({region(a,16,16,"A"),region(b,16,16,"B"),region(c,16,16,"C"),region(d,16,16,"D"),region(sync,1,8,"sync")}),16,16);
    _numeric_header["schema"]="source-views-program-v1";_numeric_header["profile"]="structure_f64";
    FormulaLogger::instance().set_algorithm("rw_probe");
    for(uint32_t k=0;k<2;++k){addr_type remote=get_operand_addr(_INPUT_OPERAND+k),local=k?b:a;
      Instruction raw{.opcode=Opcode::MOVIN,.id="load"+std::to_string(k),.dest_addr=local,.size=32,.operand_id=_INPUT_OPERAND+k,.base_addr=0,.tile_m=16,.tile_k=16,.my_tile=tile};
      for(uint32_t j=0;j<32;++j)raw.src_addrs.push_back(remote+j*64);
      tile->instructions.push_back(source_instruction(std::move(raw),json::array({view(remote,0,0,16,16)}),view(local,0,0,16,16),"rw_probe:load"));
    }
    tile->instructions.push_back(source_instruction(Instruction{.opcode=Opcode::GEMM_PRELOAD,.id="cube",.dest_addr=c,.size=32,.compute_size=16,.src_addrs={a,b},.tile_m=16,.tile_k=16,.tile_n=16,.src_from_accum=true,.my_tile=tile},json::array({view(a,0,0,16,16),view(b,0,0,16,16)}),view(c,0,0,16,16),"rw_probe:cube"));
    FormulaLogger::instance().emit_step("cube","GEMM",{"A","B"},"C",{{16,16},{16,16}},{16,16},0,"cube");
    if(scenario_=="serialized")tile->instructions.push_back(source_instruction(Instruction{.opcode=Opcode::PIPE_BARRIER,.id="sync",.dest_addr=sync,.size=1,.src_addrs={c},.src_from_accum=true,.my_tile=tile,.is_barrier=true},json::array(),nullptr,"rw_probe:sync"));
    addr_type target=scenario_=="independent"?d:scenario_=="overwrite_input"?a:c;
    tile->instructions.push_back(source_instruction(Instruction{.opcode=Opcode::DIV,.id="vector",.dest_addr=target,.size=32,.compute_size=2048,.src_addrs={a,a},.tile_m=16,.tile_k=16,.tile_n=16,.src_from_accum=true,.my_tile=tile},json::array({view(a,0,0,16,16),view(a,0,0,16,16)}),view(target,0,0,16,16),"rw_probe:vector"));
    FormulaLogger::instance().emit_step("vector","DIV",{"A","A"},scenario_=="independent"?"D":scenario_=="overwrite_input"?"A":"C",{{16,16},{16,16}},{16,16},0,"vector");
    addr_type remote=get_operand_addr(_OUTPUT_OPERAND);
    Instruction raw{.opcode=Opcode::MOVOUT,.id="store",.dest_addr=c,.size=32,.operand_id=_OUTPUT_OPERAND,.base_addr=0,.tile_m=16,.tile_k=16,.src_from_accum=true,.last_inst=true,.my_tile=tile};
    for(uint32_t j=0;j<32;++j)raw.src_addrs.push_back(remote+j*64);
    tile->instructions.push_back(source_instruction(std::move(raw),json::array({view(c,0,0,16,16)}),view(remote,0,0,16,16),"rw_probe:store"));
  }
};
}
void RwProbeModel::initialize_model(std::vector<std::unique_ptr<Tensor>>&) {
  std::map<std::string,std::string> attrs;auto op=std::make_unique<RwProbeOp>(_config,this,attrs,_model_config.at("scenario"));
  for(auto name:{"A","B","C"}){
    bool input=std::string(name)!="C";std::vector<uint32_t> shape{1,16,16};
    auto tensor=std::make_unique<Tensor>(get_root_node_id(),name_gen(get_name(),name),shape,_config.precision,input);auto id=tensor->get_id();if(input)tensor->set_produced();add_tensor(std::move(tensor));if(input)op->add_input(id);else op->add_output(id);
  }
  op->initialize_tiles(_mapping_table);_executable_layer.push_back(op.get());_operation_map[op->get_id()]=std::move(op);
}
