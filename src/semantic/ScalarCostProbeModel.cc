#include "ScalarCostProbeModel.h"
#include "operations/Operation.h"
#include "SourceViews.h"
#include "FormulaLogger.h"
#include <stdexcept>
namespace {
class ScalarProbeOp : public Operation {
 uint32_t width_;
 public:
 ScalarProbeOp(SimulationConfig c,Model* m,uint32_t w,std::map<std::string,std::string>& a):Operation(c,m,"ScalarCostProbe",a,0),width_(w){_optype="ScalarCostProbe";}
 void initialize_tiles(MappingTable&) override {
   auto t=std::make_unique<Tile>(Tile{.status=Tile::Status::INITIALIZED,.optype="ScalarCostProbe",.layer_id=_id,.batch=0,.core_id=0});initialize_instructions(t.get(),Mapping{});_tiles.push_back(std::move(t));
 }
 private:
 void initialize_instructions(Tile* t,Mapping) override {
   uint32_t p=_config.precision,n=64/p;addr_type in=SPAD_BASE,out=ACCUM_SPAD_BASE,sync=in+64;
   record_source_regions(json::array({region(in,1,n,"input"),region(out,1,n,"output"),region(sync,1,256/p,"sync")}),1,n);
   _numeric_header["schema"]="source-views-probe-v1";_numeric_header["diagnostic_only"]=true;_numeric_header["declared_compute_size"]=width_;
   FormulaLogger::instance().set_algorithm("scalar_cost_probe",1,0,1);
   uint32_t serial=0;
   auto barrier=[&](addr_type addr){t->instructions.push_back(source_instruction(Instruction{.opcode=Opcode::PIPE_BARRIER,.id="probe.sync"+std::to_string(serial++),.dest_addr=sync,.size=1,.src_addrs={addr},.src_from_accum=true,.my_tile=t,.is_barrier=true},json::array(),nullptr,"scalar_cost_probe:barrier"));sync+=64;};
   addr_type dram=get_operand_addr(_INPUT_OPERAND);
   t->instructions.push_back(source_instruction(Instruction{.opcode=Opcode::MOVIN,.id="probe.load",.dest_addr=in,.size=1,.src_addrs={dram},.operand_id=_INPUT_OPERAND,.base_addr=0,.tile_m=1,.tile_k=n,.my_tile=t},json::array({view(dram,0,0,1,n)}),view(in,0,0,1,n),"scalar_cost_probe:load"));barrier(in);
   t->instructions.push_back(source_instruction(Instruction{.opcode=Opcode::ADD,.id="probe.init",.dest_addr=out,.size=1,.compute_size=n*4,.src_addrs={in,in},.tile_m=1,.tile_k=n,.tile_n=n,.src_from_accum=true,.my_tile=t},json::array({view(in,0,0,1,n),view(in,0,0,1,n)}),view(out,0,0,1,n),"scalar_cost_probe:init"));
   FormulaLogger::instance().emit_step("probe.init","ELEMENTWISE",{"input","input"},"output",{{1,n},{1,n}},{1,n},0,"probe.init");barrier(out);
   // width16 is deliberately not a valid scalar workload contract.
   t->instructions.push_back(source_instruction(Instruction{.opcode=Opcode::SCALAR_MUL,.id="probe.scalar",.dest_addr=out,.size=1,.compute_size=width_,.src_addrs={in,in},.tile_m=1,.tile_k=1,.tile_n=1,.src_from_accum=true,.my_tile=t},json::array({view(in,0,0),view(in,0,1)}),view(out,0,0),"scalar_cost_probe:scalar"));barrier(out);
   FormulaLogger::instance().emit_step("probe.scalar","ELEMENTWISE",{"input","input"},"output",{{1,1},{1,1}},{1,1},0,"probe.scalar");
   addr_type target=get_operand_addr(_OUTPUT_OPERAND);
   t->instructions.push_back(source_instruction(Instruction{.opcode=Opcode::MOVOUT,.id="probe.store",.dest_addr=out,.size=1,.src_addrs={target},.operand_id=_OUTPUT_OPERAND,.base_addr=0,.tile_m=1,.tile_k=n,.src_from_accum=true,.last_inst=true,.my_tile=t},json::array({view(out,0,0,1,n)}),view(target,0,0,1,n),"scalar_cost_probe:store"));
 }
};
}
void ScalarCostProbeModel::initialize_model(std::vector<std::unique_ptr<Tensor>>&) {
 uint32_t width=_model_config.at("width");if((width!=1 && width!=16) || _config.precision!=2)throw std::runtime_error("probe supports precision2 and width1/16 only");
 std::map<std::string,std::string> attrs;auto op=std::make_unique<ScalarProbeOp>(_config,this,width,attrs);
 std::vector<uint32_t> shape{1,32};
 for(bool input:{true,false}){auto tensor=std::make_unique<Tensor>(get_root_node_id(),name_gen(get_name(),input?"input":"output"),shape,_config.precision,input);auto id=tensor->get_id();if(input)tensor->set_produced();add_tensor(std::move(tensor));if(input)op->add_input(id);else op->add_output(id);}
 op->initialize_tiles(_mapping_table);_executable_layer.push_back(op.get());_operation_map[op->get_id()]=std::move(op);
}
