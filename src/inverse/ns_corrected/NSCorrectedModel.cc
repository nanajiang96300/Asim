#include "NSCorrectedModel.h"
#include "operations/Operation.h"
#include "semantic/SourceViews.h"
#include "FormulaLogger.h"
#include <stdexcept>

namespace {
class NSCorrectedOperation : public Operation {
  uint32_t M,U,R,B,K;json task_;
 public:
  NSCorrectedOperation(SimulationConfig c,Model* m,uint32_t nr,uint32_t nt,uint32_t rhs,uint32_t batches,uint32_t k,json task,std::map<std::string,std::string>& attrs)
    :Operation(c,m,"NSCorrectedV1",attrs,0),M(2*nr),U(2*nt),R(2*rhs),B(batches),K(k),task_(task){_optype="NSCorrectedV1";}
  void initialize_tiles(MappingTable&) override {
    if(_config.precision!=8 || _config.dram_req_size!=64 || _config.num_cores!=24 || !B || B%24 || B>384 || !K || K>32 || !U || !R || M<U)
      throw std::runtime_error("NS corrected requires structure precision8/request64/core24 and valid dimensions");
    for(uint32_t b=0;b<B;++b){
      auto tile=std::make_unique<Tile>(Tile{.status=Tile::Status::INITIALIZED,.optype=_optype,.layer_id=_id,.batch=b,.core_id=static_cast<int>(b%24)});
      initialize_instructions(tile.get(),Mapping{});_tiles.push_back(std::move(tile));
    }
  }
 private:
  void initialize_instructions(Tile* tile,Mapping) override {
    addr_type next=SPAD_BASE;json regions=json::array();
    auto alloc=[&](const std::string& n,uint32_t rows,uint32_t cols){addr_type a=next;next+=ceil_div(uint64_t(rows)*cols*8,64)*64;regions.push_back(region(a,rows,cols,n));return a;};
    auto h=alloc("H",M,U),y=alloc("Y",M,R),reg=alloc("RegI",U,U),eye=alloc("Identity",U,U);
    auto a=alloc("A",U,U),rhs=alloc("rhs",U,R),x=alloc("X",U,U),two=alloc("twoI",U,U);
    auto t=alloc("T",U,U),negT=alloc("negT",U,U),res=alloc("residual",U,U);
    auto trace=alloc("trace",1,1),zero=alloc("zero",1,1),neg=alloc("negative_one",1,1);
    addr_type sync=next;next+=32768;regions.push_back(region(sync,1,4096,"sync"));
    addr_type out=ACCUM_SPAD_BASE;regions.push_back(region(out,U,R,"xhat"));
    record_source_regions(regions,M,U);_numeric_header["profile"]="structure_f64";_numeric_header["schema"]="source-views-program-v1";
    _numeric_header["task_contract"]=task_;_numeric_header["algorithm"]={{"id","ns_corrected_v1"},{"iterations",K},{"initialization","I/trace(real_block_A)"}};
    FormulaLogger::instance().set_algorithm("ns_corrected_v1",0,K,U);
    uint32_t serial=0;std::string prefix="batch"+std::to_string(tile->batch)+".";
    auto barrier=[&](addr_type written){
      auto i=source_instruction(Instruction{.opcode=Opcode::PIPE_BARRIER,.id=prefix+"sync"+std::to_string(serial++),.dest_addr=sync,.size=1,.src_addrs={written},.src_from_accum=true,.my_tile=tile,.is_barrier=true},json::array(),nullptr,"ns_corrected:sync");
      sync+=64;tile->instructions.push_back(std::move(i));
    };
    auto arithmetic=[&](Opcode code,const std::string& id,json src,json dst,bool broadcast=false,uint32_t inner=0){
      uint32_t rows=dst["rows"],cols=dst["cols"];bool scalar=code==Opcode::SCALAR_ADD || code==Opcode::SCALAR_SUB || code==Opcode::SCALAR_MUL;
      bool gemm=code==Opcode::GEMM_PRELOAD;
      Instruction raw{.opcode=code,.id=prefix+id,.dest_addr=dst["base"].get<addr_type>(),.size=uint32_t(ceil_div(uint64_t(rows)*cols*8,64)),
         .compute_size=scalar?1u:gemm?cols:rows*cols*8,.tile_m=rows,.tile_k=gemm?inner:cols,.tile_n=cols,.src_from_accum=true,.my_tile=tile};
      std::vector<std::string> names;std::vector<std::vector<uint32_t>> shapes;
      for(const auto& v:src){raw.src_addrs.push_back(v["base"]);names.push_back(std::to_string(v["base"].get<addr_type>()));uint32_t m=v["rows"],n=v["cols"];if(v["transpose"].get<bool>())std::swap(m,n);shapes.push_back({m,n});}
      auto inst=source_instruction(std::move(raw),src,dst,"ns_corrected:"+id);
      if(broadcast)inst->numeric_semantics["broadcast"]="scalar_rhs";
      FormulaLogger::instance().emit_step(prefix+id,gemm?"GEMM":"ELEMENTWISE",names,std::to_string(dst["base"].get<addr_type>()),shapes,{rows,cols},tile->batch,prefix+id);
      tile->instructions.push_back(std::move(inst));barrier(dst["base"]);
    };
    auto load=[&](uint32_t slot,addr_type local,uint32_t rows,uint32_t cols,bool batched){
      addr_type dram=get_operand_addr(_INPUT_OPERAND+slot)+(batched?uint64_t(tile->batch)*rows*cols*8:0);
      Instruction raw{.opcode=Opcode::MOVIN,.id=prefix+"load"+std::to_string(slot),.dest_addr=local,.size=uint32_t(ceil_div(uint64_t(rows)*cols*8,64)),.operand_id=_INPUT_OPERAND+slot,.base_addr=0,.tile_m=rows,.tile_k=cols,.my_tile=tile};
      for(uint32_t j=0;j<raw.size;++j)raw.src_addrs.push_back(dram+uint64_t(j)*64);
      tile->instructions.push_back(source_instruction(std::move(raw),json::array({view(dram,0,0,rows,cols)}),view(local,0,0,rows,cols),"ns_corrected:load"));barrier(local);
    };
    load(0,h,M,U,true);load(1,y,M,R,true);load(2,reg,U,U,false);load(3,eye,U,U,false);
    arithmetic(Opcode::GEMM_PRELOAD,"gram",json::array({view(h,0,0,M,U,true),view(h,0,0,M,U)}),view(a,0,0,U,U),false,M);
    arithmetic(Opcode::ADD,"regularize",json::array({view(a,0,0,U,U),view(reg,0,0,U,U)}),view(a,0,0,U,U));
    arithmetic(Opcode::GEMM_PRELOAD,"rhs",json::array({view(h,0,0,M,U,true),view(y,0,0,M,R)}),view(rhs,0,0,U,R),false,M);
    arithmetic(Opcode::SCALAR_MUL,"trace0",json::array({view(a,0,0),view(eye,0,0)}),view(trace,0,0));
    for(uint32_t d=1;d<U;++d)arithmetic(Opcode::SCALAR_ADD,"trace"+std::to_string(d),json::array({view(trace,0,0),view(a,d,d)}),view(trace,0,0));
    arithmetic(Opcode::SCALAR_SUB,"zero",json::array({view(eye,0,0),view(eye,0,0)}),view(zero,0,0));
    arithmetic(Opcode::SCALAR_SUB,"neg1",json::array({view(zero,0,0),view(eye,0,0)}),view(neg,0,0));
    arithmetic(Opcode::ADD,"twoI",json::array({view(eye,0,0,U,U),view(eye,0,0,U,U)}),view(two,0,0,U,U));
    arithmetic(Opcode::DIV,"initialize",json::array({view(eye,0,0,U,U),view(trace,0,0)}),view(x,0,0,U,U),true);
    for(uint32_t k=0;k<K;++k){
      std::string id="iter"+std::to_string(k)+".";
      arithmetic(Opcode::GEMM_PRELOAD,id+"AX",json::array({view(a,0,0,U,U),view(x,0,0,U,U)}),view(t,0,0,U,U),false,U);
      arithmetic(Opcode::MUL,id+"negate",json::array({view(t,0,0,U,U),view(neg,0,0)}),view(negT,0,0,U,U),true);
      arithmetic(Opcode::ADD,id+"residual",json::array({view(two,0,0,U,U),view(negT,0,0,U,U)}),view(res,0,0,U,U));
      arithmetic(Opcode::GEMM_PRELOAD,id+"update",json::array({view(x,0,0,U,U),view(res,0,0,U,U)}),view(x,0,0,U,U),false,U);
    }
    arithmetic(Opcode::GEMM_PRELOAD,"apply",json::array({view(x,0,0,U,U),view(rhs,0,0,U,R)}),view(out,0,0,U,R),false,U);
    addr_type dram=get_operand_addr(_OUTPUT_OPERAND)+uint64_t(tile->batch)*U*R*8;
    Instruction store{.opcode=Opcode::MOVOUT,.id=prefix+"store",.dest_addr=out,.size=uint32_t(ceil_div(uint64_t(U)*R*8,64)),.operand_id=_OUTPUT_OPERAND,.base_addr=0,.tile_m=U,.tile_k=R,.src_from_accum=true,.last_inst=true,.my_tile=tile};
    for(uint32_t j=0;j<store.size;++j)store.src_addrs.push_back(dram+uint64_t(j)*64);
    tile->instructions.push_back(source_instruction(std::move(store),json::array({view(out,0,0,U,R)}),view(dram,0,0,U,R),"ns_corrected:store"));
  }
};
}
void NSCorrectedModel::initialize_model(std::vector<std::unique_ptr<Tensor>>&) {
  uint32_t M=_model_config.value("matrix_m",64u),U=_model_config.value("matrix_k",16u),R=_model_config.value("nrhs",8u),B=_model_config.value("batch_size",96u),K=_model_config.value("iterations",16u);
  std::map<std::string,std::string> attrs;
  auto op=std::make_unique<NSCorrectedOperation>(_config,this,M,U,R,B,K,_model_config.at("task_contract"),attrs);
  for(auto entry:std::vector<std::pair<std::string,std::vector<uint32_t>>>{{"H",{B,2*M,2*U}},{"Y",{B,2*M,2*R}},{"RegI",{2*U,2*U}},{"Identity",{2*U,2*U}},{"xhat",{B,2*U,2*R}}}){
    bool input=entry.first!="xhat";auto tensor=std::make_unique<Tensor>(get_root_node_id(),name_gen(get_name(),entry.first),entry.second,_config.precision,input);
    uint32_t id=tensor->get_id();if(input)tensor->set_produced();add_tensor(std::move(tensor));if(input)op->add_input(id);else op->add_output(id);
  }
  op->initialize_tiles(_mapping_table);_executable_layer.push_back(op.get());_operation_map[op->get_id()]=std::move(op);
}
